/*
 * Copyright (c) 2015 Open Grid Computing, Inc. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the BSD-type
 * license below:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *      Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *      Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *      Neither the name of Open Grid Computing nor the names of any
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *      Modified source versions must be plainly marked as such, and
 *      must not be misrepresented as being the original software.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \page sos_part_migrate Migrate the object schema uuid from one directory to another
 *
 * sos_part_migrate -p <path> -s <src-dir> -d <dst-dir>
 *
 */
#define _GNU_SOURCE
#include <sys/time.h>
#include <limits.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <ftw.h>
#include <uuid/uuid.h>
#include <assert.h>
#include <pthread.h>
#include <sos/sos.h>
#include <ods/ods_rbt.h>
#include "sos_priv.h"
const char *short_options = "p:s:d:P:v";
int verbose = 0;
long progress = 0;
struct option long_options[] = {
	{"help",        no_argument,       0, '?'},
	{"path",	required_argument, 0, 'p'},
	{"src-dir",	required_argument, 0, 's'},
	{"dst-dir",	required_argument, 0, 'd'},
	{"progress",	required_argument, 0, 'P'},
	{"verbose",	no_argument,	   0, 'v'},

	{0,		0,		   0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_part_migrate -p <path> -s <dir> -d <dir>\n");
	printf("    -p <path>	The partition path.\n");
	printf("    -s <dir>	The source schema directory path.\n");
	printf("    -d <dir>	The destination schema directory path.\n");
	printf("    -P <secs>   Show progress information every <secs> seconds.\n");
	printf("    -v          Show modified objects\n.");
	exit(1);
}

struct ods_rbt uuid_rbt;	/* schema indexed by UUID */
struct ods_rbt name_rbt;	/* schema indexed by name */
struct export_obj_iter_args_s {
	sos_part_t part;		/* the partition */
	char *name;			/* the schema name */
	char *src_dir;		/* directory containing src templates */
	char *dst_dir;		/* directory containing dst templates */
	int64_t visited_count;
	int64_t missing_mapping_count;
	int64_t unchanged_count;
	int64_t updated_count;
};

struct schema_entry {
	char *name;
	char *src_uuid_str;
	uuid_t src_uuid;
	char *dst_uuid_str;
	uuid_t dst_uuid;
	struct ods_rbn uuid_rbn;
	struct ods_rbn name_rbn;
};

static int __export_callback_fn(sos_part_t part, ods_obj_t ods_obj, sos_obj_t sos_obj, void *arg)
{
	struct export_obj_iter_args_s *uarg = arg;
	char uuid_str[48];
	struct ods_rbn *rbn;
	struct schema_entry *entry;

	uuid_unparse(SOS_OBJ(ods_obj)->schema_uuid, uuid_str);
	rbn = ods_rbt_find(&uuid_rbt, SOS_OBJ(ods_obj)->schema_uuid);
	if (!rbn) {
		if (verbose)
			printf("Ignoring object schema '%s' without mapping data.\n", uuid_str);
		uarg->missing_mapping_count += 1;
		uarg->unchanged_count += 1;
		goto out;
	}
	entry = container_of(rbn, struct schema_entry, uuid_rbn);
	if (0 == uuid_compare(SOS_OBJ(ods_obj)->schema_uuid, entry->dst_uuid)) {
		if (verbose)
			printf("Source and destination uuid '%s:%s' are identical.\n",
				entry->name, entry->dst_uuid_str);
		uarg->unchanged_count += 1;
		goto out;
	}
	uuid_copy(SOS_OBJ(ods_obj)->schema_uuid, entry->dst_uuid);
	ods_obj_update(ods_obj);
	uarg->updated_count += 1;
	if (verbose)
		printf("%s --> %s\n", entry->name, entry->dst_uuid_str);
out:
        ods_obj_put(ods_obj);
        uarg->visited_count += 1;
	return 0;
}

int64_t sos_part_migrate(sos_part_t part, void *args)
{
	int rc = sos_part_obj_iter(part, NULL, __export_callback_fn, args);
	sos_part_close(part);
	return rc;
}

int64_t uuid_comparator(void *tree_key, const void *key, void *arg)
{
	return uuid_compare(tree_key, key);
}

int64_t name_comparator(void *tree_key, const void *key, void *arg)
{
	return strcmp(tree_key, key);
}

int src_phs_dir_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
{
	struct schema_entry *entry;
	struct ods_rbn *rbn;
	uuid_t uuid;
	const char *uuid_str = &fpath[ftwbuf->base];

	if (FTW_F != typeflag)
		return 0;
	/* See if the uuid is already in the tree */
	if (uuid_parse(uuid_str, uuid)) {
		printf("Ignoring file-name '%s' with invalid uuid", fpath);
		return 0;
	}
	rbn = ods_rbt_find(&uuid_rbt, (void *)uuid);
	if (rbn)
		assert(0 == "entry already present");
	entry = calloc(1, sizeof *entry);
	entry->src_uuid_str = strdup(uuid_str);
	uuid_copy(entry->src_uuid, uuid);
	ods_rbn_init(&entry->uuid_rbn, entry->src_uuid);
	ods_rbt_ins(&uuid_rbt, &entry->uuid_rbn);
	return 0;
}

int src_sl_dir_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
{
	char link_path[PATH_MAX];
	struct schema_entry *entry;
	struct ods_rbn *rbn;
	ssize_t sz;
	uuid_t uuid;
	char *uuid_str;

	if (FTW_SL != typeflag)
		return 0;

	sz = readlink(fpath, link_path, sizeof(link_path));
	if (sz < 0) {
		printf("Ignoring symbolic link '%s' that could not be read\n", fpath);
		return 0;
	}
	link_path[sz] = '\0';
	uuid_str = basename(link_path);
	int rc = uuid_parse(uuid_str, uuid);
	if (rc) {
		printf("Ignoring symbolic link '%s' pointing to file-name '%s', with invalid uuid",
			fpath, uuid_str);
		return 0;
	}
	rbn = ods_rbt_find(&uuid_rbt, uuid);
	if (!rbn)
		assert(NULL == "Schema entry missing from uuid tree.");
	entry = container_of(rbn, struct schema_entry, uuid_rbn);
	strcpy(link_path, fpath);
	entry->name = strdup(basename(link_path));
	ods_rbn_init(&entry->name_rbn, entry->name);
	ods_rbt_ins(&name_rbt, &entry->name_rbn);
	return 0;
}

int dst_sl_dir_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
{
	char link_path[PATH_MAX];
	char *uuid_str;
	struct schema_entry *entry;
	struct ods_rbn *rbn;
	ssize_t sz;
	uuid_t uuid;

	if (FTW_SL != typeflag)
		return 0;

	sz = readlink(fpath, link_path, sizeof(link_path));
	if (sz < 0) {
		printf("Ignoring symbolic link '%s' that could not be read\n", fpath);
		return 0;
	}
	link_path[sz] = '\0';
	uuid_str = basename(link_path);
	int rc = uuid_parse(uuid_str, uuid);
	if (rc) {
		printf("Ignoring symbolic link '%s' pointing to file-name '%s', with invalid uuid",
			fpath, uuid_str);
		return 0;
	}
	char *name = strdup(fpath);
	rbn = ods_rbt_find(&name_rbt, basename(name));
	free(name);
	if (!rbn)
		assert(NULL == "Schema entry missing from uuid tree.");
	entry = container_of(rbn, struct schema_entry, name_rbn);
	entry->dst_uuid_str = strdup(uuid_str);
	uuid_copy(entry->dst_uuid, uuid);
	return 0;
}

void *status_fn(void *arg)
{
	struct export_obj_iter_args_s *uargs = arg;
	long duration = 0;
	if (!progress)
		return NULL;
	while (1) {
		sleep(progress);
		duration += progress;
		printf("%ld objects visited/s.\n", uargs->visited_count / duration);
	}
	return NULL;
}

int main(int argc, char **argv)
{
	pthread_t status_thread;
	struct export_obj_iter_args_s uargs;
	char *part_path = NULL;
	char *src_dir = NULL;
	char *dst_dir = NULL;
	int o, rc;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'p':
			part_path = strdup(optarg);
			break;
		case 's':
			src_dir = strdup(optarg);
			break;
		case 'd':
			dst_dir = strdup(optarg);
			break;
		case 'v':
			verbose = 1;
			break;
		case 'P':
			progress = strtol(optarg, NULL, 0);
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (part_path == NULL)
		usage(argc, argv);
	sos_part_t part = sos_part_open(part_path, SOS_PERM_RW);
	if (!part)
		exit(1);
	memset(&uargs, 0, sizeof(uargs));
	uargs.src_dir = src_dir;
	uargs.dst_dir = dst_dir;
	ods_rbt_init(&uuid_rbt, uuid_comparator, NULL);
	ods_rbt_init(&name_rbt, name_comparator, NULL);

	rc = pthread_create(&status_thread, NULL, status_fn, &uargs);

	/* Add schema entries for all the UUID files */
	rc = nftw(src_dir, src_phs_dir_cb, 1024, FTW_PHYS);

	/* Update the schema entries based on the symbolic links */
	rc = nftw(src_dir, src_sl_dir_cb, 1024, FTW_PHYS);

	/* Update the schema entries based on the symbolic links */
	rc = nftw(dst_dir, dst_sl_dir_cb, 1024, FTW_PHYS);

        rc = sos_part_migrate(part, &uargs);
        if (rc) {
                printf("Error %d exporting the partition's objects.\n", rc);
        }
        printf("%ld objects visited.\n", uargs.visited_count);
        printf("%ld objects were missing a schema mapping.\n", uargs.missing_mapping_count);
        printf("%ld objects were unchanged.\n", uargs.unchanged_count);
        printf("%ld objects were updated.\n", uargs.updated_count);
        free(part_path);
	return rc;
}
