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
 * \page sos_part_query Query a Container's Partitions
 *
 * sos_part_query -C <container> [-v]
 *
 * List the partitions defined in a Container.
 *
 * @param "-C PATH" Specify the PATH to the Container. This option is required.
 * @param -v Show all partitions including offline partitions
 *
 * Example
 *
 *      sos_part_query -C /NVME/0/SOS_ROOT/Test
 *      Partition Name       RefCount Status           Size     Modified         Accessed         Path
 *      -------------------- -------- ---------------- -------- ---------------- ---------------- ----------------
 *      00000000                    3 ACTIVE                 1M 2015/08/25 13:49 2015/08/25 13:51 /SOS_STAGING/Test
 *      00000001                    3 ACTIVE                 2M 2015/08/25 11:54 2015/08/25 13:51 /NVME/0/SOS_ROOT/Test
 *      00000002                    3 ACTIVE                 2M 2015/08/25 11:39 2015/08/25 13:51 /NVME/0/SOS_ROOT/Test
 *      00000003                    3 PRIMARY                2M 2015/08/25 11:39 2015/08/25 13:51 /NVME/0/SOS_ROOT/Test
 */
#define _GNU_SOURCE
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <sos/sos.h>

int verbose;

void usage(int argc, char *argv[])
{
	printf("sos_part_query [-ov] -C <path>\n");
	printf("    -C <path> The path to the container. Required for all options.\n");
	printf("    -o        Show all objects in the partition.\n");
	printf("    -v        Print offline partition information.\n");
	exit(1);
}

const char *pretty_file_size(off_t size)
{
	int i;
	static char buf[32];
	const char sz_strs[] = { ' ', 'K', 'M', 'G', 'T' };
	size_t str_cnt = sizeof(sz_strs) / sizeof(sz_strs[0]);

	for (i = 0; i < str_cnt; i++) {
		if (size < 1000 || str_cnt == 4) {
			sprintf(buf, "%ld%c", size, sz_strs[i]);
			return buf;
		}
		size = size / 1000;
	}
	return NULL;		/* NB: can't happen */
}

static int print_obj_cb(sos_part_t part, ods_obj_t ods_obj, sos_obj_t sos_obj, void *arg)
{
	sos_schema_t schema = sos_obj_schema(sos_obj);
	sos_obj_ref_str_t ref_str;
	sos_obj_ref_t ref = sos_obj_ref(sos_obj);
	sos_obj_ref_to_str(ref, ref_str);
	printf("%s %-16s\n", ref_str, sos_schema_name(schema));
	sos_obj_put(sos_obj);
	return 0;
}

void print_objects(sos_part_t part)
{
	(void)sos_part_obj_iter(part, NULL, print_obj_cb, NULL);
}

void print_partitions(sos_t sos, FILE *fp, char *part_name, int show_objects)
{
	struct sos_part_stat_s sb;
	sos_part_t part;
	sos_part_iter_t iter;

	if (!fp)
		fp = stdout;

	iter = sos_part_iter_new(sos);
	if (!iter)
		return;

	fprintf(fp, "%-20s %-6s\n", "", "Use");
	fprintf(fp, "%-20s %-6s %-8s %-8s %s\n",
		"Partition Name", "Count",
		"State", "Size", "Path");
	fprintf(fp, "-------------------- ------ -------- "
		"-------- ----------------\n");
	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {
		if (part_name) {
			if (0 != strcmp(sos_part_name(part), part_name))
				continue;
		}
		char *statestr;
		sos_part_state_t state = sos_part_state(part);
		if (state == SOS_PART_STATE_OFFLINE && !verbose)
			continue;
		char uuid_str[37];
		uuid_t uuid;
		sos_part_uuid(part, uuid);
		uuid_unparse_lower(uuid, uuid_str);
		fprintf(fp, "%-20s %6d ", sos_part_name(part), sos_part_refcount(part));
		switch (state) {
		case SOS_PART_STATE_OFFLINE:
			statestr = "OFFLINE";
			break;
		case SOS_PART_STATE_ACTIVE:
			statestr = "ACTIVE";
			break;
		case SOS_PART_STATE_PRIMARY:
			statestr = "PRIMARY";
			break;
		case SOS_PART_STATE_BUSY:
			statestr = "BUSY";
			break;
		default:
			statestr = "!UNKNOWN!";
		}
		fprintf(fp, "%-8s ", statestr);

		int rc = sos_part_stat(part, &sb);
		char datestr[80];
		struct tm *tm;

		if (!rc) {
			fprintf(fp, "%8s ", pretty_file_size(sb.size));
		} else
			fprintf(fp, "%8s ", "");
		fprintf(fp, "%s\n", sos_part_path(part));
		if (verbose) {
			fprintf(fp, "\tDescription : %s\n", sos_part_desc(part));
			fprintf(fp, "\tUUID        : %s\n", uuid_str);
			tm = localtime((const time_t *)&sb.modified);
			strftime(datestr, sizeof(datestr), "%Y/%m/%d %H:%M", tm);
			fprintf(fp, "\tModified    : %-16s\n", datestr);

			tm = localtime((const time_t *)&sb.accessed);
			strftime(datestr, sizeof(datestr), "%Y/%m/%d %H:%M", tm);
			fprintf(fp, "\tAccessed    : %-16s\n", datestr);
		}
		if (show_objects) {
			print_objects(part);
		}
		sos_part_put(part);
	}
	sos_part_iter_free(iter);
}

const char *short_options = "C:vo";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"object",      required_argument,  0,  'o'},
	{"verbose",     no_argument,        0,  'v'},
	{0,             0,                  0,  0}
};

int main(int argc, char **argv)
{
	sos_t sos;
	int opt;
	int show_objects = 0;
	char *part_name = NULL;
	char *path = NULL;
	while (0 < (opt = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (opt) {
		case 'C':
			path = strdup(optarg);
			break;
		case 'o':
			show_objects = 1;
			break;
		case 'v':
			verbose = 1;
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}

	if (!path)
		usage(argc, argv);

	if (optind < argc)
		part_name = strdup(argv[optind]);

	sos = sos_container_open(path, SOS_PERM_RD);
	if (!sos) {
		printf("Error %d opening the container %s.\n",
		       errno, argv[1]);
		exit(1);
	}
	print_partitions(sos, stdout, part_name, show_objects);
	sos_container_close(sos, SOS_COMMIT_ASYNC);
	return 0;
}
