/*
 * Copyright (c) 2013 Open Grid Computing, Inc. All rights reserved.
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

/*
 * Author: Tom Tucker tom at ogc dot us
 */
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <string.h>
#include <signal.h>
#include <assert.h>
#include <ods/ods_idx.h>

void iter_tree(ods_idx_t idx)
{
	ods_key_t key;
	ods_ref_t ref;
	ods_iter_t i = ods_iter_new(idx);
	int rc;
	char *keystr = malloc(ods_idx_key_str_size(idx));
	for (rc = ods_iter_begin(i); !rc; rc = ods_iter_next(i)) {
		key = ods_iter_key(i);
		ref = ods_iter_ref(i);
		printf("%s, ", ods_key_to_str(idx, key, keystr));
		rc = ods_idx_find(idx, key, &ref);
		ods_obj_put(key);
		assert(ref);
	}
	free(keystr);
	printf("\n");
	ods_iter_delete(i);
}

void usage(int argc, char *argv[])
{
	printf("usage: %s -p <path> -k <key_str> [-o <order>]\n"
	       "       -x <idx_type>   The index type.\n"
	       "       -k <key_str>    The key type string.\n"
	       "       -C <count>      Count of random keys to insert.\n"
	       "       -p <path>       The path to the index files.\n"
	       "       -o <order>      The order of the B+ tree (default is 5).\n"
	       "       -R              Assume the index is already created.\n",
	       argv[0]);
	exit(1);
}

const char *nlstrip(char *s)
{
	static char ss[80];
	strcpy(ss, s);
	strtok(ss, " \t\n");
}

static int print_it = 0;
void print_info(int signal, siginfo_t *info, void *arg)
{
	print_it = 1;
}

void setup_signals(void)
{
	struct sigaction action, logrotate_act;
	sigset_t sigset;
	sigemptyset(&sigset);
	sigaddset(&sigset, SIGINT);

	memset(&action, 0, sizeof(action));
	action.sa_sigaction = print_info;
	action.sa_flags = SA_SIGINFO;
	action.sa_mask = sigset;

	sigaction(SIGINT, &action, NULL);
}

#define FMT "k:p:o:x:RC:"
int main(int argc, char *argv[])
{
	ods_idx_t idx;
	int order = 5;
	uint64_t count;
	uint64_t randval;
	char *idx_path = NULL;
	char *key_str = NULL;
	char *idx_name = NULL;
	ods_key_t key;
	int rc;
	int create = 1;
	while ((rc = getopt(argc, argv, FMT)) > 0) {
		switch (rc) {
		case 'k':
			key_str = strdup(optarg);
			break;
		case 'p':
			idx_path = strdup(optarg);
			break;
		case 'o':
			order = atoi(optarg);
			break;
		case 'x':
			idx_name = strdup(optarg);
			break;
		case 'R':
			create = 0;
			break;
		case 'C':
			count = strtoul(optarg, NULL, 0);
			break;
		default:
			usage(argc, argv);
		}
	}
	if (!idx_name)
		idx_name = "BPTREE";

	if (!idx_path || !key_str)
		usage(argc, argv);

	if (!create)
		goto skip_create;

	rc = ods_idx_create(idx_path, 0660, idx_name, key_str, order);
	if (rc) {
		printf("The index '%s' of type '%s' could not be created due "
		       "to error %d.\n",
		       idx_path, idx_name, rc);
		return rc;
	}

 skip_create:
	idx = ods_idx_open(idx_path, ODS_PERM_RW);
	setup_signals();
	srandom(time(NULL));
	key = ods_key_malloc(sizeof(uint64_t));
	while (count) {
		randval = (uint64_t)random();
		ods_key_set(key, &randval, sizeof(randval));
		int rc = ods_idx_insert(idx, key, randval);
		assert(rc == 0);
		count--;
		if (print_it) {
			printf("------------------------------------------------------------\n");
			ods_idx_info(idx, stdout);
			print_it = 0;
		}
		if ((count % 1000) == 0)
			ods_idx_commit(idx, ODS_COMMIT_ASYNC);
	}
	ods_idx_info(idx, stdout);
	ods_obj_put(key);
 out:
	ods_info(ods_idx_ods(idx), stdout);
	ods_idx_close(idx, ODS_COMMIT_SYNC);
	return 0;
}
