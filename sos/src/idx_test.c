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
#include <assert.h>
#include "bpt.h"

void iter_tree(ods_idx_t idx)
{
	ods_key_t key;
	ods_ref_t ref;
	ods_iter_t i = ods_iter_new(idx);
	int rc;
	for (rc = ods_iter_begin(i); !rc; rc = ods_iter_next(i)) {
		key = ods_iter_key(i);
		ref = ods_iter_ref(i);
		printf("%s, ", ods_key_to_str(idx, key));
		rc = ods_idx_find(idx, key, &ref);
		ods_obj_put(key);
		assert(ref);
	}
	printf("\n");
	ods_iter_delete(i);
}

void usage(int argc, char *argv[])
{
	printf("usage: %s -p <path> -k <key_str> [-o <order>]\n"
	       "       -x <idx_type>   The index type.\n"
	       "       -k <key_str>    The key type string.\n"
	       "       -p <path>       The path to the index files.\n"
	       "       -i <iter_key>   The key value to test iterators.\n"
	       "       -o <order>      The order of the B+ tree (default is 5).\n",
	       argv[0]);
	exit(1);
}

const char *nlstrip(char *s)
{
	static char ss[80];
	strcpy(ss, s);
	strtok(ss, " \t\n");
}

void rebuild_tree(ods_idx_t idx)
{
	ods_key_t key;
	char *s;
	char buf[2048];
	fseek(stdin, 0, SEEK_SET);
	while ((s = fgets(buf, sizeof(buf), stdin)) != NULL) {
		long inode = strtoul(s, NULL, 0);
		if (!inode) {
			printf("Ignoring key that results in <nil> object reference.\n");
			continue;
		}
		size_t key_sz = ods_idx_key_size(idx);
		if (key_sz == -1)
			key_sz = strlen(s);
		key = ods_key_malloc(idx, key_sz);
		ods_key_from_str(idx, key, nlstrip(s));
		int rc = ods_idx_insert(idx, key, inode);
		assert(rc == 0);
		ods_obj_put(key);
	}
}

#define FMT "k:p:o:x:i:"
int main(int argc, char *argv[])
{
	ods_idx_t idx;
	int order = 5;
	char *idx_path = NULL;
	char *key_str = NULL;
	char *iter_key = NULL;
	char *idx_name = NULL;
	ods_iter_t iter;
	ods_key_t key;
	ods_ref_t ref;
	int rc;
	uint64_t inode;

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
		case 'i':
			iter_key = strdup(optarg);
			break;
		default:
			usage(argc, argv);
		}
	}
	if (!idx_name)
		idx_name = "BPTREE";

	if (!idx_path || !key_str)
		usage(argc, argv);

	rc = ods_idx_create(idx_path, 0660, idx_name, key_str, order);
	if (rc) {
		printf("The index '%s' of type '%s' could not be created due "
		       "to error %d.\n",
		       idx_path, idx_name, rc);
		return rc;
	}
	idx = ods_idx_open(idx_path);
	rebuild_tree(idx);
	idx->idx_class->prv->dbg_print_idx(idx);

	/* Find the specified key and iterate until it doesn't match */
	if (!iter_key)
		goto skip_iter_key;

	/* Find all matching keys */
	printf("All matches...\n");
	iter = ods_iter_new(idx);
	ods_key_from_str(idx, key, iter_key);
	for (rc = ods_iter_find(iter, key); !rc; rc = ods_iter_next(iter)) {
		ods_key_t k = ods_iter_key(iter);
		if (ods_key_cmp(idx, key, k))
			break;
		printf("key %s\n", ods_key_to_str(idx, k));
	}
	printf("... End matches.\n");

	/* Find GLB of key */
	printf("All GLB...\n");
	iter = ods_iter_new(idx);
	ods_key_from_str(idx, key, iter_key);
	for (rc = ods_iter_find_glb(iter, key); !rc; rc = ods_iter_next(iter)) {
		ods_key_t k = ods_iter_key(iter);
		if (ods_key_cmp(idx, key, k))
			break;
		printf("key %s obj %p\n", ods_key_to_str(idx, k),
		       (void *)ods_iter_ref(iter));
	}
	printf("... End matches.\n");

	/* Find lub of key */
	printf("All LUB...\n");
	iter = ods_iter_new(idx);
	ods_key_from_str(idx, key, iter_key);
	for (rc = ods_iter_find_lub(iter, key); !rc; rc = ods_iter_next(iter)) {
		ods_key_t k = ods_iter_key(iter);
		if (ods_key_cmp(idx, key, k))
			break;
		printf("key %s obj %p\n", ods_key_to_str(idx, k),
		       (void *)ods_iter_ref(iter));
	}
	printf("... End matches.\n");

 skip_iter_key:
	/* Delete the min in the tree until the tree is empty */
	iter = ods_iter_new(idx);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_begin(iter)) {
		ods_key_t k = ods_iter_key(iter);
		printf("delete %s\n", ods_key_to_str(idx, k));
		rc = ods_idx_delete(idx, k, &ref);
		ods_obj_put(k);
		if (rc) {
			printf("FAILURE: The key '%s' is in the iterator, "
			       "but could not be deleted.\n",
			       ods_key_to_str(idx, k));
		}
	}
	idx->idx_class->prv->dbg_print_idx(idx);

	/* Build another tree */
	rebuild_tree(idx);
	idx->idx_class->prv->dbg_print_idx(idx);

	/* Delete the max in the tree until the tree is empty */
	for (rc = ods_iter_end(iter); !rc; rc = ods_iter_end(iter)) {
		ods_key_t k = ods_iter_key(iter);
		printf("delete %s\n", ods_key_to_str(idx, k));
		rc = ods_idx_delete(idx, k, &ref);
		ods_obj_put(k);
		if (rc) {
			printf("FAILURE: The key '%s' is in the iterator, "
			       "but could not be deleted.\n",
			       ods_key_to_str(idx, k));
		}
	}
	idx->idx_class->prv->dbg_print_idx(idx);

	/* Build another tree */
	rebuild_tree(idx);
	idx->idx_class->prv->dbg_print_idx(idx);

	/* Delete an interior key until the tree is empty */
	bpt_t t = idx->priv;
	ods_obj_t root;
	while (t->root_ref) {
		root = ods_ref_as_obj(t->ods, t->root_ref);
		int cnt = NODE(root)->count >> 1;
		ods_ref_t key_ref = NODE(root)->entries[cnt].key;
		ods_key_t k = ods_ref_as_obj(t->ods, key_ref);
		printf("delete %s\n", ods_key_to_str(idx, k));
		rc = ods_idx_delete(idx, k, &ref);
		ods_obj_put(k);
		if (rc) {
			printf("FAILURE: The key '%s' is in the iterator, "
			       "but could not be deleted.\n",
			       ods_key_to_str(idx, k));
		}
		ods_obj_put(root);
	}
	idx->idx_class->prv->dbg_print_idx(idx);
	rebuild_tree(idx);
#if 0
	static char *keys[] = {
		"-1", "0", "1", "3", "5", "7", "9", "11", "13", "15", "17", "19", "21"
	};
#else
	static char *keys[] = {
		"4", "5", "6", "7", "8", "9"
	};
#endif
#if 0
	for (rc = 1; rc < sizeof(keys) / sizeof(keys[0]); rc++) {
		size_t key_sz = ods_idx_key_size(idx);
		if (key_sz == -1)
			key_sz = strlen(keys[rc]);
		key = ods_key_malloc(idx, key_sz);
		ods_key_from_str(idx, key, keys[rc]);
		inode = strtoul(keys[rc], NULL, 0);
		ods_idx_insert(idx, key, inode);
		ods_obj_put(key);
	}
	idx->idx_class->prv->dbg_print_idx(idx);
	for (rc = 0; rc < sizeof(keys) / sizeof(keys[0]); rc++) {
		ods_ref_t obj_ref;
		int rrc;
		int k = strtoul(keys[rc], NULL, 0);
		char ks[32];
		sprintf(ks, "%d", k);
		size_t key_sz = ods_idx_key_size(idx);
		if (key_sz == -1)
			key_sz = strlen(keys[rc]);
		key = ods_key_malloc(idx, key_sz);
		ods_key_from_str(idx, key, ks);
		rrc = ods_idx_delete(idx, key, &obj_ref);
		if (rrc)
			printf("error key %s not found %d\n", ks, rrc);
		ods_obj_put(key);
	}
	idx->idx_class->prv->dbg_print_idx(idx);
	for (rc = 1; rc < sizeof(keys) / sizeof(keys[0]); rc++) {
		size_t key_sz = ods_idx_key_size(idx);
		if (key_sz == -1)
			key_sz = strlen(keys[rc]);
		key = ods_key_malloc(idx, key_sz);
		ods_key_from_str(idx, key, keys[rc]);
		inode = strtoul(keys[rc], NULL, 0);
		ods_idx_insert(idx, key, inode);
		ods_obj_put(key);
	}
#endif
	idx->idx_class->prv->dbg_print_idx(idx);
	for (rc = 0; rc < sizeof(keys) / sizeof(keys[0]); rc++) {
		int rrc;
		ods_ref_t lub;
		ods_ref_t glb;
		int k = strtoul(keys[rc], NULL, 0) + 1;
		char ks[32];
		sprintf(ks, "%d", k);
		size_t key_sz = ods_idx_key_size(idx);
		if (key_sz == -1)
			key_sz = strlen(keys[rc]);
		key = ods_key_malloc(idx, key_sz);
		ods_key_from_str(idx, key, ks);
		rrc = ods_idx_find_glb(idx, key, &glb);
		if (!rrc)
			printf("%d ", (int)glb);
		else
			printf("<none> ");
		printf("<= %s <= ", ks);
		rrc = ods_idx_find_lub(idx, key, &lub);
		if (!rrc)
			printf("%d\n", (int)lub);
		else
			printf("<none>\n");
		ods_obj_put(key);
		idx->idx_class->prv->dbg_print_idx(idx);
		ods_info();
	}
	idx->idx_class->prv->dbg_print_idx(idx);
	return 0;
}
