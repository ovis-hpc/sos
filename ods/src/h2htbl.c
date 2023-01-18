/*
 * Copyright (c) 2018 Open Grid Computing, Inc. All rights reserved.
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
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <string.h>
#include <assert.h>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <ods/ods.h>
#include <string.h>
#include <time.h>
#include "h2htbl.h"

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

static void print_idx(ods_idx_t idx, FILE *fp)
{
}

static void print_info(ods_idx_t idx, FILE *fp)
{
	h2htbl_t t = idx->priv;
	fprintf(fp, "%*s : %d\n", 12, "Table Size", t->udata->table_size);
	fprintf(fp, "%*s : %d\n", 12, "Lock", t->udata->lock);
	fprintf(fp, "%*s : %d\n", 12, "Cardinality", 0); /* TODO sum each table */
	fprintf(fp, "%*s : %d\n", 12, "Duplicates", 0);
	fflush(fp);
}

static void *hash_root_fn(void *);
static int h2htbl_open(ods_idx_t idx)
{
	char path_buf[PATH_MAX];
	const char *path;
	const char *base = NULL;
	ods_obj_t udata_obj;
	h2htbl_t t;
	int i, rc;
	struct stat dir_sb;

	udata_obj = ods_get_user_data(idx->ods);
	if (!udata_obj)
		return EINVAL;
	t = calloc(1, sizeof *t);
	if (!t)
		goto err_0;
	t->hash_fn = fnv_hash_a1_32; /* compiler food */
	t->hash_fn = fnv_hash_a1_64;
	t->udata_obj = udata_obj;
	t->udata = H2UDATA(udata_obj);
	t->idx_table = calloc(t->udata->table_size, sizeof *t->idx_table);
	if (!t->idx_table)
		goto err_1;
	t->ods_idx = idx;
	idx->priv = t;
	path = ods_path(idx->ods);
	rc = stat(path, &dir_sb);
	if (rc)
		goto out;
	strcpy(path_buf, path); /* basename may modify path */
	base = basename(path_buf);
	base = strdup(base);

	/* Open each hash root */
	for (i = 0; i < t->udata->table_size; i++) {
		sprintf(path_buf, "%s/bkt_%d/%s", path, i, base);
		t->idx_table[i].idx = ods_idx_open(path_buf, ODS_PERM_RW);
		if (!t->idx_table[i].idx) {
			rc = errno;
			goto out;
		}
		t->idx_table[i].mq = mq_new(H2HTBL_QUEUE_DEPTH, H2HTBL_MAX_MSG_SIZE, 1);
		if (!t->idx_table[i].mq) {
			rc = errno;
			goto out;
		}
	}
	free((void *)base);
	return 0;
 out:
	if (base)
		free((void *)base);
	return rc;
 err_1:
	free(t);
 err_0:
	ods_obj_put(udata_obj);
	return ENOMEM;
}

static unsigned long arg_int_value(const char *arg_str, const char *name_str)
{
	extern char *strcasestr(const char *haystack, const char *needle);
	char order_arg[ODS_IDX_ARGS_LEN];
	char *name, *value, *arg;

	if (!arg_str)
		return 0;

	arg = strcasestr(arg_str, name_str);
	if (!arg)
		return 0;

	strcpy(order_arg, arg);
	name = strtok(order_arg, "=");
	if (name) {
		value = strtok(NULL, "=");
		if (value)
			return strtoul(value, NULL, 0);
	}
	return 0;
}

static int h2htbl_init(ods_t ods, const char *type, const char *key, const char *argp)
{
	char path_buf[PATH_MAX];
	const char *path;
	const char *base;
	ods_obj_t udata;
	uint32_t htlen = 0;
	uint32_t seed = 0;
	int i, rc;
	struct stat sb;
	mode_t dir_mode;

	rc = ods_stat(ods, &sb);
	if (rc)
		return rc;

	dir_mode = sb.st_mode;
	if (dir_mode | S_IRUSR)
		dir_mode |= S_IXUSR;
	if (dir_mode | S_IRGRP)
		dir_mode |= S_IXGRP;
	if (dir_mode | S_IROTH)
		dir_mode |= S_IXOTH;
	rc = mkdir(ods_path(ods), dir_mode);
	if (rc)
		return rc;

	udata = ods_get_user_data(ods);
	if (!udata)
		return EINVAL;

	srandom(time(NULL));
	seed = arg_int_value(argp, "SEED");
	if (!seed)
		seed = (uint32_t)random();

	htlen = arg_int_value(argp, "SIZE");
	if (!htlen)
		htlen = H2HTBL_DEFAULT_TABLE_SIZE;

	H2UDATA(udata)->table_size = htlen;
	H2UDATA(udata)->hash_seed = seed;
	H2UDATA(udata)->lock = 0;

	/* create each hash root */
	path = ods_path(ods);
	strcpy(path_buf, path); /* basename may modify path */
	base = basename(path_buf);
	base = strdup(base);
	for (i = 0; i < htlen; i++) {
		sprintf(path_buf, "%s/bkt_%d", path, i);
		rc = mkdir(path_buf, dir_mode);
		if (rc)
			goto out;
		sprintf(path_buf, "%s/bkt_%d/%s", path, i, base);
		rc = ods_idx_create(path_buf, ods->o_perm, sb.st_mode, "HTBL", key, argp);
		if (rc)
			goto out;
	}
 out:
	ods_obj_put(udata);
	free((void *)base);
	return rc;
}

static void h2htbl_close(ods_idx_t idx)
{
	h2htbl_t t = idx->priv;
	int bkt;
	assert(t);
	idx->priv = NULL;
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		t->idx_table[bkt].state = H2HTBL_STATE_STOPPED;
		if (t->idx_table[bkt].thread)
			pthread_cancel(t->idx_table[bkt].thread);
	}
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		if (t->idx_table[bkt].thread) {
			pthread_join(t->idx_table[bkt].thread, NULL);
		}
		ods_idx_close(t->idx_table[bkt].idx, ODS_COMMIT_ASYNC);
	}
	ods_obj_put(t->udata_obj);
	free(t->idx_table);
	free(t);
}


/** a - b */
static double tv_diff(struct timeval *tv_a, struct timeval *tv_b)
{
	double a, b;
	a = ((double)tv_a->tv_sec * 1000000.0) + (double)tv_a->tv_usec;
	b = ((double)tv_b->tv_sec * 1000000.0) + (double)tv_b->tv_usec;
	return a - b;
}

void *hash_root_fn(void *arg)
{
	struct h2htbl_idx_s *root = arg;
	struct timeval tv_start, tv_end;
	mq_t mq = root->mq;
	mq_msg_t msg;

	msg = mq_get_cons_msg(mq);
 running:
	if (!msg)
		goto wait;

	ods_idx_lock(root->idx, NULL);
	gettimeofday(&tv_start, NULL);
	while (msg) {
		int rc = msg->msg_work_fn(mq, msg);
		if (rc)
			ods_lerror("Error %d processing work element.\n", rc);
		mq_post_cons_msg(mq);
		gettimeofday(&tv_end, NULL);
		if (tv_diff(&tv_end, &tv_start) > H2HTBL_DUTY_CYCLE)
			break;
		msg = mq_get_cons_msg(mq);
	}
	ods_idx_unlock(root->idx);
	usleep(0);
 wait:
	msg = mq_get_cons_msg_wait(mq);
	if (root->state == H2HTBL_STATE_RUNNING)
		goto running;

	return NULL;
}

static uint64_t hash_key(h2htbl_t t, ods_key_t key)
{
	ods_key_value_t kv = ods_key_value(key);
	return t->hash_fn((const char *)kv->value,
			  kv->len,
			  t->udata->hash_seed) % t->udata->table_size;
}

int visit_fn(struct mq_s *mq, struct mq_msg_s *_msg)
{
	visit_msg_t msg = (visit_msg_t)_msg;
	int rc = ods_idx_visit(msg->idx->idx, msg->key, msg->cb_fn, msg->ctxt);
	if (msg->key->as.ptr != &msg->key_)
		ods_obj_put(msg->key);
	return rc;
}

static int h2htbl_visit(ods_idx_t idx, ods_key_t key,
		       ods_visit_cb_fn_t cb_fn, void *ctxt)
{
	visit_msg_t visit_msg;
	h2htbl_t t = idx->priv;
	uint64_t bkt = hash_key(t, key);
	size_t key_sz = ods_key_len(key);

	if (0 == (t->rt_opts & ODS_IDX_OPT_VISIT_ASYNC))
		return ods_idx_visit(t->idx_table[bkt].idx, key, cb_fn, ctxt);

	visit_msg = (visit_msg_t)mq_get_prod_msg_wait(t->idx_table[bkt].mq);
	visit_msg->hdr.msg_type = WQE_VISIT;
	visit_msg->hdr.msg_work_fn = visit_fn;
	visit_msg->hdr.msg_size = sizeof(*visit_msg);
	visit_msg->idx = &t->idx_table[bkt];
	if (key_sz > VISIT_KEY_SIZE) {
		visit_msg->key = ods_key_malloc(key_sz);
	} else {
		visit_msg->key = ODS_OBJ_INIT(visit_msg->key_obj_, &visit_msg->key_, sizeof(visit_msg->key_));
	}
	ods_key_set(visit_msg->key, key->as.key->value, key->as.key->len);
	visit_msg->cb_fn = cb_fn;
	visit_msg->ctxt = ctxt;
	mq_post_prod_msg(t->idx_table[bkt].mq);
	return EINPROGRESS;
}

static int h2htbl_update(ods_idx_t idx, ods_key_t key, ods_idx_data_t data)
{
	h2htbl_t t = idx->priv;
	uint64_t bkt = hash_key(t, key);
	return ods_idx_update(t->idx_table[bkt].idx, key, data);
}

static int h2htbl_find(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	h2htbl_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_find(t->idx_table[hash].idx, key, data);
}

static int h2htbl_find_lub(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	h2htbl_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_find_lub(t->idx_table[hash].idx, key, data);
}

static int h2htbl_find_glb(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	h2htbl_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_find_glb(t->idx_table[hash].idx, key, data);
}

static int h2htbl_insert(ods_idx_t idx, ods_key_t new_key, ods_idx_data_t data)
{
	h2htbl_t t = idx->priv;
	uint64_t bkt = hash_key(t, new_key);
	return ods_idx_insert(t->idx_table[bkt].idx, new_key, data);
}

static int h2htbl_insert_no_lock(ods_idx_t idx, ods_key_t new_key, ods_idx_data_t data)
{
	return h2htbl_insert(idx, new_key, data);
}

static int h2htbl_max(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	int bkt;
	h2htbl_t t = idx->priv;
	ods_key_t max_key = NULL;
	ods_key_t idx_key;
	struct ods_idx_data_s idx_data;
	int rc = ENOENT;

	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		rc = ods_idx_max(t->idx_table[bkt].idx, &idx_key, &idx_data);
		if (rc)
			continue;
		rc = 0;
		if (max_key) {
			if (ods_key_cmp(t->idx_table[bkt].idx, idx_key, max_key) > 0) {
				ods_obj_put(max_key);
				max_key = idx_key;
			} else {
				ods_obj_put(idx_key);
			}
		} else {
			max_key = idx_key;
		}
	}
	if (!rc) {
		if (key)
			*key = idx_key;
		if (data)
			*data = idx_data;
	}
	return rc;
}

static int h2htbl_min(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	int bkt;
	h2htbl_t t = idx->priv;
	ods_key_t min_key = NULL;
	ods_key_t idx_key;
	struct ods_idx_data_s idx_data;
	int rc = ENOENT;

	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		rc = ods_idx_min(t->idx_table[bkt].idx, &idx_key, &idx_data);
		if (rc)
			continue;
		rc = 0;
		if (min_key) {
			if (ods_key_cmp(t->idx_table[bkt].idx, idx_key, min_key) > 0) {
				ods_obj_put(min_key);
				min_key = idx_key;
			} else {
				ods_obj_put(idx_key);
			}
		} else {
			min_key = idx_key;
		}
	}
	if (!rc) {
		if (key)
			*key = idx_key;
		if (data)
			*data = idx_data;
	}
	return rc;
}

static int h2htbl_delete(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	h2htbl_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_delete(t->idx_table[hash].idx, key, data);
}

typedef struct iter_entry_s {
	h2htbl_iter_t iter;
	ods_key_t key;
	ods_idx_data_t data;
	int bkt;
	struct ods_rbn rbn;
} *iter_entry_t;

static int64_t entry_cmp(void *tree_key, const void *key, void *arg)
{
	iter_entry_t a = tree_key;
	iter_entry_t b = (iter_entry_t)key;
	ods_iter_t oi = (ods_iter_t)a->iter;
	return ods_key_cmp(oi->idx, a->key, b->key);
}

static void iter_cleanup(h2htbl_t t, h2htbl_iter_t iter)
{
	struct ods_rbn *rbn;
	while ((rbn = ods_rbt_min(&iter->next_tree))) {
		iter_entry_t ent = container_of(rbn, struct iter_entry_s, rbn);
		ods_rbt_del(&iter->next_tree, rbn);
		ods_obj_put(ent->key);
		free(ent);
	}
}

static void h2htbl_iter_delete(ods_iter_t i)
{
	int bkt;
	h2htbl_t t = i->idx->priv;
	h2htbl_iter_t iter = (h2htbl_iter_t)i;

	/* Empty the RBT */
	iter_cleanup(t, iter);
	ods_atomic_dec(&i->idx->ref_count);

	/* Destroy each sub-iter */
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		if (iter->iter_table[bkt])
			ods_iter_delete(iter->iter_table[bkt]);
	}
	free(iter);
}

static ods_iter_t h2htbl_iter_new(ods_idx_t idx)
{
	int bkt;
	h2htbl_t t = idx->priv;
	h2htbl_iter_t iter;
	size_t iter_size = sizeof(*iter) + (t->udata->table_size * sizeof(ods_idx_t));
	iter = calloc(1, iter_size);
	if (!iter)
		return NULL;
	ods_rbt_init(&iter->next_tree, entry_cmp, NULL);
	ods_atomic_inc(&idx->ref_count);
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		iter->iter_table[bkt] = ods_iter_new(t->idx_table[bkt].idx);
		if (!iter->iter_table[bkt]) {
			h2htbl_iter_delete((ods_iter_t)iter);
			return NULL;
		}
	}
	return (ods_iter_t)iter;
}

static iter_entry_t alloc_ent(h2htbl_iter_t iter, int bkt)
{
	iter_entry_t ent = malloc(sizeof *ent);
	if (!ent)
		goto out;
	ent->iter = iter;
	ent->key = ods_iter_key(iter->iter_table[bkt]);
	ent->data = ods_iter_data(iter->iter_table[bkt]);
	ent->bkt = bkt;
	assert(ent->key);
	ods_rbn_init(&ent->rbn, ent);
 out:
	return ent;
}

static int h2htbl_iter_begin(ods_iter_t oi)
{
	int bkt, rc, rv = ENOENT;
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	h2htbl_t t = oi->idx->priv;

	iter_cleanup(t, iter);
	iter->dir = H2HTBL_ITER_FWD;

	/*
	 * Run through every iterator in the hash table and insert the
	 * key into the tree.
	 */
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		rc = ods_iter_begin(iter->iter_table[bkt]);
		if (rc)
			continue;
		rv = 0;
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		ods_rbt_ins(&iter->next_tree, &ent->rbn);
	}
	return rv;
}

static int h2htbl_iter_end(ods_iter_t oi)
{
	int bkt, rc, rv = ENOENT;
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	h2htbl_t t = oi->idx->priv;

	iter_cleanup(t, iter);
	iter->dir = H2HTBL_ITER_REV;

	/*
	 * Run through every iterator in the hash table and insert the
	 * key into the tree.
	 */
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		rc = ods_iter_end(iter->iter_table[bkt]);
		if (rc)
			continue;
		rv = 0;
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		ods_rbt_ins(&iter->next_tree, &ent->rbn);
	}
	return rv;
}

static ods_idx_data_t NULL_DATA;

static iter_entry_t h2htbl_iter_entry_fwd(ods_iter_t oi)
{
	h2htbl_iter_t iter = (typeof(iter))oi;
	struct ods_rbn *rbn = ods_rbt_min(&iter->next_tree);
	if (!rbn)
		return NULL;
	return container_of(rbn, struct iter_entry_s, rbn);
}

static iter_entry_t h2htbl_iter_entry_rev(ods_iter_t oi)
{
	h2htbl_iter_t iter = (typeof(iter))oi;
	struct ods_rbn *rbn = ods_rbt_max(&iter->next_tree);
	if (!rbn)
		return NULL;
	return container_of(rbn, struct iter_entry_s, rbn);
}

/*
 * Return the min valued key in the rbt of the iterator
 */
static ods_key_t h2htbl_iter_key_fwd(ods_iter_t oi)
{
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	struct ods_rbn *rbn;
	rbn = ods_rbt_min(&iter->next_tree);
	if (!rbn)
		return NULL;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	return ods_obj_get(ent->key);
}

static ods_key_t h2htbl_iter_key_rev(ods_iter_t oi)
{
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	struct ods_rbn *rbn;
	rbn = ods_rbt_max(&iter->next_tree);
	if (!rbn)
		return NULL;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	return ods_obj_get(ent->key);
}

static ods_key_t h2htbl_iter_key(ods_iter_t oi)
{
	h2htbl_iter_t iter = (typeof(iter))oi;
	switch (iter->dir) {
	case H2HTBL_ITER_FWD:
		return h2htbl_iter_key_fwd(oi);
	case H2HTBL_ITER_REV:
		return h2htbl_iter_key_rev(oi);
	default:
		assert(0 == "Invalid dir field in iter");
	}
	return NULL;
}

static ods_idx_data_t h2htbl_iter_data_fwd(ods_iter_t oi)
{
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	struct ods_rbn *rbn = ods_rbt_min(&iter->next_tree);
	if (!rbn)
		return NULL_DATA;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	return ent->data;
}

static ods_idx_data_t h2htbl_iter_data_rev(ods_iter_t oi)
{
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	struct ods_rbn *rbn = ods_rbt_max(&iter->next_tree);
	if (!rbn)
		return NULL_DATA;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	return ent->data;
}

static ods_idx_data_t h2htbl_iter_data(ods_iter_t oi)
{
	h2htbl_iter_t iter = (typeof(iter))oi;
	switch (iter->dir) {
	case H2HTBL_ITER_FWD:
		return h2htbl_iter_data_fwd(oi);
	case H2HTBL_ITER_REV:
		return h2htbl_iter_data_rev(oi);
	default:
		assert(0 == "Invalid dir field in iter");
	}
	return NULL_DATA;
}

static int h2htbl_iter_find_first(ods_iter_t oi, ods_key_t key)
{
	int rc, rv = ENOENT;
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	h2htbl_t t = oi->idx->priv;
	uint64_t bkt = hash_key(t, key);

	iter_cleanup(t, iter);
	iter->dir = H2HTBL_ITER_FWD;

	rc = ods_iter_find_first(iter->iter_table[bkt], key);
	if (rc)
		return ENOENT;
	ent = alloc_ent(iter, bkt);
	if (!ent)
		return ENOMEM;
	ods_rbt_ins(&iter->next_tree, &ent->rbn);
	return rv;
}

static int h2htbl_iter_find_last(ods_iter_t oi, ods_key_t key)
{
	int rc;
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	h2htbl_t t = oi->idx->priv;
	uint64_t bkt = hash_key(t, key);

	iter_cleanup(t, iter);
	iter->dir = H2HTBL_ITER_REV;

	rc = ods_iter_find_last(iter->iter_table[bkt], key);
	if (rc)
		return ENOENT;
	ent = alloc_ent(iter, bkt);
	if (!ent)
		return ENOMEM;
	ods_rbt_ins(&iter->next_tree, &ent->rbn);
	return 0;
}

static int h2htbl_iter_find(ods_iter_t oi, ods_key_t key)
{
	int rc;
	iter_entry_t ent;
	h2htbl_iter_t iter = (typeof(iter))oi;
	h2htbl_t t = oi->idx->priv;
	uint64_t bkt = hash_key(t, key);

	iter_cleanup(t, iter);
	iter->dir = H2HTBL_ITER_FWD;

	rc = ods_iter_find(iter->iter_table[bkt], key);
	if (rc)
		return ENOENT;
	ent = alloc_ent(iter, bkt);
	if (!ent)
		return ENOMEM;
	ods_rbt_ins(&iter->next_tree, &ent->rbn);
	return 0;
}

/*
 * Get the LUB from each iterator, and return the maximum. The tree
 * will either contain a single entry, which is the max of all the
 * iterators, or it will be empty.
 *
 * Note that since the tree only contains a single entry, the setting
 * of the iterator direction does not affect the value returned by
 * thee _iter_key() or _iter_data() functions.
 *
 * If the tree is not empty, 0 is returned, otherwise, ENOENT/ENOMEM
 * is returned.
 *
 */
static int h2htbl_iter_find_lub(ods_iter_t oi, ods_key_t key)
{
	h2htbl_t t = oi->idx->priv;
	h2htbl_iter_t iter = (typeof(iter))oi;
	iter_entry_t ent;
	int rv = ENOENT;
	int bkt;

	iter_cleanup(t, iter);
	iter->dir = H2HTBL_ITER_FWD;

	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		ods_iter_flags_set(iter->iter_table[bkt], oi->flags);
		if (ods_iter_find_lub(iter->iter_table[bkt], key))
			continue;
		rv = 0;
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		ods_rbt_ins(&iter->next_tree, &ent->rbn);
	}
	return rv;
}

static int h2htbl_iter_find_glb(ods_iter_t oi, ods_key_t key)
{
	h2htbl_t t = oi->idx->priv;
	h2htbl_iter_t iter = (typeof(iter))oi;
	iter_entry_t ent;
	int rv = ENOENT;
	int bkt;

	iter_cleanup(t, iter);
	iter->dir = H2HTBL_ITER_REV;

	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		ods_iter_flags_set(iter->iter_table[bkt], oi->flags);
		if (ods_iter_find_glb(iter->iter_table[bkt], key))
			continue;
		rv = 0;
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		ods_rbt_ins(&iter->next_tree, &ent->rbn);
	}
	return rv;
}

static int h2htbl_iter_next(ods_iter_t oi)
{
	h2htbl_iter_t iter = (typeof(iter))oi;
	struct ods_rbn *rbn;
	iter_entry_t ent;
	ods_key_t key;
	int bkt, rc;
	h2htbl_t t = oi->idx->priv;

	if (iter->dir == H2HTBL_ITER_REV) {
		/* need to reverse all sub iterators and rebuild the tree */
		iter->dir = H2HTBL_ITER_FWD;
		iter_cleanup(t, iter);
		for (bkt = 0; bkt < t->udata->table_size; bkt++) {
			key = ods_iter_key(iter->iter_table[bkt]);
			if (key) {
				/* valid iterator */
				ods_obj_put(key);
				rc = ods_iter_next(iter->iter_table[bkt]);
			} else {
				/* depleted iterator, must start from
				 * the beginning */
				rc = ods_iter_begin(iter->iter_table[bkt]);
			}
			if (rc)
				continue;
			ent = alloc_ent(iter, bkt);
			if (!ent)
				return ENOMEM;
			ods_rbt_ins(&iter->next_tree, &ent->rbn);
		}
		goto out;
	}

	assert(iter->dir == H2HTBL_ITER_FWD); /* direction must be forward here */

	/* Delete the min from the tree. */
	rbn = ods_rbt_min(&iter->next_tree);
	if (!rbn)
		return ENOENT;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	ods_rbt_del(&iter->next_tree, rbn);
	ods_obj_put(ent->key);
	bkt = ent->bkt;
	free(ent);

	/* Get the next entry for this bucket and insert into the tree */
	if (0 == ods_iter_next(iter->iter_table[bkt])) {
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		ods_rbt_ins(&iter->next_tree, &ent->rbn);
	}

 out:
	if (ods_rbt_empty(&iter->next_tree))
		return ENOENT;
	return 0;
}

static int h2htbl_iter_prev(ods_iter_t oi)
{
	h2htbl_iter_t iter = (typeof(iter))oi;
	struct ods_rbn *rbn;
	iter_entry_t ent;
	ods_key_t key;
	int bkt, rc;
	h2htbl_t t = oi->idx->priv;

	if (iter->dir == H2HTBL_ITER_FWD) {
		/* need to reverse all sub iterators and rebuild the tree */
		iter->dir = H2HTBL_ITER_REV;
		iter_cleanup(t, iter);
		for (bkt = 0; bkt < t->udata->table_size; bkt++) {
			key = ods_iter_key(iter->iter_table[bkt]);
			if (key) {
				/* valid iterator */
				ods_obj_put(key);
				rc = ods_iter_prev(iter->iter_table[bkt]);
			} else {
				/* depleted iterator, must start from
				 * the beginning */
				rc = ods_iter_end(iter->iter_table[bkt]);
			}
			if (rc)
				continue;
			ent = alloc_ent(iter, bkt);
			if (!ent)
				return ENOMEM;
			ods_rbt_ins(&iter->next_tree, &ent->rbn);
		}
		goto out;
	}

	assert(iter->dir == H2HTBL_ITER_REV);

	/* Delete the max from the tree. */
	rbn = ods_rbt_max(&iter->next_tree);
	if (!rbn)
		return ENOENT;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	ods_rbt_del(&iter->next_tree, rbn);
	ods_obj_put(ent->key);
	bkt = ent->bkt;
	free(ent);

	/* Get the previous entry for this bucket and insert into the tree */
	if (0 == ods_iter_prev(iter->iter_table[bkt])) {
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		ods_rbt_ins(&iter->next_tree, &ent->rbn);
	}

 out:
	if (ods_rbt_empty(&iter->next_tree))
		return ENOENT;
	return 0;
}

static int h2htbl_iter_pos_put(ods_iter_t oi, ods_pos_t pos)
{
	h2htbl_t t = oi->idx->priv;
	uint32_t ht_idx;
	h2htbl_iter_t iter = (typeof(iter))oi;
	ods_obj_t pos_obj = ods_ref_as_obj(oi->idx->ods, pos->ref);
	if (!pos_obj)
		return ENOMEM;
	ht_idx = POS(pos_obj)->htbl_iter_idx;
	if (ht_idx < t->udata->table_size)
		(void)ods_iter_pos_put(iter->iter_table[ht_idx],
				       &POS(pos_obj)->htbl_iter_pos);
	ods_obj_delete(pos_obj);
	ods_obj_put(pos_obj);
	return 0;
}

static int h2htbl_iter_pos_set(ods_iter_t oi, const ods_pos_t pos)
{
	h2htbl_t t = oi->idx->priv;
	h2htbl_iter_t iter = (typeof(iter))oi;
	ods_key_t key;
	iter_entry_t ent;
	uint32_t ht_idx;
	int i, rc = EINVAL;
	ods_obj_t pos_obj;
	int (*iter_find_fn)(ods_iter_t oi, ods_key_t key);

	pos_obj = ods_ref_as_obj(oi->idx->ods, pos->ref);
	if (!pos_obj)
		return rc;

	/* Verify the hash table index */
	ht_idx = POS(pos_obj)->htbl_iter_idx;
	if (ht_idx >= t->udata->table_size)
		goto err_0;

	/* Verify the direction */
	switch (POS(pos_obj)->dir) {
	case H2HTBL_ITER_FWD:
		iter_find_fn = ods_iter_find_lub;
		break;
	case H2HTBL_ITER_REV:
		iter_find_fn = ods_iter_find_glb;
		break;
	default:
		goto err_0;
	}

	/* clean-up the rbt and reconstruct it from the given positions */
	iter_cleanup(t, iter);

	rc = ods_iter_pos_set(iter->iter_table[ht_idx],
			      &POS(pos_obj)->htbl_iter_pos);
	if (rc)
		goto err_0;

	ent = alloc_ent(iter, ht_idx);
	if (!ent) {
		rc = ENOMEM;
		goto err_1;
	}
	ods_rbt_ins(&iter->next_tree, &ent->rbn);

	key = ods_iter_key(iter->iter_table[ht_idx]);
	if (!key) {
		rc = errno;
		goto err_1;
	}

	for (i = 0; i < t->udata->table_size; i++) {
		if (ht_idx == i)
			continue;
		rc = iter_find_fn(iter->iter_table[i], key);
		if (rc) {
			if (rc == ENOENT)
				continue;
			goto err_2;
		}
		ent = alloc_ent(iter, i);
		if (!ent) {
			rc = ENOMEM;
			goto err_2;
		}
		ods_rbt_ins(&iter->next_tree, &ent->rbn);
	}
	iter->dir = POS(pos_obj)->dir;
	ods_obj_put(key);
	ods_obj_delete(pos_obj);
	ods_obj_put(pos_obj);
	return 0;
 err_2:
	ods_obj_put(key);
 err_1:
	iter_cleanup(t, iter);
 err_0:
	ods_obj_delete(pos_obj);
	ods_obj_put(pos_obj);
	return rc;
}

static int h2htbl_iter_pos_get(ods_iter_t oi, ods_pos_t pos)
{
	iter_entry_t ent;
	ods_obj_t pos_obj;
	h2htbl_t t = oi->idx->priv;
	h2htbl_iter_t iter = (typeof(iter))oi;
	int rc;
	size_t sz = sizeof(struct h2htbl_pos_s);

	switch (iter->dir) {
	case H2HTBL_ITER_FWD:
		ent = h2htbl_iter_entry_fwd(oi);
		break;
	case H2HTBL_ITER_REV:
		ent = h2htbl_iter_entry_rev(oi);
		break;
	default:
		return EINVAL;
	}
	if (!ent)
		return ENOENT;
	pos_obj = ods_obj_alloc_extend(t->ods_idx->ods, sz, H2HTBL_EXTEND_SIZE);
	if (!pos_obj)
		return ENOMEM;

	rc = ods_iter_pos_get(iter->iter_table[ent->bkt],
			      &POS(pos_obj)->htbl_iter_pos);
	if (rc)
		goto err_0;

	POS(pos_obj)->dir = iter->dir;
	POS(pos_obj)->htbl_iter_idx = ent->bkt;
	pos->ref = ods_obj_ref(pos_obj);
	ods_obj_put(pos_obj);
	return 0;
 err_0:
	ods_obj_delete(pos_obj);
	ods_obj_put(pos_obj);
	return rc;
}

static int h2htbl_iter_entry_delete(ods_iter_t oi, ods_idx_data_t *data)
{
	return ENOENT;
}

static const char *h2htbl_get_type(void)
{
	return "H2HTBL";
}

static void h2htbl_commit(ods_idx_t idx)
{
	h2htbl_t t = idx->priv;
	int bkt;
	ods_idx_commit(t->ods_idx, ODS_COMMIT_SYNC);
	for (bkt = 0; bkt < t->udata->table_size; bkt++)
		ods_idx_commit(t->idx_table[bkt].idx, ODS_COMMIT_SYNC);
}

int h2htbl_stat(ods_idx_t idx, ods_idx_stat_t idx_sb)
{
	struct ods_idx_stat_s bkt_sb;
	h2htbl_t t = idx->priv;
	int bkt;
	memset(idx_sb, 0, sizeof(*idx_sb));
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		ods_idx_stat(t->idx_table[bkt].idx, &bkt_sb);
		idx_sb->cardinality += bkt_sb.cardinality;
		idx_sb->duplicates += bkt_sb.duplicates;
		idx_sb->size += bkt_sb.size;
	}
	return 0;
}

int h2htbl_rt_opts_set(ods_idx_t idx, ods_idx_rt_opts_t opt, va_list ap)
{
	h2htbl_t t = idx->priv;
	int i, rc;
	switch (opt) {
	case ODS_IDX_OPT_MP_UNSAFE:
		t->rt_opts |= opt;
		for (i = 0; i < t->udata->table_size; i++)
			ods_idx_rt_opts_set(t->idx_table[i].idx, opt);
		break;
	case ODS_IDX_OPT_VISIT_ASYNC:
		t->rt_opts |= opt;
		for (i = 0; i < t->udata->table_size; i++) {
			ods_idx_rt_opts_set(t->idx_table[i].idx, opt);
			t->idx_table[i].state = H2HTBL_STATE_RUNNING;
			rc = pthread_create(&t->idx_table[i].thread, NULL,
					    hash_root_fn, &t->idx_table[i]);
			if (rc) {
				ods_lerror("Error %d creating async completion thread %d\n",
					   errno, i);
				goto err_0;
			}
		}
		break;
	default:
		return EINVAL;
	}
	return 0;
 err_0:
	/* Kill all the threads and join */
	for (i = 0; i < t->udata->table_size; i++)
		t->idx_table[i].state = H2HTBL_STATE_STOPPED;
	for (i = 0; i < t->udata->table_size; i++) {
		if (t->idx_table[i].thread) {
			pthread_join(t->idx_table[i].thread, NULL);
		}
	}
	return ENOMEM;
}

ods_idx_rt_opts_t h2htbl_rt_opts_get(ods_idx_t idx)
{
	h2htbl_t t = idx->priv;
	return t->rt_opts;
}

static struct ods_idx_provider h2htbl_provider = {
	.get_type = h2htbl_get_type,
	.init = h2htbl_init,
	.open = h2htbl_open,
	.close = h2htbl_close,
	.rt_opts_set = h2htbl_rt_opts_set,
	.rt_opts_get = h2htbl_rt_opts_get,
	.commit = h2htbl_commit,
	.insert = h2htbl_insert,
	.insert_no_lock = h2htbl_insert_no_lock,
	.visit = h2htbl_visit,
	.update = h2htbl_update,
	.delete = h2htbl_delete,
	.max = h2htbl_max,
	.min = h2htbl_min,
	.find = h2htbl_find,
	.find_lub = h2htbl_find_lub,
	.find_glb = h2htbl_find_glb,
	.stat = h2htbl_stat,
	.iter_new = h2htbl_iter_new,
	.iter_delete = h2htbl_iter_delete,
	.iter_find = h2htbl_iter_find,
	.iter_find_lub = h2htbl_iter_find_lub,
	.iter_find_glb = h2htbl_iter_find_glb,
	.iter_find_first = h2htbl_iter_find_first,
	.iter_find_last = h2htbl_iter_find_last,
	.iter_begin = h2htbl_iter_begin,
	.iter_end = h2htbl_iter_end,
	.iter_next = h2htbl_iter_next,
	.iter_prev = h2htbl_iter_prev,
	.iter_pos_set = h2htbl_iter_pos_set,
	.iter_pos_get = h2htbl_iter_pos_get,
	.iter_pos_put = h2htbl_iter_pos_put,
	.iter_entry_delete = h2htbl_iter_entry_delete,
	.iter_key = h2htbl_iter_key,
	.iter_data = h2htbl_iter_data,
	.print_idx = print_idx,
	.print_info = print_info
};

struct ods_idx_provider *get(void)
{
	return &h2htbl_provider;
}

static void __attribute__ ((constructor)) h2htbl_lib_init(void)
{
}

static void __attribute__ ((destructor)) h2htbl_lib_term(void)
{
}
