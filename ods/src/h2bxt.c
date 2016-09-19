/*
 * Copyright (c) 2014-2015 Open Grid Computing, Inc. All rights reserved.
 *
 * Confidential and Proprietary
 */

/*
 * Author: Tom Tucker tom at ogc dot us
 */
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
#include <ods/ods.h>
#include <string.h>
#include <time.h>
#include "h2bxt.h"

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#define H2BXT_DEFAULT_ORDER		5
#define H2BXT_DEFAULT_TABLE_SIZE	19

#if 0
void dump_node(h2bxt_t t, ods_idx_t idx, ods_obj_t n, int ent, int indent, FILE *fp)
{
	int i;
	if (fp) fprintf(fp, "%p - %*s%s[%d] | %p : ", (void *)(unsigned long)NODE(n)->parent,
	       indent, "", "NODE", ent, n->as.ptr);
	for (i = 0; i < NODE(n)->count; i++) {
		ods_key_t key = ods_ref_as_obj(t->ods, N_ENT(n,i).key_ref);
		size_t keylen = ods_idx_key_str_size(idx, key);
		char *keystr = malloc(keylen);
		if (key) {
			ods_key_to_str(idx, key, keystr, keylen);
		} else {
			keystr[0] = '-';
			keystr[1] = '\0';
		}
		if (fp) fprintf(fp, "%s:%p, ",
				keystr,
		       (void *)(unsigned long)N_ENT(n, i).node_ref);
		free(keystr);
		ods_obj_put(key);
	}
	if (fp) fprintf(fp, "\n");
}

void dump_leaf(h2bxt_t t, ods_idx_t idx, ods_obj_t n, int ent, int indent, FILE *fp)
{
	int i;
	if (fp) fprintf(fp, "%p - %*s%s[%d] | %p :\n", (void *)(unsigned long)NODE(n)->parent,
	       indent-2, "", "LEAF", ent, n->as.ptr);
	for (i = 0; i < NODE(n)->count; i++) {
		ods_obj_t rec;
		ods_key_t key;
		ods_ref_t tail = L_ENT(n,i).tail_ref;
		ods_ref_t head = L_ENT(n,i).head_ref;
		if (fp) fprintf(fp, "%*sENTRY#%2d : head %p tail %p\n", indent+11, "", i,
			(void *)(unsigned long)head,
			(void *)(unsigned long)tail);
		do {
			rec = ods_ref_as_obj(t->ods, head);
			key = ods_ref_as_obj(t->ods, REC(rec)->key_ref);
			size_t keylen = ods_idx_key_str_size(idx, key);
			char *keystr = malloc(keylen);
			if (key) {
				ods_key_to_str(idx, key, keystr, keylen);
			} else {
				keystr[0] = '-';
				keystr[1] = '\0';
			}
			if (fp) fprintf(fp,
				"%*srec_ref %p key %s ref %p user %p prev_ref %p next_ref %p\n",
				indent+16, "",
				(void *)(unsigned long)head,
				keystr,
				(void *)*(unsigned long *)&REC(rec)->value.bytes[0],
				(void *)*(unsigned long *)&REC(rec)->value.bytes[8],
				(void *)(unsigned long)REC(rec)->prev_ref,
				(void *)(unsigned long)REC(rec)->next_ref);
			free(keystr);
			if (head == tail)
				break;
			head = REC(rec)->next_ref;
			ods_obj_put(rec);
		} while (head);
		ods_obj_put(rec);
	}
	if (fp) fprintf(fp, "\n");
}

static void print_node(ods_idx_t idx, int ent, ods_obj_t n, int indent, FILE *fp)
{
	h2bxt_t t = idx->priv;
	int i;

	if (!n) {
		if (fp) fprintf(fp, "<nil>\n");
		return;
	}

	/* Print this node */
	if (NODE(n)->is_leaf && NODE(n)->parent)
		indent += 4;
	if (NODE(n)->is_leaf)
		dump_leaf(t, idx, n, ent, indent, fp);
	else
		dump_node(t, idx, n, ent, indent, fp);
	fflush(stdout);
	if (NODE(n)->is_leaf)
		return;
	/* Now print all it's children */
	for (i = 0; i < NODE(n)->count; i++) {
		ods_obj_t node = ods_ref_as_obj(t->ods, N_ENT(n,i).node_ref);
		print_node(idx, i, node, indent + 2, fp);
		ods_obj_put(node);
	}
}

#endif
static void print_idx(ods_idx_t idx, FILE *fp)
{
#if 0
	h2bxt_t t = idx->priv;
	ods_obj_t node;
	int i;
	for (i = 0; i < t->udata->table_size; i++) {
		node = ods_ref_as_obj(t->ods, t->udata->hash_table[i]);
		print_node(idx, 0, node, 0, fp);
		ods_obj_put(node);
	}
#endif
}

static void print_info(ods_idx_t idx, FILE *fp)
{
	h2bxt_t t = idx->priv;
	fprintf(fp, "%*s : %d\n", 12, "Table Size", t->udata->table_size);
	fprintf(fp, "%*s : %d\n", 12, "Tree Order", t->udata->order);
	fprintf(fp, "%*s : %d\n", 12, "Lock", t->udata->lock);
	fprintf(fp, "%*s : %d\n", 12, "Cardinality", 0); /* TODO sum each table */
	fprintf(fp, "%*s : %d\n", 12, "Duplicates", 0);
	fflush(fp);
}

static int h2bxt_open(ods_idx_t idx)
{
	char path_buf[PATH_MAX];
	const char *path;
	const char *base = NULL;
	ods_obj_t udata_obj;
	h2bxt_t t;
	int i, rc;
	struct stat dir_sb;

	udata_obj = ods_get_user_data(idx->ods);
	if (!udata_obj)
		return EINVAL;
	t = calloc(1, sizeof *t);
	if (!t)
		goto err_0;
	t->hash_fn = fnv_hash_a1_64;
	t->udata_obj = udata_obj;
	t->udata = H2UDATA(udata_obj);
	t->idx_table = calloc(t->udata->table_size, sizeof *t->idx_table);
	if (!t->idx_table)
		goto err_1;
	ods_spin_init(&t->lock, &t->udata->lock);
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
		t->idx_table[i] = ods_idx_open(path_buf, ODS_PERM_RW);
		if (!t->idx_table[i]) {
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

static int h2bxt_init(ods_t ods, const char *type, const char *key, const char *argp)
{
	char path_buf[PATH_MAX];
	const char *path;
	const char *base;
	ods_obj_t udata;
	char *value;
	uint32_t htlen = 0;
	uint32_t order = 0;
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

	order = arg_int_value(argp, "ORDER");
	if (!order)
		order = H2BXT_DEFAULT_ORDER;

	srandom(time(NULL));
	seed = arg_int_value(argp, "SEED");
	if (!seed)
		seed = (uint32_t)random();

	htlen = arg_int_value(argp, "SIZE");
	if (!htlen)
		htlen = H2BXT_DEFAULT_TABLE_SIZE;

	H2UDATA(udata)->table_size = htlen;
	H2UDATA(udata)->hash_seed = seed;
	H2UDATA(udata)->order = order;
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
		rc = ods_idx_create(path_buf, sb.st_mode, "BXTREE", key, argp);
		if (rc)
			goto out;
	}
 out:
	ods_obj_put(udata);
	free((void *)base);
	return rc;
}

static void h2bxt_close(ods_idx_t idx)
{
	h2bxt_t t = idx->priv;
	int bkt;
	assert(t);
	idx->priv = NULL;
	for (bkt = 0; bkt < t->udata->table_size; bkt++)
		ods_idx_close(t->idx_table[bkt], ODS_COMMIT_ASYNC);
	ods_obj_put(t->udata_obj);
	ods_idx_close(t->ods_idx, ODS_COMMIT_ASYNC);
	free(t->idx_table);
	free(t);
}

static uint64_t hash_key(h2bxt_t t, ods_key_t key)
{
	ods_key_value_t kv = ods_key_value(key);
	return t->hash_fn((const char *)kv->value, kv->len, t->udata->hash_seed) % t->udata->table_size;
}

static int h2bxt_visit(ods_idx_t idx, ods_key_t key, ods_visit_cb_fn_t cb_fn, void *ctxt)
{
	h2bxt_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_visit(t->idx_table[hash], key, cb_fn, ctxt);
}

static int h2bxt_update(ods_idx_t idx, ods_key_t key, ods_idx_data_t data)
{
	h2bxt_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_update(t->idx_table[hash], key, data);
}

static int h2bxt_find(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	h2bxt_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_find(t->idx_table[hash], key, data);
}

static int h2bxt_find_lub(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	h2bxt_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_find_lub(t->idx_table[hash], key, data);
}

static int h2bxt_find_glb(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	h2bxt_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_find_glb(t->idx_table[hash], key, data);
}

static int h2bxt_insert(ods_idx_t idx, ods_key_t new_key, ods_idx_data_t data)
{
	h2bxt_t t = idx->priv;
	uint64_t hash = hash_key(t, new_key);
	return ods_idx_insert(t->idx_table[hash], new_key, data);
}

static int h2bxt_max(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	int bkt;
	h2bxt_t t = idx->priv;
	ods_key_t max_key = NULL;
	ods_key_t idx_key;
	struct ods_idx_data_s idx_data;
	int rc = ENOENT;

	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		rc = ods_idx_max(t->idx_table[bkt], &idx_key, &idx_data);
		if (rc)
			continue;
		rc = 0;
		if (max_key) {
			if (ods_key_cmp(t->idx_table[bkt], idx_key, max_key) > 0) {
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

static int h2bxt_min(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	int bkt;
	h2bxt_t t = idx->priv;
	ods_key_t min_key = NULL;
	ods_key_t idx_key;
	struct ods_idx_data_s idx_data;
	int rc = ENOENT;

	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		rc = ods_idx_min(t->idx_table[bkt], &idx_key, &idx_data);
		if (rc)
			continue;
		rc = 0;
		if (min_key) {
			if (ods_key_cmp(t->idx_table[bkt], idx_key, min_key) > 0) {
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

static int h2bxt_delete(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	h2bxt_t t = idx->priv;
	uint64_t hash = hash_key(t, key);
	return ods_idx_delete(t->idx_table[hash], key, data);
}

typedef struct iter_entry_s {
	h2bxt_iter_t iter;
	ods_key_t key;
	ods_idx_data_t data;
	int bkt;
	struct rbn rbn;
} *iter_entry_t;

static int entry_cmp(void *tree_key, void *key)
{
	iter_entry_t a = tree_key;
	iter_entry_t b = key;
	ods_iter_t oi = (ods_iter_t)a->iter;
	return ods_key_cmp(oi->idx, a->key, b->key);
}

static void iter_cleanup(h2bxt_t t, h2bxt_iter_t iter)
{
	struct rbn *rbn;
	while ((rbn = rbt_min(&iter->next_tree))) {
		iter_entry_t ent = container_of(rbn, struct iter_entry_s, rbn);
		rbt_del(&iter->next_tree, rbn);
		ods_obj_put(ent->key);
		free(ent);
	}
}

static void h2bxt_iter_delete(ods_iter_t i)
{
	int bkt;
	h2bxt_t t = i->idx->priv;
	h2bxt_iter_t iter = (h2bxt_iter_t)i;

	/* Empty the RBT */
	iter_cleanup(t, iter);

	/* Destroy each sub-iter */
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		if (iter->iter_table[bkt])
			ods_iter_delete(iter->iter_table[bkt]);
	}
	free(iter);
}

static ods_iter_t h2bxt_iter_new(ods_idx_t idx)
{
	int bkt;
	h2bxt_t t = idx->priv;
	h2bxt_iter_t iter;
	size_t iter_size = sizeof(*iter) + (t->udata->table_size * sizeof(ods_idx_t));
	iter = calloc(1, iter_size);
	if (!iter)
		return NULL;
	rbt_init(&iter->next_tree, entry_cmp);
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		iter->iter_table[bkt] = ods_iter_new(t->idx_table[bkt]);
		if (!iter->iter_table[bkt]) {
			h2bxt_iter_delete((ods_iter_t)iter);
			return NULL;
		}
	}
	return (ods_iter_t)iter;
}

static iter_entry_t alloc_ent(h2bxt_iter_t iter, int bkt)
{
	iter_entry_t ent = malloc(sizeof *ent);
	if (!ent)
		goto out;
	ent->iter = iter;
	ent->key = ods_iter_key(iter->iter_table[bkt]);
	ent->data = ods_iter_data(iter->iter_table[bkt]);
	ent->bkt = bkt;
	assert(ent->key);
	rbn_init(&ent->rbn, ent);
 out:
	return ent;
}

static int h2bxt_iter_begin(ods_iter_t oi)
{
	int bkt, rc, rv = ENOENT;
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	h2bxt_t t = oi->idx->priv;

	iter_cleanup(t, iter);
	iter->dir = H2BXT_ITER_FWD;

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
		rbt_ins(&iter->next_tree, &ent->rbn);
	}
	return rv;
}

static int h2bxt_iter_end(ods_iter_t oi)
{
	int bkt, rc, rv = ENOENT;
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	h2bxt_t t = oi->idx->priv;

	iter_cleanup(t, iter);
	iter->dir = H2BXT_ITER_REV;

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
		rbt_ins(&iter->next_tree, &ent->rbn);
	}
	return rv;
}

static ods_idx_data_t NULL_DATA = {
	.bytes = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

/*
 * Return the min valued key in the rbt of the iterator
 */
static ods_key_t h2bxt_iter_key_fwd(ods_iter_t oi)
{
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	struct rbn *rbn;
	rbn = rbt_min(&iter->next_tree);
	if (!rbn)
		return NULL;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	return ods_obj_get(ent->key);
}

static ods_key_t h2bxt_iter_key_rev(ods_iter_t oi)
{
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	struct rbn *rbn;
	rbn = rbt_max(&iter->next_tree);
	if (!rbn)
		return NULL;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	return ods_obj_get(ent->key);
}
static ods_key_t h2bxt_iter_key(ods_iter_t oi)
{
	h2bxt_iter_t iter = (typeof(iter))oi;
	switch (iter->dir) {
	case H2BXT_ITER_FWD:
		return h2bxt_iter_key_fwd(oi);
	case H2BXT_ITER_REV:
		return h2bxt_iter_key_rev(oi);
	default:
		assert(0 == "Invalid dir field in iter");
	}
	return NULL;
}

static ods_idx_data_t h2bxt_iter_data_fwd(ods_iter_t oi)
{
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	struct rbn *rbn = rbt_min(&iter->next_tree);
	if (!rbn)
		return NULL_DATA;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	return ent->data;
}
static ods_idx_data_t h2bxt_iter_data_rev(ods_iter_t oi)
{
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	struct rbn *rbn = rbt_max(&iter->next_tree);
	if (!rbn)
		return NULL_DATA;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	return ent->data;
}
static ods_idx_data_t h2bxt_iter_data(ods_iter_t oi)
{
	h2bxt_iter_t iter = (typeof(iter))oi;
	switch (iter->dir) {
	case H2BXT_ITER_FWD:
		return h2bxt_iter_data_fwd(oi);
	case H2BXT_ITER_REV:
		return h2bxt_iter_data_rev(oi);
	default:
		assert(0 == "Invalid dir field in iter");
	}
	return NULL_DATA;
}

static int h2bxt_iter_find_first(ods_iter_t oi, ods_key_t key)
{
	int rc, rv = ENOENT;
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	h2bxt_t t = oi->idx->priv;
	uint64_t bkt = hash_key(t, key);

	iter_cleanup(t, iter);
	iter->dir = H2BXT_ITER_FWD;

	rc = ods_iter_find_first(iter->iter_table[bkt], key);
	if (rc)
		return ENOENT;
	ent = alloc_ent(iter, bkt);
	if (!ent)
		return ENOMEM;
	rbt_ins(&iter->next_tree, &ent->rbn);
	return rv;
}

static int h2bxt_iter_find_last(ods_iter_t oi, ods_key_t key)
{
	int rc;
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	h2bxt_t t = oi->idx->priv;
	uint64_t bkt = hash_key(t, key);

	iter_cleanup(t, iter);
	iter->dir = H2BXT_ITER_REV;

	rc = ods_iter_find_last(iter->iter_table[bkt], key);
	if (rc)
		return ENOENT;
	ent = alloc_ent(iter, bkt);
	if (!ent)
		return ENOMEM;
	rbt_ins(&iter->next_tree, &ent->rbn);
	return 0;
}

static int h2bxt_iter_find(ods_iter_t oi, ods_key_t key)
{
	int rc;
	iter_entry_t ent;
	h2bxt_iter_t iter = (typeof(iter))oi;
	h2bxt_t t = oi->idx->priv;
	uint64_t bkt = hash_key(t, key);

	iter_cleanup(t, iter);
	iter->dir = H2BXT_ITER_FWD;

	rc = ods_iter_find(iter->iter_table[bkt], key);
	if (rc)
		return ENOENT;
	ent = alloc_ent(iter, bkt);
	if (!ent)
		return ENOMEM;
	rbt_ins(&iter->next_tree, &ent->rbn);
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
static int h2bxt_iter_find_lub(ods_iter_t oi, ods_key_t key)
{
	h2bxt_t t = oi->idx->priv;
	h2bxt_iter_t iter = (typeof(iter))oi;
	iter_entry_t ent;
	int rv = ENOENT;
	int bkt;

	iter_cleanup(t, iter);
	iter->dir = H2BXT_ITER_FWD;

	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		if (ods_iter_find_lub(iter->iter_table[bkt], key))
			continue;
		rv = 0;
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		rbt_ins(&iter->next_tree, &ent->rbn);
	}
	return rv;
}

static int h2bxt_iter_find_glb(ods_iter_t oi, ods_key_t key)
{
	h2bxt_t t = oi->idx->priv;
	h2bxt_iter_t iter = (typeof(iter))oi;
	iter_entry_t ent;
	int rv = ENOENT;
	int bkt;

	iter_cleanup(t, iter);
	iter->dir = H2BXT_ITER_FWD;

	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		if (ods_iter_find_glb(iter->iter_table[bkt], key))
			continue;
		rv = 0;
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		rbt_ins(&iter->next_tree, &ent->rbn);
	}
	return rv;
}

static int h2bxt_iter_next(ods_iter_t oi)
{
	h2bxt_t t = oi->idx->priv;
	h2bxt_iter_t iter = (typeof(iter))oi;
	struct rbn *rbn;
	iter_entry_t ent;
	int bkt;

	iter->dir = H2BXT_ITER_FWD; /* force direction to forward */

	/* Delete the min from the tree. */
	rbn = rbt_min(&iter->next_tree);
	if (!rbn)
		return ENOENT;
	ent = container_of(rbn, struct iter_entry_s, rbn);
	rbt_del(&iter->next_tree, rbn);
	ods_obj_put(ent->key);
	bkt = ent->bkt;
	free(ent);

	/* Get the next entry for this bucket and insert into the tree */
	if (0 == ods_iter_next(iter->iter_table[bkt])) {
		ent = alloc_ent(iter, bkt);
		if (!ent)
			return ENOMEM;
		rbt_ins(&iter->next_tree, &ent->rbn);
	}
	if (rbt_empty(&iter->next_tree))
		return ENOENT;
	return 0;
}

static int h2bxt_iter_prev(ods_iter_t oi)
{
	return ENOENT;
}

#define POS_PAD 0x48324258 /* 'H2BX' */

struct h2bxt_pos_s {
	uint32_t pad;
	uint32_t ent;
	ods_ref_t rec_ref;
};

static int h2bxt_iter_set(ods_iter_t oi, const ods_pos_t pos_)
{
	return 0;
}

static int h2bxt_iter_pos(ods_iter_t oi, ods_pos_t pos_)
{
	return 0;
}

static int h2bxt_iter_pos_delete(ods_iter_t oi, ods_pos_t pos_)
{
	return ENOENT;
}

static const char *h2bxt_get_type(void)
{
	return "H2BXT";
}

static void h2bxt_commit(ods_idx_t idx)
{
	h2bxt_t t = idx->priv;
	int bkt;
	ods_idx_commit(t->ods_idx, ODS_COMMIT_SYNC);
	for (bkt = 0; bkt < t->udata->table_size; bkt++)
		ods_idx_commit(t->idx_table[bkt], ODS_COMMIT_SYNC);
}

int h2bxt_stat(ods_idx_t idx, ods_idx_stat_t idx_sb)
{
	struct ods_idx_stat_s bkt_sb;
	h2bxt_t t = idx->priv;
	int bkt;
	memset(idx_sb, 0, sizeof(*idx_sb));
	for (bkt = 0; bkt < t->udata->table_size; bkt++) {
		ods_idx_stat(t->idx_table[bkt], &bkt_sb);
		idx_sb->cardinality += bkt_sb.cardinality;
		idx_sb->duplicates += bkt_sb.duplicates;
		idx_sb->size += bkt_sb.size;
	}
	return 0;
}

static struct ods_idx_provider h2bxt_provider = {
	.get_type = h2bxt_get_type,
	.init = h2bxt_init,
	.open = h2bxt_open,
	.close = h2bxt_close,
	.commit = h2bxt_commit,
	.insert = h2bxt_insert,
	.visit = h2bxt_visit,
	.update = h2bxt_update,
	.delete = h2bxt_delete,
	.max = h2bxt_max,
	.min = h2bxt_min,
	.find = h2bxt_find,
	.find_lub = h2bxt_find_lub,
	.find_glb = h2bxt_find_glb,
	.stat = h2bxt_stat,
	.iter_new = h2bxt_iter_new,
	.iter_delete = h2bxt_iter_delete,
	.iter_find = h2bxt_iter_find,
	.iter_find_lub = h2bxt_iter_find_lub,
	.iter_find_glb = h2bxt_iter_find_glb,
	.iter_find_first = h2bxt_iter_find_first,
	.iter_find_last = h2bxt_iter_find_last,
	.iter_begin = h2bxt_iter_begin,
	.iter_end = h2bxt_iter_end,
	.iter_next = h2bxt_iter_next,
	.iter_prev = h2bxt_iter_prev,
	.iter_set = h2bxt_iter_set,
	.iter_pos = h2bxt_iter_pos,
	.iter_pos_delete = h2bxt_iter_pos_delete,
	.iter_key = h2bxt_iter_key,
	.iter_data = h2bxt_iter_data,
	.print_idx = print_idx,
	.print_info = print_info
};

struct ods_idx_provider *get(void)
{
	return &h2bxt_provider;
}

static void __attribute__ ((constructor)) h2bxtlib_init(void)
{
}

static void __attribute__ ((destructor)) h2bxtlib_term(void)
{
}
