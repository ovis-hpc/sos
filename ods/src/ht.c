/*
 * Copyright (c) 2014-2021 Open Grid Computing, Inc. All rights reserved.
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
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <string.h>
#include <assert.h>
#include <ods/ods.h>
#include "ht.h"
#include "fnv_hash.h"

static void delete_entry(ht_t t, ods_obj_t ent, int64_t bkt);

/* #define HT_DEBUG */
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

void print_bkt(ods_idx_t idx, ht_t t, int64_t bkt, FILE *fp)
{
	ods_obj_t ent;
	ods_ref_t next_ref;
	ht_tbl_t ht = t->htable;
	fprintf(fp, "%ld : ", bkt);
	for (ent = ods_ref_as_obj(t->ods, ht->table[bkt].head_ref); ent;
	     ent = ods_ref_as_obj(t->ods, next_ref)) {
		ods_key_t key = ods_ref_as_obj(t->ods, HENT(ent)->key_ref);
		size_t keylen = ods_idx_key_str_size(idx, key);
		char *keystr = malloc(keylen);
		if (key) {
			ods_key_to_str(idx, key, keystr, keylen);
		} else {
			keystr[0] = '-';
			keystr[1] = '\0';
		}
		if (fp) fprintf(fp, "%s:%p, ",
				keystr,	(void *)(unsigned long)ods_obj_ref(ent));
		free(keystr);
		ods_obj_put(key);
		next_ref = HENT(ent)->next_ref;
		ods_obj_put(ent);
	}
}

static void print_idx(ods_idx_t idx, FILE *fp)
{
	ht_t t = idx->priv;
	ht_tbl_t ht = t->htable;
	int64_t bkt;
	for (bkt = ht->first_bkt; bkt <= ht->last_bkt; bkt++) {
		if (ht->table[bkt].head_ref)
			print_bkt(idx, t, bkt, fp);
	}
}
static void print_info(ods_idx_t idx, FILE *fp)
{
	ht_t t = idx->priv;
	fprintf(fp, "%*s : %lx\n", 12, "Hash Table Ref", t->udata->htable_ref);
	fprintf(fp, "%*s : %lu\n", 12, "Table Size", t->udata->htable_size);
	fprintf(fp, "%*s : %d\n", 12, "Hash Seed", t->udata->hash_seed);
	fprintf(fp, "%*s : %lu\n", 12, "Hash Type", t->udata->hash_type);
	fprintf(fp, "%*s : %d\n", 12, "Client Count", t->udata->client_count);
	fprintf(fp, "%*s : %d\n", 12, "Lock", t->udata->lock);
	fprintf(fp, "%*s : %lu\n", 12, "Max Bucket", t->udata->max_bkt);
	fprintf(fp, "%*s : %lu\n", 12, "Max Bucket Len", t->udata->max_bkt_len);
	fprintf(fp, "%*s : %d\n", 12, "Cardinality", t->udata->card);
	fprintf(fp, "%*s : %d\n", 12, "Duplicates", t->udata->dups);
	fflush(fp);

	int bkt;
	int max_bkt_len = t->udata->max_bkt_len;
	uint32_t *counts = calloc(max_bkt_len + 1, sizeof(uint32_t));
	for (bkt = 0; bkt <= t->udata->max_bkt; bkt++) {
		uint32_t cnt = t->htable->table[bkt].count;
		if (cnt)
			counts[cnt] += 1;
	}
	for (bkt = 1; bkt <= max_bkt_len; bkt++) {
		fprintf(fp, "%12d %24d\n", bkt, counts[bkt]);
	}
	fflush(fp);
	free(counts);
}

#ifdef HT_DEBUG
static void verify_node(ht_t t, ods_obj_t node)
{
	int i, j;
	ods_obj_t rec;
	ods_ref_t rec_ref;
	if (!node)
	    return;
	assert(NODE(node)->count <= t->udata->order);
	for (i = 0; i < NODE(node)->count; i++) {
		if (NODE(node)->is_leaf) {
			for (j = i + 1; j < NODE(node)->count; j++) {
				assert(L_ENT(node,i).head_ref != L_ENT(node,j).head_ref);
				assert(L_ENT(node,i).tail_ref != L_ENT(node,j).tail_ref);
			}
			assert(L_ENT(node,i).head_ref && L_ENT(node,i).tail_ref);
			rec_ref = L_ENT(node, i).head_ref;
			rec = ods_ref_as_obj(t->ods, rec_ref);
			assert(REC(rec)->next_ref != rec_ref);
			assert(REC(rec)->prev_ref != rec_ref);
			ods_obj_put(rec);
		} else {
			assert(N_ENT(node,i).node_ref);
			if (i)
				assert(N_ENT(node,i).node_ref != N_ENT(node,i-1).node_ref);
		}
	}
}
#endif

static int ht_open(ods_idx_t idx)
{
	ods_obj_t udata;
	ht_t t;
	udata = ods_get_user_data(idx->ods);
	if (!udata)
		return EINVAL;
	t = calloc(1, sizeof *t);
	if (!t) {
		ods_obj_put(udata);
		return ENOMEM;
	}
	t->ods = idx->ods;
	t->udata_obj = udata;
	t->udata = UDATA(udata);
	t->htable_obj = ods_ref_as_obj(t->ods, UDATA(udata)->htable_ref);
	t->htable = HTBL(t->htable_obj);
	t->ods = idx->ods;
	t->comparator = idx->idx_class->cmp->compare_fn;
	switch (t->udata->hash_type) {
	case HT_HASH_FNV_32:
		t->hash_fn = fnv_hash_a1_32;
		break;
	case HT_HASH_FNV_64:
		t->hash_fn = fnv_hash_a1_64;
		break;
	default:
		assert(0 == "Hash table udata is corrupted.");
	}
	idx->priv = t;
	ods_atomic_inc(&t->udata->client_count);
	return 0;
}

static int ht_init(ods_t ods, const char *idx_type, const char *key_type, const char *argp)
{
	char arg_buf[ODS_IDX_ARGS_LEN];
	ods_obj_t udata;
	char *name, *type;
	int hash_type = HT_HASH_FNV_32;
	int64_t htable_size = HT_DEF_TBL_SIZE;
	udata = ods_get_user_data(ods);
	if (!udata)
		return EINVAL;

	if (argp) {
		char *arg = strcasestr(argp, "TYPE");
		if (arg) {
			strcpy(arg_buf, arg);
			name = strtok(arg_buf, "=");
			type = strtok(NULL, "=");
			if (name && (0 == strcasecmp(name, "TYPE"))) {
				if (type && (0 == strncasecmp(type, "fnv_32", 6)))
					hash_type = HT_HASH_FNV_32;
				else if (type && (0 == strncasecmp(type, "fnv_64", 6)))
					hash_type = HT_HASH_FNV_64;
				else
					return EINVAL;
			}
		}
		arg = strcasestr(argp, "SIZE");
		if (arg) {
			strcpy(arg_buf, arg);
			name = strtok(arg_buf, "=");
			if (name && (0 == strcasecmp(name, "SIZE")))
				htable_size = strtoul(strtok(NULL, "="), NULL, 0);
		}
	}
	ods_obj_t ht;
	size_t ht_size = sizeof(struct ht_tbl_s) + (htable_size * sizeof(struct ht_bkt_s));
	ht = ods_obj_alloc_extend(ods, ht_size, HT_EXTEND_SIZE);
	if (!ht)
		return ENOMEM;

	HTBL(ht)->first_bkt = -1;
	HTBL(ht)->last_bkt = -1;
	UDATA(udata)->htable_ref = ods_obj_ref(ht);
	UDATA(udata)->htable_size = htable_size;
	UDATA(udata)->hash_type = hash_type;
	UDATA(udata)->client_count = 0;
	UDATA(udata)->lock = 0;
	UDATA(udata)->card = 0;
	UDATA(udata)->dups = 0;
	ods_obj_update(udata);
	ods_obj_put(udata);
	ods_obj_update(ht);
	ods_obj_put(ht);
	return 0;
}

static void ht_close_(ht_t t)
{
	ods_atomic_dec(&t->udata->client_count);
	ods_obj_put(t->udata_obj);
	ods_obj_put(t->htable_obj);
	free(t);
}

static void ht_close(ods_idx_t idx)
{
	ht_t t = idx->priv;
	assert(t);
	idx->priv = NULL;
	ht_close_(t);
}

static int64_t hash_bkt(ht_t t, const char *key, size_t key_len)
{
	uint64_t hash = t->hash_fn(key, key_len, t->udata->hash_seed);
	int64_t bkt = (int64_t)(hash % t->udata->htable_size);
	return bkt;
}

static ods_obj_t find_entry(ht_t t, ods_key_t key, int64_t *p_bkt)
{
	int64_t c;
	ods_key_t entry_key;
	ods_obj_t ent;
	ods_ref_t ref;
	ht_tbl_t ht = t->htable;
	ods_key_value_t kv = HKEY(key);
	int64_t bkt = hash_bkt(t, (const char *)kv->value, kv->len);
	if (p_bkt)
		/* Return bkt regardless of whether key matches */
		*p_bkt = bkt;
	if (!ht->table[bkt].head_ref)
		return NULL;
	/* Search the bucket list for a match */
	ref = ht->table[bkt].head_ref;
	for (ref = ht->table[bkt].head_ref; ref; ) {
		ent = ods_ref_as_obj(t->ods, ref);
		entry_key = ods_ref_as_obj(t->ods, HENT(ent)->key_ref);
		c = t->comparator(entry_key, key);
		ods_obj_put(entry_key);
		if (0 == c)
			return ent;
		ref = HENT(ent)->next_ref;
		ods_obj_put(ent);
	}
	return NULL;
}

static int ht_find(ods_idx_t o_idx, ods_key_t key, ods_idx_data_t *data)
{
	int rc;
	ht_t t = o_idx->priv;
	ods_obj_t ent;

#ifdef HT_THREAD_SAFE
	if (ods_lock(o_idx->ods, 0, NULL))
		return EBUSY;
#endif
	ent = find_entry(t, key, NULL);
	if (ent) {
		*data = HENT(ent)->value;
		rc = 0;
		ods_obj_put(ent);
	} else {
		rc = ENOENT;
	}
#ifdef HT_THREAD_SAFE
	ods_unlock(o_idx->ods, 0);
#endif
	return rc;
}

static int ht_update(ods_idx_t o_idx, ods_key_t key, ods_idx_data_t data)
{
	ht_t t = o_idx->priv;
	int rc;
	ods_obj_t ent;

#ifdef HT_THREAD_SAFE
	if (ods_lock(o_idx->ods, 0, NULL))
		return EBUSY;
#endif
	ent = find_entry(t, key, NULL);
	if (ent) {
		HENT(ent)->value = data;
		ods_obj_put(ent);
		rc = 0;
	} else {
		rc = ENOENT;
	}
#ifdef HT_THREAD_SAFE
	ods_unlock(o_idx->ods, 0);
#endif
	return rc;
}

static int ht_find_lub(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	return ht_find(idx, key, data);
}

static int ht_find_glb(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	return ht_find(idx, key, data);
}

static ods_key_t key_new(ods_idx_t idx, ods_key_t key)
{
	ods_key_t akey = ods_key_alloc(idx, ods_key_len(key));
	if (!akey) {
		if (0 == ods_extend(idx->ods, ods_size(idx->ods) * 2))
			akey = ods_key_alloc(idx, ods_key_len(key));
	}
	if (akey)
		ods_key_copy(akey, key);
	return akey;
}

static ods_obj_t entry_new(ods_idx_t idx)
{
	return ods_obj_alloc_extend(idx->ods, sizeof(struct ht_entry_s), HT_EXTEND_SIZE);
}

static int insert_in_bucket(ods_idx_t o_idx, ods_key_t key,
			    ods_idx_data_t data, ods_obj_t ent, int64_t bkt)
{
	ht_t t = o_idx->priv;
	ht_tbl_t ht = t->htable;
	ods_obj_t next;

	assert(bkt < t->udata->htable_size);
	HENT(ent)->value = data;
	HENT(ent)->key_ref = ods_obj_ref(key);
	HENT(ent)->prev_ref = 0;
	HENT(ent)->next_ref = ht->table[bkt].head_ref;
	if (!ht->table[bkt].head_ref) {
		ht->table[bkt].tail_ref = ods_obj_ref(ent);
	} else {
		next = ods_ref_as_obj(t->ods, ht->table[bkt].head_ref);
		HENT(next)->prev_ref = ods_obj_ref(ent);
		ods_obj_update(next);
		ods_obj_put(next);
	}
	ht->table[bkt].head_ref = ods_obj_ref(ent);
	ht->table[bkt].count++;
	ods_obj_update_offset(t->htable_obj,
		(uint64_t)&((ht_tbl_t)0)->table[bkt], sizeof(ht->table[bkt]));
	if (ht->table[bkt].count > t->udata->max_bkt_len) {
		t->udata->max_bkt_len = ht->table[bkt].count;
		t->udata->max_bkt = bkt;
	}
	t->udata->card++;
	if (ht->first_bkt < 0) {
		ht->first_bkt = ht->last_bkt = bkt;
	} else {
		if (ht->first_bkt > bkt)
			ht->first_bkt = bkt;
		if (ht->last_bkt < bkt)
			ht->last_bkt = bkt;
	}
	ods_obj_update_offset(t->htable_obj, 0, sizeof(struct ht_tbl_s));
	ods_obj_update(t->udata_obj);
	return 0;
}

static int ht_insert(ods_idx_t o_idx, ods_key_t key, ods_idx_data_t data)
{
	ht_t t = o_idx->priv;
	int rc = ENOMEM;
	ods_key_t new_key;
	ods_obj_t ent;
	int64_t bkt;
	ods_key_value_t kv;
#ifdef HT_THREAD_SAFE
	if (ods_lock(o_idx->ods, 0, NULL))
		return EBUSY;
#endif
	ent = entry_new(o_idx);
	if (!ent)
		goto out_0;
	new_key = key_new(o_idx, key);
	if (!new_key) {
		ods_obj_delete(ent);
		goto out_1;
	}
	kv = HKEY(new_key);
	bkt = hash_bkt(t, (const char *)kv->value, kv->len);
	rc = insert_in_bucket(o_idx, new_key, data, ent, bkt);
	ods_obj_update(new_key);
	ods_obj_put(new_key);
 out_1:
	ods_obj_update(ent);
	ods_obj_put(ent);
 out_0:
#ifdef HT_THREAD_SAFE
	ods_unlock(o_idx->ods, 0);
#endif
	return rc;
}

static int ht_visit(ods_idx_t idx, ods_key_t key, ods_visit_cb_fn_t cb_fn, void *ctxt)
{
	ht_t t = idx->priv;
	int rc = ENOMEM;
	ods_idx_data_t data;
	ods_key_t new_key;
	ods_obj_t ent;
	int64_t bkt;
	int found;
	ods_visit_action_t act;
#ifdef HT_THREAD_SAFE
	if (ods_lock(idx->ods, 0, NULL))
		return EBUSY;
#endif
	ent = find_entry(t, key, &bkt);
	if (ent) {
		data = HENT(ent)->value;
		found = 1;
	} else {
		found = 0;
		memset(&data, 0, sizeof(data));
	}
	act = cb_fn(idx, key, &data, found, ctxt);
	switch (act) {
	case ODS_VISIT_ADD:
		if (ent) {
			/* App adding a dup */
			ods_obj_put(ent);
		}
		rc = ENOMEM;
		ent = entry_new(idx);
		if (!ent)
			break;
		new_key = key_new(idx, key);
		if (!new_key) {
			ods_obj_delete(ent);
			ods_obj_put(ent);
		}
		rc = insert_in_bucket(idx, new_key, data, ent, bkt);
		if (rc) {
			ods_obj_delete(new_key);
			ods_obj_delete(ent);
		}
		ods_obj_put(new_key);
		ods_obj_put(ent);
		break;
	case ODS_VISIT_UPD:
		if (!found) {
			rc = ENOENT;
		} else {
			rc = 0;
			HENT(ent)->value = data;
			ods_obj_put(ent);
		}
		break;
	case ODS_VISIT_DEL:
		if (found) {
			delete_entry(t, ent, bkt);
			rc = 0;
		} else {
			rc = ENOENT;
		}
		break;
	case ODS_VISIT_NOP:
		rc = 0;
		if (ent)
			ods_obj_put(ent);
		break;
	}
#ifdef HT_THREAD_SAFE
	ods_unlock(idx->ods, 0);
#endif
	return rc;
}

static int ht_max(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	int rc = ENOENT;
	ht_t t = idx->priv;
	ht_tbl_t ht = t->htable;
	ods_obj_t ent;
#ifdef HT_THREAD_SAFE
	ods_lock(idx->ods, 0, NULL);
#endif
	if (ht->last_bkt >= 0) {
		ent = ods_ref_as_obj(t->ods, ht->table[ht->first_bkt].tail_ref);
		if (ent) {
			if (data)
				*data = HENT(ent)->value;
			if (key)
				*key = ods_ref_as_obj(t->ods, HENT(ent)->key_ref);
			rc = 0;
			ods_obj_put(ent);
		} else
			rc = ENOMEM;
	}
#ifdef HT_THREAD_SAFE
	ods_unlock(idx->ods, 0);
#endif
	return rc;
}

static int ht_min(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	int rc = ENOENT;
	ht_t t = idx->priv;
	ht_tbl_t ht = t->htable;
	ods_obj_t ent;
#ifdef HT_THREAD_SAFE
	ods_lock(idx->ods, 0, NULL);
#endif
	if (ht->first_bkt >= 0) {
		ent = ods_ref_as_obj(t->ods, ht->table[ht->first_bkt].head_ref);
		if (ent) {
			if (data)
				*data = HENT(ent)->value;
			if (key)
				*key = ods_ref_as_obj(t->ods, HENT(ent)->key_ref);
			rc = 0;
			ods_obj_put(ent);
		} else
			rc = ENOMEM;
	}
#ifdef HT_THREAD_SAFE
	ods_unlock(idx->ods, 0);
#endif
	return rc;
}

static void delete_entry(ht_t t, ods_obj_t ent, int64_t bkt)
{
	ht_tbl_t ht = t->htable;
	ods_obj_t next, prev;
	next = ods_ref_as_obj(t->ods, HENT(ent)->next_ref);
	prev = ods_ref_as_obj(t->ods, HENT(ent)->prev_ref);
	if (prev)
		HENT(prev)->next_ref = HENT(ent)->next_ref;
	else
		ht->table[bkt].head_ref = HENT(ent)->next_ref;
	if (next)
		HENT(next)->prev_ref = HENT(ent)->prev_ref;
	else
		ht->table[bkt].tail_ref = HENT(ent)->prev_ref;
	ods_ref_delete(t->ods, HENT(ent)->key_ref);
	ods_obj_update(next);
	ods_obj_update(next);
	ods_obj_put(next);
	ods_obj_put(next);
	ods_obj_delete(ent);
	ods_obj_put(ent);
}

static int ht_delete(ods_idx_t o_idx, ods_key_t key, ods_idx_data_t *data)
{
	ht_t t = o_idx->priv;
	ods_obj_t ent;
	int64_t bkt;

#ifdef HT_THREAD_SAFE
	if (ods_lock(o_idx->ods, 0, NULL))
		return EBUSY;
#endif
	ent = find_entry(t, key, &bkt);
	if (!ent)
		goto noent;
	*data = HENT(ent)->value;
	delete_entry(t, ent, bkt);
#ifdef HT_THREAD_SAFE
	ods_unlock(o_idx->ods, 0);
#endif
#ifdef HT_DEBUG
	print_idx(idx, NULL);
#endif
	return 0;
 noent:
#ifdef HT_THREAD_SAFE
	ods_unlock(o_idx->ods, 0);
#endif
	return ENOENT;
}

static ods_iter_t ht_iter_new(ods_idx_t idx)
{
	ht_iter_t hi = calloc(1, sizeof *hi);
	return (ods_iter_t)hi;
}

static void ht_iter_delete(ods_iter_t i)
{
	ht_iter_t hi = (ht_iter_t)i;
	if (hi->ent)
		ods_obj_put(hi->ent);
	free(i);
}

static int ht_iter_begin(ods_iter_t oi)
{
	ht_iter_t hi = (ht_iter_t)oi;
	ht_t t = hi->iter.idx->priv;
	ht_tbl_t ht = t->htable;
	int64_t bkt;
	ods_obj_t ent;
	if (hi->ent) {
		ods_obj_put(hi->ent);
		hi->ent = NULL;
	}
	if (ht->first_bkt >= 0) {
		ent = ods_ref_as_obj(t->ods, ht->table[ht->first_bkt].head_ref);
		bkt = ht->first_bkt;
	} else
		return ENOENT;

	hi->bkt = bkt;
	hi->ent = ent;
	return 0;
}

static int ht_iter_end(ods_iter_t oi)
{
	ht_iter_t hi = (ht_iter_t)oi;
	ht_t t = hi->iter.idx->priv;
	ht_tbl_t ht = t->htable;
	int64_t bkt;
	ods_obj_t ent;

	if (hi->ent) {
		ods_obj_put(hi->ent);
		hi->ent = NULL;
	}
	if (ht->last_bkt >= 0) {
		ent = ods_ref_as_obj(t->ods, ht->table[ht->last_bkt].tail_ref);
		bkt = ht->last_bkt;
	} else
		return ENOENT;

	hi->bkt = bkt;
	hi->ent = ent;
	return 0;
}

static ods_key_t ht_iter_key(ods_iter_t oi)
{
	ht_iter_t hi = (ht_iter_t)oi;
	ht_t t = hi->iter.idx->priv;
	if (!hi->ent)
		return NULL;
	return ods_ref_as_obj(t->ods, HENT(hi->ent)->key_ref);
}

static struct ods_idx_data_s NO_DATA;
static ods_idx_data_t ht_iter_data(ods_iter_t oi)
{
	ht_iter_t hi = (ht_iter_t)oi;
	if (!hi->ent)
		return NO_DATA;
	return HENT(hi->ent)->value;
}

static int ht_iter_find_first(ods_iter_t oi, ods_key_t key)
{
	return ENOSYS;
}

static int ht_iter_find_last(ods_iter_t oi, ods_key_t key)
{
	return ENOSYS;
}

static int ht_iter_find(ods_iter_t oi, ods_key_t key)
{
	ht_iter_t hi = (ht_iter_t)oi;
	ht_t t = hi->iter.idx->priv;
	int64_t bkt;
	ods_obj_t ent = find_entry(t, key, &bkt);

	if (!ent)
		return ENOENT;
	if (hi->ent)
		ods_obj_put(hi->ent);
	hi->bkt = bkt;
	hi->ent = ent;
	return 0;
}

static int ht_iter_find_lub(ods_iter_t oi, ods_key_t key)
{
	return ht_iter_find(oi, key);
}
static int ht_iter_find_glb(ods_iter_t oi, ods_key_t key)
{
	return ht_iter_find(oi, key);
}

static int64_t next_bucket(ht_t t, int64_t bkt)
{
	ht_tbl_t ht = t->htable;
	for (bkt++; bkt <= ht->last_bkt; bkt++) {
		if (ht->table[bkt].head_ref)
			return bkt;
	}
	return -1;
}


static int __iter_next(ht_t t, ht_iter_t hi)
{
	ht_tbl_t ht = t->htable;
	ods_ref_t next_ref;
	ods_obj_t next_obj;

	if (!hi->ent)
		return ENOENT;

	next_ref = HENT(hi->ent)->next_ref;
	ods_obj_put(hi->ent);
	hi->ent = NULL;
	if (next_ref) {
		/* Still more objects in this bucket */
		next_obj = ods_ref_as_obj(t->ods, next_ref);
		if (!next_obj)
			return ENOMEM;
		hi->ent = next_obj;
		return 0;
	}
	int64_t bkt = next_bucket(t, hi->bkt);
	if (bkt < 0)
		return ENOENT;
	next_obj = ods_ref_as_obj(t->ods, ht->table[bkt].head_ref);
	if (!next_obj)
		return ENOMEM;
	hi->ent = next_obj;
	hi->bkt = bkt;
	return 0;
}

static int ht_iter_next(ods_iter_t oi)
{
	ht_iter_t hi = (ht_iter_t)oi;
	ht_t t = hi->iter.idx->priv;
	int rc;

#ifdef HT_THREAD_SAFE
	if (ods_lock(oi->idx->ods, 0, NULL))
		return EBUSY;
#endif

	rc = __iter_next(t, hi);

#ifdef HT_THREAD_SAFE
	ods_unlock(oi->idx->ods, 0);
#endif
	return rc;
}

static int64_t prev_bucket(ht_t t, int64_t bkt)
{
	ht_tbl_t ht = t->htable;
	for (bkt--; bkt >= ht->first_bkt; bkt--) {
		if (ht->table[bkt].tail_ref)
			return bkt;
	}
	return -1;
}

static int ht_iter_prev(ods_iter_t oi)
{
	ht_iter_t hi = (ht_iter_t)oi;
	ht_t t = hi->iter.idx->priv;
	ht_tbl_t ht = t->htable;
	ods_ref_t prev_ref;
	ods_obj_t prev_obj;
	if (!hi->ent)
		return ENOENT;
	prev_ref = HENT(hi->ent)->prev_ref;
	ods_obj_put(hi->ent);
	hi->ent = NULL;
	if (prev_ref) {
		/* Still more objects in this bucket */
		prev_obj = ods_ref_as_obj(t->ods, prev_ref);
		if (!prev_obj)
			return ENOMEM;
		hi->ent = prev_obj;
		return 0;
	}
	int64_t bkt = prev_bucket(t, hi->bkt);
	if (bkt < 0)
		return ENOENT;
	prev_obj = ods_ref_as_obj(t->ods, ht->table[bkt].tail_ref);
	if (!prev_obj)
		return ENOMEM;
	hi->ent = prev_obj;
	hi->bkt = bkt;
	return 0;
}

static int ht_iter_pos_set(ods_iter_t oi, const ods_pos_t pos_)
{
	ht_iter_t hi = (ht_iter_t)oi;
	ht_t t = hi->iter.idx->priv;
	ods_obj_t pos;

	pos = ods_ref_as_obj(t->ods, pos_->ref);
	if (!pos)
		return EINVAL;

	if (hi->ent)
		ods_obj_put(hi->ent);

	hi->ent = ods_ref_as_obj(t->ods, POS(pos)->ent_ref);
	if (!hi->ent)
		goto err_0;

	hi->bkt = POS(pos)->bkt;

	ods_obj_delete(pos);	/* POS are 1-time use */
	ods_obj_put(pos);
	return 0;
 err_0:
	ods_obj_put(pos);	/* don't delete, could be garbage */
	return EINVAL;
}

static int ht_iter_pos_get(ods_iter_t oi, ods_pos_t pos_)
{
	ht_iter_t i = (ht_iter_t)oi;
	ht_t t = i->iter.idx->priv;
	ods_obj_t pos;

	if (!i->ent)
		return ENOENT;

	pos = ods_obj_alloc_extend(t->ods, sizeof(struct ht_pos_s), HT_EXTEND_SIZE);
	if (!pos)
		return ENOMEM;

	POS(pos)->ent_ref = ods_obj_ref(i->ent);
	POS(pos)->bkt = i->bkt;
	pos_->ref = ods_obj_ref(pos);
	ods_obj_put(pos);
	return 0;
}

static int ht_iter_pos_put(ods_iter_t oi, ods_pos_t pos_)
{
	void __ods_obj_delete(ods_obj_t obj);
	ods_obj_t obj;
	ht_iter_t i = (ht_iter_t)oi;
	ht_t t = i->iter.idx->priv;

	obj = ods_ref_as_obj(t->ods, pos_->ref);
	if (!obj)
		return EINVAL;

	__ods_obj_delete(obj);
	ods_obj_put(obj);
	return 0;
}

static int ht_iter_entry_delete(ods_iter_t oi, ods_idx_data_t *data)
{
	ht_iter_t hi = (ht_iter_t)oi;
	ht_t t = hi->iter.idx->priv;
	int rc = ENOENT;
	ods_obj_t ent;
	uint32_t status;
	int64_t bkt;

#ifdef HT_THREAD_SAFE
	if (ods_lock(oi->idx->ods, 0, NULL))
		return EBUSY;
#endif

	if (!hi->ent)
		goto out_0;

	/* Reposition the iterator at the next entry */
	ent = ods_obj_get(hi->ent);
	bkt = hi->bkt;
	(void)__iter_next(t, hi);

	/* Check if the entry is already gone */
	status = ods_ref_status(t->ods, ods_obj_ref(ent));
	if (status & ODS_REF_STATUS_FREE)
	    goto out_1;

	*data = HENT(ent)->value;
	delete_entry(t, ent, bkt);
	rc = 0;
 out_1:
	ods_obj_put(ent);
 out_0:
#ifdef HT_THREAD_SAFE
	ods_unlock(oi->idx->ods, 0);
#endif
	return rc;
}

static const char *ht_get_type(void)
{
	return "HTREE";
}

static void ht_commit(ods_idx_t idx)
{
	ods_commit(idx->ods, ODS_COMMIT_SYNC);
}

int ht_stat(ods_idx_t idx, ods_idx_stat_t idx_sb)
{
	struct stat sb;
	ht_t t = idx->priv;
	idx_sb->cardinality = t->udata->card;
	idx_sb->duplicates = t->udata->dups;
	ods_stat(idx->ods, &sb);
	idx_sb->size = sb.st_size;
	return 0;
}

static struct ods_idx_provider ht_provider = {
	.get_type = ht_get_type,
	.init = ht_init,
	.open = ht_open,
	.close = ht_close,
	.commit = ht_commit,
	.insert = ht_insert,
	.visit = ht_visit,
	.update = ht_update,
	.delete = ht_delete,
	.max = ht_max,
	.min = ht_min,
	.find = ht_find,
	.find_lub = ht_find_lub,
	.find_glb = ht_find_glb,
	.stat = ht_stat,
	.iter_new = ht_iter_new,
	.iter_delete = ht_iter_delete,
	.iter_find = ht_iter_find,
	.iter_find_lub = ht_iter_find_lub,
	.iter_find_glb = ht_iter_find_glb,
	.iter_find_first = ht_iter_find_first,
	.iter_find_last = ht_iter_find_last,
	.iter_begin = ht_iter_begin,
	.iter_end = ht_iter_end,
	.iter_next = ht_iter_next,
	.iter_prev = ht_iter_prev,
	.iter_pos_set = ht_iter_pos_set,
	.iter_pos_get = ht_iter_pos_get,
	.iter_pos_put = ht_iter_pos_put,
	.iter_entry_delete = ht_iter_entry_delete,
	.iter_key = ht_iter_key,
	.iter_data = ht_iter_data,
	.print_idx = print_idx,
	.print_info = print_info
};

struct ods_idx_provider *get(void)
{
	return &ht_provider;
}

static void __attribute__ ((constructor)) ht_lib_init(void)
{
}

static void __attribute__ ((destructor)) ht_lib_term(void)
{
}
