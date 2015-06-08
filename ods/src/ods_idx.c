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
#include <sys/errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/queue.h>
#include <sys/types.h>
#include <pthread.h>
#include <dlfcn.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <ods/rbt.h>
#include <ods/ods_idx.h>
#include <ods/ods_atomic.h>
#include "ods_idx_priv.h"

static pthread_spinlock_t ods_idx_lock;

struct ods_idx_type {
	struct ods_idx_provider *provider;
	struct ods_idx_key_provider *key;
	struct rbn rb_node;
};

int dylib_comparator(void *a, void *b)
{
	return strcmp(a, b);
}

static struct rbt dylib_tree = { 0, dylib_comparator };

void *load_library(const char *library, const char *pfx, const char *sym)
{
	char *errstr;
	static char libpath[PATH_MAX];
	void *p = NULL;

	sprintf(libpath, "lib%s_%s.so", pfx, library);
	void *d = dlopen(libpath, RTLD_LAZY);
	if (!d) {
		/* The library doesn't exist */
		printf("dlopen: %s\n", dlerror());
		goto err;
	}
	dlerror();
	void *(*get)() = dlsym(d, sym);
	errstr = dlerror();
	if (errstr || !get) {
		printf("dlsym: %s\n", errstr);
		goto err;
	}

	p = get();
 err:
	return p;
}

struct ods_idx_class *get_idx_class(const char *type, const char *key)
{
	struct rbn *rbn;
	struct ods_idx_class *idx_class = NULL;
	struct ods_idx_provider *prv;
	struct ods_idx_comparator *cmp;
	char *idx_classname;

	idx_classname = malloc(strlen(type) + strlen(key) + 2);
	if (!idx_classname)
		return NULL;
	sprintf(idx_classname, "%s:%s", type, key);

	pthread_spin_lock(&ods_idx_lock);
	rbn = rbt_find(&dylib_tree, (void *)idx_classname);
	if (rbn) {
		idx_class = container_of(rbn, struct ods_idx_class, rb_node);
		goto out;
	}

	/* Attempt to load the provider */
	prv = (struct ods_idx_provider *)
		load_library(type, "idx", "get");
	if (!prv) {
		errno = ENOENT;
		goto out;
	}

	/* Load the comparator */
	cmp = (struct ods_idx_comparator *)
		load_library(key, "key", "get");
	if (!cmp) {
		errno = ENOENT;
		goto out;
	}

	/* Create provider type and add it to the tree */
	idx_class = calloc(1, sizeof *idx_class);
	if (!idx_class) {
		errno = ENOMEM;
		goto out;
	}

	idx_class->prv = prv;
	idx_class->cmp = cmp;
	rbn_init(&idx_class->rb_node, strdup(idx_classname));
	rbt_ins(&dylib_tree, &idx_class->rb_node);
	free(idx_classname);

 out:
	pthread_spin_unlock(&ods_idx_lock);
	return idx_class;
}

int ods_idx_create(const char *path, int mode,
		   const char *type, const char *key,
		   ...)
{
	va_list argp;
	ods_obj_t obj;
	struct ods_idx_class *idx_class;
	struct ods_idx_meta_data *udata;
	size_t udata_sz;
	ods_t ods;

	va_start(argp, key);

	/* Get the class that handles this index type/key combination */
	idx_class = get_idx_class(type, key);
	if (!idx_class)
		return ENOENT;

	int rc = ods_create(path, mode);
	if (rc)
		return rc;

	/* Open the object store containing the index  */
	ods = ods_open(path, O_RDWR);
	if (!ods) {
		errno = ENOENT;
		goto out;
	}

	/* Set up the IDX meta data in the ODS store. */
	obj = ods_get_user_data(ods);
	udata_sz = ods_obj_size(obj);
	udata = ods_obj_as_ptr(obj);
	memset(udata, 0, udata_sz);
	strcpy(udata->signature, ODS_IDX_SIGNATURE);
	strcpy(udata->type_name, type);
	strcpy(udata->key_name, key);
	errno = idx_class->prv->init(ods, argp);
	ods_obj_put(obj);
	ods_close(ods, ODS_COMMIT_ASYNC);
 out:
	return errno;
}

ods_idx_t ods_idx_open(const char *path, ods_perm_t o_perm)
{
	ods_idx_t idx;
	struct ods_idx_class *idx_class;
	struct ods_idx_meta_data *udata;
	size_t udata_sz;
	ods_obj_t obj;
	idx = calloc(1, sizeof *idx);
	if (!idx)
		return NULL;
	idx->o_perm = o_perm;
	idx->ods = ods_open(path, o_perm);
	if (!idx->ods)
		goto err_0;

	obj = ods_get_user_data(idx->ods);
	udata = ods_obj_as_ptr(obj);
	if (strcmp(udata->signature, ODS_IDX_SIGNATURE)) {
		/* This file doesn't point to an index */
		errno = EBADF;
		goto err_1;
	}
	idx_class = get_idx_class(udata->type_name, udata->key_name);
	if (!idx_class) {
		/* The libraries necessary to handle this index
		   type/key combinationare not present */
		errno = ENOENT;
		goto err_1;
	}
	idx->idx_class = idx_class;
	if (idx_class->prv->open(idx))
		goto err_1;
	ods_obj_put(obj);
	return idx;
 err_1:
	ods_obj_put(obj);
	ods_close(idx->ods, ODS_COMMIT_ASYNC);
 err_0:
	free(idx);
	return NULL;

}

int ods_idx_stat(ods_idx_t idx, ods_idx_stat_t stat)
{
	return idx->idx_class->prv->stat(idx, stat);
}

void ods_idx_close(ods_idx_t idx, int flags)
{
	if (!idx)
		return;
	idx->idx_class->prv->close(idx);
	ods_close(idx->ods, flags);
}

void ods_idx_commit(ods_idx_t idx, int flags)
{
	if (!idx)
		return;
	ods_commit(idx->ods, flags);
}

int ods_idx_insert(ods_idx_t idx, ods_key_t key, ods_ref_t obj)
{
	if (!idx->o_perm)
		return EPERM;
	return idx->idx_class->prv->insert(idx, key, obj);
}

int ods_idx_update(ods_idx_t idx, ods_key_t key, ods_ref_t obj)
{
	if (!idx->o_perm)
		return EPERM;
	return idx->idx_class->prv->update(idx, key, obj);
}

int ods_idx_delete(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	if (!idx->o_perm)
		return EPERM;
	return idx->idx_class->prv->delete(idx, key, ref);
}

int ods_idx_find(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	return idx->idx_class->prv->find(idx, key, ref);
}

int ods_idx_find_lub(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	return idx->idx_class->prv->find_lub(idx, key, ref);
}

int ods_idx_find_glb(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	return idx->idx_class->prv->find_glb(idx, key, ref);
}

ods_iter_t ods_iter_new(ods_idx_t idx)
{
	return idx->idx_class->prv->iter_new(idx);
}

ods_idx_t ods_iter_idx(ods_iter_t iter)
{
	return iter->idx;
}

int ods_iter_flags_set(ods_iter_t i, ods_iter_flags_t flags)
{
	if (flags & ~ODS_ITER_F_MASK)
		return EINVAL;
	i->flags = flags;
	return 0;
}

ods_iter_flags_t ods_iter_flags_get(ods_iter_t i)
{
	return i->flags;
}

void ods_iter_delete(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_delete(iter);
}

int ods_iter_find(ods_iter_t iter, ods_key_t key)
{
	return iter->idx->idx_class->prv->iter_find(iter, key);
}

int ods_iter_find_lub(ods_iter_t iter, ods_key_t key)
{
	return iter->idx->idx_class->prv->iter_find_lub(iter, key);
}

int ods_iter_find_glb(ods_iter_t iter, ods_key_t key)
{
	return iter->idx->idx_class->prv->iter_find_glb(iter, key);
}

int ods_iter_begin(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_begin(iter);
}

int ods_iter_end(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_end(iter);
}

int ods_iter_next(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_next(iter);
}

int ods_iter_prev(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_prev(iter);
}

int ods_iter_pos(ods_iter_t iter, ods_pos_t pos)
{
	return iter->idx->idx_class->prv->iter_pos(iter, pos);
}

int ods_iter_set(ods_iter_t iter, const ods_pos_t pos)
{
	return iter->idx->idx_class->prv->iter_set(iter, pos);
}

ods_key_t ods_iter_key(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_key(iter);
}

ods_ref_t ods_iter_ref(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_ref(iter);
}

ods_obj_t ods_iter_obj(ods_iter_t iter)
{
	ods_ref_t ref = iter->idx->idx_class->prv->iter_ref(iter);
	return ods_ref_as_obj(iter->idx->ods, ref);
}

ods_key_t _ods_key_alloc(ods_idx_t idx, size_t sz)
{
	if (!idx->o_perm)
		return NULL;

	ods_key_t key =
		ods_obj_alloc(idx->ods,
			      sz + sizeof(struct ods_key_value_s));
	if (key)
		*key->as.uint16 = sz;
	return key;
}

void ods_key_copy(ods_key_t dst, ods_key_t src)
{
	ods_key_value_t value = ods_key_value(src);
	memcpy(dst->as.ptr, value, value->len + sizeof(*value));
}

size_t ods_key_set(ods_key_t key, void *value, size_t sz)
{
	ods_key_value_t v = key->as.ptr;
	size_t count = (sz < ods_key_size(key) ? sz : ods_key_size(key));
	memcpy(&v->value, value, count);
	v->len = count;
	return count;
}

const char *ods_key_to_str(ods_idx_t idx, ods_key_t key, char *buf)
{
	return idx->idx_class->cmp->to_str(key, buf);
}

int ods_key_from_str(ods_idx_t idx, ods_key_t key, const char *str)
{
	return idx->idx_class->cmp->from_str(key, str);
}

int ods_key_cmp(ods_idx_t idx, ods_key_t a, ods_key_t b)
{
	return idx->idx_class->cmp->compare_fn(a, b);
}

size_t ods_idx_key_size(ods_idx_t idx)
{
	return idx->idx_class->cmp->size();
}

size_t ods_idx_key_str_size(ods_idx_t idx)
{
	return idx->idx_class->cmp->str_size();
}

size_t ods_key_size(ods_key_t key)
{
	return ods_obj_size(key) - sizeof(struct ods_key_value_s);
}

size_t ods_key_len(ods_key_t key)
{
	return ((ods_key_value_t)key->as.ptr)->len;
}

void ods_idx_print(ods_idx_t idx, FILE *fp)
{
	idx->idx_class->prv->print_idx(idx, fp);
}

ods_t ods_idx_ods(ods_idx_t idx)
{
	return idx->ods;
}

void ods_idx_info(ods_idx_t idx, FILE *fp)
{
	idx->idx_class->prv->print_info(idx, fp);
}

static void __attribute__ ((constructor)) bxt_lib_init(void)
{
	pthread_spin_init(&ods_idx_lock, 1);
}

