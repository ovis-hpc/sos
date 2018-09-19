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

static pthread_spinlock_t __ods_idx_lock;

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

	pthread_spin_lock(&__ods_idx_lock);
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
 out:
	free(idx_classname);
	pthread_spin_unlock(&__ods_idx_lock);
	return idx_class;
}

int ods_idx_create(const char *path, int mode,
		   const char *type, const char *key,
		   const char *args)
{
	ods_obj_t obj;
	struct ods_idx_class *idx_class;
	struct ods_idx_meta_data *udata;
	size_t udata_sz;
	ods_t ods;

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
	if (!obj)
		goto err_1;
	udata_sz = ods_obj_size(obj);
	udata = ods_obj_as_ptr(obj);
	memset(udata, 0, udata_sz);
	strcpy(udata->signature, ODS_IDX_SIGNATURE);
	strcpy(udata->type_name, type);
	strcpy(udata->key_name, key);
	ods_obj_put(obj);
	errno = idx_class->prv->init(ods, type, key, args);
	if (errno)
		goto err_1;
	ods_close(ods, ODS_COMMIT_ASYNC);
	errno = 0;
 out:
	return errno;
 err_1:
	ods_close(ods, ODS_COMMIT_ASYNC);
	return errno;
}

int ods_idx_destroy(const char *path)
{
	return ods_destroy(path);
}

ods_idx_t ods_idx_open(const char *path, ods_perm_t o_perm)
{
	ods_idx_t idx;
	struct ods_idx_class *idx_class;
	struct ods_idx_meta_data *udata;
	ods_obj_t obj;
	idx = calloc(1, sizeof *idx);
	if (!idx)
		return NULL;
	idx->o_perm = o_perm;
	idx->ods = ods_open(path, o_perm);
	if (!idx->ods)
		goto err_0;

	obj = ods_get_user_data(idx->ods);
	if (!obj)
		goto err_1;
	udata = ods_obj_as_ptr(obj);
	if (strcmp(udata->signature, ODS_IDX_SIGNATURE)) {
		/* This file doesn't point to an index */
		errno = EBADF;
		goto err_2;
	}
	idx_class = get_idx_class(udata->type_name, udata->key_name);
	if (!idx_class) {
		/* The libraries necessary to handle this index
		   type/key combinationare not present */
		errno = ENOENT;
		goto err_2;
	}
	idx->idx_class = idx_class;
	if (idx_class->prv->open(idx))
		goto err_2;
	ods_obj_put(obj);
	idx->ref_count = 1;
	return idx;
 err_2:
	ods_obj_put(obj);
 err_1:
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
	if (0 == ods_atomic_dec(&idx->ref_count)) {
		ods_lwarn("%s index is closing.\n", idx->ods->path);
		idx->idx_class->prv->close(idx);
		ods_close(idx->ods, flags);
		free(idx);
	} else {
		ods_ldebug("%s NOT closing due to %d open iterators.\n",
			   idx->ods->path, idx->ref_count);
		assert((ods_atomic_t)-1 != idx->ref_count);
	}
}

int ods_idx_lock(ods_idx_t idx, struct timespec *wait)
{
	if (idx->idx_class->prv->lock)
		return idx->idx_class->prv->lock(idx, wait);
	return 0;

}

void ods_idx_unlock(ods_idx_t idx)
{
	if (idx->idx_class->prv->unlock)
		return idx->idx_class->prv->unlock(idx);
}

int ods_idx_rt_opts_set(ods_idx_t idx, ods_idx_rt_opts_t opt, ...)
{
	va_list ap;
	va_start(ap, opt);
	if (idx->idx_class->prv->rt_opts_set)
		return idx->idx_class->prv->rt_opts_set(idx, opt, ap);
	return EINVAL;
}

ods_idx_rt_opts_t ods_idx_rt_opts_get(ods_idx_t idx)
{
	if (idx->idx_class->prv->rt_opts_get)
		return idx->idx_class->prv->rt_opts_get(idx);
	return (ods_idx_rt_opts_t)-1;
}

void ods_idx_commit(ods_idx_t idx, int flags)
{
	if (!idx)
		return;
	ods_commit(idx->ods, flags);
}

int ods_idx_visit(ods_idx_t idx, ods_key_t key, ods_visit_cb_fn_t cb_fn, void *arg)
{
	return idx->idx_class->prv->visit(idx, key, cb_fn, arg);
}

int ods_idx_insert(ods_idx_t idx, ods_key_t key, ods_idx_data_t data)
{
	if (!idx->o_perm)
		return EPERM;
	return idx->idx_class->prv->insert(idx, key, data);
}

int ods_idx_update(ods_idx_t idx, ods_key_t key, ods_idx_data_t data)
{
	if (!idx->o_perm)
		return EPERM;
	return idx->idx_class->prv->update(idx, key, data);
}

int ods_idx_delete(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	if (!idx->o_perm)
		return EPERM;
	return idx->idx_class->prv->delete(idx, key, data);
}

int ods_idx_min(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	return idx->idx_class->prv->min(idx, key, data);
}

int ods_idx_max(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	return idx->idx_class->prv->max(idx, key, data);
}

int ods_idx_find(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	return idx->idx_class->prv->find(idx, key, data);
}

int ods_idx_find_lub(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	return idx->idx_class->prv->find_lub(idx, key, data);
}

int ods_idx_find_glb(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	return idx->idx_class->prv->find_glb(idx, key, data);
}

ods_iter_t ods_iter_new(ods_idx_t idx)
{
	ods_iter_t iter = idx->idx_class->prv->iter_new(idx);
	if (iter) {
		ods_atomic_inc(&idx->ref_count);
		iter->idx = idx;
	}
	return iter;
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
	ods_idx_t idx = iter->idx;
	iter->idx->idx_class->prv->iter_delete(iter);
	if (0 == ods_atomic_dec(&idx->ref_count))
		ods_idx_close(idx, ODS_COMMIT_ASYNC);
}

int ods_iter_find(ods_iter_t iter, ods_key_t key)
{
	return iter->idx->idx_class->prv->iter_find(iter, key);
}

int ods_iter_find_first(ods_iter_t iter, ods_key_t key)
{
	return iter->idx->idx_class->prv->iter_find_first(iter, key);
}

int ods_iter_find_last(ods_iter_t iter, ods_key_t key)
{
	return iter->idx->idx_class->prv->iter_find_last(iter, key);
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

int ods_iter_pos_get(ods_iter_t iter, ods_pos_t pos)
{
	return iter->idx->idx_class->prv->iter_pos_get(iter, pos);
}

int ods_iter_pos_set(ods_iter_t iter, const ods_pos_t pos)
{
	return iter->idx->idx_class->prv->iter_pos_set(iter, pos);
}

ods_key_t ods_iter_key(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_key(iter);
}

ods_idx_data_t ods_iter_data(ods_iter_t iter)
{
	return iter->idx->idx_class->prv->iter_data(iter);
}

int ods_iter_pos_put(ods_iter_t iter, ods_pos_t pos)
{
	return iter->idx->idx_class->prv->iter_pos_put(iter, pos);
}

int ods_iter_entry_delete(ods_iter_t iter, ods_idx_data_t *data)
{
	return iter->idx->idx_class->prv->iter_entry_delete(iter, data);
}

ods_key_t _ods_key_alloc(ods_idx_t idx, size_t sz, const char *func, int line)
{
	if (!idx->o_perm)
		return NULL;

	ods_key_t key =
		_ods_obj_alloc(idx->ods,
			       sz + sizeof(struct ods_key_value_s),
			       func, line);
	if (key)
		key->as.key->len = sz;
	return key;
}

ods_key_t _ods_key_malloc(size_t sz, const char *func, int line)
{
	ods_key_t key =
		_ods_obj_malloc(sz + sizeof(struct ods_key_value_s),
				func, line);
	if (key)
		*key->as.uint16 = sz;
	return key;
}

void ods_key_copy(ods_key_t dst, ods_key_t src)
{
	ods_key_value_t value = ods_key_value(src);
	memcpy(dst->as.ptr, value, value->len + sizeof(*value));
}

size_t ods_key_set(ods_key_t key, const void *value, size_t sz)
{
	ods_key_value_t v = key->as.ptr;
	size_t count = (sz < ods_key_size(key) ? sz : ods_key_size(key));
	memcpy(&v->value, value, count);
	v->len = count;
	return count;
}

const char *ods_key_to_str(ods_idx_t idx, ods_key_t key, char *buf, size_t len)
{
	return idx->idx_class->cmp->to_str(key, buf, len);
}

int ods_key_from_str(ods_idx_t idx, ods_key_t key, const char *str)
{
	return idx->idx_class->cmp->from_str(key, str);
}

int64_t ods_key_cmp(ods_idx_t idx, ods_key_t a, ods_key_t b)
{
	return idx->idx_class->cmp->compare_fn(a, b);
}

size_t ods_idx_key_size(ods_idx_t idx)
{
	return idx->idx_class->cmp->size();
}

size_t ods_idx_key_str_size(ods_idx_t idx, ods_key_t key)
{
	return idx->idx_class->cmp->str_size(key);
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

static void __attribute__ ((constructor)) ods_idx_init(void)
{
	pthread_spin_init(&__ods_idx_lock, 1);
}

/*
 * This code cleans up index providers. There is an issue, however,
 * with dlopen()/dlclose() and library destructor invocation. In
 * particular, dependent libraries such as libsos may have pointers to
 * the idx_class structures. Dependening on the order in which the
 * library destructors are called, it is possible to have sos
 * containers pointing to idx_class memory after this destructor is
 * called. For this reason, the class memory is not freed.
 */
static void __attribute__ ((destructor)) ods_idx_term(void)
{
	struct rbn *rbn;
	while ((rbn = rbt_min(&dylib_tree))) {
		rbt_del(&dylib_tree, rbn);
		free(rbn->key);
	}
}

