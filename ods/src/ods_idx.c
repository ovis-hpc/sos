/* -*- c-basic-offset : 8 -*-
 * Copyright (c) 2020 Open Grid Computing, Inc. All rights reserved.
 *
 * See the file COPYING at the top of this source tree for the terms
 * of the Copyright.
 */
#define _GNU_SOURCE
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
#include <inttypes.h>
#include <ods/ods_rbt.h>
#include <ods/ods_idx.h>
#include <ods/ods_atomic.h>
#include "ods_idx_priv.h"

static void __ods_idx_flush(ods_idx_t idx);
static pthread_mutex_t __ods_idx_lock = PTHREAD_MUTEX_INITIALIZER;
static __thread int64_t thread_id = -1;
static int64_t next_thread_id;
static int64_t get_thread_id()
{
	if (thread_id == -1) {
		thread_id = __sync_fetch_and_add(&next_thread_id, 1);
	}
	return thread_id;
}

struct ods_idx_type {
	struct ods_idx_provider *provider;
	struct ods_idx_key_provider *key;
	struct ods_rbn rb_node;
};

int64_t dylib_comparator(void *a, const void *b, void *arg)
{
	return strcmp(a, b);
}

static struct ods_rbt dylib_tree = { 0, dylib_comparator };

void *load_library(const char *library, const char *pfx, const char *sym)
{
	char *errstr;
	static char libpath[PATH_MAX];
	void *p = NULL;

	sprintf(libpath, "lib%s_%s.so", pfx, library);
	void *d = dlopen(libpath, RTLD_LAZY | RTLD_DEEPBIND);
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
	struct ods_rbn *ods_rbn;
	struct ods_idx_class *idx_class = NULL;
	struct ods_idx_provider *prv;
	struct ods_idx_comparator *cmp;
	char *idx_classname;

	idx_classname = malloc(strlen(type) + strlen(key) + 2);
	if (!idx_classname)
		return NULL;
	sprintf(idx_classname, "%s:%s", type, key);

	pthread_mutex_lock(&__ods_idx_lock);
	ods_rbn = ods_rbt_find(&dylib_tree, (void *)idx_classname);
	if (ods_rbn) {
		idx_class = container_of(ods_rbn, struct ods_idx_class, rb_node);
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
	ods_rbn_init(&idx_class->rb_node, strdup(idx_classname));
	ods_rbt_ins(&dylib_tree, &idx_class->rb_node);
 out:
	free(idx_classname);
	pthread_mutex_unlock(&__ods_idx_lock);
	return idx_class;
}

int ods_idx_create(const char *path, int perm, int mode,
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

	/* Open the object store containing the index  */
	ods = ods_open(path, perm|ODS_PERM_RW|ODS_PERM_CREAT, mode);
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
	ods_obj_update(obj);
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

const char *ods_idx_path(ods_idx_t idx)
{
	return ods_path(idx->ods);
}

ods_idx_t ods_idx_open(const char *path, ods_perm_t o_flags)
{
	ods_idx_t idx;
	struct ods_idx_class *idx_class;
	struct ods_idx_meta_data *udata;
	ods_obj_t obj;
	idx = calloc(1, sizeof *idx);
	if (!idx)
		return NULL;
	idx->o_perm = o_flags;
	idx->ods = ods_open(path, o_flags & ~ODS_PERM_CREAT);
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
	__ods_idx_flush(idx);
	if (0 == ods_atomic_dec(&idx->ref_count)) {
		ods_ldebug("%s index is closing.\n", ods_path(idx->ods));
		idx->idx_class->prv->close(idx);
		ods_close(idx->ods, flags);
		free(idx);
	} else {
		ods_lerror("%s: %s NOT closing due to %d open iterators.\n",
			   __func__, ods_path(idx->ods), idx->ref_count);
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

int ods_idx_rt_opts_set_va(ods_idx_t idx, ods_idx_rt_opts_t opt, va_list ap)
{	int rc;
	struct ods_bulk_ins_opt_s *kvq_opt;

	if (opt != ODS_IDX_OPT_BULK_INS) {
		if (idx->idx_class->prv->rt_opts_set)
			rc = idx->idx_class->prv->rt_opts_set(idx, opt, ap);
		if (!rc)
			idx->idx_opts |= opt;
		return rc;
	}

	switch (opt) {
	case ODS_IDX_OPT_BULK_INS:
		kvq_opt = va_arg(ap, struct ods_bulk_ins_opt_s *);
		idx->kvq_opt = *kvq_opt;
		idx->kvq_count = 64;
		idx->kvq = calloc(idx->kvq_count, sizeof(ods_kvq_t));
		if (!idx->kvq)
			rc = ENOMEM;
		else
			rc = 0;
		break;
	default:
		rc = EINVAL;
	}
	if (!rc)
		idx->idx_opts |= opt;
	return rc;
}

int ods_idx_rt_opts_set(ods_idx_t idx, ods_idx_rt_opts_t opt, ...)
{	int rc;
	va_list ap;
	va_start(ap, opt);
	rc = ods_idx_rt_opts_set_va(idx, opt, ap);
	va_end(ap);
	return rc;
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

static void __kvq_rebalance(ods_kvq_t kvq)
{
	kvq->threshold = (random() % (kvq->size / 2)) + kvq->size / 2;
}

static ods_kvq_t __ods_kvq_alloc(int count)
{
	int i;
	ods_kvq_t kvq = malloc(sizeof(*kvq) + (count * sizeof(struct ods_kvq_entry_s)));
	kvq->size = count;
	kvq->current = 0;
	kvq->threshold = random() % count;
	for (i = 0; i < count; i++)
		kvq->entry[i].key = ods_key_malloc(512);
	clock_gettime(CLOCK_REALTIME, &kvq->last_flush);
	__kvq_rebalance(kvq);
	return kvq;
}

static void __ods_kvq_queue(ods_kvq_t q, ods_key_t key, ods_idx_data_t data)
{
	int cur = q->current;
	ods_key_copy(q->entry[cur].key, key);
	q->entry[cur].data = data;
	q->current += 1;
}

static void __ods_kvq_flush_no_lock(ods_idx_t idx, ods_kvq_t q)
{
	int rc;
	int i;
	for (i = 0; i < q->current; i++) {
		rc = idx->idx_class->prv->insert_no_lock(
						idx,
						q->entry[i].key,
						q->entry[i].data
						);

		if (rc) {
			fprintf(stderr, "Error %d flushing entry %d to index %s", rc, i, ods_path(idx->ods));
		}
	}
	q->current = 0;
	__kvq_rebalance(q);
	clock_gettime(CLOCK_REALTIME, &q->last_flush);
	q->current = 0;
}

static void __ods_kvq_flush(ods_idx_t idx, ods_kvq_t q)
{
	ods_idx_lock(idx, NULL);
	__ods_kvq_flush_no_lock(idx, q);
	ods_idx_unlock(idx);
}

static void __ods_idx_flush(ods_idx_t idx)
{
	int i;
	if (0 == (idx->idx_opts & ODS_IDX_OPT_BULK_INS))
		return;
	ods_idx_lock(idx, NULL);
	for (i = 0; i < idx->kvq_count; i++) {
		if (idx->kvq[i])
			__ods_kvq_flush_no_lock(idx, idx->kvq[i]);
	}
	ods_idx_unlock(idx);
}

int ods_idx_bulk_insert(ods_idx_t idx, ods_key_t key, ods_idx_data_t data)
{
	ods_kvq_t q;
	int rc = 0;
	int thread_id;
	if (!idx->o_perm)
		return EPERM;
	/* Find/allocate the Q for this thread */
	thread_id = get_thread_id();

	/* Check if new threads have been added */
	ods_idx_lock(idx, NULL);
	if (thread_id >= idx->kvq_count) {
		int i = idx->kvq_count;
		idx->kvq_count *= 2;
		idx->kvq = realloc(idx->kvq, idx->kvq_count * sizeof(void *));
		if (!idx->kvq)
			return ENOMEM;
		for (; i < idx->kvq_count; i++)
			idx->kvq[i] = NULL;
	}
	if (!idx->kvq[thread_id]) {
		idx->kvq[thread_id] = __ods_kvq_alloc(idx->kvq_opt.depth);
		if (!idx->kvq[thread_id])
			return ENOMEM;
		q = idx->kvq[thread_id];
		q->timeout = idx->kvq_opt.timeout;
	} else {
		q = idx->kvq[thread_id];
	}
	ods_idx_unlock(idx);
	__ods_kvq_queue(q, key, data);
	/* Check if the element count triggers a flush */
	if (q->current >= q->threshold) {
		__ods_kvq_flush(idx, q);
		goto out;
	}
	/* Check if a timeout triggers a flush */
	struct timespec ts;
	int secs;
	clock_gettime(CLOCK_REALTIME, &ts);
	secs = ts.tv_sec - q->last_flush.tv_sec;
	if (secs > q->timeout.tv_sec ||
	    (secs == q->timeout.tv_sec
		&& q->timeout.tv_nsec < ts.tv_nsec)) {
		__ods_kvq_flush(idx, q);
	}
out:
	return rc;
}

int ods_idx_insert(ods_idx_t idx, ods_key_t key, ods_idx_data_t data)
{
	if (!idx->o_perm)
		return EPERM;
	if (0 == (idx->idx_opts & ODS_IDX_OPT_BULK_INS))
		return idx->idx_class->prv->insert(idx, key, data);
	return ods_idx_bulk_insert(idx, key, data);
}

int ods_idx_update(ods_idx_t idx, ods_key_t key, ods_idx_data_t data)
{
	if (!idx->o_perm)
		return EPERM;
	__ods_idx_flush(idx);
	return idx->idx_class->prv->update(idx, key, data);
}

int ods_idx_delete(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	if (!idx->o_perm)
		return EPERM;
	__ods_idx_flush(idx);
	return idx->idx_class->prv->delete(idx, key, data);
}

int ods_idx_min(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	__ods_idx_flush(idx);
	return idx->idx_class->prv->min(idx, key, data);
}

int ods_idx_max(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data)
{
	__ods_idx_flush(idx);
	return idx->idx_class->prv->max(idx, key, data);
}

int ods_idx_find(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	__ods_idx_flush(idx);
	return idx->idx_class->prv->find(idx, key, data);
}

int ods_idx_find_lub(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	__ods_idx_flush(idx);
	return idx->idx_class->prv->find_lub(idx, key, data);
}

int ods_idx_find_glb(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	__ods_idx_flush(idx);
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
	__ods_idx_flush(idx);
	iter->idx->idx_class->prv->iter_delete(iter);
	assert(idx->ref_count);
	ods_atomic_dec(&idx->ref_count);
}

int ods_iter_find(ods_iter_t iter, ods_key_t key)
{
	__ods_idx_flush(iter->idx);
	return iter->idx->idx_class->prv->iter_find(iter, key);
}

int ods_iter_find_first(ods_iter_t iter, ods_key_t key)
{
	__ods_idx_flush(iter->idx);
	return iter->idx->idx_class->prv->iter_find_first(iter, key);
}

int ods_iter_find_last(ods_iter_t iter, ods_key_t key)
{
	__ods_idx_flush(iter->idx);
	return iter->idx->idx_class->prv->iter_find_last(iter, key);
}

int ods_iter_find_lub(ods_iter_t iter, ods_key_t key)
{
	__ods_idx_flush(iter->idx);
	return iter->idx->idx_class->prv->iter_find_lub(iter, key);
}

int ods_iter_find_glb(ods_iter_t iter, ods_key_t key)
{
	__ods_idx_flush(iter->idx);
	return iter->idx->idx_class->prv->iter_find_glb(iter, key);
}

int ods_iter_begin(ods_iter_t iter)
{
	__ods_idx_flush(iter->idx);
	return iter->idx->idx_class->prv->iter_begin(iter);
}

int ods_iter_end(ods_iter_t iter)
{
	__ods_idx_flush(iter->idx);
	return iter->idx->idx_class->prv->iter_end(iter);
}

int ods_iter_next(ods_iter_t iter)
{
	__ods_idx_flush(iter->idx);
	return iter->idx->idx_class->prv->iter_next(iter);
}

int ods_iter_prev(ods_iter_t iter)
{
	__ods_idx_flush(iter->idx);
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
	if (!idx->o_perm) {
		errno = EACCES;
		return NULL;
	}

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
	__ods_idx_flush(idx);
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

int ods_idx_verify(ods_idx_t idx, FILE *fp)
{
	__ods_idx_flush(idx);
	if (idx->idx_class->prv->verify_idx)
		return idx->idx_class->prv->verify_idx(idx, fp);
	return 0;
}

static void __attribute__ ((constructor)) ods_idx_init(void)
{
	pthread_mutex_init(&__ods_idx_lock, NULL);
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
	struct ods_rbn *ods_rbn;
	while ((ods_rbn = ods_rbt_min(&dylib_tree))) {
		ods_rbt_del(&dylib_tree, ods_rbn);
		struct ods_idx_class *idx_class =
			container_of(ods_rbn, struct ods_idx_class, rb_node);
		free(ods_rbn->key);
		free(idx_class);
	}
}
