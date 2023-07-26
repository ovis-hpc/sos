/*
 * Copyright (c) 2021 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2013 Sandia Corporation. All rights reserved.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S. Government.
 * Export of this program may require a license from the United States
 * Government.
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
 *      Neither the name of Sandia nor the names of any contributors may
 *      be used to endorse or promote products derived from this software
 *      without specific prior written permission.
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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/queue.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <pthread.h>
#include <ods/ods.h>
#include <ods/ods_rbt.h>
#include "config.h"
#include "ods_priv.h"
#include "ods_log.h"

static pthread_mutex_t __ods_tree_lock = PTHREAD_MUTEX_INITIALIZER;
int64_t ods_cmp(void *a, const void *b, void *arg)
{
	return strcmp(a, b);
}
static struct ods_rbt __ods_tree = ODS_RBT_INITIALIZER(ods_cmp);

#if defined(ODS_DEBUG)
int __ods_debug = 1;
#else
int __ods_debug = 0;
#endif

struct ods_version_s ods_version(ods_t ods)
{
	return ods->version(ods);
}

size_t ods_lock_count(ods_t ods)
{
	size_t sz;
	(void)ods->get(ods, ODS_LOCK_COUNT, &sz);
	return sz;
}

int ods_lock(ods_t ods, int lock_id, struct timespec *wait)
{
	return ods->lock_(ods, lock_id, wait);
}

void ods_unlock(ods_t ods, int lock_id)
{
	ods->unlock(ods, lock_id);
}

int ods_lock_cleanup(const char *path)
{
	ods_t ods = ods_open(path, ODS_PERM_RD, 0660);
	if (!ods)
		return errno;
	int rc = ods->lock_cleanup(path);
	ods_close(ods, ODS_COMMIT_ASYNC);
	return rc;
}

int ods_begin_x(ods_t ods, struct timespec *wait)
{
	return ods->begin_x(ods, wait);
}

int ods_end_x(ods_t ods)
{
	return ods->end_x(ods);
}

pid_t ods_test_x(ods_t ods)
{
	return ods->test_x(ods);
}

int ods_lock_info(const char *path, FILE *fp)
{
	ods_t ods = ods_open(path, ODS_PERM_RD, 0660);
	if (!ods)
		return errno;
	int rc = ods->lock_info(path, fp);
	ods_close(ods, ODS_COMMIT_ASYNC);
	return rc;
}

void ods_info(ods_t ods, FILE *fp, int flags)
{
	ods->info(ods, fp, flags);
}

/*
 * Take a reference on an object
 */
ods_obj_t ods_obj_get(ods_obj_t obj)
{
	ods_atomic_inc(&obj->refcount);
	return obj;
}

void _ods_obj_put(ods_obj_t obj, const char *func, int line)
{
	if (!obj)
		return;

	obj->thread = pthread_self();
	obj->put_line = line;
	obj->put_func = func;

	if (!ods_atomic_dec(&obj->refcount)) {
		if (obj->ods) {
			ods_atomic_dec(&obj->ods->obj_count);
			__ods_lock(obj->ods);
			LIST_REMOVE(obj, entry);
			__ods_unlock(obj->ods);
			obj->ods->obj_put(obj);
		}
		free(obj);
	}
}

/*
 * Verify that an object is valid.
 */
int ods_obj_valid(ods_t ods, ods_obj_t obj)
{
	/* Iterate through the list of active objects to see if obj is present */
	ods_obj_t o;
	LIST_FOREACH(o, &ods->obj_list, entry) {
		if (o == obj)
			return 1;
	}
	return 0;
}

ods_atomic_t ods_obj_count(ods_t ods)
{
	return ods->obj_count;
}

ods_obj_t _ods_ref_as_obj(ods_t ods, ods_ref_t ref, const char *func, int line)
{
	uint64_t ref_sz;
	void *context;
	void *ptr;
	ods_obj_t obj;
	if (!ref || !ods)
		return NULL;

	ptr = ods->ref_as_ptr(ods, ref, &ref_sz, &context);
	if (!ptr) {
		errno = ENOMEM;
		return NULL;
	}

	obj = calloc(1, sizeof *obj);
	if (!obj)
		return NULL;

	obj->ods = ods;
	obj->as.ptr = ptr;
	obj->ref = ref;
	obj->refcount = 1;
	obj->context = context;
	obj->size = ref_sz;
	obj->thread = pthread_self();
	obj->alloc_func = func;
	obj->alloc_line = line;

	__ods_lock(ods);
	ods_atomic_inc(&ods->obj_count);
	LIST_INSERT_HEAD(&ods->obj_list, obj, entry);
	__ods_unlock(ods);

	return obj;
}

/*
 * Return an object's reference
 */
ods_ref_t ods_obj_ref(ods_obj_t obj)
{
	return (obj ? obj->ref : 0);
}

ods_t ods_obj_ods(ods_obj_t obj)
{
	return (obj ? obj->ods : NULL);
}

LIST_HEAD(map_list_head, ods_map_s);
int ods_extend(ods_t ods, size_t sz)
{
	return ods->extend(ods, sz);
}

ods_stat_t ods_stat_buf_new(ods_t ods)
{
	return ods->stat_buf_new(ods);
}

void ods_stat_buf_del(ods_t ods, ods_stat_t buf)
{
	ods->stat_buf_del(ods, buf);
}

int ods_stat_get(ods_t ods, ods_stat_t osb)
{
	return ods->stat_get(ods, osb);
}

int ods_stat(ods_t ods, struct stat *sb)
{
	return ods->fstat_get(ods, sb);
}

int ods_destroy(const char *path)
{
	ods_t ods = ods_open(path, ODS_PERM_RD);
	if (!ods)
		return ENOENT;
	return ods->destroy(ods);
}

const char *ods_path(ods_t ods)
{
	return ods->path;
}

ods_backend_type_t __ods_backend_type(const char *path)
{
	char tmp_path[PATH_MAX];
	int pg_fd;
	struct ods_backend_s be;

	sprintf(tmp_path, "%s%s", path, ODS_BE_SUFFIX);
	pg_fd = open(tmp_path, O_RDWR);
	if (pg_fd < 0) {
		errno = ENOENT;
		return 0;
	}
	int cnt = read(pg_fd, &be, sizeof(be));
	if (cnt < sizeof(be)) {
		errno = EINVAL;
		close(pg_fd);
		return 0;
	}
	if (memcmp(be.be_signature, ODS_BE_SIGNATURE, sizeof(be.be_signature))) {
		errno = EINVAL;
		close(pg_fd);
		return 0;
	}
	close(pg_fd);
	return be.be_type;
}

ods_t ods_open(const char *path, ods_perm_t o_flags, ...)
{
	int be;
	ods_t ods = NULL;
	struct ods_rbn *rbn;
	int o_mode = 0660;
	va_list ap;

	/*
	 * Check if the ODS is already open. If it is, bump the open count
	 * and return the ODS
	 */
	pthread_mutex_lock(&__ods_tree_lock);
	rbn = ods_rbt_find(&__ods_tree, path);
	if (rbn) {
		ods = container_of(rbn, struct ods_s, rbn);
		ods_atomic_inc(&ods->open_count);
	}
	if (ods)
		goto out;

	if (o_flags & ODS_PERM_CREAT) {
		be = o_flags & (ODS_BE_LSOS | ODS_BE_MMAP);
		va_start(ap, o_flags);
		o_mode = va_arg(ap, int);
		va_end(ap);
	} else {
		be = __ods_backend_type(path);
	}
	switch (be) {
	case ODS_BE_MMAP:
		ods = ods_mmap_open(path, o_flags, o_mode);
		break;
	case ODS_BE_LSOS:
		ods = ods_lsos_open(path, o_flags, o_mode);
		break;
	default:
		ods = ods_mmap_open(path, o_flags, o_mode);
		break;
	}
	if (ods) {
		ods->open_count = 1;
		pthread_mutex_init(&ods->lock, NULL);
		LIST_INIT(&ods->obj_list);
		ods_rbn_init(&ods->rbn, ods->path);
		ods_rbt_ins(&__ods_tree, &ods->rbn);
	}
out:
	pthread_mutex_unlock(&__ods_tree_lock);
	return ods;
}

ods_obj_t _ods_get_user_data(ods_t ods, const char *func, int line)
{
	ods_ref_t ref = ods->get_user_data(ods);
	return _ods_ref_as_obj(ods, ref, func, line);
}

ods_obj_t _ods_obj_alloc(ods_t ods, size_t sz, const char *func, int line)
{
	ods_ref_t ref;

	if (0 == (ods->o_perm & ODS_PERM_RW)) {
		errno = EACCES;
		return NULL;
	}

	ref = ods->alloc(ods, sz);
	if (!ref) {
		errno = ENOMEM;
		return NULL;
	}

	return _ods_ref_as_obj(ods, ref, func, line);
}

ods_obj_t _ods_obj_alloc_extend(ods_t ods, size_t sz, size_t extend_sz, const char *func, int line)
{
	ods_obj_t obj = _ods_obj_alloc(ods, sz, func, line);
	if (!obj) {
		extend_sz = (sz < extend_sz ? extend_sz : ODS_ROUNDUP(sz, ODS_PAGE_SIZE));
		if (0 == ods->extend(ods, ods_size(ods) + extend_sz))
			obj = _ods_obj_alloc(ods, sz, func, line);
	}
	return obj;
}

ods_perm_t ods_backend_type_get(ods_t ods)
{
	return ods->be_type;
}

ods_obj_t _ods_obj_malloc(size_t sz, const char *func, int line)
{
	ods_obj_t obj;
	obj = malloc(sz + sizeof(struct ods_obj_s));
	if (!obj)
		return NULL;
	obj->ods = NULL;
	obj->as.ptr = obj + 1;
	obj->ref = 0;
	obj->refcount = 1;
	obj->context = NULL;
	obj->size = sz;
	obj->thread = pthread_self();
	obj->alloc_line = line;
	obj->alloc_func = func;
	return obj;
}

ods_obj_t _ods_obj_realloc(ods_obj_t obj, size_t sz, const char *func, int line)
{
	obj = realloc(obj, sz + sizeof(struct ods_obj_s));
	if (!obj)
		return NULL;
	obj->ods = NULL;
	obj->as.ptr = obj + 1;
	obj->ref = 0;
	obj->refcount = 1;
	obj->context = NULL;
	obj->size = sz;
	obj->thread = pthread_self();
	obj->alloc_line = line;
	obj->alloc_func = func;
	return obj;
}

void ods_obj_update(ods_obj_t obj)
{
	if (obj && obj->ods)
		obj->ods->update(obj->ods, obj->ref, 0, obj->size);
}

void ods_obj_update_offset(ods_obj_t obj, uint64_t offset, size_t len)
{
	if (obj && obj->ods)
		obj->ods->update(obj->ods, obj->ref, offset, len);
}

int ods_ref_valid(ods_t ods, ods_ref_t ref)
{
	return ods->ref_valid(ods, ref);
}

size_t ods_size(ods_t ods)
{
	size_t size;
	(void)ods->get(ods, ODS_OBJ_SIZE, &size);
	return size;
}

/*
 * This function is thread safe.
 */
void ods_commit(ods_t ods, int flags)
{
	return ods->commit(ods, flags);
}

int ods_close(ods_t ods, int flags)
{
	if (!ods)
		return EINVAL;

	pthread_mutex_lock(&__ods_tree_lock);
	if (!ods->open_count) {
		ods_lfatal("%s: Use after free of ODS %p.\n", __func__, ods);
		pthread_mutex_unlock(&__ods_tree_lock);
		return EBADF;
	}

	if (ods_atomic_dec(&ods->open_count)) {
		ods_commit(ods, flags);
		pthread_mutex_unlock(&__ods_tree_lock);
		return 0;
	}
#if defined(ODS_DEBUG)
	if (!LIST_EMPTY(&ods->obj_list)) {
		ods_lwarn("%s: The ODS %s has %d active objects, "
			"close is leaking resources.\n",
			__func__, ods->path, ods->obj_count);
		ods->info(ods, __ods_log_fp, ODS_INFO_ALL);
	}
#endif
	/* Remove the ODS from the open list */
	ods_rbt_del(&__ods_tree, &ods->rbn);
	pthread_mutex_unlock(&__ods_tree_lock);

	return ods->close(ods, flags);
}

uint32_t ods_ref_status(ods_t ods, ods_ref_t ref)
{
	return ods->ref_status(ods, ref);
}

/*
 * This function is thread safe
 */
void ods_ref_delete(ods_t ods, ods_ref_t ref)
{
	ods->delete(ods, ref);
}

/*
 * This function is thread safe
 */
void ods_obj_delete(ods_obj_t obj)
{
	if (!obj->ods)
		return;
	if (0 == (obj->ods->o_perm & ODS_PERM_WR))
		return;
	obj->ods->delete(obj->ods, obj->ref);
	obj->ref = 0;
	obj->context = 0;
	obj->as.ptr = NULL;
	obj->size = 0;
}

void ods_dump(ods_t ods, FILE *fp)
{
	return ods->dump(ods, fp);
}

void ods_obj_iter_pos_init(ods_obj_iter_pos_t pos)
{
	pos->page_no = 1;		      /* first page is udata */
	pos->blk = 0;
}

/*
 * This function is _not_ thread safe
 */
int ods_obj_iter(ods_t ods, ods_obj_iter_pos_t pos,
		 ods_obj_iter_fn_t iter_fn, void *arg)
{
	return ods->obj_iter(ods, NULL, iter_fn, arg);
}

static void flush_all_data_fn(void *arg)
{
	ods_t ods;
	struct ods_rbn *rbn;
	pthread_mutex_lock(&__ods_tree_lock);
	ODS_RBT_FOREACH(rbn, &__ods_tree) {
		ods = container_of(rbn, struct ods_s, rbn);
		ods->release_dead_locks(ods);
		(void)ods->flush_data(ods, 0);
	}
	pthread_mutex_unlock(&__ods_tree_lock);
}
time_t __ods_gc_timeout = ODS_DEF_GC_TIMEOUT;
static pthread_t gc_thread;
static void *gc_thread_fn(void *arg)
{
	ods_t ods;
	struct ods_rbn *rbn;
	uint64_t mapped;

	pthread_cleanup_push(flush_all_data_fn, NULL);
	pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
	do {
		pthread_testcancel();
		sleep(__ods_gc_timeout);
		pthread_mutex_lock(&__ods_tree_lock);
		pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
		mapped = 0;
		ODS_RBT_FOREACH(rbn, &__ods_tree) {
			ods = container_of(rbn, struct ods_s, rbn);
			ods->release_dead_locks(ods);
			mapped = ods->flush_data(ods, 2 * __ods_gc_timeout);
		}
		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
		pthread_mutex_unlock(&__ods_tree_lock);
		ods_ldebug("Total cached data is %ld MB\n", mapped / 1024 / 1024);
	} while (1);
	pthread_cleanup_pop(1);
	return NULL;
}

static void __attribute__ ((constructor)) ods_lib_init(void)
{
	const char *env;

	env = getenv("ODS_DEBUG");
	if (env)
		__ods_debug = atoi(env);

	/* Set up the ODS log file pointer */
	__ods_log_fp = stdout;
	env = getenv("ODS_LOG_MASK");
	if (env) {
		ods_log_mask_set(atoi(env));
		if (__ods_log_mask) {
			env = getenv("ODS_LOG_FILE");
			if (env)
				__ods_log_fp = fopen(env, "w+");
			if (!__ods_log_fp)
				__ods_log_fp = stderr;
		}
	}

	/* Instantiate the memory management thread */
	env = getenv("ODS_GC_TIMEOUT");
	if (env) {
		__ods_gc_timeout = atoi(env);
		if (__ods_gc_timeout <= 0)
			__ods_gc_timeout = ODS_DEF_GC_TIMEOUT;
	}
	int rc = pthread_create(&gc_thread, NULL, gc_thread_fn, NULL);
	if (!rc)
		pthread_setname_np(gc_thread, "ods:unmap");
}
