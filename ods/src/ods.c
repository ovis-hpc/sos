/*
 * Copyright (c) 2013 Open Grid Computing, Inc. All rights reserved.
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

#include <sys/types.h>
#include <sys/stat.h>
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
#include <ods/rbt.h>
#include "ods_priv.h"
#define ODS_OBJ_SUFFIX		".OBJ"
#define ODS_PGTBL_SUFFIX	".PG"
#define ODS_PANIC(__msg) \
{ \
	fprintf(stderr, "Fatal Error: %s[%d]\n  " #__msg, \
		__FUNCTION__, __LINE__); \
	exit(1); \
}

static pthread_spinlock_t ods_lock;
static LIST_HEAD(ods_list_head, ods_s) ods_list;

static size_t ods_ref_size(ods_map_t map, ods_ref_t ref);
static void free_ref(ods_t ods, ods_ref_t ref);

struct ods_pg_s mem_pg_table;
struct ods_obj_data_s mem_obj_table;
struct ods_map_s mem_map;
/* #define ODS_DEBUG */
#ifdef ODS_DEBUG
int ods_debug = 0;
#else
int ods_debug = 1;
#endif

static inline size_t ods_page_count(ods_map_t map, size_t sz)
{
	return ((sz + (ODS_PAGE_SIZE - 1)) & ODS_PAGE_MASK) >> ODS_PAGE_SHIFT;
}

static void *ods_ref_to_ptr(ods_map_t map, uint64_t off)
{
	if (!off)
		return NULL;
	assert(off < map->obj_sz);
	return (void *)((uint64_t)off + (uint64_t)map->obj_data);
}

static ods_ref_t ods_ptr_to_ref(ods_map_t map, void *p)
{
	ods_ref_t ref;
	if (!p)
		return 0;
	ref = (uint64_t)p - (uint64_t)map->obj_data;
	assert(ref < map->obj_sz);
	return ref;
}

static inline uint64_t ods_ref_to_page(ods_map_t map, ods_ref_t ref)
{
	assert(ref < map->obj_sz);
	return ref >> ODS_PAGE_SHIFT;
}

static inline uint64_t ods_ptr_to_page(ods_map_t map, void *p)
{
	return ods_ref_to_page(map, ods_ptr_to_ref(map, p));
}

static inline int ods_bkt(ods_map_t map, size_t sz)
{
	size_t bkt_sz;
	int bkt = 0;
	for (bkt_sz = ODS_GRAIN_SIZE; sz > bkt_sz; bkt_sz <<= 1, bkt++);
	assert(bkt < (ODS_PAGE_SHIFT - ODS_GRAIN_SHIFT));
	return bkt;
}

static inline size_t ods_bkt_to_size(int bkt)
{
	return ODS_GRAIN_SIZE << bkt;
}

static int init_pgtbl(int pg_fd)
{
	static struct ods_pgt_s pgt;
	size_t min_sz;
	int rc;
	unsigned char pge;
	int count;

	count = ODS_OBJ_MIN_SZ >> ODS_PAGE_SHIFT;
	min_sz = count + sizeof(struct ods_pgt_s);
	ftruncate(pg_fd, min_sz);

	memset(&pgt, 0, sizeof pgt);
	memcpy(pgt.signature, ODS_PGT_SIGNATURE, sizeof ODS_PGT_SIGNATURE);
	pgt.gen = 1;
	pgt.count = count;

	rc = lseek(pg_fd, 0, SEEK_SET);
	if (rc < 0)
		return -1;
	rc = write(pg_fd, &pgt, sizeof pgt);
	if (rc != sizeof pgt)
		return -1;

	/* Page 0 is OBJ file header */
	pge = ODS_F_ALLOCATED;
	rc = write(pg_fd, &pge, 1);
	if (rc != 1)
		return errno;
	pge = 0;
	while (count--) {
		rc = write(pg_fd, &pge, 1);
		if (rc != 1)
			return errno;
	}
	return 0;
}

static int init_obj(int obj_fd)
{
	static struct obj_hdr {
		struct ods_obj_data_s obj;
		unsigned char pad[ODS_PAGE_SIZE - sizeof(struct ods_obj_data_s)];
		struct ods_pg_s pg;
	} hdr;
	int rc;

	memset(&hdr, 0, sizeof(hdr));
	memcpy(hdr.obj.signature, ODS_OBJ_SIGNATURE, sizeof ODS_OBJ_SIGNATURE);
	memcpy(&hdr.obj.version, ODS_OBJ_VERSION, sizeof hdr.obj.version);
	hdr.obj.gen = 1;

	/* Page 0 is the header page */
	hdr.obj.pg_free = 1 << ODS_PAGE_SHIFT;

	memset(hdr.obj.blk_free, 0, sizeof hdr.obj.blk_free);

	hdr.pg.next = 0;
	hdr.pg.count = (ODS_OBJ_MIN_SZ >> ODS_PAGE_SHIFT) - 1;

	rc = lseek(obj_fd, 0, SEEK_SET);
	if (rc < 0)
		return -1;
	rc = write(obj_fd, &hdr, sizeof hdr);
	if (rc != sizeof(hdr))
		return -1;

	if (ftruncate(obj_fd, ODS_OBJ_MIN_SZ))
		return -1;
	return 0;
}

int ods_spin_lock(ods_spin_t spin, int timeout)
{
	time_t start = time(NULL);
	time_t now = time(NULL);
	while (timeout < 0 || ((now - start) < timeout)) {
		int rc;
		assert(*spin->lock_p >= 0);
		assert(*spin->lock_p < 100);
		rc = ods_atomic_inc(spin->lock_p);
		if (rc == 1)
			return 0;
		ods_atomic_dec(spin->lock_p);
		now = time(NULL);
		sleep(0);
	}
	return -1;
}

void ods_spin_unlock(ods_spin_t spin)
{
	ods_atomic_dec(spin->lock_p);
}

/* Must be called with the ODS lock held */
static ods_map_t map_new(ods_t ods)
{
	int rc;
	void *pg_map;
	void *obj_map;
	struct stat sb;
	ods_map_t map;

	map = calloc(1, sizeof *map);
	if (!map)
		goto err_0;
	map->ods = ods;
	map->refcount = 1;

	rc = fstat(ods->pg_fd, &sb);
	if (rc)
		goto err_1;
	pg_map = mmap(NULL, sb.st_size,
		      PROT_READ | PROT_WRITE,
		      MAP_FILE | MAP_SHARED, /* | MAP_POPULATE, */
		      ods->pg_fd, 0);
	if (pg_map == MAP_FAILED)
		goto err_1;

	map->pg_sz = sb.st_size;
	map->pg_table = pg_map;
	map->pg_gen = map->pg_table->gen; /* cache gen from mapped memory */

	rc = fstat(ods->obj_fd, &sb);
	if (rc)
		goto err_2;

	obj_map = mmap(0, sb.st_size,
		       PROT_READ | PROT_WRITE,
		       MAP_FILE | MAP_SHARED, /* | MAP_POPULATE, */
		       ods->obj_fd, 0);
	if (obj_map == MAP_FAILED)
		goto err_2;

	map->obj_sz = sb.st_size;
	map->obj_data = obj_map;
	map->obj_gen = map->obj_data->gen; /* cache gen from mapped memory */

	LIST_INSERT_HEAD(&ods->map_list, map, entry);
	return map;

 err_2:
	munmap(pg_map, map->pg_sz);
 err_1:
	free(map);
 err_0:
	return NULL;
}

static inline int map_is_ok(ods_map_t map)
{
	/* Make certain the generation number is still good */
	if ((map->pg_gen != map->pg_table->gen) ||
	    (map->obj_gen != map->obj_data->gen))
		return 0;
	return 1;
}

static inline ods_map_t map_get(ods_map_t map)
{
	ods_atomic_inc(&map->refcount);
	assert(map->refcount > 1);
	return map;
}

void show_stackframe()
{
	extern size_t backtrace(void *buf, size_t buf_size);
	extern char **backtrace_symbols(void *buf, size_t buf_size);
	void *trace[16];
	char **messages = (char **)NULL;
	int i, trace_size = 0;

	trace_size = backtrace(trace, 16);
	messages = (char **)backtrace_symbols(trace, trace_size);
	printf("[bt] Execution path:\n");
	for (i=0; i< trace_size; ++i)
		printf("%s\n", messages[i]);
}

static inline void map_put(ods_map_t map)
{
	if (!map)
		return;
	if (map->refcount <= 0)
		show_stackframe();
	if (!ods_atomic_dec(&map->refcount)) {
		if (ods_debug) {
			/* DEBUG: run through the object list and ensure no
			   object has this as a reference */
			ods_obj_t obj;
			LIST_FOREACH(obj, &map->ods->obj_list, entry) {
				if (obj->map == map) {
					printf("FATAL ERROR: obj %p map %p ods %p\n",
					       obj, map, map->ods);
					ods_info(map->ods, stdout);
					assert(1);
				}
			}
		}
		munmap(map->pg_table, map->pg_sz);
		munmap(map->obj_data, map->obj_sz);
		LIST_REMOVE(map, entry);
		if (map->ods->map == map)
			map->ods->map = NULL;
		free(map);
	}
}

static ods_map_t _ods_map_get(ods_t ods)
{
	if (ods->map) {
		if (!map_is_ok(ods->map)) {
			map_put(ods->map);
			ods->map = map_new(ods);
		}
	} else
		ods->map = map_new(ods);
	return ods->map;
}

static ods_map_t ods_map_get(ods_t ods)
{
	ods_map_t map;
	pthread_spin_lock(&ods->lock);
	map = _ods_map_get(ods);
	pthread_spin_unlock(&ods->lock);
	return map;
}

static void ods_map_put(ods_map_t map)
{
	pthread_spin_lock(&map->ods->lock);
	map_put(map);
	pthread_spin_unlock(&map->ods->lock);
}

int dirty_print_fn(struct rbn *rbn, void *udata, int level)
{
	FILE *fp = udata;
	ods_dirty_t dirt = container_of(rbn, struct ods_dirty_s, rbn);
	fprintf(fp, "%14p %14p\n",
		(void *)(unsigned long)dirt->start,
		(void *)(unsigned long)dirt->end);
	return 0;
}

void ods_info(ods_t ods, FILE *fp)
{
	ods_obj_t obj;
	ods_map_t map;

	fprintf(fp, "\nODS: %p path: %s "
		"obj_fd: %d obj_size: %lu "
		"pg_fd: %d pg_size: %lu\n\n",
		ods, ods->path, ods->obj_fd, ods->obj_sz, ods->pg_fd, ods->pg_sz);
	fprintf(fp, "Active Maps\n");
	fprintf(fp, "               Ref   Obj     Obj            Obj            Page    Page           Page\n");
	fprintf(fp, "Map            Count GN      Size           Data           GN      Size           Data\n");
	fprintf(fp, "-------------- ----- ------- -------------- -------------- ------- -------------- --------------\n");
	LIST_FOREACH(map, &ods->map_list, entry) {
		fprintf(fp, "%14p %5d %3d/%3d %14zu %14p %3d/%3d %14zu %14p\n",
		       map,
		       map->refcount,
		       (int)map->obj_gen, (int)map->obj_data->gen, map->obj_sz, map->obj_data,
		       (int)map->pg_gen, (int)map->pg_table->gen, map->pg_sz, map->pg_table);
	}
	fprintf(fp, "\n");

	fprintf(fp, "Active Objects\n");
	fprintf(fp, "              Ref            ODS            ODS            ODS                           Alloc Alloc\n");
	fprintf(fp, "Object        Count Size     Reference      Pointer        Map            Thread         Line  Func\n");
	fprintf(fp, "-------------- ---- -------- -------------- -------------- -------------- -------------- ----- ------------\n");
	LIST_FOREACH(obj, &ods->obj_list, entry) {
		fprintf(fp, "%14p %4d %8zu 0x%012lx %14p %14p %14p %5d %s\n",
			obj,
			obj->refcount,
			obj->size, obj->ref, obj->as.ptr, obj->map,
			(void *)obj->thread,
			obj->alloc_line, obj->alloc_func);
	}
	fprintf(fp, "\n");

	fprintf(fp, "Free Objects\n");
	fprintf(fp, "                        ODS                           Put   Put\n");
	fprintf(fp, "Object         Size     Reference      Thread         Line  Func\n");
	fprintf(fp, "-------------- -------- -------------- -------------- ----- ------------\n");
	LIST_FOREACH(obj, &ods->obj_free_list, entry) {
		fprintf(fp, "%14p %8zu 0x%012lx %14p %5d %s\n",
			obj,
			obj->size, obj->ref,
			(void *)obj->thread,
			obj->put_line, obj->put_func);
	}
	fprintf(fp, "\n");
	fprintf(fp, "Dirty Tree\n\n");
	fprintf(fp, "Start          End\n");
	fprintf(fp, "-------------- --------------\n");
	rbt_traverse(&ods->dirty_tree, dirty_print_fn, fp);
	fprintf(fp, "\n");
	fflush(fp);
}

static int obj_init(ods_obj_t obj, ods_map_t map, ods_ref_t ref)
{
	assert(ref);
	obj->as.ptr = ods_ref_to_ptr(map, ref);
	if (!obj->as.ptr)
		return 1;
	obj->ref = ref;
	obj->refcount = 1;
	obj->map = map_get(map);
	/*
	 * NB: This check is handling a special case where the
	 * reference refers to the udata section in the header of the
	 * object store. In that case, the ref is not in the section
	 * managed by allocation/deallocation, instead it is
	 * essentially an address constant.
	 */
	if (ref != sizeof(struct ods_obj_data_s))
		obj->size = ods_ref_size(map, ref);
	else
		obj->size = ODS_UDATA_SIZE;
	return 0;
}

/*
 * Take a reference on an object
 */
ods_obj_t ods_obj_get(ods_obj_t obj)
{
	ods_atomic_inc(&obj->refcount);
	return obj;
}

/*
 * Release a reference to an object
 */
void _ods_obj_put(ods_obj_t obj)
{
	if (obj && !ods_atomic_dec(&obj->refcount)) {
		if (!obj->ods) {
			/* This is a memory object */
			free(obj);
			return;
		}
		pthread_spin_lock(&obj->ods->lock);
		assert(obj->refcount == 0);
		LIST_REMOVE(obj, entry);
		LIST_INSERT_HEAD(&obj->ods->obj_free_list, obj, entry);
		map_put(obj->map);
		pthread_spin_unlock(&obj->ods->lock);
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

static ods_obj_t obj_new(ods_t ods)
{
	ods_obj_t obj;
	if (!LIST_EMPTY(&ods->obj_free_list)) {
		obj = LIST_FIRST(&ods->obj_free_list);
		LIST_REMOVE(obj, entry);
	} else {
		ods_atomic_inc(&ods->obj_count);
		obj = malloc(sizeof *obj);
	}
	return obj;
}

static void update_dirty(ods_t ods, ods_ref_t ref, size_t size)
{
#ifndef __notyet__
	return;
#else /* This actually slowed things down */
	ods_ref_t start;
	ods_ref_t end;
	struct rbn *n;
	ods_dirty_t dirty;

	start = ref & ODS_PAGE_MASK;
	end = (ref + size + ODS_PAGE_SIZE - 1) & ODS_PAGE_MASK;

	n = rbt_find_glb(&ods->dirty_tree, &start);
	if (n) {
		dirty = container_of(n, struct ods_dirty_s, rbn);
		/*
		 * If this object is contained in this range, nothing
		 * need be done
		 */
		if (start >= dirty->start && end <= dirty->end)
			return;
		/*
		 * If the dirty range adjoins this page, extend it's range to
		 * cover this page.
		 */
		if (dirty->end == start) {
			dirty->end = end;
			return;
		}
	}
	n = rbt_find_lub(&ods->dirty_tree, &end);
	if (n) {
		/*
		 * If our end touches the LUB start, extend it down to
		 * our start
		*/
		dirty = container_of(n, struct ods_dirty_s, rbn);
		if (end == dirty->start) {
			/*
			 * Remove it from the tree, change it's start
			 * and add it back to the tree.
			*/
			rbt_del(&ods->dirty_tree, n);
			dirty->start = start;
			rbt_ins(&ods->dirty_tree, n);
			return;
		}
	}
	dirty = calloc(1, sizeof *dirty);
	assert(dirty);
	dirty->start = start;
	dirty->end = end;
	dirty->ods = ods;
	rbn_init(&dirty->rbn, &dirty->start);
	rbt_ins(&ods->dirty_tree, &dirty->rbn);
#endif
}

/*
 * Create a memory object from a persistent reference
 */
ods_obj_t _ods_ref_as_obj_with_lock(ods_t ods, ods_ref_t ref, ods_map_t map)
{
	ods_obj_t obj;
	if (!ref)
		return NULL;

	obj = obj_new(ods);
	if (!obj)
		goto err_0;
	obj->ods = ods;
	if (obj_init(obj, map, ref))
		goto err_1;

	update_dirty(ods, obj->ref, ods_obj_size(obj));
	LIST_INSERT_HEAD(&ods->obj_list, obj, entry);
	return obj;
 err_1:
	assert(0);
	LIST_INSERT_HEAD(&obj->ods->obj_free_list, obj, entry);
 err_0:
	return NULL;
}

ods_obj_t _ods_ref_as_obj(ods_t ods, ods_ref_t ref)
{
	ods_obj_t obj;
	if (!ref)
		return NULL;

	pthread_spin_lock(&ods->lock);
	obj = obj_new(ods);
	if (!obj)
		goto err_0;
	obj->ods = ods;
	ods_map_t map = _ods_map_get(ods);
	assert(map_is_ok(map));
	if (obj_init(obj, map, ref))
		goto err_1;

	update_dirty(ods, obj->ref, ods_obj_size(obj));
	LIST_INSERT_HEAD(&ods->obj_list, obj, entry);
	pthread_spin_unlock(&ods->lock);
	return obj;
 err_1:
	assert(0);
	LIST_INSERT_HEAD(&obj->ods->obj_free_list, obj, entry);
 err_0:
	pthread_spin_unlock(&ods->lock);
	return NULL;
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

int ods_extend(ods_t ods, size_t sz)
{
	ods_map_t map;
	struct ods_pg_s *pg;
	size_t n_pages;
	size_t n_sz;
	size_t new_pg_off;
	int rc;

	if (!ods->o_perm)
		return EPERM;

	pthread_spin_lock(&ods->lock);
	/*
	 * Extend the page table first, that way if we fail extending
	 * the object file, we can simply adjust the page_count back
	 * down and leave the object store in a consistent state
	 */
	new_pg_off = ods->obj_sz;
	n_sz = ods->obj_sz + sz;
	n_pages = n_sz >> ODS_PAGE_SHIFT;
	if (n_pages > (ods->pg_sz - sizeof(struct ods_pgt_s))) {
		rc = ftruncate(ods->pg_fd, n_pages + sizeof(struct ods_pgt_s));
		if (rc)
			goto out;
	}
	/* Now extend the obj file */
	rc = ftruncate(ods->obj_fd, n_sz);
	if (rc)
		goto out;

	/* Drop the old map and acquire a new one */
	map_put(ods->map);
	map = ods->map = map_new(ods);
	if (!map) {
		/*
		 * Without the map, the meta-data cannot be
		 * updated. Truncate the files back down to the
		 * original sizes.
		 */
		(void)ftruncate(ods->obj_fd, ods->obj_sz);
		(void)ftruncate(ods->pg_fd, ods->pg_sz);
		rc = ENOMEM;
		goto out;
	}
	pg = ods_ref_to_ptr(map, new_pg_off);
	pg->count = n_pages - map->pg_table->count;
	assert(map->obj_data->pg_free < map->obj_sz);
	pg->next = map->obj_data->pg_free; /* prepend the page free list */
	map->obj_data->pg_free = new_pg_off; /* new page free head */
	map->pg_table->count = n_pages;

	/* Update the cached file sizes. */
	ods->obj_sz = n_sz;
	ods->pg_sz = n_pages + sizeof(struct ods_pgt_s);

	/* Update the generation numbers so older maps will see the change */
	map->pg_gen = ++map->pg_table->gen;
	map->obj_gen = ++map->obj_data->gen;
	rc = 0;
 out:
	pthread_spin_unlock(&ods->lock);
	return rc;
}

int ods_stat(ods_t ods, struct stat *sb)
{
	int rc = fstat(ods->obj_fd, sb);
	if (rc)
		return rc;
	return 0;
}

int ods_create(const char *path, int o_mode)
{
	char tmp_path[PATH_MAX];
	struct stat sb;
	int obj_fd = -1;
	int pg_fd = -1;
	int rc;

	/* Check if the obj file already exists */
	sprintf(tmp_path, "%s%s", path, ODS_OBJ_SUFFIX);
	rc = stat(tmp_path, &sb);
	if ((rc < 0 && errno != ENOENT) || (!rc))
		return (rc ? errno : EEXIST);

	/* Check if the page file already exists */
	sprintf(tmp_path, "%s%s", path, ODS_PGTBL_SUFFIX);
	rc = stat(tmp_path, &sb);
	if (rc < 0 && errno != ENOENT)
		return errno;

	/* Create the object file */
	sprintf(tmp_path, "%s%s", path, ODS_OBJ_SUFFIX);
	obj_fd = creat(tmp_path, o_mode);
	if (obj_fd < 0)
		return errno;

	/* Create the page file */
	sprintf(tmp_path, "%s%s", path, ODS_PGTBL_SUFFIX);
	pg_fd = creat(tmp_path, o_mode);
	if (pg_fd < 0)
		goto out;

	rc = init_obj(obj_fd);
	if (rc)
		goto out;

	rc = init_pgtbl(pg_fd);
	if (rc)
		goto out;

 out:
	if (obj_fd >= 0)
		close(obj_fd);
	if (pg_fd >= 0)
		close(pg_fd);
	return rc;
}

int ods_destroy(const char *path)
{
	char tmp_path[PATH_MAX];
	int rc;

	/* Destroy the obj file */
	sprintf(tmp_path, "%s%s", path, ODS_OBJ_SUFFIX);
	rc = unlink(tmp_path);
	if (rc < 0 && errno != ENOENT)
		return errno;

	/* Destroy the page file */
	sprintf(tmp_path, "%s%s", path, ODS_PGTBL_SUFFIX);
	if (rc < 0 && errno != ENOENT)
		return errno;
	rc = unlink(tmp_path);
	if (rc < 0 && errno != ENOENT)
		return errno;
	return 0;
}

static int dirty_cmp(void *akey, void *bkey)
{
	return (*(ods_ref_t*)akey - *(ods_ref_t*)bkey);
}

ods_t ods_open(const char *path, ods_perm_t o_perm)
{
	char tmp_path[PATH_MAX];
	struct stat sb;
	ods_t ods;
	int obj_fd = -1;
	int pg_fd = -1;
	int rc;

	ods = calloc(1, sizeof *ods);
	if (!ods) {
		errno = ENOMEM;
		return NULL;
	}
	ods->o_perm = o_perm;
	pthread_spin_init(&ods->lock, 0);

	/* Open the obj file */
	sprintf(tmp_path, "%s%s", path, ODS_OBJ_SUFFIX);
	obj_fd = open(tmp_path, O_RDWR);
	if (obj_fd < 0)
		goto err;
	ods->obj_fd = obj_fd;

	/* Open the page table file */
	sprintf(tmp_path, "%s%s", path, ODS_PGTBL_SUFFIX);
	pg_fd = open(tmp_path, O_RDWR);
	if (pg_fd < 0)
		goto err;
	ods->pg_fd = pg_fd;

	ods->path = strdup(path);
	if (!ods->path)
		goto err;

	rc = fstat(ods->obj_fd, &sb);
	if (rc)
		goto err;
	if (sb.st_size < ODS_OBJ_MIN_SZ)
		goto err;

	/*
	 * Save the obj and page file sizes in the ODS, so that
	 * ods_extend will not require a map until after resizing
	 */
	rc = fstat(ods->obj_fd, &sb);
	if (rc)
		goto err;
	ods->obj_sz = sb.st_size;
	rc = fstat(ods->pg_fd, &sb);
	if (rc)
		goto err;
	ods->pg_sz = sb.st_size;

	ods->obj_count = 0;
	LIST_INIT(&ods->obj_list);
	LIST_INIT(&ods->obj_free_list);
	LIST_INIT(&ods->map_list);
	rbt_init(&ods->dirty_tree, dirty_cmp);

	pthread_spin_lock(&ods_lock);
	LIST_INSERT_HEAD(&ods_list, ods, entry);
	pthread_spin_unlock(&ods_lock);
	return ods;

 err:
	rc = errno;
	if (ods->path)
		free(ods->path);
	if (pg_fd >= 0)
		close(pg_fd);
	if (obj_fd >= 0)
		close(obj_fd);
	free(ods);
	errno = rc;
	return NULL;
}

ods_obj_t _ods_get_user_data(ods_t ods)
{
	/* User data starts immediately after the object data header */
	ods_ref_t user_ref = sizeof(struct ods_obj_data_s);
	ods_obj_t obj;

	pthread_spin_lock(&ods->lock);
	obj = obj_new(ods);
	if (!obj)
		goto err_0;
	obj->ods = ods;

	LIST_INSERT_HEAD(&ods->obj_list, obj, entry);
	/* If the map is present and is still current, use it */
	if (ods->map) {
		if (obj_init(obj, ods->map, user_ref))
			goto err_1;
	} else {
		ods->map = map_new(ods);
		if (!ods->map)
			goto err_1;
		if (obj_init(obj, ods->map, user_ref))
		    goto err_1;
	}
	pthread_spin_unlock(&ods->lock);
	return obj;
 err_1:
	LIST_REMOVE(obj, entry);
	free(obj);
 err_0:
	pthread_spin_unlock(&ods->lock);
	return NULL;
}

static void update_page_table(ods_map_t map, ods_pg_t pg, size_t count, int bkt)
{
	unsigned char flags;
	uint64_t page;
	/*
	 * Update page table so that all pages in pg are now
	 * allocated.
	 */
	flags = ODS_F_ALLOCATED;
	if (bkt >= 0)
		flags |= bkt | ODS_F_IDX_VALID;;
	if (count > 1)
		flags |= ODS_F_NEXT;
	for (page = ods_ptr_to_page(map, pg);
	     count; count--,page++) {
		if (count == 1)
			flags &= ~ODS_F_NEXT;
		map->pg_table->pages[page] = flags;
		flags |= ODS_F_PREV;
	}
}

static void *alloc_pages(ods_map_t map, size_t sz, int bkt)
{
	int pg_needed = ods_page_count(map, sz);
	ods_pg_t pg, p_pg, n_pg;
	uint64_t pg_off;
	uint64_t page;

	p_pg = NULL;
	for (pg = ods_ref_to_ptr(map, map->obj_data->pg_free);
	     pg;
	     p_pg = pg, pg = ods_ref_to_ptr(map, pg->next)) {
		if (pg_needed <= pg->count)
			break;
	}
	if (!pg) {
		errno = ENOMEM;
		goto out;
	}

	if (pg->count > pg_needed) {
		page = ods_ptr_to_page(map, pg);
		page += pg_needed;

		/* Fix-up the new page */
		n_pg = ods_page_to_ptr(map, page);
		n_pg->count = pg->count - pg_needed;
		assert(pg->next < map->obj_sz);
		n_pg->next = pg->next;

		pg_off = page << ODS_PAGE_SHIFT;
	} else {
		/* pg gets removed altogether */
		pg_off = pg->next;
	}
	update_page_table(map, pg, pg_needed, bkt);

	/* p_pg is either NULL or pointing to a previous list member */
	assert(pg_off < map->obj_sz);
	if (p_pg)
		p_pg->next = pg_off;
	else
		map->obj_data->pg_free = pg_off;
 out:
	return pg;
}

static void replenish_bkt(ods_map_t map, int bkt)
{
	size_t count;
	uint64_t sz;
	ods_blk_t blk;
	uint64_t off;
	void *p = alloc_pages(map, 1, bkt);
	if (!p)
		/* Errors are caught in alloc_bkt */
		return;

	sz = ods_bkt_to_size(bkt);
	off = ods_ptr_to_ref(map, p);
	for (count = ODS_PAGE_SIZE / sz; count; count--, off += sz) {
		blk = ods_ref_to_ptr(map, off);
		blk->next = map->obj_data->blk_free[bkt];
		assert(0 == (off & (sz-1)));
		map->obj_data->blk_free[bkt] = off;
	}
}

static void *alloc_bkt(ods_map_t map, int bkt)
{
	ods_ref_t off = map->obj_data->blk_free[bkt];
	ods_blk_t blk = ods_ref_to_ptr(map, off);
	uint64_t sz = ods_bkt_to_size(bkt);
	assert(0 == (off & (sz - 1)));
	if (!blk) {
		errno = ENOMEM;
		goto out;
	}
	assert(0 == (blk->next & (sz -1)));
	map->obj_data->blk_free[bkt] = blk->next;
#ifdef ODS_DEBUG
	sz -= sizeof(struct ods_blk_s);
	memset(blk+1, 0xAA, sz);
#endif
 out:
	return blk;
}

static void *alloc_blk(ods_map_t map, size_t sz)
{
	int bkt = ods_bkt(map, sz);
	if (!map->obj_data->blk_free[bkt])
		replenish_bkt(map, bkt);
	return alloc_bkt(map, bkt);
}

ods_obj_t _ods_obj_alloc(ods_t ods, size_t sz)
{
	ods_map_t map;
	ods_obj_t obj = NULL;
	ods_ref_t ref = 0;

	if (!ods->o_perm) {
		errno = EPERM;
		return NULL;
	}

	pthread_spin_lock(&ods->lock);
	map = _ods_map_get(ods);
	if (!map)
		goto out;
	if (sz < (ODS_PAGE_SIZE >> 1))
		ref = ods_ptr_to_ref(map, alloc_blk(map, sz));
	else
		ref = ods_ptr_to_ref(map, alloc_pages(map, sz, -1));
	if (ref) {
		obj = _ods_ref_as_obj_with_lock(ods, ref, map);
		if (!obj)
			free_ref(ods, ref);
	}
 out:
	pthread_spin_unlock(&ods->lock);
	if (obj)
		assert(0 == (obj->ref & 0x1f));
	return obj;
}

ods_obj_t _ods_obj_malloc(size_t sz)
{
	ods_obj_t obj;
	obj = malloc(sz + sizeof(struct ods_obj_s));
	if (!obj)
		return NULL;
	obj->ods = NULL;
	obj->as.ptr = obj + 1;
	obj->ref = 0;
	obj->refcount = 1;
	obj->map = NULL;
	obj->size = sz;
	return obj;
}

static size_t ods_ref_size(ods_map_t map, ods_ref_t ref)
{
	uint64_t page = ods_ref_to_page(map, ref);
	unsigned char flags = map->pg_table->pages[page];
	uint64_t count;

	if (flags & ODS_F_IDX_VALID)
		return ods_bkt_to_size(flags & ODS_M_IDX);

	/*
	 * If the first page has the prev bit set, obj is in the
	 * middle of an allocation
	 */
	if (flags & ODS_F_PREV)
		return -1;

	count = 1;
	while (flags & ODS_F_NEXT) {
		page++;
		count++;
		flags = map->pg_table->pages[page];
	}
	return count << ODS_PAGE_SHIFT;
}

size_t ods_size(ods_t ods)
{
	return ods->obj_sz;
}

uint64_t ods_get_alloc_size(ods_map_t map, uint64_t size)
{
	if (size < ODS_PAGE_SIZE)
		return ods_bkt_to_size(ods_bkt(map, size));
	/* allocated size is in pages. */
	return ods_page_count(map, size) * ODS_PAGE_SIZE;
}

static void free_blk(ods_map_t map, ods_ref_t ref)
{
	uint64_t page;
	int bkt;
	unsigned char flags;
	ods_blk_t blk;

	page = ods_ref_to_page(map, ref);
	flags = map->pg_table->pages[page];
	bkt = flags & ODS_M_IDX;
	if (0 == (flags & ODS_F_ALLOCATED) ||
	    0 == (flags & ODS_F_IDX_VALID) ||
	    bkt < 0 ||
	    bkt > (ODS_PAGE_SHIFT - ODS_GRAIN_SHIFT)) {
		ODS_PANIC("Pointer specified to free is invalid.\n");
		return;
	}
	blk = ods_ref_to_ptr(map, ref);
	blk->next = map->obj_data->blk_free[bkt];
	map->obj_data->blk_free[bkt] = ref;
#ifdef ODS_DEBUG
	uint64_t sz = ods_bkt_to_size(bkt);
	assert(0 == (ref & (sz - 1)));
	sz -= sizeof(struct ods_blk_s);
	blk++;
	memset(blk, 0xFF, sz);
#endif
}

static void free_pages(ods_map_t map, ods_ref_t ref)
{
	ods_pg_t pg;
	uint64_t count;
	uint64_t page;
	unsigned char flags;

	page = ods_ref_to_page(map, ref);
	flags = map->pg_table->pages[page];

	/*
	 * If the first page has the prev bit set, ptr is in the
	 * middle of a previous allocation
	 */
	if (flags & ODS_F_PREV) {
		ODS_PANIC("Freeing in middle of page allocation\n");
		return;
	}
	count = 1;
	for (; page; page++, count++) {
		flags = map->pg_table->pages[page];
		map->pg_table->pages[page] = 0; /* free */

		if (0 == (flags & ODS_F_NEXT))
			break;
	}
	pg = ods_ref_to_ptr(map, ref);
	pg->count = count;
	pg->next = map->obj_data->pg_free;
	assert(pg->next < map->obj_sz);
	map->obj_data->pg_free = ref;
}

/*
 * This function is thread safe.
 */
void ods_commit(ods_t ods, int flags)
{
	int mflag = (flags ? MS_SYNC : MS_ASYNC);
	ods_map_t map = NULL;

	pthread_spin_lock(&ods->lock);
	if (ods->map)
		map = map_get(ods->map);
	pthread_spin_unlock(&ods->lock);

	if (!map)
		return;

#ifndef __notyet__
	struct rbn *rbn;
	while (NULL != (rbn = rbt_min(&ods->dirty_tree))) {
		ods_dirty_t dirt = container_of(rbn, struct ods_dirty_s, rbn);
		size_t size = dirt->end - dirt->start;
		msync(ods_ref_to_ptr(map, dirt->start), size, mflag);
		msync(map->pg_table, map->pg_sz, mflag);
		rbt_del(&ods->dirty_tree, rbn);
	}
#else
	msync(map->obj_data, map->obj_sz, mflag);
	msync(map->pg_table, map->pg_sz, mflag);
#endif
	pthread_spin_lock(&ods->lock);
	map_put(map);
	pthread_spin_unlock(&ods->lock);
}

/*
 * This function is not thread-safe. The caller must know that there
 * are no other users of the ODS.
 */
void ods_close(ods_t ods, int flags)
{
	ods_obj_t obj;
	ods_map_t map;
	if (!ods)
		return;

	/* Remove the ODS from the open list */
	pthread_spin_lock(&ods_lock);
	LIST_REMOVE(ods, entry);
	pthread_spin_unlock(&ods_lock);

	ods_commit(ods, flags);
	close(ods->pg_fd);
	close(ods->obj_fd);
	free(ods->path);

	/* Clean up objects left around by poorly behaving apps */
	while (!LIST_EMPTY(&ods->obj_list)) {
		obj = LIST_FIRST(&ods->obj_list);
		LIST_REMOVE(obj, entry);
		map_put(obj->map);
		free(obj);
	}

	/* Free objects in the cache */
	while (!LIST_EMPTY(&ods->obj_free_list)) {
		obj = LIST_FIRST(&ods->obj_free_list);
		LIST_REMOVE(obj, entry);
		free(obj);
	}

	/* Clean up any maps */
	while (!LIST_EMPTY(&ods->map_list)) {
		map = LIST_FIRST(&ods->map_list);
		LIST_REMOVE(map, entry);
		munmap(map->pg_table, map->pg_sz);
		munmap(map->obj_data, map->obj_sz);
		free(map);
	}

	free(ods);
}

static void free_ref(ods_t ods, ods_ref_t ref)
{
	uint64_t page;
	ods_map_t map = _ods_map_get(ods);
	assert(ref < map->obj_sz);
	/* Get the page this ptr is in */
	page = ods_ref_to_page(map, ref);
	if (map->pg_table->pages[page] & ODS_F_IDX_VALID)
		free_blk(map, ref);
	else
		free_pages(map, ref);
}

uint32_t ods_ref_status(ods_t ods, ods_ref_t ref)
{
	uint64_t page;
	ods_map_t map;
	int bkt;
	unsigned char flags;
	ods_blk_t blk;
	uint32_t status = 0;

	pthread_spin_lock(&ods->lock);
	map = _ods_map_get(ods);
	if (ref >= map->obj_sz) {
		status |= ODS_REF_STATUS_INVALID;
		goto out;
	}

	/* Get the page this ptr is in */
	page = ods_ref_to_page(map, ref);
	flags = map->pg_table->pages[page];
	if (flags & ODS_F_IDX_VALID) {
		bkt = flags & ODS_M_IDX;
		if (0 == (flags & ODS_F_ALLOCATED))
			status |= ODS_REF_STATUS_FREE;
		if (bkt < 0 || bkt > (ODS_PAGE_SHIFT - ODS_GRAIN_SHIFT))
			flags |= ODS_REF_STATUS_CORRUPT;
		status |= ods_bkt_to_size(bkt);
	} else {
		if (0 == (flags & ODS_F_ALLOCATED))
			flags |= ODS_REF_STATUS_FREE;
		if (flags & ODS_F_PREV)
			flags |= ODS_REF_STATUS_INTERIOR;
	}
 out:
	pthread_spin_unlock(&ods->lock);
	return status;
}

/*
 * This function is thread safe
 */
void ods_ref_delete(ods_t ods, ods_ref_t ref)
{
	pthread_spin_lock(&ods->lock);
	free_ref(ods, ref);
	pthread_spin_unlock(&ods->lock);
}

/*
 * This function is thread safe
 */
void ods_obj_delete(ods_obj_t obj)
{
	ods_ref_t ref;
	if (!obj->ods)
		return;
	if (!obj->ods->o_perm) {
		errno = EPERM;
		return;
	}
	pthread_spin_lock(&obj->ods->lock);
	ref = ods_obj_ref(obj);
	if (ref)
		free_ref(obj->ods, ref);
	obj->ref = 0;
	obj->as.ptr = NULL;
	obj->size = 0;
	pthread_spin_unlock(&obj->ods->lock);
}

char *bits(ods_t ods, unsigned char mask)
{
	static char mask_str[80];
	mask_str[0] = '\0';
	if (mask & ODS_F_IDX_VALID) {
		int bkt = mask & ODS_M_IDX;
		sprintf(mask_str, "IDX[%zu] ", ods_bkt_to_size(bkt));
	}
	if (mask & ODS_F_PREV)
		strcat(mask_str, "PREV ");
	if (mask & ODS_F_NEXT)
		strcat(mask_str, "NEXT ");
	return mask_str;
}

int blk_is_free(ods_map_t map, int bkt, void *ptr)
{
	ods_blk_t blk;
	if (!map->obj_data->blk_free[bkt])
		return 0;
	for (blk = ods_ref_to_ptr(map, map->obj_data->blk_free[bkt]);
	     blk; blk = ods_ref_to_ptr(map, blk->next)) {
		if (blk == ptr)
			return 1;
	}
	return 0;
}

void ods_dump(ods_t ods, FILE *fp)
{
	ods_map_t map;
	map = ods_map_get(ods);
	fprintf(fp, "------------------------------- ODS Dump --------------------------------\n");
	fprintf(fp, "%-32s : \"%s\"\n", "Path", ods->path);
	fprintf(fp, "%-32s : %d\n", "Object File Fd", ods->obj_fd);
	fprintf(fp, "%-32s : %zu\n", "Object File Size", ods->obj_sz);
	fprintf(fp, "%-32s : %d\n", "Page File Fd", ods->pg_fd);
	fprintf(fp, "%-32s : %zu\n", "Page File Size", ods->pg_sz);

	ods_pg_t pg;

	fprintf(fp, "--------------------------- Allocated Pages ----------------------------\n");
	uint64_t i;
	uint64_t count = 0;
	for(i = 0; i < map->pg_table->count; i++) {
		uint64_t start;
		if (!(map->pg_table->pages[i] & ODS_F_ALLOCATED))
			continue;
		start = i;
		while (map->pg_table->pages[i] & ODS_F_NEXT) i++;
		if (start == i)
			fprintf(fp, "%ld %s\n", start, bits(ods, map->pg_table->pages[i]));
		else
			fprintf(fp, "%ld..%ld\n", start, i);
		count += (i - start + 1);
	}
	fprintf(fp, "Total Allocated Pages: %ld\n", count);
	fprintf(fp, "--------------------------- Allocated Blocks ----------------------------\n");
	for(i = 0; i < map->pg_table->count; i++) {
		if (!(map->pg_table->pages[i] & ODS_F_ALLOCATED))
			continue;
		if (map->pg_table->pages[i] & ODS_F_IDX_VALID) {
			int bkt = map->pg_table->pages[i] & ODS_M_IDX;
			size_t sz = ods_bkt_to_size(bkt);
			char *blk = (char *)ods_page_to_ptr(map, i);
			char *next = (char *)ods_page_to_ptr(map, i+1);
			printf("======== Size %zu ========\n", sz);
			for (count = 0; blk < next; blk += sz, count++) {
				if (blk_is_free(map, bkt, blk))
					continue;
				fprintf(fp, "%p\n", blk);
			}
			printf("Count %ld\n", count);
		}
	}
	fprintf(fp, "------------------------------ Free Pages ------------------------------\n");
	count = 0;
	for (pg = ods_ref_to_ptr(map, map->obj_data->pg_free);
	     pg;
	     pg = ods_ref_to_ptr(map, pg->next)) {
		fprintf(fp, "%-32s : 0x%016lx\n", "Page Offset",
			ods_ptr_to_ref(map, pg));
		fprintf(fp, "%-32s : %zu\n", "Page Count",
			pg->count);
		count += pg->count;
	}
	fprintf(fp, "Total Free Pages: %ld\n", count);
	fprintf(fp, "------------------------------ Free Blocks -----------------------------\n");
	int bkt;
	ods_blk_t blk;
	for (bkt = 0; bkt < (ODS_PAGE_SHIFT - ODS_GRAIN_SHIFT); bkt++) {
		if (!map->obj_data->blk_free[bkt])
			continue;
		count = 0;
		fprintf(fp, "%-32s : %zu\n", "Block Size",
			ods_bkt_to_size(bkt));
		for (blk = ods_ref_to_ptr(map, map->obj_data->blk_free[bkt]);
		     blk;
		     blk = ods_ref_to_ptr(map, blk->next)) {

			fprintf(fp, "    %-32s : 0x%016lx\n", "Block Offset",
				ods_ptr_to_ref(map, blk));
			count++;
		}
		fprintf(fp, "Total Free %zu blocks: %ld\n",
					ods_bkt_to_size(bkt), count);
	}
	fprintf(fp, "==============================- ODS End =================================\n");
	ods_map_put(map);
}

int ods_pack(ods_t ods)
{
	int i;
	ods_pg_t pg;
	ods_pg_t pg_prev;
	ods_pg_t pg_cur;
	size_t obj_sz, pg_sz;
	int rc;
	ods_map_t map = ods_map_get(ods);
	if (!map)
		return 0;
	errno = 0;
	for (i = map->pg_table->count-1; i > -1; i--) {
		if (map->pg_table->pages[i])
			break;
	}
	i++;
	if (i < 0 || map->pg_table->count <= i)
		goto out;

	pg_sz = sizeof(*map->pg_table) + i;
	pg = ods_ref_to_ptr(map, i);
	pg_prev = 0;
	pg_cur = ods_ref_to_ptr(map, map->obj_data->pg_free);
	while (pg_cur && pg_cur != pg) {
		pg_prev = pg_cur;
		pg_cur = ods_ref_to_ptr(map, pg_cur->next);
	}
	if (!pg_cur) {
		errno = ENOENT;
		goto out;
	}
	assert(pg->next < map->obj_sz);
	if (!pg_prev)
		map->obj_data->pg_free = pg->next;
	else
		pg_prev->next = pg->next;
	map->pg_table->count = i;
	ods_commit(ods, ODS_COMMIT_SYNC);
	/* offset of the trailing page is the new file size */
	obj_sz = (size_t)ods_ptr_to_ref(map, pg);
	rc = ftruncate(ods->pg_fd, pg_sz);
	if (rc)
		goto out;
	rc = ftruncate(ods->obj_fd, obj_sz);
	if (rc)
		goto out;
 out:
	ods_map_put(map);
	return errno;
}

/*
 * This function is _not_ thread safe
 */
void ods_iter(ods_t ods, ods_iter_fn_t iter_fn, void *arg)
{
	ods_pg_t pg;
	uint64_t i, start, end;
	int bkt;
	char *blk;
	char *next;
	size_t sz;
	ods_map_t map;
	ods_obj_t obj;

	map = ods_map_get(ods);
	for(i = 1; i < map->pg_table->count; i++) {
		if (!(map->pg_table->pages[i] & ODS_F_ALLOCATED))
			continue;
		if (map->pg_table->pages[i] & ODS_F_IDX_VALID) {
			bkt = map->pg_table->pages[i] & ODS_M_IDX;
			sz = ods_bkt_to_size(bkt);
			blk = (char *)ods_page_to_ptr(map, i);
			next = (char *)ods_page_to_ptr(map, i+1);
			for (; blk < next; blk += sz) {
				if (blk_is_free(map, bkt, blk))
					continue;
				obj = ods_ref_as_obj(ods, ods_ptr_to_ref(map, blk));
				iter_fn(ods, obj, arg);
				ods_obj_put(obj);
			}
		} else {
			for (start = end = i;
			     (end < map->pg_table->count) &&
				     (0 != (map->pg_table->pages[end] & ODS_F_NEXT));
			     end++);
			pg = ods_page_to_ptr(map, start);
			obj = ods_ref_as_obj(ods, ods_ptr_to_ref(map, pg));
			iter_fn(ods, obj, arg);
			ods_obj_put(obj);
			i = end;
		}
	}
}

static void __attribute__ ((constructor)) ods_lib_init(void)
{
	pthread_spin_init(&ods_lock, 1);
}
