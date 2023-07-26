/* -*- c-basic-offset : 8 -*-
 * Copyright (c) 2021 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2013 Sandia Corporation. All rights reserved.
 *
 * See the file COPYING at the top of this source tree for the terms
 * of the Copyright.
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
#include <libgen.h>
#include <ods/ods.h>
#include <ods/ods_rbt.h>
#include "config.h"
#include "ods_priv.h"
#include "ods_log.h"
#include "ods_mmap.h"
#define ODS_OBJ_SUFFIX		".OBJ"
#define ODS_PGTBL_SUFFIX	ODS_BE_SUFFIX

static size_t ref_size(ods_mmap_t ods, ods_ref_t ref);
static void free_ref(ods_mmap_t ods, ods_ref_t ref);
static ods_map_t map_new(ods_mmap_t ods, loff_t loff, uint64_t *ref_sz);
static ods_pgt_t pgt_map(ods_mmap_t ods);
static void pgt_unmap(ods_mmap_t ods);
static int ref_valid(ods_mmap_t ods, ods_ref_t ref);
static void free_pages(ods_mmap_t ods, uint64_t pg_no);
static void __lock_init(ods_lock_t *lock);
static int __pgt_lock(ods_mmap_t ods);
static void __pgt_unlock(ods_mmap_t ods);
static inline void map_put(ods_map_t map);

extern int __ods_debug;
#define MMOS_DEF_MAP_SZ	(1024 * 1024)

/*
 * Bit vectors are 0-based
 */
static inline int test_bit(uint64_t *bits, int bit)
{
	int word = bit >> 6;
	bit %= 64;
	return (bits[word] & (1L << bit) ? 1 : 0);
}

static inline int alloc_bit(uint64_t *bits, size_t bit_count)
{
	int i, bit, words = bit_count >> 6;
	if (!words)
		words = 1;
	for (bit = -1, i = 0; i < words; i++) {
		int wit = ffsl(bits[i]);
		if (wit) {
			wit -= 1;	      /* ffsl returns 1 based bit #s */
			bits[i] &= ~(1L << wit);
			bit = (i << 6) + wit;
			break;
		}
	}
	return bit;
}

static inline void set_bit(uint64_t *bits, int bit)
{
	int word;
	word = bit >> 6;
	bit %= 64;
	bits[word] |= (1L << bit);
}

static inline void reset_bit(uint64_t *bits, int bit)
{
	int word = bit >> 6;
	bit %= 64;
	bits[word] &= ~(1L << bit);
}

static ods_pgt_t pgt_get(ods_mmap_t ods)
{
	ods_pgt_t pgt;
	/* Get and/or refresh the page table */
	pgt = ods->pg_table;
	if (pgt->pg_gen != ods->pg_gen) {
		pgt_unmap(ods);
		pgt = pgt_map(ods);
	}
	return pgt;
}

static inline uint64_t page_count(size_t sz)
{
	return (sz + (ODS_PAGE_SIZE-1)) >> ODS_PAGE_SHIFT;
}

static inline ods_map_t map_get(ods_map_t map)
{
	if (__builtin_expect(!!(map), 1)) {
		assert(map->refcount >= 1);
		ods_atomic_inc(&map->refcount);
	}
	return map;
}

static inline uint64_t ref_to_page_no(ods_ref_t ref)
{
	return ref >> ODS_PAGE_SHIFT;
}

static inline int size_to_bkt(size_t sz)
{
	sz = ODS_ROUNDUP(sz, ODS_GRAIN_SIZE);
	return (sz >> ODS_GRAIN_SHIFT) - 1;
}

static inline size_t bkt_to_size(int bkt)
{
	if (bkt > ODS_BKT_TABLE_SZ)
		printf("holy shit\n");
	assert(bkt < ODS_BKT_TABLE_SZ);
	return (bkt + 1) << ODS_GRAIN_SHIFT;
}

static int init_pgtbl(int pg_fd)
{
	struct ods_pgt_s pgt;
	size_t min_sz;
	int rc, i;
	struct ods_pg_s pge;
	int count;

	count = ODS_OBJ_MIN_SZ >> ODS_PAGE_SHIFT;
	min_sz = (count * sizeof(struct ods_pg_s)) + sizeof(struct ods_pgt_s);
	rc = ftruncate(pg_fd, min_sz);
	if (rc)
		return -1;

	memset(&pgt, 0, sizeof pgt);
	memcpy(pgt.base.be_signature, ODS_PGT_SIGNATURE,
	       sizeof(pgt.base.be_signature));
	pgt.base.be_vers.major = ODS_VER_MAJOR;
	pgt.base.be_vers.minor = ODS_VER_MINOR;
	pgt.base.be_vers.fix = ODS_VER_FIX;
	pgt.base.be_type = ODS_BE_MMAP;
	strncpy((char *)pgt.base.be_vers.commit_id, ODS_COMMIT_ID,
		sizeof(pgt.base.be_vers.commit_id));
	pgt.pg_gen = 1;
	pgt.pg_count = count;
	pgt.pg_free = 1;

	/* Initialize the bucket table */
	for (i = 0; i < ODS_BKT_TABLE_SZ; i++) {
		pgt.bkt_table[i].pg_next = 0;
	}

	/* Initialize all the lock entries */
	pthread_mutexattr_t attr;
	pthread_mutexattr_init(&attr);
	pthread_mutexattr_setpshared(&attr, 1);
	pthread_mutex_init(&pgt.pgt_lock.mutex, &attr);
	for (i = 0; i < ODS_LOCK_CNT; i++)
		__lock_init(&pgt.lck_tbl[i]);

	rc = lseek(pg_fd, 0, SEEK_SET);
	if (rc < 0)
		return -1;

	rc = write(pg_fd, &pgt, sizeof pgt);
	if (rc != sizeof pgt)
		return -1;

	/* Initialize the page entry for the OBJ header */
	memset(&pge, 0, sizeof(pge));
	pge.pg_flags = ODS_F_ALLOCATED;
	pge.pg_next = 1;
	pge.pg_count = 1;
	rc = write(pg_fd, &pge, sizeof(pge));
	if (rc != sizeof(pge))
		return errno;


	/* Initialize the free entry */
	pge.pg_flags = 0;
	pge.pg_next = 0;
	pge.pg_count = --count;
	rc = write(pg_fd, &pge, sizeof(pge));
	if (rc != sizeof(pge))
		return errno;

	/* Initialize the remainder of the page table */
	memset(&pge, 0, sizeof(pge));
	while (--count) {
		rc = write(pg_fd, &pge, sizeof(pge));
		if (rc != sizeof(pge))
			return errno;
	}

	return 0;
}

static int ods_mmap_get(ods_t ods_, int cfg_id, ...)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	va_list ap;
	size_t *size_p;
	uint64_t *uint64_p;

	va_start(ap, cfg_id);
	switch (cfg_id) {
	case ODS_OBJ_SIZE:
		size_p = va_arg(ap, size_t *);
		*size_p = ods->obj_sz;
		break;
	case ODS_LOCK_COUNT:
		size_p = va_arg(ap, size_t *);
		*size_p = ODS_LOCK_CNT;
		break;
	case ODS_MAP_SIZE:
		uint64_p = va_arg(ap, uint64_t *);
		*uint64_p = ods->obj_map_sz;
		break;
	default:
		return EINVAL;
	}
	va_end(ap);
	return 0;
}

static int ods_mmap_set(ods_t ods_, int cfg_id, ...)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	va_list ap;
	uint64_t uint64;
	int rc = 0;

	va_start(ap, cfg_id);
	switch (cfg_id) {
	case ODS_MAP_SIZE:
		uint64 = va_arg(ap, uint64_t);
		ods->obj_map_sz = uint64;
		break;
	default:
		rc = EINVAL;
	}
	va_end(ap);
	return rc;
}

static struct ods_version_s ods_mmap_version(ods_t ods_)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_pgt_t pgt;
	struct ods_version_s ver = { 0, 0, 0, "........................................" };

	__ods_lock(ods_);
	__pgt_lock(ods);
	pgt = pgt_get(ods);
	if (!pgt)
		goto out;
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return pgt->base.be_vers;
 out:
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return ver;
}

static int init_obj(int obj_fd)
{
	struct obj_hdr {
		struct ods_obj_data_s obj;
		unsigned char pad[ODS_PAGE_SIZE - sizeof(struct ods_obj_data_s)];
	} hdr;
	int rc;

	memset(&hdr, 0, sizeof(hdr));
	memcpy(hdr.obj.obj_signature, ODS_OBJ_SIGNATURE, sizeof(hdr.obj.obj_signature));

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

static void __lock_init(ods_lock_t *lock)
{
	pthread_mutexattr_t attr;
	int rc = pthread_mutexattr_init(&attr);
	assert(rc == 0);
	rc = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
	assert(rc == 0);
	rc = pthread_mutex_init(&lock->mutex, &attr);
	assert(rc == 0);
}

static struct timespec default_wait = {
	.tv_sec = 1,
	.tv_nsec = 0
};

static int __take_lock(ods_lock_t *lock, struct timespec *wait)
{
	int rc;
	if (!wait)
		wait = &default_wait;
	do {
		rc = pthread_mutex_timedlock(&lock->mutex, wait);
		if (wait != &default_wait)
			return rc;
	} while (rc != 0);
	return rc;
}

static void __release_lock(ods_lock_t *lock)
{
	pthread_mutex_unlock(&lock->mutex);
}

static int ods_mmap_lock(ods_t ods_, int lock_id, struct timespec *wait)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_lock_t *lock;
	ods_pgt_t pgt = ods->lck_table;
	assert(pgt);

	if (lock_id < 0 || lock_id >= ODS_LOCK_CNT)
		return EINVAL;

	lock = &pgt->lck_tbl[lock_id];
	return __take_lock(lock, wait);
}

static void ods_mmap_unlock(ods_t ods_, int lock_id)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_lock_t *lock;
	ods_pgt_t pgt = ods->lck_table;
	assert(pgt);

	if (lock_id < 0 || lock_id >= ODS_LOCK_CNT)
		return;

	lock = &pgt->lck_tbl[lock_id];
	__release_lock(lock);
}

static int __pgt_lock(ods_mmap_t ods)
{
	ods_pgt_t pgt = ods->lck_table;
	assert(pgt);
	return __take_lock(&pgt->pgt_lock, NULL);
}

static void __pgt_unlock(ods_mmap_t ods)
{
	ods_pgt_t pgt = ods->lck_table;
	assert(pgt);
	return __release_lock(&pgt->pgt_lock);
}

/* Returns a 2^x value based on the current ODS container size */
static inline size_t new_obj_map_sz(ods_mmap_t ods)
{
	uint64_t computed_size = ods->obj_sz >> 3;
	int i = __builtin_clz(computed_size);
	computed_size = 1LL << (63 - i);
	if (computed_size < ODS_MIN_MAP_SZ)
		return ODS_MIN_MAP_SZ;
	if (computed_size > ODS_MAX_MAP_SZ)
		return ODS_MAX_MAP_SZ;
	return computed_size;
}

/*
 * The loff parameter specifies the offset in the file the map must
 * include.
 *
 * The ref_sz parameter returns the size of the object to which loff
 * refers. If the object size is greater than ods->obj_map_sz, it will
 * be used for the map size rounded up to the next ODS_PAGE_SZ
 * boundary.
 */
static ods_map_t map_new(ods_mmap_t ods, loff_t loff, uint64_t *ref_sz)
{
	void *obj_map;
	struct ods_rbn *rbn;
	struct map_key_s key;
	ods_map_t map;
	uint64_t map_off;
	uint64_t map_len;
	ods_pgt_t pgt;
	uint64_t sz;

	/* Opportunistically check if last map will work */
	__ods_lock(&ods->base);

	/* Get the PGT in case it has been resized by another process */
	__pgt_lock(ods);
	pgt = pgt_get(ods);
	if (!pgt)
		goto err_1;

	sz = *ref_sz = ref_size(ods, loff);
	if (loff + sz > ods->obj_sz)
		goto err_1;

	map = map_get(ods->last_map);
	if (map
	    && (loff >= map->map.off)
	    && ((map->map.off + map->map.len) >= (loff + sz))) {
		__pgt_unlock(ods);
		__ods_unlock(&ods->base);
		return map;
	} else if (map) {
		map_put(map);
	}

	/* Find the largest map that will support loff */
	key.off = loff;
	key.len = 0xffffffff;
	rbn = ods_rbt_find_glb(&ods->map_tree, &key);
	if (rbn) {
		map = container_of(rbn, struct ods_map_s, rbn);
		if (map->map.len > ods->obj_sz && map->map.len < ods->obj_map_sz)
			/* Replace this map and let it age out */
			goto skip;
		if ((map->map.off + map->map.len) >= (loff + sz)) {
			ods_map_t last_map = ods->last_map;
			map->last_used = time(NULL);
			map = map_get(map);
			if (__sync_bool_compare_and_swap(&ods->last_map, last_map, map)) {
				/* We replaced the last_map, drop the ODS ref on it */
				map_put(last_map);
				/* Take a ref on the new last_map */
				map_get(map);
			}
			__pgt_unlock(ods);
			__ods_unlock(&ods->base);
			return map;
		}
		/* Found a map, but it wasn't big enough */
	}
 skip:
	map = calloc(1, sizeof *map);
	if (!map) {
		ods_lerror("Memory allocation failure in %s for %d bytes\n",
			   __func__, sizeof *map);
		goto err_1;
	}
	map->ods = ods;
	map->refcount = 1;

	ods->obj_map_sz = new_obj_map_sz(ods);
	map_off = loff & ~(ods->obj_map_sz - 1);
	map_len = ods->obj_map_sz;
	if ((map_off + map_len) < (loff + sz))
		/* after rounding the offset down, the default map len is too small */
		map_len = ODS_ROUNDUP(loff + sz, map_len);

	if (map_off + map_len > ods->obj_sz)
		/* Truncate map to file size */
		map_len = ods->obj_sz - map_off;

	obj_map = mmap(0, map_len,
		       PROT_READ | PROT_WRITE,
		       MAP_FILE | MAP_SHARED, /* | MAP_POPULATE, */
		       ods->obj_fd, map_off);
	if (obj_map == MAP_FAILED) {
		ods_lerror("Map failure for %zu bytes in %s on fd %d\n",
			   map_len, ods->base.path, ods->obj_fd);
		ods_info(&ods->base, __ods_log_fp, ODS_INFO_ALL);
		goto err_2;
	}

	map->map.len = map_len;
	map->map.off = map_off;
	map->data = obj_map;
	map->last_used = time(NULL);

	ods_rbn_init(&map->rbn, &map->map);
	ods_rbt_ins(&ods->map_tree, &map->rbn);
	__pgt_unlock(ods);
	__ods_unlock(&ods->base);
	return map_get(map);	/* The map_tree consumes a reference */

 err_2:
	free(map);
 err_1:
	__pgt_unlock(ods);
	__ods_unlock(&ods->base);
	return NULL;
}

static void pgt_unmap(ods_mmap_t ods)
{
	int rc = munmap(ods->pg_table, ods->pg_sz);
	assert(rc == 0);
	ods->pg_table = NULL;
}

static ods_pgt_t pgt_map(ods_mmap_t ods)
{
	int rc;
	ods_pgt_t pgt_map;
	struct stat sb;

	rc = fstat(ods->pg_fd, &sb);
	if (rc)
		goto err_0;

	pgt_map = mmap(NULL, sb.st_size,
		       PROT_READ | PROT_WRITE,
		       MAP_FILE | MAP_SHARED, /* | MAP_POPULATE, */
		       ods->pg_fd, 0);
	if (pgt_map == MAP_FAILED)
		goto err_0;

	/* Check the Page Table signature */
	if (memcmp(pgt_map->base.be_signature, ODS_PGT_SIGNATURE,
		   sizeof(pgt_map->base.be_signature))) {
		ods_lerror("The signature in the file '%s' does not match"
			   " '%s', this is a corrupted Page Table.\n",
			   ods->base.path, ODS_PGT_SIGNATURE);
		errno = EINVAL;
		goto err_1;
	}

	/* Check the ODS version to see if the container is compatible */
	if (pgt_map->base.be_vers.major != ODS_VER_MAJOR) {
		ods_lerror("Unsupported container version %d.%d.%d; "
			   "this library is version %d.%d.%d\n",
			   pgt_map->base.be_vers.major,
			   pgt_map->base.be_vers.minor,
			   pgt_map->base.be_vers.fix,
			   ODS_VER_MAJOR, ODS_VER_MINOR, ODS_VER_FIX);
		errno = EINVAL;
		goto err_1;
	}

	ods->pg_sz = sb.st_size;
	ods->pg_table = pgt_map;
	ods->pg_gen = ods->pg_table->pg_gen; /* cache gen from mapped memory */

	/* Update the object file size */
	rc = fstat(ods->obj_fd, &sb);
	if (rc)
		goto err_1;
	ods->obj_sz = sb.st_size;
	return pgt_map;
 err_1:
	munmap(pgt_map, sb.st_size);
 err_0:
	return NULL;
}

static ods_pgt_t lck_map(ods_mmap_t ods)
{
	void *lck_map;

	lck_map = mmap(NULL, ODS_PAGE_SIZE,
		       PROT_READ | PROT_WRITE,
		       MAP_FILE | MAP_SHARED, /* | MAP_POPULATE, */
		       ods->pg_fd, 0);
	if (lck_map == MAP_FAILED)
		goto err_0;

	ods->lck_table = lck_map;
	return ods->lck_table;
 err_0:
	return NULL;
}

static inline int pgt_is_ok(ods_mmap_t ods)
{
	/* Make certain the generation number is still good */
	if (ods->pg_gen != ods->pg_table->pg_gen)
		return 0;
	return 1;
}

static void show_stackframe()
{
	extern size_t backtrace(void *buf, size_t buf_size);
	extern char **backtrace_symbols(void *buf, size_t buf_size);
	void *trace[16];
	char **messages = (char **)NULL;
	int i, trace_size = 0;

	trace_size = backtrace(trace, 16);
	messages = (char **)backtrace_symbols(trace, trace_size);
	ods_ldebug("[bt] Execution path:\n");
	for (i=0; i< trace_size; ++i)
		ods_ldebug("%s\n", messages[i]);
}

static inline void map_put(ods_map_t map)
{
	if (!map)
		return;

	if ((int)map->refcount <= 0) {
		show_stackframe();
		ods_lerror("Putting a map %p with a zero refcount\n", map);
	}

	if (!ods_atomic_dec(&map->refcount)) {
		int rc = munmap(map->data, map->map.len);
		if (rc) {
			ods_lerror("Error %d unmapping %p:%ld]\n",
						errno, map->data, map->map.len);
		}
		if (__ods_debug) {
			/*
			 * DEBUG: run through the object list and ensure no
			 * object has this as a reference
			 */
			ods_obj_t obj;
			LIST_FOREACH(obj, &map->ods->base.obj_list, entry) {
				if (obj->context == map) {
					ods_lfatal("obj %p map %p ods %p\n",
						   obj, map, map->ods);
					ods_info(&map->ods->base, __ods_log_fp, ODS_INFO_ALL);
					assert(0 == "Deleting map with active object references");
				}
			}
		}
		free(map);
	}
}

static int print_map(struct ods_rbn *rbn, void *arg, int l)
{
	FILE *fp = arg;
	ods_map_t map = container_of(rbn, struct ods_map_s, rbn);

	fprintf(fp, "%14p %5d %14p %14zu %p\n",
		map, map->refcount, (void *)map->map.off, map->map.len, map->data);

	return 0;
}

static int check_lock(pthread_mutex_t *mtx, int cleanup)
{
	char proc_path[80];
	struct stat sb;
	int rc;

	if (mtx->__data.__lock == 0)
		return 0;

	/* Check if the holder of the lock is an active process */
	sprintf(proc_path, "/proc/%d", mtx->__data.__owner);
	rc = stat(proc_path, &sb);
	if (rc) {
		if (cleanup)
			pthread_mutex_unlock(mtx);
		return 1;
	}
	return 0;
}

static void do_lock_header(const char *path, FILE *fp)
{
	fprintf(fp, "%s\n", path);
	fprintf(fp, "%-2s %-6s %-8s %-8s %-12s %-8s\n", "Id", "Type", "Lock",
		"Count", "Owner", "Users");
	fprintf(fp, "-- ------ -------- -------- ------------ --------\n");
}

static void print_lock(FILE *fp, int *do_hdr, const char *path, const char *lck_type,
		       int id, pthread_mutex_t *mtx)
{
	if (mtx->__data.__lock == 0)
		return;
	if (*do_hdr) {
		do_lock_header(path, fp);
		*do_hdr = 0;
	}
	fprintf(fp, "%2d %6s %8d %8d %12d %8d",
		id,
		lck_type,
		mtx->__data.__lock,
		mtx->__data.__count,
		mtx->__data.__owner,
		mtx->__data.__nusers);

	if (check_lock(mtx, 0)) {
		printf("  <--- DEAD LOCK: process %d does not exist ---\n",
		       mtx->__data.__owner);
	} else {
		fprintf(fp, "\n");
	}
}

static int ods_mmap_lock_cleanup(const char *path)
{
	char tmp_path[PATH_MAX];
	int id, pg_fd, rc;
	ods_pgt_t pgt;
	pthread_mutex_t *mtx;

	/* Open the page table file */
	sprintf(tmp_path, "%s%s", path, ODS_PGTBL_SUFFIX);
	pg_fd = open(tmp_path, O_RDWR);
	if (pg_fd < 0) {
		rc = errno;
		goto err_0;
	}

	pgt = mmap(NULL, ODS_PAGE_SIZE,
		   PROT_READ | PROT_WRITE,
		   MAP_FILE | MAP_SHARED, pg_fd, 0);
	if (pgt == MAP_FAILED) {
		rc = errno;
		goto err_1;
	}

	mtx = &pgt->pgt_lock.mutex;
	check_lock(mtx, 1);

	for (id = 0; id < ODS_LOCK_CNT; id++) {
		mtx = &pgt->lck_tbl[id].mutex;
		check_lock(mtx, 1);
	}
	munmap(pgt, ODS_PAGE_SIZE);
	rc = 0;
 err_1:
	close(pg_fd);
 err_0:
	return rc;
}

static int ods_mmap_lock_info(const char *path, FILE *fp)
{
	char tmp_path[PATH_MAX];
	int id, pg_fd, rc;
	ods_pgt_t pgt;
	pthread_mutex_t *mtx;
	int do_hdr = 1;

	/* Open the page table file */
	sprintf(tmp_path, "%s%s", path, ODS_PGTBL_SUFFIX);
	pg_fd = open(tmp_path, O_RDWR);
	if (pg_fd < 0) {
		rc = errno;
		goto err_0;
	}

	pgt = mmap(NULL, ODS_PAGE_SIZE,
		   PROT_READ | PROT_WRITE,
		   MAP_FILE | MAP_PRIVATE, pg_fd, 0);
	if (pgt == MAP_FAILED) {
		rc = errno;
		goto err_1;
	}

	mtx = &pgt->pgt_lock.mutex;
	print_lock(fp, &do_hdr, tmp_path, "Global", 0, mtx);

	for (id = 0; id < ODS_LOCK_CNT; id++) {
		mtx = &pgt->lck_tbl[id].mutex;
		print_lock(fp, &do_hdr, tmp_path, "User", id, mtx);
	}
	munmap(pgt, ODS_PAGE_SIZE);
	rc = 0;
 err_1:
	close(pg_fd);
 err_0:
	return rc;
}

static void __active_map_info(ods_mmap_t ods, FILE *fp)
{
	fprintf(fp, "Active Maps\n");
	fprintf(fp, "               Ref   Map            Map            Obj\n");
	fprintf(fp, "Map            Count Offset         Len            Data\n");
	fprintf(fp, "-------------- ----- -------------- -------------- --------------\n");
	ods_rbt_traverse(&ods->map_tree, print_map, fp);
	fprintf(fp, "\n");
}

static void __active_object_info(ods_t ods, FILE *fp)
{
	struct sos_part_ref_data_s {
		char name[256];
		char path[1024];
		uint32_t state;
	};
	ods_obj_t obj;

	fprintf(fp, "Active Objects\n");
	fprintf(fp, "              Ref            ODS            ODS            ODS                           Alloc Alloc\n");
	fprintf(fp, "Object        Count Size     Reference      Pointer        Map            Thread         Line  Func\n");
	fprintf(fp, "-------------- ---- -------- -------------- -------------- -------------- -------------- ----- ------------\n");
	LIST_FOREACH(obj, &ods->obj_list, entry) {
		fprintf(fp, "%14p %4d %8zu 0x%012lx %14p %14p %14p %5d %s\n",
			obj,
			obj->refcount,
			obj->size, obj->ref, obj->as.ptr, obj->context,
			(void *)obj->thread,
			obj->alloc_line, obj->alloc_func);
	}
	fprintf(fp, "\n");
}

static void __lock_info(ods_mmap_t ods, FILE *fp)
{
	pthread_mutex_t *mtx;
	int do_hdr = 1;
	int id;

	mtx = &ods->pg_table->pgt_lock.mutex;
	print_lock(fp, &do_hdr, ods->base.path, "Global", 0, mtx);

	for (id = 0; id < ODS_LOCK_CNT; id++) {
		mtx = &ods->pg_table->lck_tbl[id].mutex;
		print_lock(fp, &do_hdr, ods->base.path, "User", id, mtx);
	}
}

static void ods_mmap_info(ods_t ods_, FILE *fp, int flags)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	if (!fp)
		fp = __ods_log_fp;
	fprintf(fp, "\nODS: %p path: %s "
		"obj_fd: %d obj_size: %lu "
		"pg_fd: %d pg_size: %lu\n\n",
		ods, ods->base.path, ods->obj_fd, ods->obj_sz,
		ods->pg_fd, ods->pg_sz);
	if (flags & ODS_INFO_MAP)
		__active_map_info(ods, fp);
	if (flags & ODS_INFO_ACTIVE_OBJ)
		__active_object_info(&ods->base, fp);
	if (flags & ODS_INFO_LOCK)
		__lock_info(ods, fp);
	fflush(fp);
}

static void ods_mmap_obj_put(ods_obj_t obj)
{
	assert(obj && obj->ods && obj->refcount == 0);
	map_put((ods_map_t)obj->context);
}

static void *ods_mmap_ref_as_ptr(ods_t ods_, ods_ref_t ref,
	uint64_t *ref_sz, void **context)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_map_t map;
	if (!ref || !ods_)
		return NULL;
	map = map_new(ods, ref, ref_sz);
	if (!map)
		return NULL;
	*context = map;
	assert(ref >= map->map.off);
	return &(map->data[ref - map->map.off]);
}

LIST_HEAD(map_list_head, ods_map_s);
static void empty_del_list(struct map_list_head *del_q)
{
	ods_map_t map;
	/* Run through the delete queue and destroy all maps */
	while (!LIST_EMPTY(del_q)) {
		map = LIST_FIRST(del_q);
		LIST_REMOVE(map, entry);
		ods_ldebug("Unmapping %p len %ld MB\n", map->data, map->map.len/1024/1024);
		/* Drop the tree reference and remove it from the tree */
		ods_rbt_del(&map->ods->map_tree, &map->rbn);
		map_put(map);
	}
}

struct del_fn_arg_s {
	time_t timeout;
	struct map_list_head *del_q;
	uint64_t mapped;
};

static int del_map_fn(struct ods_rbn *rbn, void *arg, int l)
{
	struct del_fn_arg_s *darg = arg;
	ods_map_t map = container_of(rbn, struct ods_map_s, rbn);
	LIST_INSERT_HEAD(darg->del_q, map, entry);
	int rc = msync(map->data, map->map.len, MS_ASYNC | MS_INVALIDATE);
	if (rc) {
		ods_lerror("Error %d in %s msyncing map %p of length %ld\n",
			   rc, __func__, map->data, map->map.len);
	}
	return 0;
}

static int ods_mmap_extend(ods_t ods_, size_t sz)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	struct map_list_head del_list;
	struct del_fn_arg_s fn_arg;
	struct stat pg_sb, obj_sb;
	struct ods_pg_s *pg;
	size_t n_pages;
	size_t n_sz;
	int rc;

	if (0 == (ods_->o_perm & ODS_PERM_WR))
		return EPERM;

	__ods_lock(ods_);
	__pgt_lock(ods);
	/*
	 * Update our cached sizes in case they have been changed
	 * since our last map_new()
	 */
	rc = fstat(ods->pg_fd, &pg_sb);
	if (rc)
		goto out;
	rc = fstat(ods->obj_fd, &obj_sb);
	if (rc)
		goto out;

	/*
	 * Extend the page table first, that way if we fail extending
	 * the object file, we can simply adjust the page_count back
	 * down and leave the object store in a consistent state
	 */
	n_pages = (sz + ODS_PAGE_SIZE - 1) >> ODS_PAGE_SHIFT;
	n_sz = n_pages << ODS_PAGE_SHIFT;
	uint64_t pg_sz = (uint64_t)&((struct ods_pgt_s *)0)->
		pg_pages[n_pages + ods->pg_table->pg_count];
	rc = ftruncate(ods->pg_fd, pg_sz);
	if (rc)
		goto out;

	/* Now extend the obj file */
	rc = ftruncate(ods->obj_fd, n_sz + obj_sb.st_size);
	if (rc) {
		/* Restore page file to its original size */
		n_sz = ftruncate(ods->pg_fd, pg_sb.st_size);
		goto out;
	}

	/* Drop the old page map and acquire a new one */
	pgt_unmap(ods);
	ods->pg_table = pgt_map(ods);
	if (!ods->pg_table) {
		/*
		 * Without the map, the meta-data cannot be
		 * updated. Truncate the files back down to the
		 * original sizes.
		 */
		n_sz = ftruncate(ods->obj_fd, obj_sb.st_size);
		n_sz = ftruncate(ods->pg_fd, pg_sb.st_size);
		rc = ENOMEM;
		goto out;
	}
	/* Update the page map to include the new pages */
	pg = &ods->pg_table->pg_pages[ods->pg_table->pg_count];
	pg->pg_count = n_pages;
	pg->pg_next = ods->pg_table->pg_free;
	ods->pg_table->pg_free = ods->pg_table->pg_count;
	ods->pg_table->pg_count += n_pages;

	/* Update the cached file sizes. */
	ods->obj_sz = obj_sb.st_size + n_sz;
	ods->pg_sz = pg_sz;

	/* Update the generation number so older maps will see the change */
	ods->pg_table->pg_gen += 1;
	ods->pg_gen = ods->pg_table->pg_gen;

	/* Opportunistically discard maps in the rbt that are unused */
	LIST_INIT(&del_list);
	fn_arg.del_q = &del_list;
	ods_rbt_traverse(&ods->map_tree, del_map_fn, &fn_arg);
	empty_del_list(&del_list);
	rc = 0;
 out:
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return rc;
}

static ods_stat_t ods_mmap_stat_buf_new(ods_t ods)
{
	size_t buf_size = sizeof(struct ods_stat) +
		(ODS_BKT_TABLE_SZ * 2 * sizeof(uint64_t));
	ods_stat_t buf = malloc(buf_size);
	if (!buf)
		return NULL;
	memset(buf, 0, buf_size);
	return buf;
}

static void ods_mmap_stat_buf_del(ods_t ods, ods_stat_t buf)
{
	free(buf);
}

static int ods_mmap_stat_get(ods_t ods_, ods_stat_t osb)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	int rc, bkt, blk;
	uint64_t allocb, freeb, pg_no;
	ods_pgt_t pgt;
	ods_pg_t pg;
	struct stat sb;
	__ods_lock(ods_);
	__pgt_lock(ods);
	pgt = pgt_get(ods);
	if (!pgt) {
		__pgt_unlock(ods);
		__ods_unlock(ods_);
		return ENOMEM;
	}
	rc = fstat(ods->obj_fd, &sb);
	if (rc) {
		__pgt_unlock(ods);
		__ods_unlock(ods_);
		return rc;
	}

	osb->st_atime = sb.st_atime;
	osb->st_mtime = sb.st_mtime;
	osb->st_ctime = sb.st_ctime;
	osb->st_size = sb.st_size;

	osb->st_pg_count = pgt->pg_count;
	osb->st_pg_free = pgt->pg_free;
	osb->st_pg_size = ODS_PAGE_SIZE;
	osb->st_bkt_count = ODS_BKT_TABLE_SZ;
	osb->st_grain_size = ODS_GRAIN_SIZE;

	for (bkt = 0; bkt < ODS_BKT_TABLE_SZ; bkt ++) {
		allocb = freeb = 0;
		for (pg_no = pgt->bkt_table[bkt].pg_next; pg_no; pg_no = pg->pg_next) {
			pg = &pgt->pg_pages[pg_no];
			for (blk = 0; ODS_BKT_TABLE_SZ; blk ++) {
				if (test_bit(pg->pg_bits, blk)) {
					freeb += 1;
				} else {
					allocb += 1;
				}
			}
		}
		osb->st_blk_alloc[bkt] = allocb;
		osb->st_blk_free[bkt] = freeb;
		osb->st_total_blk_free += freeb;
		osb->st_total_blk_alloc += allocb;
	}
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return 0;
}

static int ods_mmap_fstat_get(ods_t ods_, struct stat *sb)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	return fstat(ods->obj_fd, sb);
}

static int __ods_create(const char *path, int o_mode)
{
	char tmp_path[PATH_MAX];
	struct stat sb;
	int obj_fd = -1;
	int pg_fd = -1;
	int rc;
	mode_t oumask;

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
	oumask = umask(0);
	obj_fd = open(tmp_path, O_CREAT | O_WRONLY, o_mode);
	umask(oumask);
	if (obj_fd < 0)
		return errno;

	/* Create the page file */
	sprintf(tmp_path, "%s%s", path, ODS_PGTBL_SUFFIX);
	oumask = umask(0);
	pg_fd = open(tmp_path, O_CREAT | O_WRONLY, o_mode);
	(void)umask(oumask);
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

static int64_t map_cmp(void *akey, const void *bkey, void *arg)
{
	struct map_key_s *amap = akey;
	struct map_key_s const *bmap = bkey;

	if (amap->off < bmap->off)
		return -1;
	if (amap->off > bmap->off)
		return 1;
	if (amap->len < bmap->len)
		return -1;
	if (amap->len > bmap->len)
		return 1;
	return 0;
}

static ods_ref_t ods_mmap_get_user_data(ods_t ods)
{
	/* User data starts immediately after the object data header */
	return sizeof(struct ods_obj_data_s);
}

static uint64_t alloc_pages(ods_mmap_t ods, size_t pg_needed)
{
	uint64_t pg_no, p_pg_no, n_pg_no;
	ods_pgt_t pgt = ods->pg_table;

	p_pg_no = 0;
	/* Search for an extent large enough to satisfy pg_needed */
	for (pg_no = pgt->pg_free; pg_no; pg_no = pgt->pg_pages[pg_no].pg_next) {
		if (pg_needed <= pgt->pg_pages[pg_no].pg_count) {
			break;
		}
		/* keep track of the previous extent */
		p_pg_no = pg_no;
	}
	if (!pg_no) {
		/* No extents exist or are large enough */
		errno = ENOMEM;
		goto out;
	}

	n_pg_no = pgt->pg_pages[pg_no].pg_next;
	if (pg_needed < pgt->pg_pages[pg_no].pg_count) {
		/* Update the extent created by splitting this one */
		n_pg_no = pg_no + pg_needed;
		pgt->pg_pages[n_pg_no].pg_count =
			pgt->pg_pages[pg_no].pg_count - pg_needed;
		pgt->pg_pages[n_pg_no].pg_next =
			pgt->pg_pages[pg_no].pg_next;
		pgt->pg_pages[n_pg_no].pg_flags = 0;
	}

	/* Update the newly allocated extent */
	pgt->pg_pages[pg_no].pg_count = pg_needed;
	pgt->pg_pages[pg_no].pg_next = 0;
	pgt->pg_pages[pg_no].pg_flags = ODS_F_ALLOCATED;

	if (p_pg_no)
		/* Link the previous extent to the new extent */
		pgt->pg_pages[p_pg_no].pg_next = n_pg_no;
	else {
		/* We used the first extent, update pg_free */
		if (n_pg_no < pgt->pg_count) {
			pgt->pg_free = n_pg_no;
		} else {
			pgt->pg_free = 0;
		}
	}
 out:
	assert(pg_no < ods->pg_table->pg_count);
	return pg_no;
}

static struct bkt_bits {
	int blk_idx;
	size_t blk_sz;
	size_t blk_cnt;
	uint64_t mask_0;
	uint64_t mask_1;
} bkt_bits[] = {
	/* Blk, Size,  Cnt,            Mask[0],            Mask[1] */
	{    0,   32,  128, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF},
	{    1,   64,   64, 0xFFFFFFFFFFFFFFFF, 0x0000000000000000},
	{    2,   96,   42, 0x000003FFFFFFFFFF, 0x0000000000000000},
	{    3,  128,   32, 0x00000000FFFFFFFF, 0x0000000000000000},
	{    4,  160,   25, 0x0000000001FFFFFF, 0x0000000000000000},
	{    5,  192,   21, 0x00000000001FFFFF, 0x0000000000000000},
	{    6,  224,   18, 0x000000000003FFFF, 0x0000000000000000},
	{    7,  256,   16, 0x000000000000FFFF, 0x0000000000000000},
	{    8,  288,   14, 0x0000000000003FFF, 0x0000000000000000},
	{    9,  320,   12, 0x0000000000000FFF, 0x0000000000000000},
	{   10,  352,   11, 0x00000000000007FF, 0x0000000000000000},
	{   11,  384,   10, 0x00000000000003FF, 0x0000000000000000},
	{   12,  416,    9, 0x00000000000001FF, 0x0000000000000000},
	{   13,  448,    9, 0x00000000000001FF, 0x0000000000000000},
	{   14,  480,    8, 0x00000000000000FF, 0x0000000000000000},
	{   15,  512,    8, 0x00000000000000FF, 0x0000000000000000},
	{   16,  544,    7, 0x000000000000007F, 0x0000000000000000},
	{   17,  576,    7, 0x000000000000007F, 0x0000000000000000},
	{   18,  608,    6, 0x000000000000003F, 0x0000000000000000},
	{   19,  640,    6, 0x000000000000003F, 0x0000000000000000},
	{   20,  672,    6, 0x000000000000003F, 0x0000000000000000},
	{   21,  704,    5, 0x000000000000001F, 0x0000000000000000},
	{   22,  736,    5, 0x000000000000001F, 0x0000000000000000},
	{   23,  768,    5, 0x000000000000001F, 0x0000000000000000},
	{   24,  800,    5, 0x000000000000001F, 0x0000000000000000},
	{   25,  832,    4, 0x000000000000000F, 0x0000000000000000},
	{   26,  864,    4, 0x000000000000000F, 0x0000000000000000},
	{   27,  896,    4, 0x000000000000000F, 0x0000000000000000},
	{   28,  928,    4, 0x000000000000000F, 0x0000000000000000},
	{   29,  960,    4, 0x000000000000000F, 0x0000000000000000},
	{   30,  992,    4, 0x000000000000000F, 0x0000000000000000},
	{   31, 1024,    4, 0x000000000000000F, 0x0000000000000000},
	{   32, 1056,    3, 0x0000000000000007, 0x0000000000000000},
	{   33, 1088,    3, 0x0000000000000007, 0x0000000000000000},
	{   34, 1120,    3, 0x0000000000000007, 0x0000000000000000},
	{   35, 1152,    3, 0x0000000000000007, 0x0000000000000000},
	{   36, 1184,    3, 0x0000000000000007, 0x0000000000000000},
	{   37, 1216,    3, 0x0000000000000007, 0x0000000000000000},
	{   38, 1248,    3, 0x0000000000000007, 0x0000000000000000},
	{   39, 1280,    3, 0x0000000000000007, 0x0000000000000000},
	{   40, 1312,    3, 0x0000000000000007, 0x0000000000000000},
	{   41, 1344,    3, 0x0000000000000007, 0x0000000000000000},
	{   42, 1376,    2, 0x0000000000000003, 0x0000000000000000},
	{   43, 1408,    2, 0x0000000000000003, 0x0000000000000000},
	{   44, 1440,    2, 0x0000000000000003, 0x0000000000000000},
	{   45, 1472,    2, 0x0000000000000003, 0x0000000000000000},
	{   46, 1504,    2, 0x0000000000000003, 0x0000000000000000},
	{   47, 1536,    2, 0x0000000000000003, 0x0000000000000000},
	{   48, 1568,    2, 0x0000000000000003, 0x0000000000000000},
	{   49, 1600,    2, 0x0000000000000003, 0x0000000000000000},
	{   50, 1632,    2, 0x0000000000000003, 0x0000000000000000},
	{   51, 1664,    2, 0x0000000000000003, 0x0000000000000000},
	{   52, 1696,    2, 0x0000000000000003, 0x0000000000000000},
	{   53, 1728,    2, 0x0000000000000003, 0x0000000000000000},
	{   54, 1760,    2, 0x0000000000000003, 0x0000000000000000},
	{   55, 1792,    2, 0x0000000000000003, 0x0000000000000000},
	{   56, 1824,    2, 0x0000000000000003, 0x0000000000000000},
	{   57, 1856,    2, 0x0000000000000003, 0x0000000000000000},
	{   58, 1888,    2, 0x0000000000000003, 0x0000000000000000},
	{   59, 1920,    2, 0x0000000000000003, 0x0000000000000000},
	{   60, 1952,    2, 0x0000000000000003, 0x0000000000000000},
	{   61, 1984,    2, 0x0000000000000003, 0x0000000000000000},
	{   62, 2016,    2, 0x0000000000000003, 0x0000000000000000},
	{   63, 2048,    2, 0x0000000000000003, 0x0000000000000000},
};

static uint64_t replenish_bkt(ods_mmap_t ods, ods_pgt_t pgt, int bkt)
{
	uint64_t pg_no = alloc_pages(ods, 1);
	if (!pg_no)
		return 0;
	ods_pg_t pg = &pgt->pg_pages[pg_no];
	pg->pg_next = pgt->bkt_table[bkt].pg_next;
	pgt->bkt_table[bkt].pg_next = pg_no;
	pg->pg_flags |= ODS_F_IDX_VALID | ODS_F_IN_BKT;
	pg->pg_bkt_idx = bkt;
	pg->pg_bits[0] = bkt_bits[bkt].mask_0;
	pg->pg_bits[1] = bkt_bits[bkt].mask_1;
	return pg_no;
}

static void del_bkt_tbl_pg(ods_pgt_t pgt, int bkt, uint32_t pg_no)
{
	uint64_t bkt_pg, prev_pg;
	ods_pg_t pg;

	/* Remove the page from the block list */
	prev_pg = 0;
	for (bkt_pg = pgt->bkt_table[bkt].pg_next; bkt_pg;
	     prev_pg = bkt_pg, bkt_pg = pg->pg_next) {
		pg = &pgt->pg_pages[bkt_pg];
		if (bkt_pg == pg_no) {
			assert(pg->pg_flags & ODS_F_IN_BKT);
			pg->pg_flags &= ~ODS_F_IN_BKT;
			if (prev_pg) {
				pgt->pg_pages[prev_pg].pg_next = pg->pg_next;
			} else {
				pgt->bkt_table[bkt].pg_next = pg->pg_next;
			}
			pg->pg_next = 0;
			return;
		}
	}
	assert(0 == "Attempt to remove a block that was not on the free list");
}

static ods_ref_t alloc_blk(ods_mmap_t ods, ods_pgt_t pgt, uint64_t sz)
{
	int blk, bkt = size_to_bkt(sz);
	ods_pg_t pg;
	uint64_t pg_no = pgt->bkt_table[bkt].pg_next;
	do {
		if (!pg_no) {
			pg_no = replenish_bkt(ods, pgt, bkt);
			if (!pg_no)
				return 0;
		}
		pg = &pgt->pg_pages[pg_no];
		assert(pg->pg_flags & (ODS_F_IDX_VALID | ODS_F_IN_BKT));
		if (pg->pg_bits[0] || pg->pg_bits[1]) {
			blk = alloc_bit(pg->pg_bits, bkt_bits[bkt].blk_cnt);
			if (0 == pg->pg_bits[0] && 0 == pg->pg_bits[1])
				/* The last bit was consumed, take it off the bucket
				 * list to avoid searching it next time */
				del_bkt_tbl_pg(pgt, bkt, pg_no);
			if (blk >= 0)
				break;
		}
		pg_no = pg->pg_next;
	} while (1);
	ods_ref_t ref = pg_no << ODS_PAGE_SHIFT | (bkt_to_size(bkt) * blk);
	assert(ref_valid(ods, ref));
	return ref;
}

static ods_ref_t ods_mmap_alloc(ods_t ods_, size_t sz)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	uint64_t pg_no;
	ods_ref_t ref = 0;
	ods_pgt_t pgt;

	if (0 == (ods->base.o_perm & ODS_PERM_WR)) {
		errno = EPERM;
		return 0;
	}

	__ods_lock(ods_);
	__pgt_lock(ods);

	/* Get and/or refresh the page table */
	pgt = pgt_get(ods);
	if (!pgt)
		goto out;

	if (sz < (ODS_PAGE_SIZE >> 1)) {
		ref = alloc_blk(ods, pgt, sz);
		if (!ref)
			goto out;
	} else {
		pg_no = alloc_pages(ods, page_count(sz));
		if (!pg_no) {
			ref = 0;
			goto out;
		}
		ref = pg_no << ODS_PAGE_SHIFT;
	}
 out:
	if (__ods_debug && ref)
		assert(ref_valid(ods, ref));
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return ref;
}

static void ods_mmap_update(ods_t ods, ods_ref_t ref, loff_t offset, size_t len)
{
}

static size_t ref_size(ods_mmap_t ods, ods_ref_t ref)
{
	uint64_t pg_no;
	ods_pg_t pg;

	/*
	 * NB: This check is handling a special case where the
	 * reference refers to the udata section in the header of the
	 * object store. In that case, the ref is not in the section
	 * managed by allocation/deallocation, instead it is
	 * essentially an address constant.
	 */
	if (ref == sizeof(struct ods_obj_data_s))
		return ODS_UDATA_SIZE;

	pg_no = ref_to_page_no(ref);
	assert(pg_no < ods->pg_table->pg_count);
	pg = &ods->pg_table->pg_pages[pg_no];

	if (pg->pg_flags & ODS_F_IDX_VALID)
		return bkt_to_size(pg->pg_bkt_idx);

	return pg->pg_count << ODS_PAGE_SHIFT;
}

static inline int ref_to_blk_no(ods_mmap_t ods, ods_ref_t ref)
{
	uint64_t sz = ref_size(ods, ref);
	return (ref & ~ODS_PAGE_MASK) / sz;
}

/*
 * Return True(!0) if the ref points to the start of an allocation. If
 * unallocated or inside an allocation (i.e. not the start), return
 * False (0)
 */
static int ref_valid(ods_mmap_t ods, ods_ref_t ref)
{
	uint64_t pg_no = ref_to_page_no(ref);
	ods_pgt_t pgt = ods->pg_table;

	if (0 == (pgt->pg_pages[pg_no].pg_flags & ODS_F_ALLOCATED))
		return 0;

	if (pgt->pg_pages[pg_no].pg_flags & ODS_F_IDX_VALID) {
		int blk_no = ref_to_blk_no(ods, ref);
		int ref_sz = bkt_to_size(pgt->pg_pages[pg_no].pg_bkt_idx);
		if (blk_no >= ODS_PAGE_SIZE / bkt_to_size(pgt->pg_pages[pg_no].pg_bkt_idx))
			/* Doesn't fit in page */
			return 0;
		if (0 != ((ref & ~ODS_PAGE_MASK) % ref_sz))
			/* Must be aligned on size boundary */
			return 0;
		/* Must not be free */
		return 0 == test_bit(pgt->pg_pages[pg_no].pg_bits, blk_no);
	}
	/* This is a page allocation, the page offset should be zero */
	if (ref & ~ODS_PAGE_MASK)
		return 0;

	/*
	 * For allocated pages, only the first page has the count set,
	 * all internal pages have a count of zero.
	 */
	return pgt->pg_pages[pg_no].pg_count;
}

static int ods_mmap_ref_valid(ods_t ods_, ods_ref_t ref)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	int rc;
	__pgt_lock(ods);
	rc = ref_valid(ods, ref);
	__pgt_unlock(ods);
	return rc;
}

static void free_blk(ods_mmap_t ods, ods_ref_t ref)
{
	ods_pgt_t pgt = ods->pg_table;
	ods_pg_t pg;
	int bkt, blk_no, pg_no;

	pg_no = ref_to_page_no(ref);
	pg = &pgt->pg_pages[pg_no];
	bkt = pg->pg_bkt_idx;
	blk_no = ref_to_blk_no(ods, ref);

	if (test_bit(pg->pg_bits, blk_no)) {
		/* ref is already free */
		ods_lerror("Ref %p is already free\n", (void *)ref);
		return;
	}
	if (0 == (pg->pg_flags & ODS_F_ALLOCATED) || /* ref page not allocated */
	    0 == (pg->pg_flags & ODS_F_IDX_VALID) || /* ref is not in bucket page */
	    ((ref & ~ODS_PAGE_MASK) % bkt_to_size(bkt))|| /* ref not aligned to size */
	    bkt < 0 || bkt >= ODS_BKT_TABLE_SZ       /* bkt index is invalid */
	    ) {
		ods_lerror("Ref %p is invalid.\n", (void *)ref);
		return;
	}
	set_bit(pg->pg_bits, blk_no);

	/* Add the bucket back to the bucket list */
	if (0 == (pg->pg_flags & ODS_F_IN_BKT)) {
		pg->pg_next = pgt->bkt_table[bkt].pg_next;
		pgt->bkt_table[bkt].pg_next = pg_no;
		pg->pg_flags |= ODS_F_IN_BKT;
		return;
	}

	/* The bucket is on the list, and it not empty return */
	if (pg->pg_bits[0] != bkt_bits[bkt].mask_0
	    || pg->pg_bits[1] != bkt_bits[bkt].mask_1) {
		return;
	}

	/* If this is the only bucket remaining on the list, leave it */
	if (pgt->bkt_table[bkt].pg_next == pg_no && pg->pg_next == 0) {
		return;
	}

	/* Remove this bucket to avoid accumulating empty buckets on a bucket list */
	del_bkt_tbl_pg(pgt, bkt, pg_no);

	/* Free the page */
	free_pages(ods, pg_no);
}

static void free_pages(ods_mmap_t ods, uint64_t pg_no)
{
	ods_pgt_t pgt = ods->pg_table;
	ods_pg_t pg, next_ext;
	uint64_t count;

	if (pg_no == 0 || pg_no >= pgt->pg_count) {
		ods_lerror("Attempt to free an invalid page number: %ld\n", pg_no);
		return;
	}

	pg = &pgt->pg_pages[pg_no];
	if (0 == pg->pg_count) {
		/* If the page count is 0, this is an interior page in an extent */
		ods_lerror("Page %ld is an interior page of an extent\n", pg_no);
		return;
	}
	for (count = pg->pg_count; count; count--) {
		/* Mark all the pages in this extent as free */
		pg->pg_flags = 0;
		pg++;
	}
	pg = &pgt->pg_pages[pg_no];

	/* Insert at head of the free list */
	if ((0 == pgt->pg_free) || (pg_no < pgt->pg_free)) {
		pg->pg_next = pgt->pg_free;
		pgt->pg_free = pg_no;

		/* Coalesce adjacent extent */
		next_ext = pg + pg->pg_count; /* adjacent extent */
		if (next_ext->pg_flags == 0) {
			uint64_t next_count, next_no;
			/* Save off the info for the ext being coalesced */
			next_no = pg->pg_next;
			next_count = next_ext->pg_count;

			pg->pg_next = next_ext->pg_next;
			pg->pg_count += next_count;
			next_ext->pg_count = 0;

			next_ext = &pgt->pg_pages[next_no + next_count];
		}
		return;
	}
	uint64_t pg_prev, pg_next;
	/* Search for the free entry that is less than pg_no */
	for (pg_prev = 0, pg_next = pgt->pg_free; pg_next && pg_next < pg_no;
	     pg_prev = pg_next, pg_next = pgt->pg_pages[pg_next].pg_next);
	assert(pg_prev);

	ods_pg_t prev_ext = &pgt->pg_pages[pg_prev];
	/* See if previous extent abuts this extent */
	if ((pg_prev + prev_ext->pg_count) == pg_no) {
		/* Combine extents */
		prev_ext->pg_count += pg->pg_count;
		pg_no = pg_prev;
		pg = prev_ext;
	} else {
		pg->pg_next = prev_ext->pg_next;
		prev_ext->pg_next = pg_no;
	}
	/* See if this spanned to the next free extent */
	if ((pg_no + pg->pg_count) == pg->pg_next) {
		ods_pg_t next_ext = &pgt->pg_pages[pg->pg_next];
		pg->pg_next = next_ext->pg_next;
		pg->pg_count += next_ext->pg_count;
	}
}

static int commit_map_fn(struct ods_rbn *rbn, void *arg, int l)
{
	ods_map_t map = container_of(rbn, struct ods_map_s, rbn);
	int rc = msync(map->data, map->map.len, (int)(unsigned long)arg);
	if (rc) {
		ods_lerror("Error %d in %s msyncing map %p of length %ld\n",
			   rc, __func__, map->data, map->map.len);
	}
	return 0;
}

/*
 * This function is thread safe.
 */
static void ods_mmap_commit(ods_t ods_, int flags)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	int mflag = (flags ? MS_SYNC : MS_ASYNC);
	__ods_lock(ods_);
	ods_rbt_traverse(&ods->map_tree, commit_map_fn, (void *)(unsigned long)mflag);
	__ods_unlock(ods_);
}

/*
 * This function is racing with the garbage cleanup thread and is
 * thread-safe, but is otherwise, not thread-safe. IOW, if there are
 * references to the ODS handle and the application tries to use it
 * after this function returns, it will crash.
 */
static int ods_mmap_close(ods_t ods_, int flags)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_map_t map;
	if (!ods)
		return 0;

	ods_commit(ods_, flags);
	pgt_unmap(ods);
	if (ods->lck_table)
		munmap(ods->lck_table, ODS_PAGE_SIZE);
	close(ods->pg_fd);
	close(ods->obj_fd);
	free(ods->base.path);

	/* Clean up any maps */
	struct ods_rbn *rbn;
	while ((rbn = ods_rbt_min(&ods->map_tree))) {
		map = container_of(rbn, struct ods_map_s, rbn);
		ods_rbt_del(&ods->map_tree, rbn);
		int rc = munmap(map->data, map->map.len);
		assert(0 == rc);
		free(map);
	}

	free(ods);
	return 0;
}

static void free_ref(ods_mmap_t ods, ods_ref_t ref)
{
	uint64_t pg_no = ref_to_page_no(ref);
	ods_pgt_t pgt = ods->pg_table;

	assert(ref < ods->obj_sz);
	if (pgt->pg_pages[pg_no].pg_flags & ODS_F_IDX_VALID)
		free_blk(ods, ref);
	else
		free_pages(ods, pg_no);
}

static uint32_t ods_mmap_ref_status(ods_t ods_, ods_ref_t ref)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	uint64_t pg_no;
	ods_pgt_t pgt;
	ods_pg_t pg;
	uint32_t status = 0;

	__ods_lock(ods_);
	__pgt_lock(ods);

	if (!ref_valid(ods, ref)) {
		status |= ODS_REF_STATUS_INVALID;
		goto out;
	}

	pgt = ods->pg_table;
	pg_no = ref_to_page_no(ref);
	pg = &pgt->pg_pages[pg_no];

	if (pg->pg_flags & ODS_F_IDX_VALID) {
		if (0 == (pg->pg_flags & ODS_F_ALLOCATED))
			status |= ODS_REF_STATUS_FREE;
		if (pg->pg_bkt_idx < 0 || pg->pg_bkt_idx >= ODS_BKT_TABLE_SZ)
			status |= ODS_REF_STATUS_CORRUPT;
		status |= bkt_to_size(pg->pg_bkt_idx);
	} else {
		if (0 == (pg->pg_flags & ODS_F_ALLOCATED))
			status |= ODS_REF_STATUS_FREE;
	}
 out:
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return status;
}

/*
 * This function is thread safe
 */
static void ods_mmap_delete(ods_t ods_, ods_ref_t ref)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	__ods_lock(ods_);
	__pgt_lock(ods);
	free_ref(ods, ref);
	__pgt_unlock(ods);
	__ods_unlock(ods_);
}

static void ods_mmap_dump(ods_t ods_, FILE *fp)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	fprintf(fp, "------------------------------- ODS Dump --------------------------------\n");
	fprintf(fp, "%-32s : \"%s\"\n", "Path", ods_->path);
	fprintf(fp, "%-32s : %d\n", "Object File Fd", ods->obj_fd);
	fprintf(fp, "%-32s : %zu\n", "Object File Size", ods->obj_sz);
	fprintf(fp, "%-32s : %d\n", "Page File Fd", ods->pg_fd);
	fprintf(fp, "%-32s : %zu\n", "Page File Size", ods->pg_sz);

	ods_pgt_t pgt = ods->pg_table;
	ods_pg_t pg;
	uint64_t pg_no;

	fprintf(fp, "--------------------------- Allocated Pages ----------------------------\n");
	uint64_t count = 0;

	for(pg_no = 1; pg_no < pgt->pg_count; ) {
		pg = &pgt->pg_pages[pg_no];
		if (0 == pg->pg_flags || 0 != (pg->pg_flags & ODS_F_IDX_VALID)) {
			pg_no++;
			continue;
		}
		count += pg->pg_count;
		fprintf(fp, "PGS [%6ld]     %10ld ... %ld\n",
			pg->pg_count, pg_no, pg_no + pg->pg_count - 1);
		pg_no += pg->pg_count;
	}
	fprintf(fp, "Total Allocated Pages: %ld\n", count);

	fprintf(fp, "--------------------------- Block Usage ----------------------------\n");
	int bkt, blk, sz;
	int hdr, allocb, freeb;
	for (bkt = 0; bkt < ODS_BKT_TABLE_SZ; bkt ++) {
		hdr = 1;
		allocb = freeb = 0;
		sz = bkt_to_size(bkt);
		for (pg_no = pgt->bkt_table[bkt].pg_next; pg_no; pg_no = pg->pg_next) {
			if (hdr) {
				printf("Block Size: %6dB\n", sz);
				hdr = 0;
			}
			pg = &pgt->pg_pages[pg_no];
			printf("%10ld ", pg_no);
			for (blk = 0; blk < ODS_PAGE_SIZE / sz; blk ++) {
				if (test_bit(pg->pg_bits, blk)) {
					/* block is free */
					printf("-");
					freeb += 1;
				} else {
					printf("A");
					allocb += 1;
				}
			}
			printf("\n");
		}
		if (!hdr)
			printf("           Total: %d   Allocated/Free: %d /%d\n",
			       allocb + freeb, allocb, freeb);
	}

	fprintf(fp, "------------------------------ Free Pages ------------------------------\n");
	count = 0;
	for (pg_no = pgt->pg_free; pg_no && pg_no < pgt->pg_count; ) {
		pg = &pgt->pg_pages[pg_no];
		fprintf(fp, "%-32s : 0x%016lx / %zu\n", "Page No / Page Count", pg_no, pg->pg_count);
		count += pg->pg_count;
		pg_no = pgt->pg_pages[pg_no].pg_next;
	}
	fprintf(fp, "Total Free Pages: %ld\n", count);

	fprintf(fp, "==============================- ODS End =================================\n");
}

static void ods_mmap_obj_iter_pos_init(ods_obj_iter_pos_t pos)
{
	pos->page_no = 1;		      /* first page is udata */
	pos->blk = 0;
}

/*
 * This function is _not_ thread safe
 */
static int ods_mmap_obj_iter(ods_t ods_, ods_obj_iter_pos_t pos,
		 ods_obj_iter_fn_t iter_fn, void *arg)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_pgt_t pgt = ods->pg_table;
	ods_pg_t pg;
	uint64_t pg_no, blk;
	int bkt, rc = 0;
	size_t sz;

	if (pos) {
		pg_no = pos->page_no;
		if (!pg_no)
			pg_no = 1;
		blk = pos->blk;
	} else {
		pg_no = 1;
		blk = 0;
	}

	for(; pg_no < pgt->pg_count; ) {
		pg = &pgt->pg_pages[pg_no];
		if (0 == pg->pg_flags) {
			pg_no++;
			continue;
		}
		if (pg->pg_flags & ODS_F_IDX_VALID) {
			bkt = pg->pg_bkt_idx;
			sz = bkt_to_size(bkt);
			for (; blk < ODS_PAGE_SIZE / sz; blk ++) {
				if (test_bit(pg->pg_bits, blk))
					/* block is free */
					continue;
				rc = iter_fn(ods_, (pg_no << ODS_PAGE_SHIFT) | (blk * sz), arg);
				if (rc)
					goto out;
			}
			pg_no++;
		} else {
			blk = 0;
			rc = iter_fn(ods_, pg_no << ODS_PAGE_SHIFT, arg);
			pg_no += pg->pg_count;
			if (rc)
				goto out;
		}
		blk = 0;
	}
 out:
	if (pos) {
		pos->page_no = pg_no;
		pos->blk = blk;
	}
	return rc;
}

static void ods_mmap_release_dead_locks(ods_t ods_)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_pgt_t pgt;
	pthread_mutex_t *mtx;
	int id;

	__ods_lock(ods_);
	__pgt_lock(ods);
	pgt = pgt_get(ods);
	if (!pgt)
		goto out;

	mtx = &pgt->pgt_lock.mutex;
	check_lock(mtx, 1);

	for (id = 0; id < ODS_LOCK_CNT; id++) {
		mtx = &pgt->lck_tbl[id].mutex;
		check_lock(mtx, 1);
	}
 out:
	__pgt_unlock(ods);
	__ods_unlock(ods_);
}

static int ods_mmap_destroy(ods_t ods)
{
	char tmp_obj_path[PATH_MAX];
	char tmp_pg_path[PATH_MAX];
	int rc;
	int status = 0;

	sprintf(tmp_obj_path, "%s%s", ods->path, ODS_OBJ_SUFFIX);
	sprintf(tmp_pg_path, "%s%s", ods->path, ODS_PGTBL_SUFFIX);

	ods_close(ods, ODS_COMMIT_SYNC);

	/* Destroy the obj file */
	rc = unlink(tmp_obj_path);
	if (rc < 0 && errno != ENOENT)
		status = errno;

	/* Destroy the page file */
	rc = unlink(tmp_pg_path);
	if (rc < 0 && errno != ENOENT)
		status = errno;
	return status;
}

static int ods_mmap_begin(ods_t ods_, struct timespec *wait)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_pgt_t pgt;
	uint64_t pid = (uint64_t)getpid();
	struct timespec now;
	int rc;

 retry:
	__ods_lock(ods_);
	__pgt_lock(ods);
	pgt = pgt_get(ods);
	if (!pgt) {
		rc = EINVAL;
		goto out;
	}

	if (0 == (rc = __sync_val_compare_and_swap(&pgt->pgt_x, 0, pid)))
		goto out;

	__pgt_unlock(ods);
	__ods_unlock(ods_);
	usleep(500);

	if (!wait)
		goto retry;

	clock_gettime(CLOCK_REALTIME, &now);
	if (now.tv_sec > wait->tv_sec) {
		return ETIMEDOUT;
	} else if (now.tv_sec == wait->tv_sec) {
		if (now.tv_nsec > wait->tv_nsec) {
			return ETIMEDOUT;
		}
	}
	goto retry;

 out:
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return rc;
}

static int ods_mmap_end(ods_t ods_)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_pgt_t pgt;
	int rc = 0;

	__ods_lock(ods_);
	__pgt_lock(ods);

	pgt = pgt_get(ods);
	if (!pgt) {
		rc = EINVAL;
		goto out;
	}
	__sync_lock_release(&pgt->pgt_x, 0);
 out:
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return rc;
}

static pid_t ods_mmap_test(ods_t ods_)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	ods_pgt_t pgt;
	pid_t pid;

	__ods_lock(ods_);
	__pgt_lock(ods);

	pgt = pgt_get(ods);
	if (!pgt) {
		pid = (pid_t)-1;
		errno = EINVAL;
		goto out;
	}
	pid = (pid_t)pgt->pgt_x;
 out:
	__pgt_unlock(ods);
	__ods_unlock(ods_);
	return pid;
}

static uint64_t ods_mmap_flush_data(ods_t ods, int keep_time);
ods_t ods_mmap_open(const char *path, ods_perm_t o_perm, int o_mode)
{
	char tmp_path[PATH_MAX];
	struct stat sb;
	ods_mmap_t ods;
	int obj_fd = -1;
	int pg_fd = -1;
	int lock_fd = -1;
	int rc;

	ods = calloc(1, sizeof *ods);
	if (!ods) {
		errno = ENOMEM;
		return NULL;
	}
	ods->base.o_perm = o_perm;
	ods->base.be_type = ODS_BE_MMAP;
	ods->base.begin_x = ods_mmap_begin;
	ods->base.end_x = ods_mmap_end;
	ods->base.test_x = ods_mmap_test;
	ods->base.commit = ods_mmap_commit;
	ods->base.close = ods_mmap_close;
	ods->base.alloc =  ods_mmap_alloc;
	ods->base.delete = ods_mmap_delete;
	ods->base.ref_as_ptr = ods_mmap_ref_as_ptr;
	ods->base.update = ods_mmap_update;

	ods->base.ref_status = ods_mmap_ref_status;
	ods->base.ref_valid = ods_mmap_ref_valid;
	ods->base.extend = ods_mmap_extend;
	ods->base.dump = ods_mmap_dump;
	ods->base.obj_iter_pos_init = ods_mmap_obj_iter_pos_init;
	ods->base.obj_iter = ods_mmap_obj_iter;
	ods->base.dump = ods_mmap_dump;
	ods->base.fstat_get = ods_mmap_fstat_get;
	ods->base.stat_get = ods_mmap_stat_get;
	ods->base.destroy = ods_mmap_destroy;

	ods->base.version = ods_mmap_version;

	ods->base.lock_ = ods_mmap_lock;
	ods->base.unlock = ods_mmap_unlock;
	ods->base.lock_cleanup = ods_mmap_lock_cleanup;
	ods->base.lock_info = ods_mmap_lock_info;
	ods->base.info = ods_mmap_info;

	ods->base.get_user_data = ods_mmap_get_user_data;
	ods->base.obj_put = ods_mmap_obj_put;
	ods->base.stat_buf_new = ods_mmap_stat_buf_new;
	ods->base.stat_buf_del = ods_mmap_stat_buf_del;

	ods->base.release_dead_locks = ods_mmap_release_dead_locks;
	ods->base.flush_data = ods_mmap_flush_data;

	ods->base.get = ods_mmap_get;
	ods->base.set = ods_mmap_set;

	/* Take the ods file lock */
	char *dir = strdup(path);
	if (!dir)
		return NULL;
	char *base = strdup(path);
	if (!base)
		return NULL;
	sprintf(tmp_path, "%s/.%s.lock", dirname((char *)dir),
		basename((char *)base));
	free(dir);
	free(base);
	mode_t oumask = umask(0);
	lock_fd = open(tmp_path, O_RDWR | O_CREAT, 0666);
	umask(oumask);
	if (lock_fd < 0)
		return NULL;
	rc = flock(lock_fd, LOCK_EX);
	if (rc) {
		close(lock_fd);
		errno = rc;
		return NULL;
	}

	/* Open the obj file */
	sprintf(tmp_path, "%s%s", path, ODS_OBJ_SUFFIX);
	obj_fd = open(tmp_path, O_RDWR);
	if (obj_fd < 0) {
		if (ods->base.o_perm & ODS_PERM_CREAT) {
			rc = __ods_create(path, o_mode);
			if (rc)
				goto err;
			obj_fd = open(tmp_path, O_RDWR);
			if (obj_fd < 0)
				goto err;
		} else
			goto err;
	}
	close(lock_fd);
	ods->obj_fd = obj_fd;

	/* Open the page table file */
	sprintf(tmp_path, "%s%s", path, ODS_PGTBL_SUFFIX);
	pg_fd = open(tmp_path, O_RDWR);
	if (pg_fd < 0)
		goto err;
	ods->pg_fd = pg_fd;

	ods->base.path = strdup(path);
	if (!ods->base.path)
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
	if (!pgt_map(ods))
		goto err;
	if (!lck_map(ods))
		goto err;

	ods->obj_map_sz = MMOS_DEF_MAP_SZ;
	ods_rbt_init(&ods->map_tree, map_cmp, NULL);

	return &ods->base;

 err:
	close(lock_fd);
	rc = errno;
	if (ods->base.path)
		free(ods->base.path);
	if (pg_fd >= 0)
		close(pg_fd);
	if (obj_fd >= 0)
		close(obj_fd);
	free(ods);
	errno = rc;
	return NULL;
}

static int q4_del_fn(struct ods_rbn *rbn, void *arg, int l)
{
	struct del_fn_arg_s *darg = arg;

	ods_map_t map = container_of(rbn, struct ods_map_s, rbn);
	darg->mapped += map->map.len;
	if (map->refcount > 1) {
		map->last_used = time(NULL);
		return 0;
	}
	if (map->last_used + darg->timeout < time(NULL)) {
		/*
		 * It hasn't been used since the last collection
		 * cycle, delete the map
		 */
		LIST_INSERT_HEAD(darg->del_q, map, entry);
		int rc = msync(map->data, map->map.len, MS_ASYNC | MS_INVALIDATE);
		if (rc) {
			ods_lerror("Error %d in %s msyncing map %p of length %ld\n",
				   rc, __func__, map->data, map->map.len);
		}
	}
	return 0;
}

static uint64_t ods_mmap_flush_data(ods_t ods_, int keep_time)
{
	ods_mmap_t ods = (ods_mmap_t)ods_;
	struct map_list_head del_list;
	struct del_fn_arg_s fn_arg;
	uint64_t mapped;

	if (keep_time == 0) {
		struct ods_rbn *rbn;
		__ods_lock(ods_);
		ODS_RBT_FOREACH(rbn, &ods->map_tree) {
			ods_map_t map = container_of(rbn, struct ods_map_s, rbn);
			int rc = msync(map->data, map->map.len, MS_ASYNC | MS_INVALIDATE);
			if (rc) {
				ods_lerror("Error %d in %s msyncing map %p of length %ld MB\n",
					   rc, __func__, map->data, map->map.len);
			}
		}
		__ods_unlock(ods_);
		return 0;
	}
	/*
	* Traverse the map_tree and add the maps that can be
	* deleted to the del_list
	*/
	LIST_INIT(&del_list);
	__ods_lock(ods_);
	fn_arg.timeout = keep_time;
	fn_arg.del_q = &del_list;
	fn_arg.mapped = 0;
	ods_rbt_traverse(&ods->map_tree, q4_del_fn, &fn_arg);
	mapped = fn_arg.mapped;
	empty_del_list(&del_list);
	__ods_unlock(ods_);
	return mapped;
}
