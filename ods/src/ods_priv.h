/*
 * Copyright (c) 2020 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2013 Sandia Corporation. All rights reserved.
 *
 * See the file COPYING at the top of this source tree for the terms
 * of the Copyright.
 */
/*
 * Author: Tom Tucker tom at ogc dot us
 */

#ifndef __ODS_PRIV_H
#define __ODS_PRIV_H
#include <ods/ods_atomic.h>
#include <ods/ods_rbt.h>
#include <stdarg.h>
#include <stdint.h>

struct ods_s {
	/* The path to the file on disk */
	char *path;
	ods_atomic_t open_count;
	struct ods_rbn rbn;

	/* Open flags */
	ods_perm_t o_perm;

	/* Local lock for this ODS instance */
	pthread_mutex_t lock;

	/* The object list */
	ods_atomic_t obj_count;
	LIST_HEAD(obj_list_head, ods_obj_s) obj_list;

	int (*close)(ods_t ods, int flags);
	void (*commit)(ods_t ods, int flags);
	ods_ref_t (*alloc)(ods_t ods, size_t sz);
	void (*delete)(ods_t ods, ods_ref_t ref);
	void *(*ref_as_ptr)(ods_t od_, ods_ref_t ref, uint64_t *ref_sz, void **context);
	uint32_t (*ref_status)(ods_t ods, ods_ref_t ref);
	int (*extend)(ods_t ods, size_t sz);
	int (*ref_valid)(ods_t ods, ods_ref_t ref);
	void (*update)(ods_t ods, ods_ref_t ref, loff_t offset, size_t len);

	void (*obj_iter_pos_init)(ods_obj_iter_pos_t pos);
	int (*obj_iter)(ods_t ods, ods_obj_iter_pos_t pos,
			ods_obj_iter_fn_t iter_fn, void *arg);

	void (*dump)(ods_t ods, FILE *fp);
	ods_stat_t (*stat_buf_new)(ods_t ods);
	void (*stat_buf_del)(ods_t ods, ods_stat_t buf);
	int (*stat_get)(ods_t ods, ods_stat_t osb);
	int (*fstat_get)(ods_t ods, struct stat *);
	int (*destroy)(ods_t ods);

	struct ods_version_s (*version)(ods_t ods);
	int (*lock_)(ods_t ods, int lock_id, struct timespec *wait);
	void (*unlock)(ods_t ods, int lock_id);
	void (*dump_maps)(const char *name);
	int (*lock_cleanup)(const char *path);
	int (*lock_info)(const char *path, FILE *fp);
	void (*info)(ods_t ods, FILE *fp, int flags);
	void (*obj_put)(ods_obj_t obj);
	ods_ref_t (*get_user_data)(ods_t ods);

	void (*release_dead_locks)(ods_t ods);
	uint64_t (*flush_data)(ods_t ods, int keep_time);

	int (*get)(ods_t ods, int id, ...);
	int (*set)(ods_t ods, int id, ...);
};

#define ODS_PAGE_SIZE	 4096
#define ODS_PAGE_SHIFT	 12
#define ODS_PAGE_MASK	 ~(ODS_PAGE_SIZE-1)

#define ODS_LOCK_COUNT	1
#define ODS_MAP_SIZE	2
#define ODC_GC_TIMEOUT	3
#define ODS_OBJ_SIZE	4

static inline void __ods_lock(ods_t ods)
{
	(void)pthread_mutex_lock(&ods->lock);
}

static inline void __ods_unlock(ods_t ods)
{
	pthread_mutex_unlock(&ods->lock);
}

/* Garbage collection timeout */
#define ODS_DEF_GC_TIMEOUT	10 /* 10 seconds */

/* Default map size */
#define ODS_MIN_MAP_SZ	(64 * ODS_PAGE_SIZE)	/* 256K */
#define ODS_DEF_MAP_SZ	(256 * ODS_PAGE_SIZE)	/* 1M */
#define ODS_MAX_MAP_SZ	(512 * ODS_DEF_MAP_SZ)	/* 512M */

extern uint64_t __ods_def_map_sz;
extern time_t __ods_gc_timeout;

void __ods_obj_delete(ods_obj_t obj);
#define ODS_ROUNDUP(_sz_, _align_) (((_sz_) + (_align_) - 1) & ~((_align_)-1))

#pragma pack(4)

#define ODS_BE_SIGNATURE	"ODS_BE_1"
#define ODS_BE_SUFFIX		".BE"
struct ods_backend_s {
	char be_signature[8];	/* ODS_BE_SIGNATURE */
	struct ods_version_s be_vers; /* version */
	uint64_t be_type;	/* The backend type */
};
#pragma pack()

typedef enum ods_backend_type_e {
	ODS_BACKEND_MMAP = 1,	/* Memory Structured Object Store (mmap) */
	ODS_BACKEND_LSOS = 2	/* Log Structured Object Store */
} ods_backend_type_t;

ods_backend_type_t __ods_backend_type(const char *path);

ods_t ods_mmap_open(const char *path, ods_perm_t o_flags, int o_mode);
ods_t ods_lsos_open(const char *path, ods_perm_t o_flags, int o_mode);

#endif
