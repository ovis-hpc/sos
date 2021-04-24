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

/**
 * ods - A file backed object store
 *
 * There are two types of memory allocations supported: pagesets and
 * blocks. A pageset is a collection of ODS_PAGE sized chunks of
 * memory that are contiguous. A block is a smaller chunk of memory
 * that is part of a subdivided pageset. Blocks are kept in a table
 * indexed by i_log2(desired_size) - i_log2(ODS_MIN_SIZE). If this
 * table is empty, a ODS_PAGE chunk is allocated and subdivided in
 * blocks that are chained on a list and added to this table.
 *
 *   7   6   5   4   3   2   1   0
 * +---------------+---------------+
 * | o | o | o | o | o | o | o | o |
 * +-+---+---+---+---+---+---+---+-+
 *   |   |   |   |   |   |   |   |
 *   |   |   |   |   +---+---+---+---: If bit4==1, the Log2 size of the block
 *   |   |   |   +-------------------: The index is valid
 *   |   |   +-----------------------: The next page is part of this chunk
 *   |   +---------------------------: The prev page is part of this chunk
 *   +-------------------------------: The page is part of an allocation
 *
 *
 */

/*
 * The ODS obj map is a partial map of the ODS object file.
 */
struct ods_map_s {
	/*
	 * Reference count for the map. Every ods_obj has a
	 * reference. When the last reference is dropped, the map is a
	 * candidate to be cleaned up by the garbage collection thread
	 * and the extent munmap'd.
	 */
	ods_atomic_t refcount;

	/* The ODS for this map */
	ods_t ods;

	/* Pointer to the data in memory */
	unsigned char *data;

	/* time() last used */
	time_t last_used;

	struct map_key_s {
		loff_t off;		/* Map offset */
		size_t len;		/* This length of this map in Bytes */
	} map;
	struct ods_rbn rbn;		/* Active map tree */
	LIST_ENTRY(ods_map_s) entry;	/* Queued for deletion */
};

struct ods_s {
	/* The path to the file on disk */
	char *path;

	/* Open permissions */
	ods_perm_t o_perm;

	/* The open file descriptor */
	int obj_fd;
	size_t obj_sz;

	/* The page-file file descriptor */
	int pg_fd;
	size_t pg_sz;

	/*
	 * The mapping generation number. Used to detect if another
	 * process extended the page file and we need to remap it.
	 */
	uint64_t pg_gen;

	/* Pointer to the page-file data in memory */
	struct ods_pgt_s *lck_table; /* never grows, persistent until close */
	struct ods_pgt_s *pg_table; /* grows on ods_extend */

	/* Current ODS map size for new maps in bytes */
	size_t obj_map_sz;

	/* Tree of object maps. Key is file offset and map length. */
	ods_map_t last_map;
	struct ods_rbt map_tree;

	/* Local lock for this ODS instance */
	pthread_mutex_t lock;

	/* The object list */
	ods_atomic_t obj_count;
	LIST_HEAD(obj_list_head, ods_obj_s) obj_list;

	/* The dirty tree */
	struct ods_rbt dirty_tree;

	LIST_ENTRY(ods_s) entry;

	void (*commit)(ods_t ods, int flags);
	int (*close)(ods_t ods, int flags);

	ods_obj_t (*obj_alloc)(ods_t ods, size_t sz);
	void (*obj_delete)(ods_obj_t obj);
	void (*ref_delete)(ods_t ods, ods_ref_t ref);
	uint32_t (*ref_status)(ods_t ods, ods_ref_t ref);
	int (*ref_valid)(ods_t ods, ods_ref_t ref);
	int (*extend)(ods_t ods, size_t sz);
	ods_obj_t (*ref_as_obj)(ods_t ods, ods_ref_t ref);

	uint64_t (*get_alloc_size)(ods_map_t map, uint64_t size);
	void (*obj_iter_pos_init)(ods_obj_iter_pos_t pos);
	int (*obj_iter)(ods_t ods, ods_obj_iter_pos_t pos,
			ods_obj_iter_fn_t iter_fn, void *arg);

	void (*dump)(ods_t ods, FILE *fp);
	int (*stat_get)(ods_t ods, ods_stat_t osb);
	int (*fstat_get)(ods_t ods, struct stat *);
	int (*destroy)(ods_t ods);

	struct ods_version_s (*version)(ods_t ods);
	size_t (*lock_count)(ods_t ods);
	int (*lock_)(ods_t ods, int lock_id, struct timespec *wait);
	void (*unlock)(ods_t ods, int lock_id);
	void (*dump_maps)(const char *name);
	int (*lock_cleanup)(const char *path);
	int (*lock_info)(const char *path, FILE *fp);
	void (*info)(ods_t ods, FILE *fp, int flags);
	void (*obj_put)(ods_obj_t obj);
	int (*obj_valid)(ods_t ods, ods_obj_t obj);
	ods_stat_t (*stat_buf_new)(ods_t ods);
	void (*stat_buf_del)(ods_t ods, ods_stat_t buf);
	ods_obj_t (*get_user_data)(ods_t ods);

	void (*release_dead_locks)(ods_t ods);
	uint64_t (*flush_data)(ods_t ods, int keep_time);

};

#define ODS_OBJ_SIGNATURE "OBJSTORE"
#define ODS_PGT_SIGNATURE "PGTSTORE"

#define ODS_PAGE_SIZE	 4096
#define ODS_PAGE_SHIFT	 12
#define ODS_PAGE_MASK	 ~(ODS_PAGE_SIZE-1)
#define ODS_GRAIN_SIZE	 32
#define ODS_GRAIN_SHIFT	 5
#define ODS_BKT_TABLE_SZ 64

#define ODS_F_IDX_VALID		0x10 /* Bucket index is valid */
#define ODS_F_IN_BKT		0x20 /* In the bucket table */
#define ODS_F_ALLOCATED		0x80 /* Page is allocated */

/*
 *              Page Table
 *              +----------+
 *   pg_free -->| pg_next  o----+
 *              +----------+    |
 *     +------->| blk_next o----^--+
 *     |        +----------+	|  |
 *     |   +----o pg_next  |<---+  |
 *     |   |    +----------+	   |
 *     |   | +--o blk_next |<------+
 *     |   | |  +----------+
 *     |   | +->| blk_next |
 *     |   |    +----------+
 *     |   +--->| pg_next  |
 *     |        +----------+
 *     |        S          S
 *     |        +----------+
 *     |        |          |
 *     |        +----------+
 *     |
 *     |        Bucket Table
 *     |   ---- +----------+
 *     |    ^   |          |
 *     |    |   +----------+
 *     +----^---o blk_next |
 *          |   +----------+
 *         64   S          S
 *          |   +----------+
 *          v   |          |
 *         ---- +----------+
 *
 * Buckets are arrays of blocks of the same size. All block sizes are
 * restricted to 32B ... 2048B in 32B increments; which is 63 size
 * classes 0 == 32B, 62 == 2048B. Objects > 2048B consume an integral
 * number of pages.
 *
 * The 128b pg_bits field in the page table has a bit for each block in
 * the page which is a maximum of 128 blocks per page.
 */
typedef struct ods_pg_s {
	uint64_t pg_flags:8;	/* Indicates if the page is allocated and whether or not it is bucket list member */
	uint64_t pg_bkt_idx:8;	/* If page contains blocks, this is the index in the bucket table */
	uint64_t pg_next;	/* Page no of next extent */
	uint64_t pg_count;	/* number of pages in this extent */
	uint64_t pg_bits[2];	/* 1 if blk allocated, 0 if block is free */
} *ods_pg_t;

typedef struct ods_lock_s {
	union {
		struct ods_spin_lock_s {
			ods_atomic_t lock_word;
			ods_atomic_t contested;
			uint64_t owner;
		} spin;
		pthread_mutex_t mutex;
	};
} ods_lock_t;

#pragma pack(4)

#define ODS_PGT_PFX_SZ  (8 +				\
			 44 +				\
			 sizeof(struct ods_version_s) +	\
			 (3 * sizeof(uint64_t)) +	\
			 sizeof(ods_lock_t)		\
			 )
#define ODS_LOCK_MEM_SZ	(ODS_PAGE_SIZE - ODS_PGT_PFX_SZ)
#define ODS_LOCK_CNT	(ODS_LOCK_MEM_SZ / sizeof(ods_lock_t))

typedef struct ods_bkt_s {
	uint64_t pg_next;	/* next bucket */
} *ods_bkt_t;

typedef struct ods_pgt_s {
	char pg_signature[8];	 /* pgt signature 'PGTSTORE' */
	char pg_commit_id[41];	 /* git SHA1 hash is 40B */
	struct ods_version_s pg_vers; /* PGT version */
	uint64_t pg_gen;	 /* generation number */
	uint64_t pg_free;	 /* first free page number */
	uint64_t pg_count;	 /* count of pages */
	ods_lock_t pgt_lock;	 /* inter-process page-table lock */
	/* Inter-process locks for applications */
	union {
		unsigned char lock_mem[ODS_LOCK_MEM_SZ];
		ods_lock_t lck_tbl[0];
	};
	/* Should begin on a 4096B boundary */
	struct ods_bkt_s bkt_table[ODS_BKT_TABLE_SZ];
	struct ods_pg_s pg_pages[0];/* array of page control information */
} *ods_pgt_t;

struct ods_obj_data_s {
	char obj_signature[8];	 /* obj signature 'OBJSTORE' */
};
#pragma pack()

#define ODS_UDATA_SIZE (ODS_PAGE_SIZE - sizeof(struct ods_obj_data_s))

#define ODS_PGTBL_MIN_SZ	(4096)
#define ODS_PGTBL_MIN_SZ	(4096)
#define ODS_OBJ_MIN_SZ		(16 * 4096)

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

ods_t ods_mmap_open(const char *path, ods_perm_t o_perm, int o_mode);

#endif
