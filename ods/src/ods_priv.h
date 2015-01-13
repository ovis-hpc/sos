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

#ifndef __ODS_PRIV_H
#define __ODS_PRIV_H
#include <ods/ods_atomic.h>
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

struct ods_map_s {
	/*
	 * Reference count for the map. One ref is held by ods_open
	 * and another reference is held by any caller to
	 * ods_begin. When ods_end is called, the ods_begin reference
	 * is dropped. The ods_open reference is dropped when
	 * ods_extend re-maps the backing file and when ods_close is
	 * called.
	 */
	ods_atomic_t refcount;

	/* The ODS for this map */
	ods_t ods;

	/*
	 * The mapping generation number. Used to detect if another
	 * process extended the map and we need to remap.
	 */
	uint64_t obj_gen;

	/* Equivalent of obj_gen for pg_table mapping */
	uint64_t pg_gen;

	/* The object map size */
	size_t obj_sz;

	/* Pointer to the object data in memory */
	struct ods_obj_data_s *obj_data;

	/* The page-file map size */
	size_t pg_sz;

	/* Pointer to the page-file data in memory */
	struct ods_pgt_s *pg_table;

	LIST_ENTRY(ods_map_s) entry;
};

typedef struct ods_dirty_s {
	ods_ref_t start;
	ods_ref_t end;
	ods_t ods;
	struct rbn rbn;
} *ods_dirty_t;

struct ods_s {
	pthread_spinlock_t lock;

	/* That path to the file on disk */
	char *path;

	/* Open permissions */
	ods_perm_t o_perm;

	/* The open file descriptor */
	int obj_fd;
	size_t obj_sz;

	/* The page-file file descriptor */
	int pg_fd;
	size_t pg_sz;

	/* The current map */
	ods_map_t map;

	/* The object list */
	ods_atomic_t obj_count;
	LIST_HEAD(obj_list_head, ods_obj_s) obj_list;
	LIST_HEAD(obj_free_list_head, ods_obj_s) obj_free_list;
	LIST_HEAD(map_list_head, ods_map_s) map_list;

	/* The dirty tree */
	struct rbt dirty_tree;

	LIST_ENTRY(ods_s) entry;
};

#define ODS_OBJ_SIGNATURE "OBJSTORE"
#define ODS_PGT_SIGNATURE "PGTSTORE"
#define ODS_OBJ_VERSION   "04012014"

typedef struct ods_pg_s {
	uint64_t next;	/* Next free page range */
	uint64_t count;	/* number of pages in this page range */
} *ods_pg_t;

typedef struct ods_blk_s {
	uint64_t next;		/* next block */
} *ods_blk_t;

#define ODS_PAGE_SIZE	4096
#define ODS_PAGE_SHIFT	12
#define ODS_PAGE_MASK	~(ODS_PAGE_SIZE-1)
#define ODS_GRAIN_SIZE	32
#define ODS_GRAIN_SHIFT	5

#define ODS_M_IDX	0x0F /* Mask for block index */
#define ODS_F_IDX_VALID	0x10 /* Bucket index is valid */
#define ODS_F_NEXT	0x20 /* Next page is part of this allocation */
#define ODS_F_PREV	0x40 /* Previous page is part of this allocation */
#define ODS_F_ALLOCATED	0x80 /* Page is allocated */

struct ods_pgt_s {
	char signature[8];	 /* pgt signature 'PGTSTORE' */
	uint64_t gen;		 /* generation number */
	uint64_t count;		 /* count of pages */
	unsigned char pages[0];	 /* array of page control information */
};

struct ods_obj_data_s {
	char signature[8];	 /* obj signature 'OBJSTORE' */
	uint64_t version;	 /* The file format version number */
	uint64_t gen;		 /* generation number */
	uint64_t pg_free;	 /* first free page offset */
	uint64_t blk_free[ODS_PAGE_SHIFT - ODS_GRAIN_SHIFT];
};
#define ODS_UDATA_SIZE (ODS_PAGE_SIZE - sizeof(struct ods_obj_data_s))

static inline int ods_page_is_allocated(ods_map_t map, uint64_t page) {
	unsigned char b = map->pg_table->pages[page];
	return (0 != (b & ODS_F_ALLOCATED));
}

static inline int ods_page_is_free(ods_map_t map, uint64_t page) {
	unsigned char b = map->pg_table->pages[page];
	return (0 == (b & ODS_F_ALLOCATED));
}

static inline uint64_t ods_page_next(ods_map_t map, uint64_t page) {
	unsigned char b = map->pg_table->pages[page];
	if (0 == (b & ODS_F_NEXT))
		return 0;
	return page+1;
}

static inline uint64_t ods_page_prev(ods_map_t map, uint64_t page) {
	unsigned char b = map->pg_table->pages[page];
	if (0 == (b & ODS_F_PREV))
		return 0;
	return page-1;
}

static inline struct ods_pg_s *ods_page_to_ptr(ods_map_t map, uint64_t page) {
	uint64_t off;
	if (!page)
		return NULL;
	off = (uint64_t)map->obj_data;
	return (struct ods_pg_s *)(off + (page << ODS_PAGE_SHIFT));
}

#define ODS_PGTBL_MIN_SZ	(4096)
#define ODS_PGTBL_MIN_SZ	(4096)
#define ODS_OBJ_MIN_SZ		(16 * 4096)
#endif
