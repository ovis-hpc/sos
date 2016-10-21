/*
 * Copyright (c) 2016 Open Grid Computing, Inc. All rights reserved.
 *
 * Confidential and Proprietary
 */

/*
 * Author: Tom Tucker tom at ogc dot us
 */

#ifndef _HT_H_
#define _HT_H_

#include <ods/ods_idx.h>
#include <ods/ods.h>
#include "ods_idx_priv.h"

#pragma pack(4)

typedef struct ht_bkt_s {
	uint64_t count;		/* Depth of bucket */
	int64_t prev_bkt;	/* Previous non-empty bucket */
	int64_t next_bkt;	/* Next non-empty bucket */
	ods_ref_t head_ref;	/* Head of bucket list */
	ods_ref_t tail_ref;	/* Tail of bucket list */
} *ht_bkt_t;
typedef struct ht_tbl_s {
	int64_t first_bkt;	/* First non-empty bucket */
	int64_t last_bkt;	/* Last non-empty bucket */
	struct ht_bkt_s table[0];
} *ht_tbl_t;
typedef struct ht_entry_s {
	ods_ref_t key_ref;
	ods_idx_data_t value;
	ods_ref_t next_ref;
	ods_ref_t prev_ref;
} *ht_entry_t;

typedef uint64_t (*ht_hash_fn_t)(const char *key, int key_len, uint64_t seed);
#define HT_DEF_TBL_SIZE 1048583
#define HT_HASH_FNV_32	1
#define HT_HASH_FNV_64	2
typedef struct ht_udata {
	struct ods_idx_meta_data idx_udata;
	ods_ref_t htable_ref;	/* Pointer to the hash table */
	uint64_t hash_type;	/* Hash function type */
	uint64_t htable_size;	/* Size of the hash table */
	uint32_t hash_seed;	/* The hash seed */
	ods_atomic_t client_count;	/* Active clients */
	ods_atomic_t lock;	/* Cross-memory spin lock */
	uint64_t max_bkt;	/* Deepest bucket */
	uint64_t max_bkt_len;	/* Deepest bucket length */
	ods_atomic_t card;	/* Cardinality */
	ods_atomic_t dups;	/* Duplicate keys */
} *ht_udata_t;

/* Structure to hang on to cached node allocations */
struct ht_obj_el {
	ods_obj_t obj;
	ods_ref_t ref;
	LIST_ENTRY(ht_obj_el) entry;
};

/*
 * In memory object that refers to a Hash Table
 */
typedef struct ht_s {
	ods_t ods;		/* The ods that contains the tree */
	ods_obj_t udata_obj;
	ht_udata_t udata;
	ods_obj_t htable_obj;
	ht_tbl_t htable;
	ht_hash_fn_t hash_fn;
	ods_idx_compare_fn_t comparator;
} *ht_t;

typedef struct ht_iter {
	struct ods_iter iter;
	ods_obj_t ent;
	int64_t bkt;
} *ht_iter_t;

#define HT_SIGNATURE "HASHTBL1"
#pragma pack()

#define UDATA(_o_) ODS_PTR(struct ht_udata *, _o_)
/* Hash Table */
#define HTBL(_o_) ODS_PTR(ht_tbl_t, _o_)
/* Hash Bucket */
#define HBKT(_o_) ODS_PTR(ht_bkt_t, _o_)
/* Bucket Entry */
#define HENT(_o_) ODS_PTR(ht_entry_t, _o_)
/* Hash Key */
#define HKEY(_o_) ODS_PTR(ods_key_value_t, _o_)

#endif
