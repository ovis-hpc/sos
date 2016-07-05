/*
 * Copyright (c) 2016 Open Grid Computing, Inc. All rights reserved.
 *
 * Confidential and Proprietary
 */

/*
 * Author: Tom Tucker tom at ogc dot us
 */

#ifndef _H2BXT_H_
#define _H2BXT_H_

#include <ods/ods_idx.h>
#include <ods/ods.h>
#include <ods/rbt.h>
#include "ods_idx_priv.h"
#include "fnv_hash.h"
#include "bxt.h"

#pragma pack(4)

typedef struct h2bxt_udata {
	struct ods_idx_meta_data idx_udata;
	uint32_t order;		/* The order of each BXT node */
	ods_atomic_t lock;	/* Cross-memory spin lock */
	uint32_t hash_seed;	/* The hash function seed */
	uint32_t table_size;	/* The depth of the tree root hash table */
} *h2bxt_udata_t;

/*
 * In memory object that refers to a H2BXT
 */
typedef struct h2bxt_s {
	ods_idx_t ods_idx;
	ods_obj_t udata_obj;
	struct ods_spin_s lock;	/* H2BXT global lock */
	h2bxt_udata_t udata;
	fnv_hash_fn_t hash_fn;
	ods_idx_t *idx_table;
} *h2bxt_t;

typedef struct h2bxt_iter {
	struct ods_iter iter;
	uint64_t hash;
	enum {
		H2BXT_ITER_FWD,
		H2BXT_ITER_REV
	} dir;
	struct rbt next_tree;
	ods_iter_t iter_table[0];
} *h2bxt_iter_t;

#define H2BXT_SIGNATURE "H2BXTREE"
#pragma pack()

#define H2UDATA(_o_) ODS_PTR(struct h2bxt_udata *, _o_)

#endif
