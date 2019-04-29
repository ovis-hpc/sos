/*
 * Copyright (c) 2014-2019 Open Grid Computing, Inc. All rights reserved.
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

/*
 * Author: Tom Tucker tom at ogc dot us
 */

#ifndef _BXT_H_
#define _BXT_H_

#include <ods/ods_idx.h>
#include <ods/ods.h>
#include "ods_idx_priv.h"

#pragma pack(4)

/**
 * B+ Tree Node Entry
 *
 * Describes a key and the object to which it refers. The key is an
 * obj_ref_t which refers to an arbitrarily sized ODS object that
 * contains an opaque key. The key comparator is used to compare two
 * keys.
 */
typedef struct bxn_entry {
	union {
		struct {
			ods_ref_t head_ref;
			ods_ref_t tail_ref;
		} leaf;
		struct node {
			ods_ref_t key_ref;
			ods_ref_t node_ref;
		} node;
	} u;
} *bxn_entry_t;

typedef struct bxn_record {
	ods_ref_t key_ref;	/* The key */
	ods_idx_data_t value;	/* The value */
	ods_ref_t next_ref;	/* The next record */
	ods_ref_t prev_ref;	/* The previous record */
} *bxn_record_t;

typedef struct bxt_node {
	ods_ref_t parent;	/* NULL if root */
	uint32_t count:16;
	uint32_t is_leaf:16;
	struct bxn_entry entries[];
} *bxt_node_t;

typedef struct bxt_udata {
	struct ods_idx_meta_data idx_udata;
	uint32_t order;		/* The order or each internal node */
	ods_ref_t root_ref;	/* The root of the tree */
	ods_atomic_t depth;	/* The current tree depth */
	ods_atomic_t card;	/* Cardinality */
	ods_atomic_t dups;	/* Duplicate keys */
} *bxt_udata_t;

/* Structure to hang on to cached node allocations */
struct bxt_obj_el {
	ods_obj_t obj;
	ods_ref_t ref;
	LIST_ENTRY(bxt_obj_el) entry;
};

/*
 * In memory object that refers to a B+ Tree
 */
typedef struct bxt_s {
	ods_t ods;		/* The ods that contains the tree */
	ods_obj_t udata_obj;
	bxt_udata_t udata;
	ods_idx_compare_fn_t comparator;
	ods_idx_rt_opts_t rt_opts;	/* Run-time flags */
	/*
	 * The node_q keeps a Q of nodes for allocation.
	 */
	ods_atomic_t node_q_depth;
	LIST_HEAD(node_node_q_head, bxt_obj_el) node_q;
	LIST_HEAD(node_el_q_head, bxt_obj_el) el_q;
} *bxt_t;

typedef struct bxt_pos_s {
	ods_ref_t rec_ref;
	uint32_t ent;
} *bxt_pos_t;

typedef struct bxt_iter_s {
	struct ods_iter iter;
	ods_obj_t rec;
	ods_obj_t node;
	uint32_t ent;
} *bxt_iter_t;

#define BXT_EXTEND_SIZE	(1024 * 1024)
#define BXT_SIGNATURE "BXTREE01"
#pragma pack()

#define UDATA(_o_) ODS_PTR(struct bxt_udata *, _o_)
/* Node entry */
#define N_ENT(_o_,_i_) ODS_PTR(bxt_node_t, _o_)->entries[_i_].u.node
/* Leaf entry */
#define L_ENT(_o_,_i_) ODS_PTR(bxt_node_t, _o_)->entries[_i_].u.leaf
#define NODE(_o_) ODS_PTR(bxt_node_t, _o_)
#define REC(_o_) ODS_PTR(bxn_record_t, _o_)
#define POS(_o_) ODS_PTR(bxt_pos_t, _o_)
#endif
