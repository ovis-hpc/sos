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
#include "ods_priv.h"
#include "ods_idx_priv.h"
#include "fnv_hash.h"
#include "mq.h"

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
#define H2BXT_STATE_RUNNING	1
#define H2BXT_STATE_STOPPED	2
#define H2BXT_DUTY_CYCLE	500000 /* micro-seconds */

typedef struct h2bxt_idx_s {
	pthread_t thread;
	ods_idx_t idx;
	mq_t mq;
	int state;
} *h2bxt_idx_t;
typedef struct h2bxt_s {
	ods_idx_t ods_idx;
	ods_obj_t udata_obj;
	h2bxt_udata_t udata;
	fnv_hash_fn_t hash_fn;
	ods_idx_rt_opts_t rt_opts;
	struct h2bxt_idx_s *idx_table;
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

typedef struct h2bxt_pos_s {
	uint32_t dir;
	uint32_t bxt_iter_idx;	/* index into h2bxt_iter.iter_table */
	struct ods_pos_s bxt_iter_pos; /* pos in underlying bxt index */
} *h2bxt_pos_t;

#define WQE_VISIT	1
#define VISIT_KEY_SIZE	256
typedef struct visit_msg_s {
	struct mq_msg_s hdr;
	h2bxt_idx_t idx;
	struct key_storage {
		uint16_t len;
		unsigned char value[VISIT_KEY_SIZE];
	} key_;
	struct ods_obj_s key_obj_;
	ods_key_t key;
	ods_visit_cb_fn_t cb_fn;
	void *ctxt;
} *visit_msg_t;
#define H2BXT_MAX_MSG_SIZE	sizeof(struct visit_msg_s)
#define H2BXT_QUEUE_DEPTH	256

#define H2BXT_DEFAULT_ORDER		5
#define H2BXT_DEFAULT_TABLE_SIZE	5

#define H2BXT_EXTEND_SIZE 65536
#define H2BXT_SIGNATURE "H2BXTREE"
#pragma pack()

#define H2UDATA(_o_) ODS_PTR(struct h2bxt_udata *, _o_)
#define POS(_o_) ODS_PTR(h2bxt_pos_t, _o_)

#endif
