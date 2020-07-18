/*
 * Copyright (c) 2018 Open Grid Computing, Inc. All rights reserved.
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

#ifndef _H2HTBL_H_
#define _H2HTBL_H_

#include <ods/ods_idx.h>
#include <ods/ods.h>
#include <ods/ods_rbt.h>
#include "ods_priv.h"
#include "ods_idx_priv.h"
#include "fnv_hash.h"
#include "mq.h"

#pragma pack(4)

typedef struct h2htbl_udata {
	struct ods_idx_meta_data idx_udata;
	ods_atomic_t lock;	/* Cross-memory spin lock */
	uint32_t hash_seed;	/* The hash function seed */
	uint32_t table_size;	/* The depth of the tree root hash table */
} *h2htbl_udata_t;

/*
 * In memory object that refers to a H2HTBL
 */
#define H2HTBL_STATE_RUNNING	1
#define H2HTBL_STATE_STOPPED	2
#define H2HTBL_DUTY_CYCLE	500000 /* micro-seconds */

typedef struct h2htbl_idx_s {
	pthread_t thread;
	ods_idx_t idx;
	mq_t mq;
	int state;
} *h2htbl_idx_t;
typedef struct h2htbl_s {
	ods_idx_t ods_idx;
	ods_obj_t udata_obj;
	h2htbl_udata_t udata;
	fnv_hash_fn_t hash_fn;
	ods_idx_rt_opts_t rt_opts;
	struct h2htbl_idx_s *idx_table;
} *h2htbl_t;

typedef struct h2htbl_iter {
	struct ods_iter iter;
	uint64_t hash;
	enum {
		H2HTBL_ITER_FWD,
		H2HTBL_ITER_REV
	} dir;
	struct ods_rbt next_tree;
	ods_iter_t iter_table[0];
} *h2htbl_iter_t;

typedef struct h2htbl_pos_s {
	uint32_t dir;
	uint32_t htbl_iter_idx;	/* index into h2htbl_iter.iter_table */
	struct ods_pos_s htbl_iter_pos; /* pos in underlying HTBL index */
} *h2htbl_pos_t;

#define WQE_VISIT	1
#define VISIT_KEY_SIZE	256
typedef struct visit_msg_s {
	struct mq_msg_s hdr;
	h2htbl_idx_t idx;
	struct key_storage {
		uint16_t len;
		unsigned char value[VISIT_KEY_SIZE];
	} key_;
	struct ods_obj_s key_obj_;
	ods_key_t key;
	ods_visit_cb_fn_t cb_fn;
	void *ctxt;
} *visit_msg_t;
#define H2HTBL_MAX_MSG_SIZE	sizeof(struct visit_msg_s)
#define H2HTBL_QUEUE_DEPTH	256

#define H2HTBL_DEFAULT_ORDER		5
#define H2HTBL_DEFAULT_TABLE_SIZE	5

#define H2HTBL_EXTEND_SIZE 65536
#define H2HTBL_SIGNATURE "H2HTBL"
#pragma pack()

#define H2UDATA(_o_) ODS_PTR(struct h2htbl_udata *, _o_)
#define POS(_o_) ODS_PTR(h2htbl_pos_t, _o_)

#endif
