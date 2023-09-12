/*
 * Copyright (c) 2013 Open Grid Computing, Inc. All rights reserved.
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

#ifndef _ODS_IDX_PRIV_H_
#define _ODS_IDX_PRIV_H_
#include <stdarg.h>
#include <stdlib.h>
#include <ods/ods_rbt.h>
#include <ods/ods.h>
#include "ods_log.h"

struct ods_idx_provider {
	const char *(*get_type)(void);
	int (*init)(ods_t ods, const char *idx_type, const char *key_type, const char *args);
	int (*open)(ods_idx_t idx);
	void (*close)(ods_idx_t idx);
	int (*lock)(ods_idx_t idx, struct timespec *wait);
	void (*unlock)(ods_idx_t idx);
	int (*rt_opts_set)(ods_idx_t idx, ods_idx_rt_opts_t opt, va_list ap);
	ods_idx_rt_opts_t (*rt_opts_get)(ods_idx_t idx);
	void (*commit)(ods_idx_t idx);
	int (*insert)(ods_idx_t idx, ods_key_t uk, ods_idx_data_t data);
	int (*insert_no_lock)(ods_idx_t idx, ods_key_t uk, ods_idx_data_t data);
	int (*visit)(ods_idx_t idx, ods_key_t key, ods_visit_cb_fn_t cb_fn, void *ctxt);
	int (*update)(ods_idx_t idx, ods_key_t uk, ods_idx_data_t data);
	int (*delete)(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data);
	int (*find)(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data);
	int (*find_lub)(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data);
	int (*find_glb)(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data);
	int (*stat)(ods_idx_t idx, ods_idx_stat_t sb);
	int (*max)(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data);
	int (*min)(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *data);
	ods_iter_t (*iter_new)(ods_idx_t idx);
	void (*iter_delete)(ods_iter_t i);
	int (*iter_find)(ods_iter_t iter, ods_key_t key);
	int (*iter_find_lub)(ods_iter_t iter, ods_key_t key);
	int (*iter_find_glb)(ods_iter_t iter, ods_key_t key);
	int (*iter_find_first)(ods_iter_t iter, ods_key_t key);
	int (*iter_find_last)(ods_iter_t iter, ods_key_t key);
	int (*iter_begin)(ods_iter_t iter);
	int (*iter_end)(ods_iter_t iter);
	int (*iter_next)(ods_iter_t iter);
	int (*iter_prev)(ods_iter_t iter);
	int (*iter_pos_set)(ods_iter_t iter, const ods_pos_t pos);
	int (*iter_pos_get)(ods_iter_t iter, ods_pos_t pos);
	int (*iter_pos_put)(ods_iter_t iter, ods_pos_t pos);
	int (*iter_entry_delete)(ods_iter_t iter, ods_idx_data_t *data);
	ods_key_t (*iter_key)(ods_iter_t iter);
	ods_idx_data_t (*iter_data)(ods_iter_t iter);
	void (*print_idx)(ods_idx_t idx, FILE *fp);
	void (*print_info)(ods_idx_t idx, FILE *fp);
	int (*verify_idx)(ods_idx_t idx, FILE *fp);
};

struct ods_idx_comparator {
	/** Return the name of the comparator */
	const char *(*get_type)(void);
	/** Return a description of how the key works  */
	const char *(*get_doc)(void);
	/** Return a string representation of the key value */
	const char *(*to_str)(ods_key_t, char *buf, size_t len);
	/** Set the key value from a string */
	int (*from_str)(ods_key_t, const char *);
	/* Return the size of the key data or -1 if variable */
	size_t (*size)(void);
	/* Return the size of the key if formatted as a string */
	size_t (*str_size)(ods_key_t);
	/** Compare two keys */
	ods_idx_compare_fn_t compare_fn;
};

#pragma pack(1)
#define ODS_IDX_NAME_MAX	31
struct ods_idx_meta_data {
	char signature[ODS_IDX_NAME_MAX+1];
	char type_name[ODS_IDX_NAME_MAX+1];
	char key_name[ODS_IDX_NAME_MAX+1];
};
#pragma pack()

struct ods_idx_class {
	struct ods_idx_provider *prv;
	struct ods_idx_comparator *cmp;
	struct ods_rbn rb_node;
};

struct ods_idx {
	/** open and iterator references */
	ods_atomic_t ref_count;
	/** The index and key handler functions */
	struct ods_idx_class *idx_class;
	/** The ODS object store for the index */
	ods_t ods;
	/** The index options */
	int idx_opts;
	/** The permissions the index was opened with */
	ods_perm_t o_perm;
	/** The optional Key-Value-Array for bulk insert */
	struct ods_bulk_ins_opt_s kvq_opt;
	/* Number threads, one kvq per thread */
	int kvq_count;
	/* Array of KVQ, one for each thread */
	ods_kvq_t *kvq;
	/** Place for the index to store its private data */
	void *priv;
};

struct ods_iter {
	ods_iter_flags_t flags;
	struct ods_idx *idx;
};

static inline int ods_idx_data_null(ods_idx_data_t *data)
{
	return ((0 == data->uint64_[0]) && (0 == data->uint64_[1]));
}
static inline int ods_idx_data_equal(ods_idx_data_t *a, ods_idx_data_t *b)
{
	return 0 == memcmp(a, b, sizeof(*a));
}
#endif
