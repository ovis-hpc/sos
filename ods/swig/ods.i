/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
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
/* File: ods.i */
%module ods
%include "cpointer.i"
%{
#include <ods/ods.h>
#include <ods/ods_idx.h>
%}


typedef struct ods_s *ods_t;
typedef uint64_t ods_ref_t;
typedef struct ods_map_s *ods_map_t;
typedef struct ods_obj_s *ods_obj_t;

/*
 * Python interfaces for ODS
 */
ods_t ods_open(const char *path, int o_flags);
ods_obj_t ods_get_user_data(ods_t ods);
#define ODS_COMMIT_ASYNC	0
#define ODS_COMMIT_SYNC		1
void ods_commit(ods_t ods, int flags);
void ods_close(ods_t ods, int flags);
ods_obj_t ods_obj_alloc(ods_t ods, size_t sz);
// ods_obj_t ods_obj_malloc(ods_t ods, size_t sz);
void ods_obj_delete(ods_obj_t obj);
void ods_ref_delete(ods_t ods, ods_ref_t ref);
int ods_extend(ods_t ods, size_t sz);
size_t ods_size(ods_t ods);
void ods_dump(ods_t ods, FILE *fp);
ods_obj_t ods_obj_get(ods_obj_t obj);
void _ods_obj_put(ods_obj_t obj);
ods_obj_t ods_ref_as_obj(ods_t ods, ods_ref_t ref);
ods_ref_t ods_obj_ref(ods_obj_t obj);

typedef struct ods_spin_s *ods_spin_t;
ods_spin_t ods_spin_get(ods_atomic_t *lock_p);
int ods_spin_lock(ods_spin_t s, int timeout);
void ods_spin_unlock(ods_spin_t s);
void ods_spin_put(ods_spin_t spin);
void ods_info(ods_t ods, FILE *fp);

/*
 * Python interfaces for ODS Indices
 */
typedef struct ods_idx *ods_idx_t;
typedef struct ods_iter *ods_iter_t;

#define ODS_IDX_SIGNATURE	"ODSIDX00"
#define ODS_IDX_BPTREE		"BPTREE"
#define ODS_IDX_RADIXTREE	"RADIXTREE"
#define ODS_IDX_RBTREE		"RBTREE"

int ods_idx_create(const char *path, int mode,
		   const char *type, const char *key,
		   ...);
ods_idx_t ods_idx_open(const char *path, int o_mode);
void ods_idx_close(ods_idx_t idx, int flags);
ods_t ods_idx_ods(ods_idx_t idx);
void ods_idx_commit(ods_idx_t idx, int flags);

#pragma pack(2)
typedef struct ods_key_value_s {
	uint16_t len;
	unsigned char value[0];
} *ods_key_value_t;
#pragma pack()
typedef ods_obj_t ods_key_t;

ods_key_t ods_key_alloc(ods_idx_t idx, size_t sz);
// ods_key_t ods_key_malloc(ods_idx_t idx, size_t sz);
size_t ods_key_set(ods_key_t key, void *value, size_t sz);
static inline ods_key_value_t ods_key_value(ods_key_t key) { return key->as.ptr; }
int ods_key_from_str(ods_idx_t idx, ods_key_t key, const char *str);
const char *ods_key_to_str(ods_idx_t idx, ods_key_t key);
int ods_key_cmp(ods_idx_t idx, ods_key_t a, ods_key_t b);
size_t ods_idx_key_size(ods_idx_t idx);
size_t ods_key_size(ods_key_t key);
size_t ods_key_len(ods_key_t key);
int ods_idx_insert(ods_idx_t idx, ods_key_t key, ods_ref_t obj);
int ods_idx_update(ods_idx_t idx, ods_key_t key, ods_ref_t obj);
int ods_idx_delete(ods_idx_t idx, ods_key_t key, ods_ref_t *ref);
int ods_idx_find(ods_idx_t idx, ods_key_t key, ods_ref_t *ref);
int ods_idx_find_lub(ods_idx_t idx, ods_key_t key, ods_ref_t *ref);
int ods_idx_find_glb(ods_idx_t idx, ods_key_t key, ods_ref_t *ref);
ods_iter_t ods_iter_new(ods_idx_t idx);
void ods_iter_delete(ods_iter_t iter);
int ods_iter_find(ods_iter_t iter, ods_key_t key);
int ods_iter_find_lub(ods_iter_t iter, ods_key_t key);
int ods_iter_find_glb(ods_iter_t iter, ods_key_t key);
int ods_iter_begin(ods_iter_t iter);
int ods_iter_end(ods_iter_t iter);
int ods_iter_next(ods_iter_t iter);
int ods_iter_prev(ods_iter_t iter);
ods_key_t ods_iter_key(ods_iter_t iter);
ods_ref_t ods_iter_ref(ods_iter_t iter);
void ods_idx_print(ods_idx_t idx, FILE* fp);
void ods_idx_info(ods_idx_t idx, FILE* fp);
