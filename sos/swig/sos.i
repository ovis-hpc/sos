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
%module sos
%include "cpointer.i"
%include "cstring.i"
%{
#include <stdio.h>
#include <sys/queue.h>
#include <sos/sos.h>

const char *value_as_str(sos_value_t value)
{
	static char buff[1024];
	if (!value)
		return "";
	return sos_value_to_str(value, buff, sizeof(buff));
}

void container_info(sos_t c)
{
	sos_container_info(c, stdout);
}

int pos_from_str(sos_pos_t pos, const char *str)
{
	const char *src = str;
	int i;
	for (i = 0; i < sizeof(pos->data); i++) {
		int rc = sscanf(src, "%02hhX", &pos->data[i]);
		if (rc != 1)
			return EINVAL;
		src += 2;
	}
	return 0;
}

const char *pos_to_str(sos_pos_t pos)
{
	static char str[258];
	char *dst = str;
	int i;
	for (i = 0; i < sizeof(pos->data); i++) {
		sprintf(dst, "%02hhX", pos->data[i]);
		dst += 2;
	}
	return str;
}

%}

const char *value_as_str(sos_value_t value);
const char *pos_to_str(sos_pos_t pos);
int pos_from_str(sos_pos_t pos, const char *str);

/* Filters */
sos_filter_t sos_filter_new(sos_iter_t iter);
void sos_filter_free(sos_filter_t f);
int sos_filter_cond_add(sos_filter_t f,	sos_attr_t attr, enum sos_cond_e cond_e, sos_value_t value);
sos_filter_cond_t sos_filter_eval(sos_obj_t obj, sos_filter_t filt);
sos_obj_t sos_filter_begin(sos_filter_t filt);
sos_obj_t sos_filter_next(sos_filter_t filt);
sos_obj_t sos_filter_prev(sos_filter_t filt);
sos_obj_t sos_filter_end(sos_filter_t filt);
int sos_filter_pos(sos_filter_t filt, sos_pos_t pos);
int sos_filter_set(sos_filter_t filt, const sos_pos_t pos);
sos_obj_t sos_filter_obj(sos_filter_t filt);
int sos_filter_flags_set(sos_filter_t f, sos_iter_flags_t flags);

void container_info(sos_t sos);

typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

typedef struct sos_container_s *sos_t;
typedef struct sos_attr_s *sos_attr_t;
typedef struct sos_schema_s *sos_schema_t;
typedef struct sos_obj_s *sos_obj_t;

#define SOS_ATTR_NAME_LEN	64
#define SOS_SCHEMA_NAME_LEN	64

enum sos_cond_e {
	SOS_COND_LT,
	SOS_COND_LE,
	SOS_COND_EQ,
	SOS_COND_GE,
	SOS_COND_GT,
	SOS_COND_NE,
};

typedef enum sos_type_e {
	/** All types up to the arrays are fixed size */
	SOS_TYPE_INT32 = 0,
	SOS_TYPE_INT64,
	SOS_TYPE_UINT32,
	SOS_TYPE_UINT64,
	SOS_TYPE_FLOAT,
	SOS_TYPE_DOUBLE,
	SOS_TYPE_LONG_DOUBLE,
	SOS_TYPE_TIMESTAMP,
	SOS_TYPE_OBJ,
	/** Arrays are variable sized */
	SOS_TYPE_BYTE_ARRAY,
	SOS_TYPE_ARRAY = SOS_TYPE_BYTE_ARRAY,
	SOS_TYPE_INT32_ARRAY,
	SOS_TYPE_INT64_ARRAY,
	SOS_TYPE_UINT32_ARRAY,
	SOS_TYPE_UINT64_ARRAY,
	SOS_TYPE_FLOAT_ARRAY,
	SOS_TYPE_DOUBLE_ARRAY,
	SOS_TYPE_LONG_DOUBLE_ARRAY,
	SOS_TYPE_OBJ_ARRAY,
	SOS_TYPE_LAST = SOS_TYPE_OBJ_ARRAY
} sos_type_t;

#pragma pack(1)
union sos_array_element_u {
	unsigned char byte_[0];
	uint16_t uint16_[0];
	uint32_t uint32_[0];
	uint64_t uint64_[0];
	int16_t int16_[0];
	int32_t int32_[0];
	int64_t int64_[0];
	float float_[0];
	double double_[0];
	long double long_double_[0];
};

union sos_timestamp_u {
	uint64_t time;
	struct sos_timestamp_s {
		uint32_t nsecs;	/* NB: presumes LE byte order for comparison order */
		uint32_t secs;
	} fine;
};

struct sos_array_s {
	uint32_t count;
	union sos_array_element_u data;
};

union sos_primary_u {
	unsigned char byte_;
	uint16_t uint16_;
	uint32_t uint32_;
	uint64_t uint64_;
	int16_t int16_;
	int32_t int32_;
	int64_t int64_;
	float float_;
	double double_;
	long double long_double_;
	union sos_timestamp_u timestamp_;
	ods_ref_t ref_;
};

typedef union sos_value_data_u {
	union sos_primary_u prim;
	struct sos_array_s array;
} *sos_value_data_t;

typedef struct sos_value_s {
	sos_obj_t obj;
	sos_attr_t attr;
	union sos_value_data_u data_;
	sos_value_data_t data;
} *sos_value_t;

#pragma pack()
typedef enum sos_perm_e {
	SOS_PERM_RO = 0,
	SOS_PERM_RW,
} sos_perm_t;
typedef enum sos_commit_e {
	SOS_COMMIT_ASYNC,
	SOS_COMMIT_SYNC
} sos_commit_t;

typedef struct ods_obj_s *ods_obj_t;
typedef ods_obj_t ods_key_t;
typedef ods_key_t sos_key_t;

void sos_schema_print(sos_schema_t schema, FILE *fp);
sos_schema_t sos_schema_new(const char *name);
int sos_schema_add(sos_t sos, sos_schema_t schema);
sos_schema_t sos_schema_by_name(sos_t sos, const char *name);
sos_schema_t sos_schema_by_id(sos_t sos, uint32_t id);
int sos_schema_delete(sos_t sos, const char *name);
sos_schema_t sos_schema_get(sos_schema_t schema);
void sos_schema_put(sos_schema_t schema);;
const char *sos_schema_name(sos_schema_t schema);
int sos_schema_attr_count(sos_schema_t schema);
int sos_schema_attr_add(sos_schema_t schema, const char *name, sos_type_t type);
int sos_schema_index_add(sos_schema_t schema, const char *name);
int sos_schema_index_cfg(sos_schema_t schema, const char *name,
			 const char *idx_type, const char *key_type, ...);
sos_attr_t sos_schema_attr_by_name(sos_schema_t schema, const char *name);
sos_attr_t sos_schema_attr_by_id(sos_schema_t schema, int attr_id);
sos_schema_t sos_schema_first(sos_t sos);
sos_schema_t sos_schema_next(sos_schema_t schema);
sos_attr_t sos_schema_attr_first(sos_schema_t schema);
sos_attr_t sos_schema_attr_last(sos_schema_t schema);
sos_attr_t sos_schema_attr_next(sos_attr_t attr);
sos_attr_t sos_schema_attr_prev(sos_attr_t attr);

int sos_attr_id(sos_attr_t attr);
const char *sos_attr_name(sos_attr_t attr);
sos_type_t sos_attr_type(sos_attr_t attr);
int sos_attr_index(sos_attr_t attr);
sos_schema_t sos_attr_schema(sos_attr_t attr);
size_t sos_attr_size(sos_attr_t attr);

int sos_obj_attr_from_str(sos_obj_t sos_obj, sos_attr_t attr, const char *attr_value, char **endptr);
int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
				  const char *attr_name, const char *attr_value, char **endptr);
int sos_container_new(const char *path, int o_mode);
sos_t sos_container_open(const char *path, sos_perm_t o_perm);
int sos_container_delete(sos_t c);


void sos_container_close(sos_t c, sos_commit_t flags);
int sos_container_commit(sos_t c, sos_commit_t flags);
void sos_container_info(sos_t sos, FILE* fp);
sos_t sos_container_get(sos_t sos);
void sos_container_put(sos_t sos);

#define SOS_OBJ_BE	1
#define SOS_OBJ_LE	2

sos_schema_t sos_obj_schema(sos_obj_t obj);
sos_obj_t sos_obj_new(sos_schema_t schema);
void sos_obj_delete(sos_obj_t obj);
sos_obj_t sos_obj_get(sos_obj_t obj);
void sos_obj_put(sos_obj_t obj);
int sos_obj_index(sos_obj_t obj);
int sos_obj_remove(sos_obj_t obj);
sos_obj_t sos_obj_from_value(sos_t sos, sos_value_t ref_val);
sos_obj_t sos_obj_from_ref(sos_t sos, sos_ref_t ref);
sos_value_t sos_value_by_name(sos_value_t value, sos_schema_t schema, sos_obj_t obj,
			      const char *name, int *attr_id);
sos_value_t sos_value_by_id(sos_value_t value, sos_obj_t obj, int attr_id);
sos_value_t sos_value_new();
sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr);
void sos_value_free(sos_value_t);
sos_value_t sos_value_init(sos_value_t value, sos_obj_t obj, sos_attr_t attr);
int sos_value_cmp(sos_value_t a, sos_value_t b);
size_t sos_value_size(sos_value_t value);
const char *sos_obj_attr_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len);
int sos_obj_attr_from_str(sos_obj_t obj, sos_attr_t attr, const char *str, char **endptr);
void sos_value_put(sos_value_t value);
const char *sos_value_to_str(sos_value_t value, char *str, size_t len);
int sos_value_from_str(sos_value_t value, const char* str, char **endptr);

size_t sos_key_set(sos_key_t key, void *value, size_t sz);
int sos_key_from_str(sos_attr_t attr, sos_key_t key, const char *str);
%newobject sos_key_to_str;
const char *sos_key_to_str(sos_attr_t attr, sos_key_t key);
int sos_key_cmp(sos_attr_t attr, sos_key_t a, sos_key_t b);
size_t sos_attr_key_size(sos_attr_t attr);
size_t sos_key_size(sos_key_t key);
size_t sos_key_len(sos_key_t key);
unsigned char *sos_key_value(sos_key_t key);
void *sos_value_as_key(sos_value_t value);
void sos_key_put(sos_key_t key);

typedef enum sos_iter_flags_e {
	SOS_ITER_F_ALL = ODS_ITER_F_ALL,
	/** The iterator will skip duplicate keys in the index */
	SOS_ITER_F_UNIQUE = ODS_ITER_F_UNIQUE,
	SOS_ITER_F_MASK = ODS_ITER_F_MASK
} sos_iter_flags_t;
typedef struct sos_iter_s *sos_iter_t;
struct sos_pos {
	char data[16];
};
typedef struct sos_pos *sos_pos_t;
int sos_iter_pos(sos_iter_t iter, sos_pos_t pos);
int sos_iter_set(sos_iter_t iter, const sos_pos_t pos);
sos_iter_t sos_iter_new(sos_attr_t attr);
int sos_iter_flags_set(sos_iter_t iter, sos_iter_flags_t flags);
sos_iter_flags_t sos_iter_flags_get(sos_iter_t iter);
uint64_t sos_iter_card(sos_iter_t iter);
uint64_t sos_iter_dups(sos_iter_t iter);
void sos_iter_free(sos_iter_t iter);
const char *sos_iter_name(sos_iter_t iter);
sos_attr_t sos_iter_attr(sos_iter_t iter);
int sos_iter_key_cmp(sos_iter_t iter, sos_key_t other);
int sos_iter_find(sos_iter_t iter, sos_key_t key);
int sos_iter_inf(sos_iter_t i, sos_key_t key);
int sos_iter_sup(sos_iter_t i, sos_key_t key);
int sos_iter_next(sos_iter_t iter);
int sos_iter_prev(sos_iter_t i);
int sos_iter_begin(sos_iter_t i);
int sos_iter_end(sos_iter_t i);
sos_key_t sos_iter_key(sos_iter_t iter);
sos_obj_t sos_iter_obj(sos_iter_t iter);
int sos_iter_obj_remove(sos_iter_t iter);

%pythoncode %{
%}
