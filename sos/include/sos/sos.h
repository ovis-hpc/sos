 /*
 * Copyright (c) 2013-2015 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2013-2015 Sandia Corporation. All rights reserved.
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

#ifndef __SOS_H
#define __SOS_H

#include <stdint.h>
#include <stddef.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>

/** \defgroup schema SOS Schema
 * @{
 */

/** \defgroup schema_types Schema Types
 * @{
 */
typedef struct sos_container_s *sos_t;
typedef struct sos_attr_s *sos_attr_t;
typedef struct sos_index_s *sos_index_t;
typedef struct sos_schema_s *sos_schema_t;
typedef struct sos_obj_s *sos_obj_t;

#define SOS_CONTAINER_NAME_LEN  64
#define SOS_CONFIG_NAME_LEN  64
#define SOS_SCHEMA_NAME_LEN	64
#define SOS_ATTR_NAME_LEN	64

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
	uint8_t byte_[0];
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

typedef struct sos_array_s {
	uint32_t count;
	union sos_array_element_u data;
} *sos_array_t;

union sos_timestamp_u {
	uint64_t time;
	struct sos_timestamp_s {
		uint32_t usecs;	/* NB: presumes LE byte order for comparison order */
		uint32_t secs;
	} fine;
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

enum sos_cond_e {
	SOS_COND_LT,
	SOS_COND_LE,
	SOS_COND_EQ,
	SOS_COND_GE,
	SOS_COND_GT,
	SOS_COND_NE,
};

#pragma pack()
/** @} */

/** \defgroup schema_funcs Schema Functions
 * @{
 */
sos_schema_t sos_schema_new(const char *name);
sos_schema_t sos_schema_dup(sos_schema_t schema);
size_t sos_schema_count(sos_t sos);
int sos_schema_add(sos_t sos, sos_schema_t schema);

typedef struct sos_schema_template_attr {
	const char *name;
	sos_type_t type;
	int indexed;
} *sos_schema_template_attr_t;

typedef struct sos_schema_template {
	const char *name;
	struct sos_schema_template_attr attrs[];
} *sos_schema_template_t;

sos_schema_t sos_schema_from_template(sos_schema_template_t pt);
sos_schema_t sos_schema_by_name(sos_t sos, const char *name);
sos_schema_t sos_schema_by_id(sos_t sos, uint32_t id);
void sos_schema_print(sos_schema_t schema, FILE *fp);
int sos_schema_delete(sos_t sos, const char *name);
sos_schema_t sos_schema_first(sos_t sos);
sos_schema_t sos_schema_next(sos_schema_t schema);
sos_schema_t sos_schema_get(sos_schema_t schema);
void sos_schema_put(sos_schema_t schema);
const char *sos_schema_name(sos_schema_t schema);
int sos_schema_id(sos_schema_t schema);
int sos_schema_attr_count(sos_schema_t schema);
int sos_schema_attr_add(sos_schema_t schema, const char *name, sos_type_t type);
int sos_schema_index_add(sos_schema_t schema, const char *name);
int sos_schema_index_modify(sos_schema_t schema, const char *name,
			    const char *idx_type, const char *key_type, ...);
sos_attr_t sos_schema_attr_by_name(sos_schema_t schema, const char *name);
sos_attr_t sos_schema_attr_by_id(sos_schema_t schema, int attr_id);
sos_attr_t sos_schema_attr_first(sos_schema_t schema);
sos_attr_t sos_schema_attr_last(sos_schema_t schema);
sos_attr_t sos_schema_attr_next(sos_attr_t attr);
sos_attr_t sos_schema_attr_prev(sos_attr_t attr);

int sos_attr_id(sos_attr_t attr);
const char *sos_attr_name(sos_attr_t attr);
sos_type_t sos_attr_type(sos_attr_t attr);
int sos_attr_index(sos_attr_t attr);
size_t sos_attr_size(sos_attr_t attr);
sos_schema_t sos_attr_schema(sos_attr_t attr);

int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
				  const char *attr_name, const char *attr_value,
				  char **endptr);

/** @} */
/** @} */

int sos_container_new(const char *path, int o_mode);

typedef enum sos_perm_e {
	SOS_PERM_RO = 0,
	SOS_PERM_RW,
} sos_perm_t;

sos_t sos_container_open(const char *path, sos_perm_t o_perm);

int sos_container_delete(sos_t c);

typedef enum sos_commit_e {
	/** Returns immediately, the sync to storage will be completed
	 *  asynchronously */
	SOS_COMMIT_ASYNC,
	/** Does not return until the sync is complete */
	SOS_COMMIT_SYNC
} sos_commit_t;

int sos_container_extend(sos_t sos, size_t new_size);
int sos_container_stat(sos_t sos, struct stat *sb);
void sos_container_close(sos_t c, sos_commit_t flags);
int sos_container_commit(sos_t c, sos_commit_t flags);
void sos_container_info(sos_t sos, FILE* fp);

#define SOS_CONTAINER_PARTITION_ENABLE		"PARTITION_ENABLE"
#define SOS_CONTAINER_PARTITION_SIZE		"PARTITION_SIZE"
#define SOS_CONTAINER_PARTITION_PERIOD		"PARTITION_PERIOD"
#define SOS_CONTAINER_PARTITION_EXTEND		"PARTITION_EXTEND"

int sos_container_config(const char *path, const char *option, const char *value);
typedef struct sos_config_iter_s *sos_config_iter_t;
sos_config_iter_t sos_config_iter_new(const char *path);
void sos_config_iter_free(sos_config_iter_t iter);
typedef struct sos_config_data_s {
	char name[SOS_CONFIG_NAME_LEN];
	char value[0];
} *sos_config_t;
sos_config_t sos_config_first(sos_config_iter_t iter);
sos_config_t sos_config_next(sos_config_iter_t iter);
void sos_config_print(const char *path, FILE *fp);

#define SOS_PART_NAME_DEFAULT			"00000000"
#define SOS_PART_NAME_LEN			64
/** Partition is being moved up or otherwise not used */
#define SOS_PART_STATE_OFFLINE			0
/** Consulted for queries/iteration */
#define SOS_PART_STATE_ACTIVE			1
/** New objects stored here */
#define SOS_PART_STATE_PRIMARY			2

typedef struct sos_part_iter_s *sos_part_iter_t;
typedef struct sos_part_s *sos_part_t;
int sos_part_new(sos_t sos, const char *name);
sos_part_iter_t sos_part_iter_new(sos_t sos);
void sos_part_iter_free(sos_part_iter_t iter);
sos_part_t sos_part_first(sos_part_iter_t iter);
sos_part_t sos_part_next(sos_part_iter_t iter);
const char *sos_part_name_get(sos_part_t part);
uint32_t sos_part_state_get(sos_part_t part);
void sos_part_primary_set(sos_part_t part);
int sos_part_active_set(sos_part_t part, int active);
int sos_part_remove(sos_part_t part);
void sos_part_put(sos_part_t part);
void sos_part_print(sos_t sos, FILE *fp);

/** \defgroup objects SOS Objects
 * @{
 */
/**
 *
 * An object is a persistent instance of attribute values described by
 * a schema. While a schema is a collection of attributes, an object
 * is a collection of values. Each value in the object is described by
 * an attribute in the schema. The attribute identifies the type of
 * each value in the object.
 *
 * - sos_obj_new()	 Create a new object in the container
 * - sos_obj_delete()    Delete an object from the container
 * - sos_obj_get()	 Take a reference on an object
 * - sos_obj_put()	 Drop a reference on an object
 * - sos_obj_index()	 Add an object to it's indices
 * - sos_obj_remove()	 Remove an object from it's indices
 * - sos_obj_ptr()       Returns a pointer to the object's data
 * - sos_value()	 Return a value given object and attribute handles.
 * - sos_value_by_name() Get the value handle by name
 * - sos_value_by_id()   Get the value handle by id
 * - sos_value_to_str()	 Get the value as a string
 * - sos_value_from_str() Set the value from a string
 * - sos_value_init()	 Initializes a stack variable as a value.
 */

/**
 * Identifies the byte order of the objects
 */
#define SOS_OBJ_BE	1
#define SOS_OBJ_LE	2

sos_obj_t sos_obj_new(sos_schema_t schema);
sos_schema_t sos_obj_schema(sos_obj_t obj);

typedef struct sos_ref_s {
	void *part;
	ods_ref_t ref;
} sos_ref_t;
sos_ref_t sos_obj_ref(sos_obj_t obj);
#define SOS_NULL_REF(_ref_)  ((_ref_).ref == 0)

sos_obj_t sos_obj_from_value(sos_t sos, sos_value_t ref_val);
sos_obj_t sos_obj_from_ref(sos_t sos, sos_ref_t ref);
void sos_obj_delete(sos_obj_t obj);
sos_obj_t sos_obj_get(sos_obj_t obj);
void sos_obj_put(sos_obj_t obj);
int sos_obj_index(sos_obj_t obj);
int sos_obj_remove(sos_obj_t obj);
sos_value_t sos_value_by_name(sos_value_t value, sos_schema_t schema, sos_obj_t obj,
			      const char *name, int *attr_id);
sos_value_t sos_value_by_id(sos_value_t value, sos_obj_t obj, int attr_id);
sos_obj_t sos_array_obj_new(sos_t sos, sos_type_t type, size_t count);
int sos_attr_is_ref(sos_attr_t attr);
int sos_attr_is_array(sos_attr_t attr);
size_t sos_array_count(sos_value_t val);
sos_value_t sos_array_new(sos_value_t val, sos_attr_t attr, sos_obj_t obj, size_t count);
static inline void *sos_array_ptr(sos_value_t val) {
	return val->data->array.data.byte_;
}
sos_value_t sos_value_new();
void sos_value_free(sos_value_t v);
void *sos_obj_ptr(sos_obj_t obj);
sos_value_t sos_value_init(sos_value_t value, sos_obj_t obj, sos_attr_t attr);
#define SOS_VALUE(_name_)				\
	struct sos_value_s  _name_ ## __, *_name_ = &_name_ ## __;

sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr);
void sos_value_put(sos_value_t value);
int sos_value_cmp(sos_value_t a, sos_value_t b);
size_t sos_value_size(sos_value_t value);
size_t sos_value_memset(sos_value_t value, void *buf, size_t buflen);
char *sos_obj_attr_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len);
int sos_obj_attr_from_str(sos_obj_t obj, sos_attr_t attr, const char *str, char **endptr);
const char *sos_value_to_str(sos_value_t value, char *str, size_t len);
int sos_value_from_str(sos_value_t value, const char *str, char **endptr);

/** @} */

/** \defgroup indices SOS Indices
 * @{
 */
typedef struct ods_obj_s *sos_key_t;

int sos_index_new(sos_t sos, const char *name,
		  const char *idx_type, const char *key_type, ...);
sos_index_t sos_index_open(sos_t sos, const char *name);
int sos_index_insert(sos_index_t index, sos_key_t key, sos_obj_t obj);
int sos_index_obj_remove(sos_index_t index, sos_key_t key, sos_obj_t obj);
sos_obj_t sos_index_find(sos_index_t index, sos_key_t key);
sos_obj_t sos_index_find_inf(sos_index_t index, sos_key_t key);
sos_obj_t sos_index_find_sup(sos_index_t index, sos_key_t key);
int sos_index_commit(sos_index_t index, sos_commit_t flags);
int sos_index_close(sos_index_t index, sos_commit_t flags);

/**
 * \brief Define a SOS stack key
 *
 * A key that is up to 256 bytes in length that is allocated on the
 * current stack frame. If your application uses keys that are greater
 * than this length, use the sos_key_new() function or redefine the
 * SOS_STACK_KEY_SIZE macro and recompile your application.
 *
 * Do not use the sos_obj_put() function to release keys of this type,
 * they will be automatically destroyed when the containing function
 * returns.
 *
 * \param _name_	The variable name to use to refer to the key.
 */
#define SOS_STACK_KEY_SIZE 256
#define SOS_KEY(_name_)					\
	struct sos_key_value_s  ## _name_ {		\
		uint16_t len;				\
		unsigned char value[SOS_STACK_KEY_SIZE];\
	} _name_ ## _ ## data;				\
	ODS_OBJ(_name_ ## _ ## key_s, &_name_ ## _ ## data, 256);	\
	sos_key_t _name_ = &_name_ ## _ ## key_s;

/**
 * \brief Create a memory key
 *
 * A key is just a an object with a set of convenience routines to
 * help with getting and setting it's value based on the key type used
 * on an index.
 *
 * A memory key is used to look up objects in the ODS. The storage for
 * these keys comes from memory. See the sos_key_new() function for
 * keys that are stored in the Container.
 *
 * If the size of the key is known to be less than 254 bytes, the
 * SOS_KEY() macro is useful for defining a SOS key that is allocated
 * on the stack and is automatically destroyed when the containing
 * function returns.
 *
 * \param sz	The maximum size in bytes of the key value
 * \retval !0	ods_key_t pointer to the key
 * \retval 0	Insufficient resources
 */
#define sos_key_new(sz) ({		\
	ods_key_t k = ods_key_malloc(sz);	\
	if (k) {				\
		k->alloc_line = __LINE__;	\
		k->alloc_func = __func__;	\
	}					\
	k;					\
})

#define sos_key_put(key) ods_obj_put(key)
size_t sos_key_set(sos_key_t key, void *value, size_t sz);
int sos_attr_key_from_str(sos_attr_t attr, sos_key_t key, const char *str);
const char *sos_attr_key_to_str(sos_attr_t attr, sos_key_t key);
sos_key_t sos_attr_key_new(sos_attr_t attr, size_t len);
int sos_attr_key_from_str(sos_attr_t attr, sos_key_t key, const char *str);
int sos_attr_key_cmp(sos_attr_t attr, sos_key_t a, sos_key_t b);

size_t sos_attr_key_size(sos_attr_t attr);
int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
				  const char *attr_name, const char *attr_value,
				  char **endptr);
char *sos_obj_attr_by_name_to_str(sos_obj_t sos_obj, const char *attr_name,
				  char *str, size_t len);
size_t sos_key_size(sos_key_t key);
size_t sos_key_len(sos_key_t key);
unsigned char *sos_key_value(sos_key_t key);
void *sos_value_as_key(sos_value_t value);

/** @} */

/** \defgroup iter SOS Iterators
 * @{
 */
typedef struct sos_iter_s *sos_iter_t;
struct sos_pos {
	char data[16];
};
typedef struct sos_pos *sos_pos_t;

sos_iter_t sos_index_iter_new(sos_index_t index);
sos_iter_t sos_attr_iter_new(sos_attr_t attr);
void sos_iter_free(sos_iter_t iter);
int sos_iter_key_cmp(sos_iter_t iter, sos_key_t other);
int sos_iter_find(sos_iter_t iter, sos_key_t key);
int sos_iter_inf(sos_iter_t i, sos_key_t key);
int sos_iter_sup(sos_iter_t i, sos_key_t key);

typedef enum sos_iter_flags_e {
	SOS_ITER_F_ALL = ODS_ITER_F_ALL,
	/** The iterator will skip duplicate keys in the index */
	SOS_ITER_F_UNIQUE = ODS_ITER_F_UNIQUE,
	SOS_ITER_F_MASK = ODS_ITER_F_MASK
} sos_iter_flags_t;

int sos_iter_flags_set(sos_iter_t i, sos_iter_flags_t flags);
sos_iter_flags_t sos_iter_flags_get(sos_iter_t i);
uint64_t sos_iter_card(sos_iter_t i);
uint64_t sos_iter_dups(sos_iter_t i);
int sos_iter_pos(sos_iter_t i, sos_pos_t pos);
int sos_iter_set(sos_iter_t i, const sos_pos_t pos);
int sos_iter_next(sos_iter_t iter);
int sos_iter_prev(sos_iter_t i);
int sos_iter_begin(sos_iter_t i);
int sos_iter_end(sos_iter_t i);
sos_key_t sos_iter_key(sos_iter_t iter);
sos_obj_t sos_iter_obj(sos_iter_t iter);
int sos_iter_obj_remove(sos_iter_t iter);

typedef struct sos_filter_cond_s *sos_filter_cond_t;
typedef struct sos_filter_s *sos_filter_t;

sos_filter_t sos_filter_new(sos_iter_t iter);
void sos_filter_free(sos_filter_t f);
int sos_filter_cond_add(sos_filter_t f, sos_attr_t attr, enum sos_cond_e cond_e, sos_value_t value);
sos_filter_cond_t sos_filter_eval(sos_obj_t obj, sos_filter_t filt);
sos_obj_t sos_filter_begin(sos_filter_t filt);
sos_obj_t sos_filter_next(sos_filter_t filt);
sos_obj_t sos_filter_prev(sos_filter_t filt);
sos_obj_t sos_filter_end(sos_filter_t filt);
int sos_filter_pos(sos_filter_t filt, sos_pos_t pos);
int sos_filter_set(sos_filter_t filt, const sos_pos_t pos);
sos_obj_t sos_filter_obj(sos_filter_t filt);
int sos_filter_flags_set(sos_filter_t filt, sos_iter_flags_t flags);

/** @} */

#endif
