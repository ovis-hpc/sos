/*
 * Copyright (c) 2013-2021 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2013-2017 Sandia Corporation. All rights reserved.
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
#include <wchar.h>
#include <uuid/uuid.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include <stdarg.h>

/** \defgroup cont SOS Containers
 * @{
 */

/** \defgroup cont_types Container Types
 * @{
 */
typedef struct sos_container_s *sos_t;
typedef enum sos_perm_e {
	SOS_PERM_RD = ODS_PERM_RD,
	SOS_PERM_WR = ODS_PERM_WR,
	SOS_PERM_RW = ODS_PERM_RW,
	SOS_PERM_CREAT = ODS_PERM_CREAT,
	SOS_BE_MMOS = ODS_BE_MMAP,
	SOS_BE_LSOS = ODS_BE_LSOS,
	SOS_PERM_USER = 512
} sos_perm_t;

#define SOS_POS_KEEP_TIME			"POS_KEEP_TIME"

#define SOS_CONTAINER_NAME_LEN  128
#define SOS_CONFIG_NAME_LEN	128

/**
 * \brief Specifies whether to commit synchronously or asynchronously
 */
typedef enum sos_commit_e {
	/** Returns immediately, the sync to storage will be completed
	 *  asynchronously */
	SOS_COMMIT_ASYNC,
	/** Does not return until the sync is complete */
	SOS_COMMIT_SYNC
} sos_commit_t;

typedef struct sos_config_iter_s *sos_config_iter_t;
typedef struct sos_config_data_s {
	char name[SOS_CONFIG_NAME_LEN];
	char value[0];
} *sos_config_t;

/** @} */
/** \defgroup cont_funcs Container Functions
 * @{
 */
#pragma pack(1)
#define SOS_VERS_MAJOR	ODS_VER_MAJOR
#define SOS_VERS_MINOR	ODS_VER_MINOR
#define SOS_VERS_FIX	ODS_VER_FIX
struct sos_version_s {
	uint8_t major;		/* Binary compatability */
	uint8_t minor;		/* Feature availability */
	uint16_t fix;		/* Defect repair */
	char git_commit_id[44];	/* git commit id */
};
#pragma pack()
int sos_container_file_version(const char *path, struct sos_version_s *ver);
struct sos_version_s sos_container_version(sos_t sos);
int sos_container_new(const char *path, int o_mode); /* deprecated */
sos_t sos_container_open(const char *path_arg, sos_perm_t o_perm, ...);
int sos_container_clone(sos_t sos, const char *path);
int sos_container_verify(sos_t sos);
int sos_container_move(const char *path_arg, const char *new_path);
int sos_container_delete(sos_t c);
int sos_container_stat(sos_t sos, struct stat *sb);
void sos_container_close(sos_t c, sos_commit_t flags);
int sos_container_commit(sos_t c, sos_commit_t flags);
void sos_container_info(sos_t sos, FILE* fp);
int sos_container_lock_info(const char *path, FILE *fp);
int sos_container_lock_cleanup(const char *path);
void sos_inuse_obj_info(sos_t sos, FILE *fp);
void sos_free_obj_info(sos_t sos, FILE *fp);
int sos_container_config_set(const char *path, const char *option, const char *value);
char *sos_container_config_get(const char *path, const char *option);
const char *sos_container_path(sos_t sos);
sos_perm_t sos_container_perm(sos_t sos);
int sos_container_mode(sos_t sos);
sos_config_iter_t sos_config_iter_new(const char *path);
void sos_config_iter_free(sos_config_iter_t iter);
sos_config_t sos_config_first(sos_config_iter_t iter);
sos_config_t sos_config_next(sos_config_iter_t iter);
void sos_config_print(const char *path, FILE *fp);
void sos_begin_x(sos_t sos);
void sos_end_x(sos_t sos);
/** @} */
/** @} */

/** \defgroup log SOS Logging Functions
 * @{
 */

#define SOS_LOG_FATAL	0x01
#define SOS_LOG_ERROR	0x02
#define SOS_LOG_WARN	0x04
#define SOS_LOG_INFO	0x08
#define SOS_LOG_DEBUG	0x10
#define SOS_LOG_ALL	0xff

void sos_log_file_set(FILE *fp);
void sos_log_mask_set(uint64_t mask);

/** @} */

/** \defgroup schema SOS Schema
 * @{
 */

/** \defgroup schema_types Schema Types
 * @{
 */
typedef struct sos_attr_s *sos_attr_t;
typedef struct sos_index_s *sos_index_t;
typedef struct sos_schema_s *sos_schema_t;
typedef struct sos_obj_s *sos_obj_t;

#define SOS_SCHEMA_NAME_LEN	256
#define SOS_ATTR_NAME_LEN	256
#define SOS_INDEX_NAME_LEN	256
#define SOS_INDEX_KEY_TYPE_LEN	64
#define SOS_INDEX_TYPE_LEN	64
#define SOS_INDEX_ARGS_LEN	256

typedef enum sos_type_e {
	/** All types up to the arrays are fixed size */
	SOS_TYPE_INT16 = 0,
	SOS_TYPE_FIRST = SOS_TYPE_INT16,
	SOS_TYPE_INT32,
	SOS_TYPE_INT64,
	SOS_TYPE_UINT16,
	SOS_TYPE_UINT32,
	SOS_TYPE_UINT64,
	SOS_TYPE_FLOAT,
	SOS_TYPE_DOUBLE,
	SOS_TYPE_LONG_DOUBLE,
	SOS_TYPE_TIMESTAMP,
	SOS_TYPE_OBJ,
	SOS_TYPE_STRUCT,
	SOS_TYPE_JOIN,
	SOS_TYPE_BYTE_ARRAY = 32,
	SOS_TYPE_ARRAY = SOS_TYPE_BYTE_ARRAY,
	SOS_TYPE_CHAR_ARRAY,
	SOS_TYPE_STRING = SOS_TYPE_CHAR_ARRAY,
	SOS_TYPE_INT16_ARRAY,
	SOS_TYPE_INT32_ARRAY,
	SOS_TYPE_INT64_ARRAY,
	SOS_TYPE_UINT16_ARRAY,
	SOS_TYPE_UINT32_ARRAY,
	SOS_TYPE_UINT64_ARRAY,
	SOS_TYPE_FLOAT_ARRAY,
	SOS_TYPE_DOUBLE_ARRAY,
	SOS_TYPE_LONG_DOUBLE_ARRAY,
	SOS_TYPE_OBJ_ARRAY,
	SOS_TYPE_LAST = SOS_TYPE_OBJ_ARRAY
} sos_type_t;

#pragma pack(1)
typedef ods_idx_data_t sos_idx_data_t;
typedef union sos_obj_ref_s {
	ods_idx_data_t idx_data;
	struct sos_idx_ref_s {
		uuid_t part_uuid;	/* The reference to the ODS */
		ods_ref_t obj;		/* The object reference */
	} ref;
} sos_obj_ref_t;
static inline void sos_ref_reset(sos_obj_ref_t ref)
{
	uuid_clear(ref.ref.part_uuid);
	ref.ref.obj = 0;
}

union sos_array_element_u {
	char char_[0];
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
	sos_obj_ref_t ref_[0];
};

typedef struct sos_array_info_s {
	uint32_t size;
	uint32_t offset;
} *sos_array_info_t;

typedef struct sos_array_s {
	uint32_t count;
	union sos_array_element_u data;
} *sos_array_t;

struct sos_timeval_s {
	uint32_t usecs;
	uint32_t secs;
};

union sos_timestamp_u {
	uint64_t timestamp;
	struct ods_timeval_s tv;
	struct sos_timeval_s fine;
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
	union sos_obj_ref_s ref_;
	uint8_t struc_[0];
};

typedef union sos_value_data_u {
	union sos_primary_u prim;
	union sos_array_element_u struc;
	struct sos_array_info_s array_info;
	struct sos_array_s array;
	struct sos_array_s join;
	char __pad[256];
} *sos_value_data_t;

/*! Describes the value of an attribute in an object */
typedef struct sos_value_s {
	sos_obj_t obj;		/*! The object to which the value refers  */
	sos_type_t type;	/*! The value type, used when attr is NULL, i.e. this is a constant */
	sos_attr_t attr;	/*! The attribute in the object */
	sos_value_data_t data;	      /*! Points at data_ or into obj */
	union sos_value_data_u data_; /*! Memory based value data */
} *sos_value_t;

sos_array_t sos_array(sos_value_t v);
#define sos_array_count(_v_) sos_array(_v_)->count
#define sos_array_data(_v_, _m_) (sos_array(_v_)->data._m_)
size_t sos_array_data_size(sos_type_t typ, size_t count);

#define SOS_STRUCT_VALUE(_name_, _sz_)					\
	unsigned char _name_ ## value_data [_sz_ + sizeof(struct sos_value_s)]; \
	sos_value_t _name_ = (sos_value_t)_name_ ## value_data;

enum sos_cond_e {
	SOS_COND_LT,
	SOS_COND_LE,
	SOS_COND_EQ,
	SOS_COND_GE,
	SOS_COND_GT,
	SOS_COND_NE,
};
typedef enum sos_cond_e sos_cond_t;

typedef struct sos_schema_template_attr {
	const char *name;
	sos_type_t type;
	size_t size;		/*! Must be specified if the type is SOS_TYPE_STRUCT or SOS_TYPE_JOIN */
	const char **join_list;	/*! An array of attribute names */
	int indexed;
	const char *idx_type;
	const char *key_type;
	const char *idx_args;
} *sos_schema_template_attr_t;

typedef struct sos_schema_template {
	const char *name;
	const char *uuid;
	struct sos_schema_template_attr attrs[];
} *sos_schema_template_t;

#pragma pack()
/** @} */

/** \defgroup schema_funcs Schema Functions
 * @{
 */
int sos_schema_export(const char *path, FILE *fp);
sos_schema_t sos_schema_create(const char *name, const uuid_t uuid);
sos_schema_t sos_schema_new(const char *name);
void sos_schema_free(sos_schema_t schema);
sos_schema_t sos_schema_dup(sos_schema_t schema);
size_t sos_schema_count(sos_t sos);
int sos_schema_add(sos_t sos, sos_schema_t schema);
sos_schema_t sos_schema_from_template(sos_schema_template_t pt);
sos_schema_t sos_schema_by_name(sos_t sos, const char *name);
sos_schema_t sos_schema_by_uuid(sos_t sos, uuid_t id);
void sos_schema_print(sos_schema_t schema, FILE *fp);
int sos_schema_delete(sos_t sos, const char *name);
sos_schema_t sos_schema_first(sos_t sos);
sos_schema_t sos_schema_next(sos_schema_t schema);
const char *sos_schema_name(sos_schema_t schema);
void sos_schema_uuid(sos_schema_t schema, uuid_t uuid);
int sos_schema_attr_count(sos_schema_t schema);
int sos_schema_attr_add(sos_schema_t schema, const char *name, sos_type_t type, ...);
int sos_schema_index_add(sos_schema_t schema, const char *name);
int sos_schema_index_modify(sos_schema_t schema, const char *name,
			    const char *idx_type, const char *key_type,
			    const char *idx_args);
sos_attr_t sos_schema_attr_by_name(sos_schema_t schema, const char *name);
sos_attr_t sos_schema_attr_by_id(sos_schema_t schema, int attr_id);
sos_attr_t sos_schema_attr_first(sos_schema_t schema);
sos_attr_t sos_schema_attr_last(sos_schema_t schema);
sos_attr_t sos_schema_attr_next(sos_attr_t attr);
sos_attr_t sos_schema_attr_prev(sos_attr_t attr);

int sos_attr_id(sos_attr_t attr);
int sos_attr_is_indexed(sos_attr_t attr);
const char *sos_attr_name(sos_attr_t attr);
sos_type_t sos_attr_type(sos_attr_t attr);
sos_index_t sos_attr_index(sos_attr_t attr);
const char *sos_attr_idx_type(sos_attr_t attr);
const char *sos_attr_key_type(sos_attr_t attr);
const char *sos_attr_idx_args(sos_attr_t attr);
size_t sos_attr_size(sos_attr_t attr);
sos_schema_t sos_attr_schema(sos_attr_t attr);
sos_array_t sos_attr_join_list(sos_attr_t attr);
sos_value_t sos_attr_min(sos_attr_t attr);
sos_value_t sos_attr_max(sos_attr_t attr);
int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
				  const char *attr_name, const char *attr_value,
				  char **endptr);
size_t sos_obj_attr_strlen(sos_obj_t obj, sos_attr_t attr);
size_t sos_obj_attr_value_set_va(sos_obj_t sos_obj, sos_attr_t attr, va_list ap);
size_t sos_obj_attr_by_name_set(sos_obj_t sos_obj, const char *attr_name, ...);
size_t sos_obj_attr_by_id_set(sos_obj_t sos_obj, int attr_id, ...);
sos_value_t sos_obj_attr_value_get(sos_obj_t sos_obj, sos_attr_t attr);
/** @} */
/** @} */

/** \defgroup partitions SOS Partitions
 * @{
 */
/** \defgroup part_types Partition Types
 * @{
 */
/** The maximum length of a partition name */
#define SOS_PART_NAME_LEN			256
/** The maximum length of a partition description */
#define SOS_PART_DESC_LEN			1024
/** The maximum length of a partition path */
#define SOS_PART_PATH_LEN			1024
typedef enum sos_part_state_e {
	/** Partition is not being used */
	SOS_PART_STATE_OFFLINE = 0,
	/** Consulted for queries/iteration */
	SOS_PART_STATE_ACTIVE = 1,
	/** New objects stored here */
	SOS_PART_STATE_PRIMARY = 2,
	/** Partition is undergoing state change  */
	SOS_PART_STATE_BUSY = 3,
	/** Detached */
	SOS_PART_STATE_DETACHED = 4
} sos_part_state_t;

/** Describes a Partitions storage attributes */
typedef struct sos_part_stat_s {
	uint64_t size;		/*! Size of the partition in bytes */
	uint64_t accessed;	/*! Last access time as a Unix timestamp */
	uint64_t modified;	/*! Last modify time as a Unix timestamp */
	uint64_t changed;	/*! Status change time as a Unix timestamp */
	uint64_t ref_count;	/*! The number of containers using this partition */
	double rd_req_sec;	/*! Read requests per second */
	double rd_kb_sec;	/*! Read kilobytes per second */
	double wr_req_sec;	/*! Write requests per second */
	double wr_kb_sec;	/*! Write kilobytes per second */
} *sos_part_stat_t;

typedef struct sos_part_iter_s *sos_part_iter_t;
typedef struct sos_part_s *sos_part_t;
/** @} */
/**
 * \defgroup part_funcs Partition Functions
 * @{
 */
int sos_part_create(sos_t sos, const char *name, const char *path);	/* deprecated */
sos_part_t sos_part_open(const char *path, int o_perm, ...);
void sos_part_close(sos_part_t part);
int sos_part_chown(sos_part_t part, uid_t uid, gid_t gid);
int sos_part_chmod(sos_part_t part, int mode);
sos_perm_t sos_part_be_get(sos_part_t part);
int sos_part_attach(sos_t sos, const char *name, const char *path);
int sos_part_detach(sos_t sos, const char *name);

sos_part_t sos_part_find(sos_t sos, const char *name);	/* deprecated */
sos_part_t sos_part_by_name(sos_t sos, const char *name);
sos_part_t sos_part_by_path(sos_t sos, const char *path);
sos_part_t sos_part_by_uuid(sos_t sos, const uuid_t uuid);
sos_part_iter_t sos_part_iter_new(sos_t sos);
void sos_part_iter_free(sos_part_iter_t iter);
sos_part_t sos_part_first(sos_part_iter_t iter);
sos_part_t sos_part_next(sos_part_iter_t iter);
const char *sos_part_name(sos_part_t part);
const char *sos_part_path(sos_part_t part);
const char *sos_part_desc(sos_part_t part);
void sos_part_desc_set(sos_part_t part, const char *name);
uint32_t sos_part_id(sos_part_t part);
sos_part_state_t sos_part_state(sos_part_t part);
int sos_part_state_set(sos_part_t part, sos_part_state_t state);
uint32_t sos_part_refcount(sos_part_t part);
void sos_part_uuid(sos_part_t part, uuid_t uuid);
sos_part_t _sos_part_get(sos_part_t part, const char *func, int line);
#define sos_part_get(_p_) _sos_part_get(_p_, __func__, __LINE__)
void sos_part_put(sos_part_t part);
int sos_part_stat(sos_part_t part, sos_part_stat_t stat);
uid_t sos_part_uid(sos_part_t part);
gid_t sos_part_gid(sos_part_t part);
int sos_part_perm(sos_part_t part);
size_t sos_part_remap_schema_uuid(sos_part_t part, const char *dst_path, const char *src_path);

typedef char sos_obj_ref_str_t[37+32];
char *sos_obj_ref_to_str(sos_obj_ref_t ref, sos_obj_ref_str_t str);
int sos_obj_ref_from_str(sos_obj_ref_t ref, const char *value, char **endptr);

/**
 * \brief The callback function called by the sos_part_obj_iter() function
 *
 * Called by the sos_part_obj_iter() function for each object in the partition. If
 * the function wishes to cancel iteration, return !0, otherwise,
 * return 0.
 *
 * This callback function owns a reference on the provided object.
 *
 * \param sos	The container handle
 * \param ods_obj	The ODS object handle
 * \param sos_obj	The SOS object handle
 * \param sz	The size of the object
 * \param arg	The 'arg' passed into sos_part_obj_iter()
 * \retval 0	Continue iterating
 * \retval !0	Stop iterating
 */
typedef int (*sos_part_obj_iter_fn_t)(sos_part_t part, ods_obj_t ods_obj, sos_obj_t sos_obj, void *arg);
typedef struct sos_part_obj_iter_pos_s {
	struct ods_obj_iter_pos_s pos;
} *sos_part_obj_iter_pos_t;
void sos_part_obj_iter_pos_init(sos_part_obj_iter_pos_t pos);
int sos_part_obj_iter(sos_part_t part, sos_part_obj_iter_pos_t pos,
		      sos_part_obj_iter_fn_t fn, void *arg);
/** @} */
/** @} */

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
 * - sos_obj_new_with_data() Create and populate a new object with data
 * - sos_obj_malloc() Create a memory object that lives outside a container
 * - sos_obj_delete()    Delete an object from the container
 * - sos_obj_get()	 Take a reference on an object
 * - sos_obj_put()	 Drop a reference on an object
 * - sos_obj_index()	 Add an object to its indices
 * - sos_obj_remove()	 Remove an object from its indices
 * - sos_obj_ptr()       Returns a pointer to the object's data
 * - sos_obj_size()	Returns the size in bytes of the object
 * - sos_obj_find()	 Find an object based on an attribute value
 * - sos_value()	 Return a value given object and attribute handles.
 * - sos_value_by_name() Get the value handle by name
 * - sos_value_by_id()   Get the value handle by id
 * - sos_value_to_str()	 Get the value as a string
 * - sos_value_strlen()  Get the size of the buffer required by sos_value_to_str()
 * - sos_value_from_str() Set the value from a string
 * - sos_value_init()	 Initializes a stack variable as a object value.
 */

/**
 * Identifies the byte order of the objects
 */
#define SOS_OBJ_BE	1
#define SOS_OBJ_LE	2

sos_obj_t sos_obj_new(sos_schema_t schema);
sos_obj_t sos_obj_new_with_data(sos_schema_t schema, uint8_t *data, size_t data_size);
sos_obj_t sos_obj_malloc(sos_schema_t schema);
int sos_obj_commit(sos_obj_t obj);
int sos_obj_commit_part(sos_obj_t obj, sos_part_t part);
sos_schema_t sos_obj_schema(sos_obj_t obj);
size_t sos_obj_size(sos_obj_t obj);
int sos_obj_copy(sos_obj_t dst, sos_obj_t src);
int sos_obj_attr_copy(sos_obj_t dst_obj, sos_attr_t dst_attr,
					sos_obj_t src_obj, sos_attr_t src_attr);
sos_obj_ref_t sos_obj_ref(sos_obj_t obj);
sos_obj_t sos_ref_as_obj(sos_t sos, sos_obj_ref_t ref);

sos_obj_t sos_obj_from_value(sos_t sos, sos_value_t ref_val);
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
sos_value_t sos_array_new(sos_value_t val, sos_attr_t attr, sos_obj_t obj, size_t count);
sos_value_t sos_value_new();
void sos_value_free(sos_value_t v);
void *sos_obj_ptr(sos_obj_t obj);
sos_value_t sos_value_init(sos_value_t value, sos_obj_t obj, sos_attr_t attr);
sos_value_t sos_value_init_const(sos_value_t value, sos_type_t type, ...);
int sos_value_true(sos_value_t value);
sos_value_t sos_value_copy(sos_value_t dst, sos_value_t src);
#define SOS_VALUE(_name_)				\
	struct sos_value_s  _name_ ## __, *_name_ = &_name_ ## __;
sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr);
void sos_value_put(sos_value_t value);
sos_value_data_t sos_obj_attr_data(sos_obj_t obj, sos_attr_t attr, sos_obj_t *arr_obj);
sos_value_data_t sos_value_data_new(sos_type_t typ, size_t count);
void sos_value_data_del(sos_value_data_t vd);
size_t sos_value_data_set(sos_value_data_t vd, sos_type_t type, ...);
size_t sos_value_data_set_va(sos_value_data_t vd, sos_type_t type, va_list ap);
size_t sos_value_data_size(sos_value_data_t vd, sos_type_t type);
int sos_value_cmp(sos_value_t a, sos_value_t b);
int sos_value_is_array(sos_value_t value);
sos_type_t sos_value_type(sos_value_t value);
size_t sos_value_size(sos_value_t value);
size_t sos_value_memcpy(sos_value_t value, void *buf, size_t buflen);
char *sos_obj_attr_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len);
int sos_obj_attr_from_str(sos_obj_t obj, sos_attr_t attr, const char *str, char **endptr);
size_t sos_value_strlen(sos_value_t v);
const char *sos_value_to_str(sos_value_t value, char *str, size_t len);
int sos_value_from_str(sos_value_t value, const char *str, char **endptr);
int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
				const char *attr_name, const char *attr_value,
				char **endptr);
char *sos_obj_attr_by_name_to_str(sos_obj_t sos_obj, const char *attr_name,
				char *str, size_t len);
char *sos_obj_attr_by_id_to_str(sos_obj_t sos_obj, int,
				char *str, size_t len);

/** @} */

/** \defgroup indices SOS Indices
 * @{
 */
/** \defgroup index_types Index Types
 * @{
 */
typedef struct sos_index_stat_s {
	uint64_t cardinality;
	uint64_t duplicates;
	uint64_t size;
} *sos_index_stat_t;
/** @} */
/** \defgroup index_funcs Index Functions
 * @{
 */
typedef struct ods_obj_s *sos_key_t;
typedef int (*sos_ins_cb_fn_t)(sos_index_t index, sos_key_t key,
			       int missing, sos_obj_ref_t *ref, void *arg);
typedef enum sos_visit_action {
	SOS_VISIT_ADD = ODS_VISIT_ADD,	/*! Add the key and set its data to idx_data */
	SOS_VISIT_DEL = ODS_VISIT_DEL,	/*! Delete the key */
	SOS_VISIT_UPD = ODS_VISIT_UPD,	/*! Update the index data for key */
	SOS_VISIT_NOP = ODS_VISIT_NOP	/*! Do nothing */
} sos_visit_action_t;
typedef sos_visit_action_t (*sos_visit_cb_fn_t)(sos_index_t index,
						sos_key_t key, sos_idx_data_t *idx_data,
						int found,
						void *arg);
sos_obj_t sos_obj_find(sos_attr_t attr, sos_key_t key);
int sos_index_new(sos_t sos, const char *name,
		  const char *idx_type, const char *key_type,
		  const char *args);
sos_index_t sos_index_open(sos_t sos, const char *name);
typedef uint32_t sos_index_rt_opt_t;
#define SOS_INDEX_RT_OPT_MP_UNSAFE	1
#define SOS_INDEX_RT_OPT_VISIT_ASYNC	2
int sos_index_rt_opt_set(sos_index_t index, sos_index_rt_opt_t opt, ...);
int sos_index_insert(sos_index_t index, sos_key_t key, sos_obj_t obj);
int sos_index_remove(sos_index_t index, sos_key_t key, sos_obj_t obj);
int sos_index_visit(sos_index_t index, sos_key_t key, sos_visit_cb_fn_t cb_fn, void *arg);
sos_obj_t sos_index_find(sos_index_t index, sos_key_t key);
int sos_index_find_ref(sos_index_t index, sos_key_t key, sos_obj_ref_t *ref);
sos_obj_t sos_index_find_inf(sos_index_t index, sos_key_t key);
sos_obj_t sos_index_find_sup(sos_index_t index, sos_key_t key);
sos_obj_t sos_index_find_min(sos_index_t index, sos_key_t *pkey);
sos_obj_t sos_index_find_max(sos_index_t index, sos_key_t *pkey);
int sos_index_commit(sos_index_t index, sos_commit_t flags);
int sos_index_close(sos_index_t index, sos_commit_t flags);
size_t sos_index_key_size(sos_index_t index);
sos_key_t sos_index_key_new(sos_index_t index, size_t size);
int sos_index_key_from_str(sos_index_t index, sos_key_t key, const char *str);
const char *sos_index_key_to_str(sos_index_t index, sos_key_t key);
int64_t sos_index_key_cmp(sos_index_t index, sos_key_t a, sos_key_t b);
void sos_index_print(sos_index_t index, FILE *fp);
const char *sos_index_name(sos_index_t index);
int sos_index_stat(sos_index_t index, sos_index_stat_t sb);
void sos_container_index_list(sos_t sos, FILE *fp);
typedef struct sos_container_index_iter_s *sos_container_index_iter_t;
sos_container_index_iter_t sos_container_index_iter_new(sos_t sos);
void sos_container_index_iter_free(sos_container_index_iter_t iter);
sos_index_t sos_container_index_iter_first(sos_container_index_iter_t iter);
sos_index_t sos_container_index_iter_next(sos_container_index_iter_t iter);
/** @} */
/** @} */

/** \defgroup keys SOS Keys
 * @{
 */

typedef struct sos_comp_key_spec {
	sos_type_t type;
	sos_value_data_t data;
} *sos_comp_key_spec_t;

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
#define SOS_KEY_VALUE(_name_)					\
	struct sos_key_value_s  ## _name_ {			\
		uint16_t len;					\
		unsigned char value[SOS_STACK_KEY_SIZE];	\
	} _name_ ## _ ## data;

#define SOS_KEY(_name_)					\
	struct sos_key_value_s  ## _name_ {		\
		uint16_t len;				\
		unsigned char value[SOS_STACK_KEY_SIZE];\
	} _name_ ## _ ## data;				\
	ODS_OBJ(_name_ ## _ ## key_s, &_name_ ## _ ## data, SOS_STACK_KEY_SIZE);	\
	sos_key_t _name_ = &_name_ ## _ ## key_s;

#define SOS_KEY_SZ(_name_, _sz_)				\
	struct sos_key_value_s  ## _name_ {		\
		uint16_t len;				\
		unsigned char value[_sz_];\
	} _name_ ## _ ## data;				\
	ODS_OBJ(_name_ ## _ ## key_s, &_name_ ## _ ## data, _sz_);	\
	sos_key_t _name_ = &_name_ ## _ ## key_s;

sos_key_t sos_key_new(size_t sz);
sos_key_t sos_key_for_attr(sos_key_t key, sos_attr_t attr, ...);
void sos_key_put(sos_key_t key);
int sos_key_copy(sos_key_t dst, sos_key_t src);
size_t sos_key_set(sos_key_t key, void *value, size_t sz);
char *sos_key_to_str(sos_key_t key, const char *fmt, const char *sep, size_t el_sz);
int sos_key_join(sos_key_t key, sos_attr_t join_attr, ...);
int sos_key_join_va(sos_key_t key, sos_attr_t join_attr, va_list ap);
int sos_key_join_size(sos_attr_t join_attr, ...);
int sos_key_join_size_va(sos_attr_t join_attr, va_list ap);
int sos_key_split(sos_key_t key, sos_attr_t join_attr, ...);
int sos_comp_key_set(sos_key_t key, size_t len, sos_comp_key_spec_t key_spec);
sos_comp_key_spec_t sos_comp_key_get(sos_key_t key, size_t *len);
size_t sos_comp_key_size(size_t len, sos_comp_key_spec_t key_spec);
size_t sos_key_size(sos_key_t key);
size_t sos_key_len(sos_key_t key);
unsigned char *sos_key_value(sos_key_t key);
void *sos_value_as_key(sos_value_t value);

int sos_attr_key_from_str(sos_attr_t attr, sos_key_t key, const char *str);
const char *sos_attr_key_to_str(sos_attr_t attr, sos_key_t key);
sos_key_t sos_attr_key_new(sos_attr_t attr, size_t len);
int sos_attr_key_cmp(sos_attr_t attr, sos_key_t a, sos_key_t b);
size_t sos_attr_key_size(sos_attr_t attr);

/** @} */
/** @} */

/** \defgroup iter SOS Iterators
 * @{
 */
/** \defgroup iter_types Iterator Types
 * @{
 */
typedef struct sos_iter_s *sos_iter_t;
typedef uint32_t sos_pos_t;
typedef enum sos_iter_flags_e {
	SOS_ITER_F_ALL = ODS_ITER_F_ALL,
	/** The iterator will skip duplicate keys in the index */
	SOS_ITER_F_UNIQUE = ODS_ITER_F_UNIQUE,
	SOS_ITER_F_INF_LAST_DUP = ODS_ITER_F_GLB_LAST_DUP,
	SOS_ITER_F_SUP_LAST_DUP = ODS_ITER_F_LUB_LAST_DUP,
	SOS_ITER_F_MASK = ODS_ITER_F_MASK
} sos_iter_flags_t;
typedef struct sos_filter_cond_s *sos_filter_cond_t;
typedef struct sos_filter_s *sos_filter_t;

/** @} */
/** \defgroup iter_funcs Iterator Functions
 * @{
 */
sos_iter_t sos_index_iter_new(sos_index_t index);
sos_iter_t sos_attr_iter_new(sos_attr_t attr);
void sos_iter_free(sos_iter_t iter);
int64_t sos_iter_key_cmp(sos_iter_t iter, sos_key_t other);
int sos_iter_find(sos_iter_t iter, sos_key_t key);
int sos_iter_find_first(sos_iter_t iter, sos_key_t key);
int sos_iter_find_last(sos_iter_t iter, sos_key_t key);
int sos_iter_inf(sos_iter_t i, sos_key_t key);
int sos_iter_sup(sos_iter_t i, sos_key_t key);
sos_attr_t sos_iter_attr(sos_iter_t i);

int sos_iter_flags_set(sos_iter_t i, sos_iter_flags_t flags);
sos_iter_flags_t sos_iter_flags_get(sos_iter_t i);
uint64_t sos_iter_card(sos_iter_t i);
uint64_t sos_iter_dups(sos_iter_t i);

int sos_iter_next(sos_iter_t iter);
int sos_iter_prev(sos_iter_t i);
int sos_iter_begin(sos_iter_t i);
int sos_iter_end(sos_iter_t i);
sos_key_t sos_iter_key(sos_iter_t iter);
sos_obj_t sos_iter_obj(sos_iter_t iter);
sos_obj_ref_t sos_iter_ref(sos_iter_t iter);
int sos_iter_entry_remove(sos_iter_t iter);

sos_filter_t sos_filter_new(sos_iter_t iter);
void sos_filter_free(sos_filter_t f);
int sos_filter_cond_add(sos_filter_t f, sos_attr_t attr,
			enum sos_cond_e cond_e, sos_value_t value);
sos_obj_t sos_filter_begin(sos_filter_t filt);
sos_obj_t sos_filter_next(sos_filter_t filt);
sos_obj_t sos_filter_prev(sos_filter_t filt);
sos_obj_t sos_filter_end(sos_filter_t filt);
int sos_filter_miss_count(sos_filter_t filt);

sos_obj_t sos_filter_obj(sos_filter_t filt);
int sos_filter_flags_set(sos_filter_t filt, sos_iter_flags_t flags);
sos_iter_flags_t sos_filter_flags_get(sos_filter_t filt);

/** @} */
/** @} */

#endif
