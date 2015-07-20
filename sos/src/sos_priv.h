/*
 * Copyright (c) 2012-2015 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2012-2015 Sandia Corporation. All rights reserved.
 *
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

#ifndef __SOS_PRIV_H
#define __SOS_PRIV_H

#include <stdint.h>
#include <sys/queue.h>
#include <endian.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include <ods/rbt.h>

typedef enum sos_internal_schema_e {
	SOS_ISCHEMA_BYTE_ARRAY = SOS_TYPE_BYTE_ARRAY,
	SOS_ISCHEMA_INT32_ARRAY = SOS_TYPE_INT32_ARRAY,
	SOS_ISCHEMA_INT64_ARRAY = SOS_TYPE_INT64_ARRAY,
	SOS_ISCHEMA_UINT32_ARRAY = SOS_TYPE_UINT32_ARRAY,
	SOS_ISCHEMA_UINT64_ARRAY = SOS_TYPE_UINT64_ARRAY,
	SOS_ISCHEMA_FLOAT_ARRAY = SOS_TYPE_FLOAT_ARRAY,
	SOS_ISCHEMA_DOUBLE_ARRAY = SOS_TYPE_DOUBLE_ARRAY,
	SOS_ISCHEMA_LONG_DOUBLE_ARRAY = SOS_TYPE_LONG_DOUBLE_ARRAY,
	SOS_ISCHEMA_OBJ_ARRAY = SOS_TYPE_OBJ_ARRAY,
	SOS_SCHEMA_FIRST_USER = 128,
} sos_ischema_t;

#define SOS_LATEST_VERSION 0x03010000
#define SOS_SCHEMA_SIGNATURE 0x534f535348434D41 // 'SOSSCHMA'
typedef struct sos_udata_s {
	uint64_t signature;
	uint32_t version;
	ods_atomic_t last_schema_id;
} *sos_udata_t;
#define SOS_UDATA(_o_) ODS_PTR(sos_udata_t, _o_)

typedef struct sos_obj_part_s {
	ods_obj_t part_obj;
	ods_t obj_ods;
	TAILQ_ENTRY(sos_part_s) entry;
} *sos_obj_part_t;

typedef struct sos_idx_part_s {
	ods_obj_t part_obj;
	ods_idx_t index;
	TAILQ_ENTRY(sos_idx_part_s) entry;
} *sos_idx_part_t;

/*
 * An object is counted array of bytes. Everything in the ODS store is an object.
 *
 * +-----------+--~~~~~---+
 * | schema id | data...  |
 * +-----------+--~~~~~---+
 */
typedef struct sos_obj_data_s {
	uint64_t schema;	/* The unique schema identifier */
	uint8_t data[0];
} *sos_obj_data_t;

struct sos_obj_s {
	ods_atomic_t ref_count;
	sos_t sos;
	sos_schema_t schema;
	sos_obj_part_t part;
	ods_obj_t obj;
	LIST_ENTRY(sos_obj_s) entry;
};

#define SOS_OBJ(_o_) ODS_PTR(sos_obj_data_t, _o_)

typedef struct sos_obj_ref_s {
	ods_ref_t ods_ref;	/* The reference to the ODS */
	ods_ref_t obj_ref;	/* The object reference */
} *sos_obj_ref_t;
#define SOS_OBJ_REF(_o_) ODS_PTR(sos_obj_ref_t, _o_)

#define SOS_SIGNATURE "SOS_OBJ_STORE"
#define SOS_OBJ_BE	1
#define SOS_OBJ_LE	2

typedef struct sos_attr_data_s {
	char name[SOS_ATTR_NAME_LEN];
	uint32_t id;
	uint32_t type:8;
	uint32_t pad:23;
	uint32_t indexed:1;
	uint64_t offset;
} *sos_attr_data_t;

typedef size_t (*sos_value_size_fn_t)(sos_value_t);
typedef char *(*sos_value_to_str_fn_t)(sos_value_t, char *, size_t);
typedef int (*sos_value_from_str_fn_t)(sos_value_t, const char *, char **);
typedef void *(*sos_value_key_value_fn_t)(sos_value_t);

struct sos_index_s {
	sos_t sos;
	sos_idx_part_t last_part;
	TAILQ_HEAD(sos_idx_part_list, sos_idx_part_s) idx_list;
	char *idx_type;
	char *key_type;
};

struct sos_attr_s {
	sos_attr_data_t data;
	struct sos_attr_data_s data_;

	sos_schema_t schema;
	sos_index_t index;
#if 0
	sos_idx_part_t last_part;
	TAILQ_HEAD(sos_idx_part_list, sos_idx_part_s) idx_list;
#endif
	char *idx_type;
	char *key_type;
	sos_value_size_fn_t size_fn;
	sos_value_from_str_fn_t from_str_fn;
	sos_value_to_str_fn_t to_str_fn;
	sos_value_key_value_fn_t key_value_fn;

	TAILQ_ENTRY(sos_attr_s) entry;
};

typedef struct sos_schema_data_s {
	char name[SOS_SCHEMA_NAME_LEN];
	ods_atomic_t ref_count;
	uint32_t id;		/* Index into the schema dictionary */
	uint32_t attr_cnt;	/* Count of attributes in object class */
	uint32_t key_sz;	/* Size of largest indexed attribute */
	uint64_t obj_sz;	/* Size of object */
	uint64_t schema_sz;	/* Size of schema */
	struct sos_attr_data_s attr_dict[0];
} *sos_schema_data_t;

#define SOS_SCHEMA_F_INTERNAL	0x01
struct sos_schema_s {
	ods_atomic_t ref_count;
	uint32_t flags;
	sos_schema_data_t data;
	struct sos_schema_data_s data_;
	sos_t sos;
	ods_obj_t schema_obj;
	struct rbn name_rbn;
	struct rbn id_rbn;
	sos_attr_t *dict;
	LIST_ENTRY(sos_schema_s) entry;
	TAILQ_HEAD(sos_attr_list, sos_attr_s) attr_list;
};
#define SOS_SCHEMA(_o_) ODS_PTR(sos_schema_data_t, _o_)
#define SOS_CONFIG(_o_) ODS_PTR(sos_config_t, _o_)
#define SOS_PART(_o_) ODS_PTR(sos_part_t, _o_)
#define SOS_ARRAY(_o_) ODS_PTR(sos_array_t, _o_)

#define SOS_OPTIONS_PARTITION_ENABLE	1
struct sos_container_config {
	unsigned int options;
	uint64_t partition_extend;
	uint64_t max_partition_size;
	uint32_t partition_period; /* Number of seconds in partition */
	time_t partition_timestamp;
};

/*
 * The container
 */
struct sos_container_s {
	pthread_mutex_t lock;
	ods_atomic_t ref_count;

	/*
	 * "Path" to the file. This is used as a prefix for all the
	 *  real file paths.
	 */
	char *path;
	ods_perm_t o_perm;
	int o_mode;

	/*
	 * The schema dictionary and index
	 */
	ods_idx_t schema_idx;	/* Index schema by name */
	ods_t schema_ods;	/* Contains the schema definitions */
	struct rbt schema_name_rbt;	/* In memory schema tree by name */
	struct rbt schema_id_rbt;	/* In memory schema tree by id */
	size_t schema_count;

	/*
	 * Storage management configuration
	 */
	time_t container_time;	/* Current container timestamp */
	struct sos_container_config config;

	/*
	 * The object partitions
	 */
	sos_obj_part_t primary_part;
	TAILQ_HEAD(sos_obj_part_list, sos_obj_part_s) ods_list;
	ods_t part_ods;
	ods_idx_t part_idx;

	LIST_HEAD(obj_list_head, sos_obj_s) obj_list;
	LIST_HEAD(obj_free_list_head, sos_obj_s) obj_free_list;
	LIST_HEAD(schema_list, sos_schema_s) schema_list;
};

typedef int (*sos_filter_fn_t)(sos_value_t a, sos_value_t b);
struct sos_filter_cond_s {
	sos_attr_t attr;
	sos_value_t value;
	sos_iter_t iter;
	sos_filter_fn_t cmp_fn;
	enum sos_cond_e cond;
	TAILQ_ENTRY(sos_filter_cond_s) entry;
};

struct sos_filter_s {
	sos_iter_t iter;
	TAILQ_HEAD(sos_cond_list, sos_filter_cond_s) cond_list;
};

/**
 * \brief SOS extend size.
 *
 * SOS uses ODS to store its data. Once SOS failed to allocate an object from
 * ODS, it will try to extend the ODS. This value indicates the size of each
 * extension.
 *
 * \note Assumes to be 2^N.
 */
#define SOS_ODS_EXTEND_SZ (1024*1024)

/**
 * \brief Initial size of the ODS for SOS.
 */
#define SOS_INITIAL_SIZE (1024*1024)

struct sos_config_iter_s {
	ods_t config_ods;
	ods_idx_t config_idx;
	sos_config_t config;
	ods_iter_t iter;
	ods_obj_t obj;
};

struct sos_part_iter_s {
	ods_t part_ods;
	ods_idx_t part_idx;
	sos_part_t part;
	ods_iter_t iter;
	ods_obj_t obj;
};

struct sos_iter_s {
	sos_index_t index;
	ods_iter_t iter;
	sos_obj_part_t obj_part;
	sos_idx_part_t idx_part;
};

/**
 * Internal routines
 */
sos_obj_t __sos_init_obj(sos_t sos, sos_schema_t schema, ods_obj_t ods_obj, sos_obj_part_t part);
sos_value_size_fn_t __sos_attr_size_fn_for_type(sos_type_t type);
sos_value_to_str_fn_t __sos_attr_to_str_fn_for_type(sos_type_t type);
sos_value_from_str_fn_t __sos_attr_from_str_fn_for_type(sos_type_t type);
sos_value_key_value_fn_t __sos_attr_key_value_fn_for_type(sos_type_t type);
int __sos_config_init(sos_t sos);
sos_schema_t __sos_schema_init(sos_t sos, ods_obj_t schema_obj);
ods_obj_t __sos_obj_new(ods_t ods, size_t size, pthread_mutex_t *lock);
void __sos_schema_free(sos_schema_t schema);
sos_idx_part_t __sos_active_idx_part(sos_index_t index);
sos_obj_part_t __sos_active_obj_part(sos_t sos);
sos_obj_part_t __sos_primary_obj_part(sos_t sos);
sos_idx_part_t __sos_matching_idx_part(sos_index_t attr, sos_obj_part_t obj_part);
sos_part_iter_t __sos_part_iter_new(sos_t sos);
ods_obj_t __sos_part_first(sos_part_iter_t iter);
ods_obj_t __sos_part_next(sos_part_iter_t iter);
void __sos_part_iter_free(sos_part_iter_t iter);
int __sos_schema_open(sos_t sos, sos_schema_t schema, ods_obj_t part_obj);
int __sos_schema_name_cmp(void *a, void *b);

#endif
