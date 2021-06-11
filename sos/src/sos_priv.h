/*
 * Copyright (c) 2012-2017 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2012-2017 Sandia Corporation. All rights reserved.
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
#include <stdarg.h>
#include <sys/syscall.h>
#include <sys/queue.h>
#include <time.h>
#include <uuid/uuid.h>
#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include <ods/ods_rbt.h>
#define _SOS_REF_TRACK_ 1
#include "sos_ref.h"

#define SOS_ISCHEMA_BYTE_ARRAY "35190552-9177-11eb-941e-0cc47a6e9146"
#define SOS_ISCHEMA_CHAR_ARRAY "0ad74016-917b-11eb-9af8-0cc47a6e9146"
#define SOS_ISCHEMA_INT16_ARRAY "54c4ea52-917b-11eb-a4c0-0cc47a6e9146"
#define SOS_ISCHEMA_INT32_ARRAY "596f92be-917b-11eb-8e5b-0cc47a6e9146"
#define SOS_ISCHEMA_INT64_ARRAY "5e70102c-917b-11eb-b2cd-0cc47a6e9146"
#define SOS_ISCHEMA_UINT16_ARRAY "6366fb90-917b-11eb-babd-0cc47a6e9146"
#define SOS_ISCHEMA_UINT32_ARRAY "677a7860-917b-11eb-8f87-0cc47a6e9146"
#define SOS_ISCHEMA_UINT64_ARRAY "6cb66ce4-917b-11eb-9bee-0cc47a6e9146"
#define SOS_ISCHEMA_FLOAT_ARRAY "72535022-917b-11eb-b586-0cc47a6e9146"
#define SOS_ISCHEMA_DOUBLE_ARRAY "76cfdb48-917b-11eb-b643-0cc47a6e9146"
#define SOS_ISCHEMA_LONG_DOUBLE_ARRAY "7c0cac1c-917b-11eb-b40a-0cc47a6e9146"
#define SOS_ISCHEMA_OBJ_ARRAY "7fd250e0-917b-11eb-b53e-0cc47a6e9146"
#define SOS_SCHEMA_FIRST_USER "83935940-917b-11eb-a0d3-0cc47a6e9146"

#define SOS_LATEST_VERSION 0x05010100
#define SOS_VERSION_MASK 0xffff0000
#define SOS_SCHEMA_SIGNATURE 0x534f535348434D41 /* 'SOSSCHMA' */
typedef struct sos_schema_udata_s {
	uint64_t signature;
	uint32_t version;
} *sos_schema_udata_t;
#define SOS_SCHEMA_UDATA(_o_) ODS_PTR(sos_schema_udata_t, _o_)

/**
 * Partitions keep object stores and indexes together into a physical
 * sub-container that can be moved. The purpose of this is to allow
 * data to be migrated out of storage of one class (fast/expensive) to
 * another class (slow/cheap) in an organized and safe fashion.
 *
 * Paritions can be active (part of queries) action+primary (part of
 * queries and stores new data), and inactive (not part of queries or
 * new data storage). In order to be moved, a partition must
 * be inactive.
 *
 * Partitions are reference counted. If a partition is in-use, it will
 * have a ref_count > 0. A partition can only be deleted if it is
 * inactive and has a ref_count of 0. The partition iterator is
 * semi-lockless, i.e. while a partition is in use, it will have a
 * non-zero ref_count, but the partion list-lock will not be
 * held. This allows partitions to be removed from the list, even when
 * in use, however, they will not be moved or destroyed until the
 * reference count is 0.
 */
#define SOS_PART_REF_SIGNATURE 0x5041525452454631 /* 'PARTREF1' */
typedef struct sos_part_ref_udata_s {
	uint64_t signature;
	ods_atomic_t gen;	/* Generation number */
	ods_atomic_t lock;	/* Protects the partition list */
	ods_ref_t head;		/* Head of the partition list, SOS_PART_REF */
	ods_ref_t tail;		/* Tail of the partition list, SOS_PART_REF */
	ods_ref_t primary;	/* Current primary partition, SOS_PART_REF */
} *sos_part_ref_udata_t;
#define SOS_PART_REF_UDATA(_o_) ODS_PTR(sos_part_ref_udata_t, _o_)

/* The sos_part_ref_data_s structure refers to an attached partition */
typedef struct sos_part_ref_data_s {
	char name[SOS_PART_NAME_LEN];
	char path[SOS_PART_PATH_LEN];
	uint32_t state;		/* BUSY(3), PRIMARY (2), ACTIVE (1), OFFLINE (0) */
	ods_ref_t next;		/* Next partition, SOS_PART_REF */
	ods_ref_t prev;		/* Previous partition, SOS_PART_REF */
} *sos_part_ref_data_t;
#define SOS_PART_REF(_o_) ODS_PTR(sos_part_ref_data_t, _o_)

#define SOS_PART_SIGNATURE 0x534f535041525431 /* 'SOSPART1' */
typedef struct sos_part_udata_s {
	uint64_t signature;	/* SOS_PART_SIGNATURE */
	char desc[SOS_PART_DESC_LEN];
	uuid_t uuid;		/* UUID for this partition */
	ods_atomic_t ref_count;	/* Containers that are attached to this partition */
	uint32_t is_primary;	/* !0 if the partition is primary in a container */
	uint32_t is_busy;	/* !0 if another container is changing this partition */
} *sos_part_udata_t;
#define SOS_PART_UDATA(_o_) ODS_PTR(sos_part_udata_t, _o_)

/*
 * This is the in-memory partition structure that is created when the partition is opened.
 */
struct sos_part_s {
	struct sos_ref_s ref_count;
	sos_t sos;
	ods_obj_t ref_obj;		/* sos_part_ref_s object */
	ods_obj_t udata_obj;	/* sos_part_udata_s object */
	ods_t obj_ods;			/* ODS where objects are stored */
	TAILQ_ENTRY(sos_part_s) entry;
};

struct sos_part_iter_s {
	sos_t sos;
	sos_part_t part;
};

#define SOS_IDXDIR_SIGNATURE 0x534f534958444952 /* 'SOSIXDIR' */
typedef struct sos_idxdir_udata_s {
	uint64_t signature;
	ods_atomic_t lock;	/* Lock for adding/removing indices */
} *sos_idxdir_udata_t;
#define SOS_IDXDIR_UDATA(_o_) ODS_PTR(sos_idxdir_udata_t, _o_)
#define _stringify_(_x_) #_x_
#define stringify(_x_) _stringify_(_x_)

/*
 * An object is counted array of bytes. Everything in the ODS store is an object.
 *
 * +-----------+--~~~~~---+
 * | schema id | data...  |
 * +-----------+--~~~~~---+
 */
typedef struct sos_obj_data_s {
	uuid_t schema_uuid;	/* The unique schema identifier */
	uint64_t data_size;
	uint8_t data[0];
} *sos_obj_data_t;

struct sos_obj_s {
	ods_atomic_t ref_count;
	sos_t sos;
	sos_schema_t schema;
	sos_obj_ref_t obj_ref;
	ods_obj_t obj;
	LIST_ENTRY(sos_obj_s) entry;
	size_t size;
	uint64_t next_array_off;
};
#define SOS_OBJ(_o_) ODS_PTR(sos_obj_data_t, (_o_))

#define SOS_SIGNATURE "SOS_OBJ_STORE"
#define SOS_OBJ_BE	1
#define SOS_OBJ_LE	2

/* Extended data for the join attribute type */
typedef struct sos_join_data_s {
	char key[SOS_INDEX_KEY_TYPE_LEN];	/* comparator name */
	sos_obj_ref_t attr_list;		/* list of attr ids in this join */
} sos_join_data_t;

typedef struct sos_attr_data_s {
	char name[SOS_ATTR_NAME_LEN];
	uint32_t id;
	uint32_t indexed:1;	/* !0 if there is an associated index */
	uint32_t type:7;	/* Attribute type */
	uint32_t pad:24;
	uint32_t el_sz:8;	/* The size of each element if array */
	uint32_t reserved:24;	/* not used */
	uint32_t size;		/* The size of the attribute in bytes */
	uint64_t offset;	/* location of attribute in the object */
	ods_ref_t ext_ref;	/* reference to extended data */
} *sos_attr_data_t;

typedef size_t (*sos_value_size_fn_t)(sos_value_t);
typedef size_t (*sos_value_strlen_fn_t)(sos_value_t);
typedef char *(*sos_value_to_str_fn_t)(sos_value_t, char *, size_t);
typedef int (*sos_value_from_str_fn_t)(sos_value_t, const char *, char **);
typedef void *(*sos_value_key_value_fn_t)(sos_value_t);

typedef struct ods_idx_ref_s {
	ods_idx_t idx;
	sos_part_t part;
	LIST_ENTRY(ods_idx_ref_s) entry;
} *ods_idx_ref_t;

struct sos_index_s {
	char name[SOS_INDEX_NAME_LEN];
	sos_t sos;
	/*
	 * The primary index. All inserts go here.
	 */
	ods_idx_t primary_idx;
	sos_part_t primary_part;

	ods_obj_t idx_obj;

	/*
	 * The list of active partition indices. Iteration, search, etc...
	 * consult these indices.
	 */
	LIST_HEAD(sos_idx_list_head, ods_idx_ref_s) active_idx_list;
};

#define SOS_JOIN_EXT_SIZE(_count_)   ((sizeof(uint32_t) * _count_) + sizeof(uint32_t))
struct sos_attr_s {
	sos_attr_data_t data;
	struct sos_attr_data_s data_;

	sos_schema_t schema;
	sos_index_t index;
	char *idx_type;
	char *key_type;
	char *idx_args;
	sos_array_t ext_ptr;
	sos_value_size_fn_t size_fn;
	sos_value_from_str_fn_t from_str_fn;
	sos_value_to_str_fn_t to_str_fn;
	sos_value_strlen_fn_t strlen_fn;
	sos_value_key_value_fn_t key_value_fn;

	TAILQ_ENTRY(sos_attr_s) idx_entry;
	TAILQ_ENTRY(sos_attr_s) entry;
};

typedef struct sos_schema_data_s {
	char name[SOS_SCHEMA_NAME_LEN];
	ods_atomic_t ref_count;
	uuid_t uuid;		/* schema uuid */
	uint32_t attr_cnt;	/* Count of attributes in object class */
	uint32_t array_cnt;	/* Count of array attributes in the object class */
	uint32_t key_sz;	/* Size of largest indexed attribute */
	uint64_t obj_sz;	/* Size of scalars in object */
	uint64_t el_sz;		/* Size of each element if this is an array object */
	uint64_t schema_sz;	/* Size of schema, not including arrays */
	struct sos_attr_data_s attr_dict[0];
} *sos_schema_data_t;

enum sos_schema_state {
	SOS_SCHEMA_CLOSED,
	SOS_SCHEMA_OPEN,
};
#define SOS_SCHEMA_F_INTERNAL	0x01
struct sos_schema_s {
	ods_atomic_t ref_count;
	uint32_t flags;
	sos_schema_data_t data;
	struct sos_schema_data_s data_;
	sos_t sos;
	enum sos_schema_state state;
	ods_obj_t schema_obj;
	struct ods_rbn name_rbn;
	struct ods_rbn id_rbn;
	sos_attr_t *dict;
	LIST_ENTRY(sos_schema_s) entry;
	TAILQ_HEAD(sos_index_list, sos_attr_s) idx_attr_list;
	TAILQ_HEAD(sos_attr_list, sos_attr_s) attr_list;
};

typedef struct sos_idx_data_s {
	ods_atomic_t ref_count;
	uint32_t mode;
	char name[SOS_INDEX_NAME_LEN];
	char key_type[SOS_INDEX_KEY_TYPE_LEN];
	char idx_type[SOS_INDEX_TYPE_LEN];
	char args[SOS_INDEX_ARGS_LEN];
} *sos_idx__data_t;

#define SOS_SCHEMA(_o_) ODS_PTR(sos_schema_data_t, _o_)
#define SOS_CONFIG(_o_) ODS_PTR(sos_config_t, _o_)
#define SOS_PART_ATT(_o_) ODS_PTR(sos_part_att_data_t, _o_)
#define SOS_ARRAY(_o_) ODS_PTR(sos_array_t, _o_)
#define SOS_IDX(_o_) ODS_PTR(sos_idx__data_t, _o_)

#define SOS_OPTIONS_PARTITION_ENABLE	1
struct sos_container_config {
	unsigned int options;
	int pos_keep_time;
};

/*
 * The container
 */
struct sos_container_s {
	pthread_mutex_t lock;

	/*
	 * "Path" to the file. This is used as a prefix for all the
	 *  real file paths.
	 */
	char *path;
	ods_perm_t o_perm;
	int o_mode;

	/*
	 * Index dictionary - Keeps track of all indices defined on
	 * the Container.
	 */
	ods_idx_t idx_idx;
	ods_t idx_ods;
	ods_obj_t idx_udata;

	/*
	 * The schema dictionary - Keeps track of all Object schema
	 * defined for the Container.
	 */
	ods_idx_t schema_idx;	/* Index schema by name */
	ods_t schema_ods;	/* Contains the schema definitions */
	struct ods_rbt schema_name_rbt;	/* In memory schema tree by name */
	struct ods_rbt schema_id_rbt;	/* In memory schema tree by id */
	size_t schema_count;

	/*
	 * Config - Keeps a dictionary of user-specified configuration
	 * for the Container. Similar to getenv().
	 */
	time_t container_time;	/* Current container timestamp */
	struct sos_container_config config;

	/*
	 * The object partitions
	 */
	ods_atomic_t part_gn;		/* partition generation number */
	ods_obj_t part_ref_udata;	/* SOS_PART_REF_UDATA */
	ods_t part_ref_ods;
	sos_part_t primary_part;
	TAILQ_HEAD(sos_part_list, sos_part_s) part_list;

	LIST_HEAD(obj_list_head, sos_obj_s) obj_list;
	LIST_HEAD(obj_free_list_head, sos_obj_s) obj_free_list;
	LIST_HEAD(schema_list, sos_schema_s) schema_list;

	LIST_ENTRY(sos_container_s) entry;
};

typedef int (*sos_filter_fn_t)(sos_value_t a, sos_value_t b, int *ret);
struct sos_filter_cond_s {
	sos_attr_t attr;
	struct sos_value_s value_;
	sos_value_t value;
	sos_iter_t iter;
	sos_filter_fn_t cmp_fn;
	enum sos_cond_e cond;
	int ret;
	TAILQ_ENTRY(sos_filter_cond_s) entry;
};

struct sos_filter_s {
	sos_iter_t iter;
	SOS_KEY_VALUE(last_match_key);
	struct ods_obj_s last_match_obj;
	sos_key_t last_match;
	int miss_cnt;
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

typedef struct ods_iter_ref_s {
	ods_iter_t iter;
	LIST_ENTRY(ods_iter_ref_s) entry;
} *ods_iter_ref_t;

typedef struct ods_iter_obj_ref_s {
	ods_iter_t iter;
	sos_obj_ref_t ref;
	sos_part_t part;
	ods_key_t key;
	struct ods_rbn rbn;
} *ods_iter_obj_ref_t;

typedef struct sos_iter_s {
	sos_attr_t attr;	/* !NULL if this iterator is associated with an attribute */
	sos_index_t index;
	/* One iterator per ODS index in index */
	LIST_HEAD(ods_iter_list_head, ods_iter_ref_s) iter_list;
	/* Current iterator position */
	ods_iter_obj_ref_t pos;
	/* Objects inserted from each iterator */
	struct ods_rbt rbt;
} *sos_iter_t;

/**
 * Internal routines
 */
sos_schema_t __sos_get_ischema(sos_type_t type);
sos_obj_t __sos_init_obj(sos_t sos, sos_schema_t schema,
			 ods_obj_t ods_obj, sos_obj_ref_t obj_ref);
sos_obj_t __sos_init_obj_no_lock(sos_t sos, sos_schema_t schema, ods_obj_t ods_obj,
				 sos_obj_ref_t obj_ref);
void __sos_obj_put_no_lock(sos_obj_t obj);
int sos_iter_pos_put_no_lock(sos_iter_t iter, const sos_pos_t pos);
sos_value_size_fn_t __sos_attr_size_fn_for_type(sos_type_t type);
sos_value_strlen_fn_t __sos_attr_strlen_fn_for_type(sos_type_t type);
sos_value_to_str_fn_t __sos_attr_to_str_fn_for_type(sos_type_t type);
sos_value_from_str_fn_t __sos_attr_from_str_fn_for_type(sos_type_t type);
sos_value_key_value_fn_t __sos_attr_key_value_fn_for_type(sos_type_t type);
int __sos_config_init(sos_t sos);
sos_schema_t __sos_schema_init(sos_t sos, ods_obj_t schema_obj);
ods_obj_t __sos_obj_new(ods_t ods, size_t size, pthread_mutex_t *lock);
void __sos_schema_free(sos_schema_t schema);
void __sos_part_primary_set(sos_t sos, ods_obj_t part_obj);
sos_part_t __sos_primary_obj_part(sos_t sos);
int __sos_part_create(const char *part_path, const char *part_desc, sos_perm_t o_perm, int o_mode);
sos_part_iter_t __sos_part_iter_new(sos_t sos);
ods_obj_t __sos_part_obj_get(sos_t sos, ods_obj_t part_obj);
void __sos_part_obj_put(sos_t sos, ods_obj_t part_obj);
void __sos_part_iter_free(sos_part_iter_t iter);
ods_obj_t __sos_part_data_first(sos_t sos);
ods_obj_t __sos_part_data_next(sos_t sos, ods_obj_t part_obj);
void __sos_part_obj_put(sos_t sos, ods_obj_t part_obj);
ods_obj_t __sos_part_obj_get(sos_t sos, ods_obj_t part_obj);
int __sos_schema_open(sos_t sos, sos_schema_t schema);
int64_t __sos_schema_name_cmp(void *a, const void *b, void *);
int64_t __sos_schema_id_cmp(void *a, const void *b, void *);
void __sos_schema_reset(sos_t sos);
int __sos_open_partitions(sos_t sos, char *tmp_path);
int __sos_make_all_dir(const char *inp_path, mode_t omode);
sos_part_t __sos_container_part_find(sos_t sos, const char *name);
#define MAX_JOIN_ATTRS 8
ods_key_comp_t __sos_next_key_comp(ods_key_comp_t comp);
ods_key_comp_t __sos_set_key_comp(ods_key_comp_t comp, sos_value_t v, size_t *comp_len);
ods_key_comp_t __sos_set_key_comp_to_min(ods_key_comp_t comp, sos_attr_t a, size_t *comp_len);
ods_key_comp_t __sos_set_key_comp_to_max(ods_key_comp_t comp, sos_attr_t a, size_t *comp_len);
int __sos_value_is_max(sos_value_t v);
int __sos_value_is_min(sos_value_t v);
void __sos_init_array_values(sos_schema_t schema, sos_obj_t obj);
size_t __sos_obj_array_size(sos_obj_t obj);
size_t __sos_obj_total_size(sos_obj_t obj);
sos_part_t __sos_part_find_by_uuid(sos_t sos, uuid_t uuid);
sos_part_t __sos_part_find_by_ods(sos_t sos, ods_t ods);

extern FILE *__ods_log_fp;
extern uint64_t __ods_log_mask;

static inline void sos_log(int level, const char *func, int line, char *fmt, ...)
{
	va_list ap;
	pid_t tid;
	struct timespec ts;
	extern pthread_mutex_t _sos_log_lock;
	extern long syscall(long number, ...);

	if (!__ods_log_fp)
		return;

	if (0 ==  (level & __ods_log_mask))
		return;

	tid = (pid_t) syscall (SYS_gettid);
	pthread_mutex_lock(&_sos_log_lock);
	clock_gettime(CLOCK_REALTIME, &ts);
	va_start(ap, fmt);
	fprintf(__ods_log_fp, "[%d] %ld.%09ld: sosdb[%d] @ %s:%d | ",
		tid, (long)ts.tv_sec, (long)ts.tv_nsec, level, func, line);
	vfprintf(__ods_log_fp, fmt, ap);
	fflush(__ods_log_fp);
	pthread_mutex_unlock(&_sos_log_lock);
}

#define sos_fatal(fmt, ...) sos_log(SOS_LOG_FATAL, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define sos_error(fmt, ...) sos_log(SOS_LOG_ERROR, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define sos_warn(fmt, ...) sos_log(SOS_LOG_WARN, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define sos_info(fmt, ...) sos_log(SOS_LOG_INFO, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define sos_debug(fmt, ...) sos_log(SOS_LOG_DEBUG, __func__, __LINE__, fmt, ##__VA_ARGS__)

#define SOS_ROUNDUP(_sz_, _align_) (((_sz_) + (_align_) - 1) & ~((_align_)-1))

#endif
