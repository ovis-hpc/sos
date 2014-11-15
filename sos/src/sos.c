/*
 * Copyright (c) 2012-14 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2012-14 Sandia Corporation. All rights reserved.
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

#include <sys/queue.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

static uint32_t type_sizes[] = {
	[SOS_TYPE_INT32] = 4,
	[SOS_TYPE_INT64] = 8,
	[SOS_TYPE_UINT32] = 4,
	[SOS_TYPE_UINT64] = 8,
	[SOS_TYPE_FLOAT] = 4,
	[SOS_TYPE_DOUBLE] = 8,
	[SOS_TYPE_LONG_DOUBLE] = 16,
	[SOS_TYPE_OBJ] = 16,
	[SOS_TYPE_BYTE_ARRAY] = 8,
	[SOS_TYPE_INT32_ARRAY] = 8,
	[SOS_TYPE_INT64_ARRAY] = 8,
	[SOS_TYPE_UINT32_ARRAY] = 8,
	[SOS_TYPE_UINT64_ARRAY] = 8,
	[SOS_TYPE_FLOAT_ARRAY] = 8,
	[SOS_TYPE_DOUBLE_ARRAY] = 8,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = 8,
	[SOS_TYPE_OBJ_ARRAY] = 8,
};

static sos_attr_t _attr_by_name(sos_schema_t schema, const char *name)
{
	sos_attr_t attr;
	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		if (0 == strcmp(name, attr->data->name))
			return attr;
	}
	return NULL;
}

static sos_attr_t _attr_by_idx(sos_schema_t schema, int attr_id)
{
	sos_attr_t attr;
	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		if (attr->data->id == attr_id)
			return attr;
	}
	return NULL;
}

sos_schema_t sos_schema_new(const char *name)
{
	sos_schema_t schema;

	if (strlen(name) >= SOS_SCHEMA_NAME_LEN)
		return NULL;
	schema = calloc(1, sizeof(*schema));
	if (schema) {
		schema->data = &schema->data_;
		strcpy(schema->data->name, name);
		TAILQ_INIT(&schema->attr_list);
	}
	return schema;
}

int sos_schema_attr_count(sos_schema_t schema)
{
	return schema->data->attr_cnt;
}

const char *sos_schema_name(sos_schema_t schema)
{
	return schema->data->name;
}

static char * make_index_path(char *container_path,
			      char *schema_name,
			      char *attr_name)
{
	static char tmp_path[PATH_MAX];
	sprintf(tmp_path, "%s/%s_%s_idx",
		container_path, schema_name, attr_name);
	return tmp_path;
}

const char *key_types[] = {
	[SOS_TYPE_INT32] = "INT32",
	[SOS_TYPE_INT64] = "INT64",
	[SOS_TYPE_UINT32] = "UINT32",
	[SOS_TYPE_UINT64] = "UINT64",
	[SOS_TYPE_FLOAT] = "FLOAT",
	[SOS_TYPE_DOUBLE] = "DOUBLE",
	[SOS_TYPE_LONG_DOUBLE] = "LONG_DOUBLE",
	[SOS_TYPE_OBJ] = NULL,
	[SOS_TYPE_BYTE_ARRAY] = "STRING",
 	[SOS_TYPE_INT32_ARRAY] = NULL,
	[SOS_TYPE_INT64_ARRAY] = NULL,
	[SOS_TYPE_UINT32_ARRAY] = NULL,
	[SOS_TYPE_UINT64_ARRAY] = NULL,
	[SOS_TYPE_FLOAT_ARRAY] = NULL,
	[SOS_TYPE_DOUBLE_ARRAY] = NULL,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = NULL,
	[SOS_TYPE_OBJ_ARRAY] = NULL,
};

static sos_attr_t attr_new(sos_schema_t schema, sos_type_t type)
{
	sos_attr_t attr = calloc(1, sizeof *attr);
	if (!attr)
		goto out;
	attr->schema = schema;
	attr->size_fn = __sos_attr_size_fn_for_type(type);
	attr->to_str_fn = __sos_attr_to_str_fn_for_type(type);
	attr->from_str_fn = __sos_attr_from_str_fn_for_type(type);
	attr->key_value_fn = __sos_attr_key_value_fn_for_type(type);
	attr->idx_type = "BXTREE";
	attr->key_type = key_types[type];
 out:
	return attr;
}

int sos_attr_add(sos_schema_t schema, const char *name,
		 sos_type_t type, int initial_count)
{
	sos_attr_t attr;
	sos_attr_t prev = NULL;

	if (schema->schema_obj)
		return EBUSY;

	/* Search the schema to see if the name is already in use */
	attr = _attr_by_name(schema, name);
	if (attr)
		return EEXIST;

	if (type > SOS_TYPE_LAST)
		return EINVAL;

	if (!TAILQ_EMPTY(&schema->attr_list))
		prev = TAILQ_LAST(&schema->attr_list, sos_attr_list);

	attr = attr_new(schema, type);
	if (!attr)
		return ENOMEM;
	attr->data = &attr->data_;
	strcpy(attr->data->name, name);
	attr->data->type = type;
	attr->data->id = schema->data->attr_cnt++;
	attr->data->initial_count = initial_count;
	if (prev)
		attr->data->offset = prev->data->offset + type_sizes[prev->data->type];
	else
		attr->data->offset = sizeof(struct sos_obj_data_s);
	schema->data->obj_sz = attr->data->offset + type_sizes[attr->data->type];

	/* Append new attribute to tail of list */
	TAILQ_INSERT_TAIL(&schema->attr_list, attr, entry);
	return 0;
}

int sos_index_add(sos_schema_t schema, const char *name)
{
	sos_attr_t attr;

	if (schema->schema_obj)
		return EBUSY;

	/* Find the attribute */
	attr = _attr_by_name(schema, name);
	if (!attr)
		return ENOENT;

	attr->data->indexed = 1;
	return 0;
}

int sos_index_cfg_type(sos_schema_t schema, const char *name,
		       const char *idx_type, const char *key_type, ...)
{
	return ENOSYS;
}

sos_attr_t sos_attr_by_name(sos_schema_t schema, const char *name)
{
	return _attr_by_name(schema, name);
}

sos_attr_t sos_attr_by_id(sos_schema_t schema, int attr_id)
{
	return _attr_by_idx(schema, attr_id);
}

sos_type_t sos_attr_type(sos_attr_t attr)
{
	return attr->data->type;
}

int sos_attr_id(sos_attr_t attr)
{
	return attr->data->id;
}

const char *sos_attr_name(sos_attr_t attr)
{
	return attr->data->name;
}

struct sos_schema_s sos_obj_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_obj_ischema.data_,
	.data_ = {
		.name = "ISCHEMA_OBJ",
		.ref_count = 1,
		.id = SOS_ISCHEMA_OBJ,
		.obj_sz = sizeof(ods_ref_t),
	},
};
struct sos_schema_s sos_byte_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_byte_array_ischema.data_,
	.data_ = {
		.name = "BYTE_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_BYTE_ARRAY,
		.obj_sz = sizeof(char),
	},
};
struct sos_schema_s sos_int32_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_int32_array_ischema.data_,
	.data_ = {
		.name = "INT32_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_INT32_ARRAY,
		.obj_sz = sizeof(int32_t),
	},
};
struct sos_schema_s sos_int64_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_int64_array_ischema.data_,
	.data_ = {
		.name = "INT64_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_INT64_ARRAY,
		.obj_sz = sizeof(int64_t),
	},
};
struct sos_schema_s sos_uint32_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_uint32_array_ischema.data_,
	.data_ = {
		.name = "UINT32_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_BYTE_ARRAY,
		.obj_sz = sizeof(uint32_t),
	},
};
struct sos_schema_s sos_uint64_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_uint64_array_ischema.data_,
	.data_ = {
		.name = "UINT64_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_UINT64_ARRAY,
		.obj_sz = sizeof(uint64_t),
	},
};
struct sos_schema_s sos_float_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_float_array_ischema.data_,
	.data_ = {
		.name = "ISCHEMA_FLOAT_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_FLOAT_ARRAY,
		.obj_sz = sizeof(float),
	},
};
struct sos_schema_s sos_double_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_double_array_ischema.data_,
	.data_ = {
		.name = "DOUBLE_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_DOUBLE_ARRAY,
		.obj_sz = sizeof(double),
	},
};
struct sos_schema_s sos_long_double_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_long_double_array_ischema.data_,
	.data_ = {
		.name = "LONG_DOUBLE_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_LONG_DOUBLE_ARRAY,
		.obj_sz = sizeof(long double),
	},
};
struct sos_schema_s sos_obj_array_ischema = {
	.sos = (void *)-1,
	.ref_count = 1,
	.data = &sos_obj_array_ischema.data_,
	.data_ = {
		.name = "OBJ_ARRAY",
		.ref_count = 1,
		.id = SOS_ISCHEMA_OBJ_ARRAY,
		.obj_sz = sizeof(struct sos_obj_ref_s),
	},
};

sos_schema_t get_ischema(sos_type_t type)
{
	assert(type >= SOS_TYPE_OBJ);
	switch (type) {
	case SOS_TYPE_OBJ:
		return &sos_obj_ischema;
	case SOS_TYPE_BYTE_ARRAY:
		return &sos_byte_array_ischema;
	case SOS_TYPE_INT32_ARRAY:
		return &sos_int32_array_ischema;
	case SOS_TYPE_INT64_ARRAY:
		return &sos_int64_array_ischema;
	case SOS_TYPE_UINT32_ARRAY:
		return &sos_uint32_array_ischema;
	case SOS_TYPE_UINT64_ARRAY:
		return &sos_uint64_array_ischema;
	case SOS_TYPE_FLOAT_ARRAY:
		return &sos_float_array_ischema;
	case SOS_TYPE_DOUBLE_ARRAY:
		return &sos_double_array_ischema;
	case SOS_TYPE_LONG_DOUBLE_ARRAY:
		return &sos_long_double_array_ischema;
	case SOS_TYPE_OBJ_ARRAY:
		return &sos_obj_array_ischema;
	}
}

sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr)
{
	ods_obj_t array_obj;
	size_t elem_count;
	sos_schema_t schema;
	sos_value_t val = calloc(1, sizeof *val);
	if (!val)
		return NULL;

	val->obj = sos_obj_get(obj);
	val->data = (sos_value_data_t)&obj->obj->as.bytes[attr->data->offset];

	if (attr->data->type < SOS_TYPE_OBJ)
		return val;

	elem_count = attr->data->initial_count;
	array_obj = ods_ref_as_obj(obj->schema->sos->obj_ods,
				   val->data->prim.uint64_);
	schema = get_ischema(attr->data->type);
	if (!array_obj) {
		size_t size =
			sizeof(struct sos_obj_data_s)
			+ sizeof(uint32_t) /* element count */
			+ (elem_count * schema->data->obj_sz); /* array elements */
		array_obj = ods_obj_alloc(obj->schema->sos->obj_ods, size);
		if (!array_obj)
			goto err;
		struct sos_array_s *data =
			(struct sos_array_s *)&SOS_OBJ(array_obj)->data[0];
		data->count = elem_count;

		val->data->prim.uint64_ = ods_obj_ref(array_obj);
	}
	sos_obj_put(val->obj);
	val->obj = __sos_init_obj(schema, array_obj);
	if (!val->obj)
		goto err;
	val->data = (sos_value_data_t)&SOS_OBJ(array_obj)->data[0];
	return val;
 err:
	errno = ENOMEM;
	free(val);
	return NULL;
}

void sos_value_put(sos_value_t value)
{
	if (value->obj)
		sos_obj_put(value->obj);
	free(value);
}

sos_value_t sos_value_by_name(sos_schema_t schema, sos_obj_t obj,
				    const char *name, int *attr_id)
{
	int i;
	sos_attr_t attr = sos_attr_by_name(schema, name);
	if (!attr)
		return NULL;
	return sos_value(obj, attr);
}

sos_value_t sos_value_by_id(sos_obj_t obj, int attr_id)
{
	sos_attr_t attr = sos_attr_by_id(obj->schema, attr_id);
	if (!attr)
		return NULL;
	return sos_value(obj, attr);
}

int sos_attr_index(sos_attr_t attr)
{
	return attr->data->indexed;
}

size_t sos_value_size(sos_attr_t attr, sos_value_t value)
{
	return attr->size_fn(attr, value);
}

void *sos_value_as_key(sos_attr_t attr, sos_value_t value)
{
	return attr->key_value_fn(attr, value);
}

const char *sos_value_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len)
{
	return attr->to_str_fn(attr, obj, str, len);
}

int sos_value_from_str(sos_obj_t obj, sos_attr_t attr, const char *str)
{
	return attr->from_str_fn(attr, obj, str);
}

static sos_schema_t init_schema(sos_t sos, ods_obj_t schema_obj)
{
	sos_attr_t attr;
	sos_schema_t schema;
	int idx;

	if (!schema_obj)
		return NULL;
	schema = calloc(1, sizeof(*schema));
	if (!schema)
		return NULL;
	schema->ref_count = 1;
	TAILQ_INIT(&schema->attr_list);
	schema->schema_obj = schema_obj;
	schema->sos = sos_container_get(sos);
	schema->data = schema_obj->as.ptr;
	for (idx = 0; idx < schema->data->attr_cnt; idx++) {
		attr = attr_new(schema, schema->data->attr_dict[idx].type);
		if (!attr)
			goto err;
		attr->data = &schema->data->attr_dict[idx];
		TAILQ_INSERT_TAIL(&schema->attr_list, attr, entry);
		if (!attr->data->indexed)
			continue;
		attr->index = ods_idx_open(make_index_path(sos->path,
							   schema->data->name,
							   attr->data->name),
					   sos->o_flags);
		if (!attr->index)
			goto err;
	}
	rbn_init(&schema->rbn, schema->data->name);
	rbt_ins(&sos->schema_rbt, &schema->rbn);
	return schema;
 err:
	while (!TAILQ_EMPTY(&schema->attr_list)) {
		attr = TAILQ_FIRST(&schema->attr_list);
		TAILQ_REMOVE(&schema->attr_list, attr, entry);
		free(attr);
	}
	ods_obj_put(schema_obj);
	free(schema);
	return NULL;
}

sos_schema_t sos_schema_find(sos_t sos, const char *name)
{
	sos_schema_t schema;
	struct rbn *rbn = rbt_find(&sos->schema_rbt, (void *)name);
	if (!rbn)
		return NULL;
	schema = container_of(rbn, struct sos_schema_s, rbn);
	return sos_schema_get(schema);
}

int sos_schema_add(sos_t sos, sos_schema_t schema)
{
	char tmp_path[PATH_MAX];
	size_t size, key_len;
	ods_obj_t sos_obj_ref;
	ods_obj_t schema_obj;
	ods_key_t schema_key;
	sos_attr_t attr;
	int idx, rc;
	uint32_t offset;
	ods_obj_t udata;
	int o_mode;
	struct stat sb;

	/* See if this schema is already part of a container */
	if (schema->schema_obj)
		return EBUSY;

	/* Check to see if a schema by this name is already in the container */
	if (sos_schema_find(sos, schema->data->name))
		return EEXIST;

	udata = ods_get_user_data(sos->schema_ods);
	if (!udata)
		return ENOMEM;

	rc = ods_stat(sos->schema_ods, &sb);
	if (rc) {
		rc = errno;
		goto err_0;
	}
	o_mode = sb.st_mode;

	/* Compute the size of the schema data */
	size = sizeof(struct sos_schema_data_s);
	size += schema->data->attr_cnt * sizeof(struct sos_attr_data_s);
	schema_obj = ods_obj_alloc(sos->schema_ods, size);
	if (!schema_obj) {
		rc = ENOMEM;
		goto err_0;
	}
	sos_obj_ref = ods_obj_alloc(ods_idx_ods(sos->schema_idx), sizeof *sos_obj_ref);
	if (!sos_obj_ref)
		goto err_1;
	SOS_OBJ_REF(sos_obj_ref)->ods_ref = 0;
	SOS_OBJ_REF(sos_obj_ref)->obj_ref = ods_obj_ref(schema_obj);

	key_len = strlen(schema->data->name) + 1;
	schema_key = ods_key_alloc(sos->schema_idx, key_len);
	if (!schema_key) {
		rc = ENOMEM;
		goto err_2;
	}
	ods_key_set(schema_key, schema->data->name, key_len);

	schema->schema_obj = schema_obj;
	schema->sos = sos;

	strcpy(SOS_SCHEMA(schema_obj)->name, schema->data->name);
	SOS_SCHEMA(schema_obj)->ref_count = 0;
	SOS_SCHEMA(schema_obj)->schema_sz = size;
	SOS_SCHEMA(schema_obj)->obj_sz = schema->data->obj_sz;
	SOS_SCHEMA(schema_obj)->attr_cnt = schema->data->attr_cnt;
	SOS_SCHEMA(schema_obj)->id = SOS_UDATA(udata)->dict_len;

	idx = 0;
	offset = 0;
	/*
	 * Iterate through the attribute definitions and add them to
	 * the schema object
	 */
	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		sos_attr_data_t attr_data = &SOS_SCHEMA(schema_obj)->attr_dict[idx++];
		*attr_data = *attr->data;
		attr->data = attr_data;
		if (attr->data->indexed && !attr->key_type) {
			rc = EINVAL;
			goto err_3;
		}
		if (!attr->data->indexed)
			continue;
		rc = ods_idx_create(make_index_path(sos->path,
						    schema->data->name,
						    attr->data->name),
				    o_mode,
				    attr->idx_type,
				    attr->key_type,
				    5);
		if (rc)
			goto err_3;
		attr->index = ods_idx_open(make_index_path(sos->path,
							   schema->data->name,
							   attr->data->name),
					   sos->o_flags);
	}
	schema->data = schema_obj->as.ptr;
	rc = ods_idx_insert(sos->schema_idx, schema_key,
			    ods_obj_ref(sos_obj_ref));
	if (rc)
		goto err_3;
	SOS_UDATA(udata)->dict[SOS_UDATA(udata)->dict_len] = ods_obj_ref(schema_obj);
	SOS_UDATA(udata)->dict_len += 1;
	rbn_init(&schema->rbn, schema->data->name);
	rbt_ins(&sos->schema_rbt, &schema->rbn);

	ods_obj_put(sos_obj_ref);
	ods_obj_put(schema_key);
	ods_obj_put(udata);
	return 0;
 err_3:
	ods_obj_delete(schema_key);
	ods_obj_put(schema_key);
 err_2:
	ods_obj_delete(sos_obj_ref);
	ods_obj_put(sos_obj_ref);
 err_1:
	ods_obj_delete(schema_obj);
	ods_obj_put(schema_obj);
 err_0:
	ods_obj_put(udata);
	return rc;
}

/**
 * Create a new container
 */
int sos_container_new(const char *path, int o_mode)
{
	char tmp_path[PATH_MAX];
	int rc;
	int x_mode;
	ods_obj_t udata;

	/* A container is a directory */
	x_mode = o_mode;
	if (x_mode & (S_IWGRP | S_IRGRP))
		x_mode |= S_IXGRP;
	if (x_mode & (S_IWUSR | S_IRUSR))
		x_mode |= S_IXUSR;
	if (x_mode & (S_IWOTH | S_IROTH))
		x_mode |= S_IXOTH;
	rc = mkdir(path, x_mode);
	if (rc) {
		rc = errno;
		goto err_0;
	}

	/* Create the ODS to contain the schema objects */
	sprintf(tmp_path, "%s/schemas", path);
	rc = ods_create(tmp_path, o_mode);
	if (rc)
		goto err_1;
	ods_t schema_ods = ods_open(tmp_path, O_RDWR);
	if (!schema_ods)
		goto err_1;
	/* Initialize the schema dictionary */
	udata = ods_get_user_data(schema_ods);
	if (!udata) {
		rc = errno;
		goto err_2;
	}
	SOS_UDATA(udata)->signature = SOS_SCHEMA_SIGNATURE;
	SOS_UDATA(udata)->version = SOS_LATEST_VERSION;
	SOS_UDATA(udata)->dict[0] = 0xffffffff; /* invalid reference */
	SOS_UDATA(udata)->dict_len = 1;
	ods_obj_put(udata);
	ods_close(schema_ods, ODS_COMMIT_SYNC);

	/*Create the index to look up the schema names */
	sprintf(tmp_path, "%s/schema_idx", path);
	rc = ods_idx_create(tmp_path, o_mode, "BXTREE", "STRING", 5);
	if (rc)
		goto err_2;

	/* Create the ODS to contain the objects */
	sprintf(tmp_path, "%s/objects", path);
	rc = ods_create(tmp_path, o_mode);
	if (rc)
		goto err_3;

	return 0;
 err_3:
	sprintf(tmp_path, "%s/schema_idx", path);
	ods_destroy(tmp_path);
 err_2:
	sprintf(tmp_path, "%s/schemas", path);
	ods_destroy(tmp_path);
 err_1:
	rmdir(path);
 err_0:
	return rc;
}

const char *type_names[] = {
	"INT32",
	"INT64",
	"UINT32",
	"UINT64",
	"FLOAT",
	"DOUBLE",
	"LONG_DOUBLE",
	"OBJ",
	"BYTE_ARRAY",
	"INT32_ARRAY",
	"INT64_ARRAY",
	"UINT32_ARRAY",
	"UINT64_ARRAY",
	"FLOAT_ARRAY",
	"DOUBLE_ARRAY",
	"LONG_DOUBLE_ARRAY",
	"OBJ_ARRAY",
};

const char *type_name(sos_type_t t)
{
	if (t <= SOS_TYPE_LAST)
		return type_names[t];
	return "corrupted!";
}

int print_schema(struct rbn *n, void *fp_, int level)
{
	FILE *fp = fp_;
	sos_attr_t attr;

	sos_schema_t schema = container_of(n, struct sos_schema_s, rbn);
	fprintf(fp, "schema :\n");
	fprintf(fp, "    name      : %s\n", schema->data->name);
	fprintf(fp, "    schema_sz : %ld\n", schema->data->schema_sz);
	fprintf(fp, "    obj_sz    : %ld\n", schema->data->obj_sz);
	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		fprintf(fp, "    -attribute : %s\n", attr->data->name);
		fprintf(fp, "        type          : %s\n", type_name(attr->data->type));
		fprintf(fp, "        idx           : %d\n", attr->data->id);
		fprintf(fp, "        initial_count : %d\n", attr->data->initial_count);
		fprintf(fp, "        indexed       : %d\n", attr->data->indexed);
		fprintf(fp, "        offset        : %ld\n", attr->data->offset);
	}
	return 0;
}

void sos_container_info(sos_t sos, FILE *fp)
{
	rbt_traverse(&sos->schema_rbt, print_schema, fp);
}

void free_schema(sos_schema_t schema)
{
	/* Drop our reference on the schema object */
	if (schema->schema_obj)
		ods_obj_put(schema->schema_obj);

	/* Free all of our attributes and close it's indices */
	while (!TAILQ_EMPTY(&schema->attr_list)) {
		sos_attr_t attr = TAILQ_FIRST(&schema->attr_list);
		TAILQ_REMOVE(&schema->attr_list, attr, entry);
		if (attr->index)
			ods_idx_close(attr->index, ODS_COMMIT_ASYNC);
		free(attr);
	}
	free(schema);
}

void free_sos(sos_t sos, sos_commit_t flags)
{
	struct rbn *rbn;
	/* Iterate through all the schema and free each one */
	while (NULL != (rbn = rbt_min(&sos->schema_rbt))) {
		rbt_del(&sos->schema_rbt, rbn);
		sos_schema_put(container_of(rbn, struct sos_schema_s, rbn));
	}
	if (sos->path)
		free(sos->path);
	if (sos->schema_idx)
		ods_idx_close(sos->schema_idx, flags);
	if (sos->schema_ods)
		ods_close(sos->schema_ods, flags);
	if (sos->obj_ods)
		ods_close(sos->obj_ods, flags);
	free(sos);
}

int schema_cmp(void *a, void *b)
{
	return strcmp((char *)a, (char *)b);
}

int sos_container_open(const char *path, int o_flags, sos_t *p_c)
{
	char tmp_path[PATH_MAX];
	sos_t sos;
	int rc;
	ods_iter_t iter;

	sos = calloc(1, sizeof(*sos));
	if (!sos)
		return ENOMEM;
	sos->path = strdup(path);
	sos->o_flags = o_flags;
	rbt_init(&sos->schema_rbt, schema_cmp);

	/* Open the ODS containing the schema objects */
	sprintf(tmp_path, "%s/schemas", path);
	sos->schema_ods = ods_open(tmp_path, o_flags);
	if (!sos->schema_ods)
		goto err;

	/* Open the index on the schema objects */
	sprintf(tmp_path, "%s/schema_idx", path);
	sos->schema_idx = ods_idx_open(tmp_path, o_flags);
	if (!sos->schema_idx)
		goto err;

	/* Create the ODS to contain the objects */
	sprintf(tmp_path, "%s/objects", path);
	sos->obj_ods = ods_open(tmp_path, o_flags);
	if (!sos->obj_ods)
		goto err;

	/*
	 * Iterate through all the schemas and open/create the indices and
	 * repositories.
	 */
	iter = ods_iter_new(sos->schema_idx);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		int idx;
		ods_obj_t obj_ref = ods_iter_obj(iter);
		ods_obj_t schema_obj =
			ods_ref_as_obj(sos->schema_ods,
				       SOS_OBJ_REF(obj_ref)->obj_ref);
		ods_obj_put(obj_ref);
		sos_schema_t schema = init_schema(sos, schema_obj);
		if (!schema)
			goto err;
	}
	*p_c = sos;
	sos->ref_count = 1;
	return 0;
 err:
	free_sos(sos, SOS_COMMIT_ASYNC);
	return ENOENT;
}

void sos_container_close(sos_t sos, sos_commit_t flags)
{
	free_sos(sos, flags);
}

sos_obj_t __sos_init_obj(sos_schema_t schema, ods_obj_t ods_obj)
{
	sos_obj_t sos_obj;
	if (!schema->sos) {
		errno = EINVAL;
		return NULL;
	}
	sos_obj = calloc(1, sizeof *sos_obj);
	if (!sos_obj)
		return NULL;
	SOS_OBJ(ods_obj)->schema = schema->data->id;
	sos_obj->obj = ods_obj;
	ods_atomic_inc(&schema->data->ref_count);
	sos_obj->schema = sos_schema_get(schema);
	sos_obj->ref_count = 1;

	return sos_obj;
}

sos_obj_t sos_obj_new(sos_schema_t schema)
{
	ods_obj_t ods_obj;
	sos_obj_t sos_obj;
	ods_obj = ods_obj_alloc(schema->sos->obj_ods, schema->data->obj_sz);
	if (!ods_obj)
		goto err;
	sos_obj = __sos_init_obj(schema, ods_obj);
	if (!sos_obj)
		goto err;
	return sos_obj;
 err:
	ods_obj_delete(ods_obj);
	ods_obj_put(ods_obj);
	return NULL;
}

sos_t sos_container_get(sos_t sos)
{
	ods_atomic_inc(&sos->ref_count);
	return sos;
}

void sos_container_put(sos_t sos)
{
	if (sos && !ods_atomic_dec(&sos->ref_count)) {
		free_sos(sos, SOS_COMMIT_ASYNC);
	}
}

sos_schema_t sos_schema_get(sos_schema_t schema)
{
	ods_atomic_inc(&schema->ref_count);
	return schema;
}

void sos_schema_put(sos_schema_t schema)
{
	if (schema && !ods_atomic_dec(&schema->ref_count)) {
		sos_container_put(schema->sos);
		free_schema(schema);
	}
}

sos_obj_t sos_obj_get(sos_obj_t obj)
{
	ods_atomic_inc(&obj->ref_count);
	return obj;
}

void sos_obj_put(sos_obj_t obj)
{
	if (obj && !ods_atomic_dec(&obj->ref_count)) {
		sos_schema_put(obj->schema);
		ods_obj_put(obj->obj);
		free(obj);
	}
}

int sos_obj_index(sos_t sos, sos_obj_t obj)
{
	sos_value_t value;
	sos_attr_t attr;
	size_t key_sz;
	ods_key_t key;
	int rc;

	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		if (!attr->data->indexed)
			continue;
		assert(attr->index);
		value = sos_value(obj, attr);
		key_sz = sos_value_size(attr, value);
		key = ods_key_malloc(attr->index, key_sz);
		if (!key) {
			sos_value_put(value);
			return ENOMEM;
		}
		ods_key_set(key, sos_value_as_key(attr, value), key_sz);
		rc = ods_idx_insert(attr->index, key, ods_obj_ref(obj->obj));
		ods_obj_put(key);
		sos_value_put(value);
		if (rc)
			return rc;
	}

	return 0;
}

int sos_attr_from_str(sos_obj_t sos_obj, sos_attr_t attr, const char *attr_value)
{
	return sos_value_from_str(sos_obj, attr, attr_value);
}

int sos_attr_by_name_from_str(sos_schema_t schema, sos_obj_t sos_obj,
			      const char *attr_name, const char *attr_value)
{
	sos_value_t value;
	sos_attr_t attr;
	attr = sos_attr_by_name(schema, attr_name);
	if (!attr)
		return ENOENT;

	return sos_attr_from_str(sos_obj, attr, attr_value);
}

