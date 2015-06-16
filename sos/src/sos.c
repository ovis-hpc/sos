/*
 * Copyright (c) 2012-2015 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2012-2015 Sandia Corporation. All rights reserved.
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
/**
 * \mainpage Scalable Object Store Documentation
 *
 * \section intro Introduction
 *
 * The Scalable Object Storage (SOS) Service is a high performance
 * storage engine designed to efficiently store structured data to
 * persistent media.
 *
 * \subsection coll Collection
 *
 * A SOS Object Store is called a Collection. A Collection is
 * identified by a name that looks like a POSIX Filesytem path. This
 * allows Collections to be organized into a hierarchical name
 * space. This is a convenience and does not mean that a Container is
 * necessarily stored in a Filesytem.
 *
 * \subsection schema Schema
 *
 * Inside a Container are Schemas, Objects, and Indices. A Schema
 * defines the format of an Object. There can be any number of Schema
 * in the Container such that a single Container may contain Objects
 * of many different types. The Container has a directory of
 * Schemas. When Objects are created, the Schema handle is specified
 * to inform the object store of the size and format of the object and
 * whether or not one or more of it's attributes has an Index.
 *
 * \subsection object Object
 *
 * An Object is an instance of a Schema. An Object is a collection of
 * Attributes. An Attribute has a Name and a Type. There are built-in
 * types for an Attribute and user-defined types. The built in types
 * include the familiar <tt>int</tt>, <tt>long</tt>, <tt>double</tt>
 * types as well as arrays of these types. A special Attribute type is
 * <tt>SOS_TYPE_OBJ</tt>, which is a <tt>Reference</tt> to another Object. This
 * allows complex data structures like linked lists to be implemented
 * in the Container.
 *
 * The user-defined types are Objects. An Object's type is essentially
 * it's Schema ID and Schema Name.
 *
 * An Index is a strategy for quickly finding an Object in a container
 * based on the value of one of it's Attributes. Whether or not an
 * Attribute has an Index is specified by the Object's Schema.
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
	[SOS_TYPE_TIMESTAMP] = 8,
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

static uint32_t element_sizes[] = {
	[SOS_TYPE_INT32] = 4,
	[SOS_TYPE_INT64] = 8,
	[SOS_TYPE_UINT32] = 4,
	[SOS_TYPE_UINT64] = 8,
	[SOS_TYPE_FLOAT] = 4,
	[SOS_TYPE_DOUBLE] = 8,
	[SOS_TYPE_LONG_DOUBLE] = 16,
	[SOS_TYPE_OBJ] = 16,
	[SOS_TYPE_BYTE_ARRAY] = 1,
	[SOS_TYPE_INT32_ARRAY] = 4,
	[SOS_TYPE_INT64_ARRAY] = 8,
	[SOS_TYPE_UINT32_ARRAY] = 4,
	[SOS_TYPE_UINT64_ARRAY] = 8,
	[SOS_TYPE_FLOAT_ARRAY] = 4,
	[SOS_TYPE_DOUBLE_ARRAY] = 8,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = 16,
	[SOS_TYPE_OBJ_ARRAY] = 8,
};

/**
 * \page schema_overview Schema Overview
 *
 * Objects in a SOS container are described by a schema. Every object
 * in a SOS database is associated with a schema. A container may have
 * any number of schema and therefore objects of different types can
 * be stored in the same container. A schema consists of a unique name
 * and a set of attribute specifications that describe the object.
 *
 * Schema are created with the sos_schema_new() function. After a
 * schema is created it must be associated with one or more containers
 * with the sos_schema_add() function. Once a schema has been added to
 * a container, it can be used to create objects using the
 * sos_obj_new() function.
 *
 * Attributes are identified by their name and by the order in which
 * they were added to the schema. To manipulate or query a schema
 * attribute the attribute handle is required. The attribute handle is
 * returned by the sos_schema_attr_by_id() or sos_schema_attr_by_name()
 * functions.
 *
 * - sos_schema_new()	     Create a schema
 * - sos_schema_attr_add()   Add an attribute to a schema
 * - sos_schema_index_add()  Add an index to an attribute
 * - sos_schema_add()	     Add a schema to a container
 * - sos_schema_find()	     Find the schema with the specified name
 * - sos_schema_delete()     Remove a schema from the container
 * - sos_schema_count()	     Returns the number of schema in the container
 * - sos_schema_get()        Take a reference on a schema
 * - sos_schema_put()        Drop a reference on a schema
 * - sos_schema_name()	     Get the schema's name
 * - sos_schema_attr_count() Returns the number of attributes in the schema.
 * - sos_schema_attr_by_id() Returns the attribute by ordinal id
 * - sos_schema_attr_by_name() Returns the attribute with the specified name
 */

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
	if (attr_id < 0 || attr_id >= schema->data->attr_cnt)
		return NULL;
	if (schema->dict)
		return schema->dict[attr_id];
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
	schema->ref_count = 1;
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

int sos_schema_id(sos_schema_t schema)
{
	return schema->data->id;
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
	[SOS_TYPE_TIMESTAMP] = "UINT64",
	[SOS_TYPE_OBJ] = NULL,
	[SOS_TYPE_BYTE_ARRAY] = "STRING",
 	[SOS_TYPE_INT32_ARRAY] = "NONE",
	[SOS_TYPE_INT64_ARRAY] = "NONE",
	[SOS_TYPE_UINT32_ARRAY] = "NONE",
	[SOS_TYPE_UINT64_ARRAY] = "NONE",
	[SOS_TYPE_FLOAT_ARRAY] = "NONE",
	[SOS_TYPE_DOUBLE_ARRAY] = "NONE",
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = "NONE",
	[SOS_TYPE_OBJ_ARRAY] = "NONE",
};

static sos_attr_t attr_new(sos_schema_t schema, sos_type_t type)
{
	sos_attr_t attr = calloc(1, sizeof *attr);
	if (!attr)
		goto out;
	attr->data = &attr->data_;
	attr->schema = schema;
	attr->size_fn = __sos_attr_size_fn_for_type(type);
	attr->to_str_fn = __sos_attr_to_str_fn_for_type(type);
	attr->from_str_fn = __sos_attr_from_str_fn_for_type(type);
	attr->key_value_fn = __sos_attr_key_value_fn_for_type(type);
	attr->idx_type = strdup("BXTREE");
	attr->key_type = strdup(key_types[type]);
 out:
	return attr;
}

int sos_schema_attr_add(sos_schema_t schema, const char *name, sos_type_t type)
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
	strcpy(attr->data->name, name);
	attr->data->type = type;
	attr->data->id = schema->data->attr_cnt++;
	if (prev)
		attr->data->offset = prev->data->offset + type_sizes[prev->data->type];
	else
		attr->data->offset = sizeof(struct sos_obj_data_s);

	uint32_t a_size = sos_attr_size(attr);
	if (a_size > schema->data->key_sz)
		schema->data->key_sz = a_size;
	schema->data->obj_sz = attr->data->offset + type_sizes[attr->data->type];

	/* Append new attribute to tail of list */
	TAILQ_INSERT_TAIL(&schema->attr_list, attr, entry);
	return 0;
}

int sos_schema_index_add(sos_schema_t schema, const char *name)
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

int sos_schema_index_modify(sos_schema_t schema, const char *name,
			    const char *idx_type, const char *key_type, ...)
{
	sos_attr_t attr;

	if (schema->schema_obj)
		return EBUSY;

	/* Find the attribute */
	attr = _attr_by_name(schema, name);
	if (!attr)
		return ENOENT;

	attr->idx_type = strdup(idx_type);
	attr->key_type = strdup(key_type);
	return 0;
}

sos_attr_t sos_schema_attr_by_name(sos_schema_t schema, const char *name)
{
	return _attr_by_name(schema, name);
}

sos_attr_t sos_schema_attr_by_id(sos_schema_t schema, int attr_id)
{
	return _attr_by_idx(schema, attr_id);
}

sos_attr_t sos_schema_attr_first(sos_schema_t schema)
{
	return TAILQ_FIRST(&schema->attr_list);
}

sos_attr_t sos_schema_attr_last(sos_schema_t schema)
{
	return TAILQ_LAST(&schema->attr_list, sos_attr_list);
}

sos_attr_t sos_schema_attr_next(sos_attr_t attr)
{
	return TAILQ_NEXT(attr, entry);
}

sos_attr_t sos_schema_attr_prev(sos_attr_t attr)
{
	return TAILQ_PREV(attr, sos_attr_list, entry);
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

sos_schema_t sos_attr_schema(sos_attr_t attr)
{
	return attr->schema;
}

struct sos_schema_s sos_obj_ischema = {
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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
	.sos = NULL,
	.flags = SOS_SCHEMA_F_INTERNAL,
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

int sos_attr_is_ref(sos_attr_t attr)
{
	return attr->data->type >= SOS_TYPE_OBJ;
}

int sos_attr_is_array(sos_attr_t attr)
{
	return attr->data->type >= SOS_TYPE_ARRAY;
}

static sos_value_t mem_value_init(sos_value_t val, sos_attr_t attr)
{
	size_t elem_count;
	sos_schema_t schema;

	val->attr = attr;
	if (sos_attr_is_ref(attr)) {
		errno = EINVAL;
		return NULL;
	}
	val->data = &val->data_;
	return val;
}

sos_value_t sos_value_new()
{
	return calloc(1, sizeof(struct sos_value_s));
}

void sos_value_free(sos_value_t v)
{
	free(v);
}

void *sos_obj_ptr(sos_obj_t obj)
{
	sos_obj_data_t data = obj->obj->as.ptr;
	return data->data;
}

sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr)
{
	sos_value_t value = sos_value_new();
	if (value)
		value = sos_value_init(value, obj, attr);
	return value;
}

sos_value_t sos_value_init(sos_value_t val, sos_obj_t obj, sos_attr_t attr)
{
	ods_obj_t ref_obj;
	sos_schema_t schema;

	if (!obj)
		return mem_value_init(val, attr);

	val->attr = attr;
	val->obj = sos_obj_get(obj);
	val->data = (sos_value_data_t)&obj->obj->as.bytes[attr->data->offset];
	if (!sos_attr_is_array(attr))
		return val;
	/* Follow the reference to the object */
	ref_obj = ods_ref_as_obj(obj->sos->obj_ods, val->data->prim.ref_);
	sos_obj_put(val->obj);
	if (!ref_obj)
		return NULL;
	val->obj = __sos_init_obj(obj->sos, get_ischema(attr->data->type), ref_obj);
	val->data = (sos_value_data_t)&SOS_OBJ(ref_obj)->data[0];
	return val;
}

size_t sos_array_count(sos_value_t val)
{
	return val->data->array.count;
}

/*
 * Allocate an ODS object of the requested size and extend the store if necessary
 * NB: The lock protects multiple threads from attempting to extend if the
 * store requires expansion
 */
static ods_obj_t __obj_new(ods_t ods, size_t size, pthread_mutex_t *lock)
{
	int rc;
	ods_obj_t ods_obj;
	size_t extend_size = (size < SOS_ODS_EXTEND_SZ ? SOS_ODS_EXTEND_SZ : size << 4);
	pthread_mutex_lock(lock);
	ods_obj = ods_obj_alloc(ods, size);
	if (!ods_obj) {
		int rc = ods_extend(ods, extend_size);
		if (rc)
			goto err_0;
		ods_obj = ods_obj_alloc(ods, size);
	}
 err_0:
	pthread_mutex_unlock(lock);
	return ods_obj;
}

sos_value_t sos_array_new(sos_value_t val, sos_attr_t attr, sos_obj_t obj, size_t count)
{
	ods_obj_t array_obj;
	sos_schema_t schema;
	if (!sos_attr_is_array(attr)) {
		errno = EINVAL;
		return NULL;
	}
	schema = get_ischema(attr->data->type);
	size_t size =
		sizeof(struct sos_obj_data_s)
		+ sizeof(uint32_t) /* element count */
		+ (count * schema->data->obj_sz); /* array elements */

	array_obj = __obj_new(obj->sos->obj_ods, size, &obj->sos->lock);
	if (!array_obj)
		goto err;

	val->attr = attr;
	val->data = (sos_value_data_t)&obj->obj->as.bytes[attr->data->offset];

	struct sos_array_s *array = (struct sos_array_s *)&SOS_OBJ(array_obj)->data[0];
	array->count = count;
	val->data->prim.ref_ = ods_obj_ref(array_obj);
	val->obj = __sos_init_obj(obj->sos, schema, array_obj);
	if (!val->obj)
		goto err;
	val->data = (sos_value_data_t)&SOS_OBJ(array_obj)->data[0];
	return val;
 err:
	errno = ENOMEM;
	return NULL;
}

size_t sos_value_memset(sos_value_t val, void *buf, size_t buflen)
{
	void *dst;
	if (buflen > sos_value_size(val))
		buflen = sos_value_size(val);
	if (!sos_attr_is_array(val->attr))
		dst = val->data;
	else
		dst = &val->data->array.data.byte_[0];
	memcpy(dst, buf, buflen);
	return buflen;
}

void sos_value_put(sos_value_t value)
{
	if (!value)
		return;
	if (value->obj) {
		sos_obj_put(value->obj);
	} else {
		if (value->data != &value->data_)
			free(value->data);
	}
}

sos_value_t sos_value_by_name(sos_value_t value, sos_schema_t schema, sos_obj_t obj,
			      const char *name, int *attr_id)
{
	int i;
	sos_attr_t attr = sos_schema_attr_by_name(schema, name);
	if (!attr)
		return NULL;
	return sos_value_init(value, obj, attr);
}

sos_value_t sos_value_by_id(sos_value_t value, sos_obj_t obj, int attr_id)
{
	sos_attr_t attr = sos_schema_attr_by_id(obj->schema, attr_id);
	if (!attr)
		return NULL;
	return sos_value_init(value, obj, attr);
}

int sos_attr_index(sos_attr_t attr)
{
	return attr->data->indexed;
}

size_t sos_attr_size(sos_attr_t attr)
{
	return type_sizes[attr->data->type];
}

size_t sos_value_size(sos_value_t value)
{
	return value->attr->size_fn(value);
}

void *sos_value_as_key(sos_value_t value)
{
	return value->attr->key_value_fn(value);
}

char *sos_obj_attr_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len)
{
	struct sos_value_s v_;
	sos_value_t v = sos_value_init(&v_, obj, attr);
	if (!v)
		return NULL;
	char *s = v->attr->to_str_fn(v, str, len);
	sos_value_put(v);
	return s;
}

const char *sos_value_to_str(sos_value_t v, char *str, size_t len)
{
	return v->attr->to_str_fn(v, str, len);
}

int sos_obj_attr_from_str(sos_obj_t obj, sos_attr_t attr, const char *str, char **endptr)
{
	int rc;
	sos_value_t v;
	struct sos_value_s v_;
	if (!sos_attr_is_array(attr)) {
		v = sos_value_init(&v_, obj, attr);
		if (v) {
			rc = sos_value_from_str(v, str, endptr);
			sos_value_put(v);
		} else
			rc = EINVAL;
	} else if (sos_attr_type(attr) == SOS_TYPE_BYTE_ARRAY) {
		v = sos_value_init(&v_, obj, attr);
		size_t sz = strlen(str) + 1;
		if (v && sos_array_count(v) < sz) {
			/* Too short, delete and re-alloc */
			sos_obj_delete(v->obj);
			sos_obj_put(v->obj);
			v = NULL;
		}
		if (!v) {
			/* The array has not been allocated yet */
			v = sos_array_new(&v_, attr, obj, sz);
			if (!v)
				return ENOMEM;
			rc = sos_value_from_str(v, str, endptr);
			sos_value_put(v);
		}
	} else
		rc = EINVAL;
	return rc;
}

int sos_value_from_str(sos_value_t v, const char *str, char **endptr)
{
	return v->attr->from_str_fn(v, str, endptr);
}

sos_schema_t sos_schema_dup(sos_schema_t schema)
{
	sos_schema_t dup;
	sos_attr_t attr, src_attr;
	int idx;

	if (!schema)
		return NULL;
	dup = calloc(1, sizeof(*schema));
	if (!dup)
		return NULL;
	dup->ref_count = 1;
	TAILQ_INIT(&dup->attr_list);
	dup->data = &dup->data_;
	*dup->data = *schema->data;
	dup->dict = calloc(dup->data->attr_cnt, sizeof(sos_attr_t));
	if (!dup->dict)
		goto err_0;
	idx = 0;
	TAILQ_FOREACH(src_attr, &schema->attr_list, entry) {
		attr = attr_new(dup, src_attr->data->type);
		if (!attr)
			goto err_1;
		schema->dict[idx++] = attr;
		*attr->data = *src_attr->data;
		TAILQ_INSERT_TAIL(&dup->attr_list, attr, entry);
	}
	rbn_init(&dup->name_rbn, dup->data->name);
	rbn_init(&dup->id_rbn, &dup->data->id);
	return dup;
 err_1:
	free(schema->dict);
	while (!TAILQ_EMPTY(&dup->attr_list)) {
		attr = TAILQ_FIRST(&schema->attr_list);
		TAILQ_REMOVE(&dup->attr_list, attr, entry);
		free(attr->key_type);
		free(attr->idx_type);
		free(attr);
	}
 err_0:
	free(dup);
	return NULL;
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
	schema->sos = sos;
	schema->data = schema_obj->as.ptr;
	schema->dict = calloc(schema->data->attr_cnt, sizeof(sos_attr_t));
	if (!schema->dict)
		goto err_0;
	for (idx = 0; idx < schema->data->attr_cnt; idx++) {
		attr = attr_new(schema, schema->data->attr_dict[idx].type);
		if (!attr)
			goto err_1;
		schema->dict[idx] = attr;
		attr->data = &schema->data->attr_dict[idx];
		TAILQ_INSERT_TAIL(&schema->attr_list, attr, entry);
		if (!attr->data->indexed)
			continue;
		attr->index = ods_idx_open(make_index_path(sos->path,
							   schema->data->name,
							   attr->data->name),
					   sos->o_perm);
		if (!attr->index)
			goto err_1;
	}
	rbn_init(&schema->name_rbn, schema->data->name);
	rbt_ins(&sos->schema_name_rbt, &schema->name_rbn);
	rbn_init(&schema->id_rbn, &schema->data->id);
	rbt_ins(&sos->schema_id_rbt, &schema->id_rbn);
	sos->schema_count++;
	return schema;
 err_1:
	free(schema->dict);
	while (!TAILQ_EMPTY(&schema->attr_list)) {
		attr = TAILQ_FIRST(&schema->attr_list);
		TAILQ_REMOVE(&schema->attr_list, attr, entry);
		free(attr->key_type);
		free(attr->idx_type);
		free(attr);
	}
	ods_obj_put(schema_obj);
 err_0:
	free(schema);
	return NULL;
}

sos_schema_t sos_schema_by_name(sos_t sos, const char *name)
{
	sos_schema_t schema;
	struct rbn *rbn = rbt_find(&sos->schema_name_rbt, (void *)name);
	if (!rbn)
		return NULL;
	schema = container_of(rbn, struct sos_schema_s, name_rbn);
	return sos_schema_get(schema);
}

sos_schema_t sos_schema_by_id(sos_t sos, uint32_t id)
{
	sos_schema_t schema;
	struct rbn *rbn = rbt_find(&sos->schema_id_rbt, (void *)&id);
	if (!rbn)
		return NULL;
	schema = container_of(rbn, struct sos_schema_s, id_rbn);
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
	if (sos_schema_by_name(sos, schema->data->name))
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
	schema_obj = __obj_new(sos->schema_ods, size, &sos->lock);
	if (!schema_obj) {
		rc = ENOMEM;
		goto err_0;
	}
	sos_obj_ref = __obj_new(ods_idx_ods(sos->schema_idx), sizeof *sos_obj_ref, &sos->lock);
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
	schema->dict = calloc(schema->data->attr_cnt, sizeof(sos_attr_t));
	if (!schema->dict)
		goto err_3;
	/*
	 * Iterate through the attribute definitions and add them to
	 * the schema object
	 */
	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		sos_attr_data_t attr_data = &SOS_SCHEMA(schema_obj)->attr_dict[idx];
		*attr_data = *attr->data;
		attr->data = attr_data;
		if (attr->data->indexed && !attr->key_type) {
			rc = EINVAL;
			goto err_3;
		}
		schema->dict[idx++] = attr;
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
					   sos->o_perm);
	}
	schema->data = schema_obj->as.ptr;
	rc = ods_idx_insert(sos->schema_idx, schema_key,
			    ods_obj_ref(sos_obj_ref));
	if (rc)
		goto err_3;
	SOS_UDATA(udata)->dict[SOS_UDATA(udata)->dict_len] = ods_obj_ref(schema_obj);
	SOS_UDATA(udata)->dict_len += 1;

	rbn_init(&schema->name_rbn, schema->data->name);
	rbt_ins(&sos->schema_name_rbt, &schema->name_rbn);

	rbn_init(&schema->id_rbn, &schema->data->id);
	rbt_ins(&sos->schema_id_rbt, &schema->id_rbn);

	sos->schema_count++;

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

sos_schema_t sos_schema_first(sos_t sos)
{
	sos_schema_t first = LIST_FIRST(&sos->schema_list);
	if (first)
		return sos_schema_get(first);
	return NULL;
}

sos_schema_t sos_schema_next(sos_schema_t schema)
{
	sos_schema_t next = LIST_NEXT(schema, entry);
	if (next)
		return sos_schema_get(next);
	return NULL;
}

int sos_schema_delete(sos_t sos, const char *name)
{
	return ENOSYS;
}

/**
 * \page container_overview Container Overview
 *
 * SOS Container group Schmema, Objects, and Indices together into a
 * single namespace. The root of the namespace is the container's
 * name. Containers are created with the sos_container_new(). SOS
 * implements the POSIX security model. When a Container is created,
 * it inherits the owner and group of the process that created the
 * container. The sos_container_new() function takes an o_mode
 * parameter that identifies the standard POSIX umask to specify RO/RW
 * access for owner/group/other.
 *
 * The sos_container_open() function opens a previously created
 * container. The user/group of the process opening the container must
 * have adequate permission to provide the requested R/W access. The
 * sos_container_open() function returns a sos_t container handle that
 * is used in subsequent SOS API.
 *
 * Changes to a container are opportunistically commited to stable
 * storage. An application can initiate a commit to storage with the
 * sos_container_commit() function.
 *
 * - sos_container_new() Create a new Container
 * - sos_container_open() Open a previously created Container
 * - sos_container_close() Close a container
 * - sos_container_commit() Commit a Container's contents to stable storage
 * - sos_container_delete() Destroy a Container and all of it's contents
 * - sos_container_info() - Print Container information to a FILE pointer
 * NB: These may be deprecated
 * - sos_container_get() - Take a counted reference on the Container
 * - sos_container_put() - Put a counted reference on the Container
 */

/* This function effectively implements 'mkdir -p' */
static int make_all_dir(const char *inp_path, mode_t omode)
{
	struct stat sb;
	mode_t numask, oumask;
	int first, last, retval;
	char *p, *path;

	p = path = strdup(inp_path);
	if (!p) {
		errno = ENOMEM;
		return 1;
	}

	oumask = 0;
	retval = 0;
	if (p[0] == '/')
		++p;

	for (first = 1, last = 0; !last ; ++p) {
		if (p[0] == '\0')
			last = 1;
		else if (p[0] != '/')
			continue;
		*p = '\0';
		if (!last && p[1] == '\0')
			last = 1;
		if (first) {
			oumask = umask(0);
			numask = oumask & ~(S_IWUSR | S_IXUSR);
			(void)umask(numask);
			first = 0;
		}
		if (last)
			(void)umask(oumask);
		if (mkdir(path, last ? omode : S_IRWXU | S_IRWXG | S_IRWXO) < 0) {
			if (errno == EEXIST || errno == EISDIR) {
				if (stat(path, &sb) < 0) {
					retval = 1;
					break;
				} else if (!S_ISDIR(sb.st_mode)) {
					if (last)
						errno = EEXIST;
					else
						errno = ENOTDIR;
					retval = 1;
					break;
				}
			} else {
				retval = 1;
				break;
			}
		}
		if (!last)
			*p = '/';
	}
	if (!first && !last)
		(void)umask(oumask);
	free(path);
	return retval;
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
	rc = make_all_dir(path, x_mode);
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

int sos_container_delete(sos_t c)
{
	return ENOSYS;
}

int sos_container_commit(sos_t sos, sos_commit_t flags)
{
	sos_schema_t schema;
	sos_attr_t attr;
	int commit;

	if (flags == SOS_COMMIT_SYNC)
		commit = ODS_COMMIT_SYNC;
	else
		commit = ODS_COMMIT_ASYNC;

	/* Commit the schema idx and ods */
	ods_commit(sos->schema_ods, commit);
	ods_idx_commit(sos->schema_idx, commit);

	/* Commit the object ods */
	ods_commit(sos->obj_ods, commit);

	/* Commit all the attribute indices */
	LIST_FOREACH(schema, &sos->schema_list, entry) {
		TAILQ_FOREACH(attr, &schema->attr_list, entry) {
			if (attr->index)
				ods_idx_commit(attr->index, commit);
		}
	}
}

const char *type_names[] = {
	[SOS_TYPE_INT32] = "INT32",
	[SOS_TYPE_INT64] = "INT64",
	[SOS_TYPE_UINT32] = "UINT32",
	[SOS_TYPE_UINT64] = "UINT64",
	[SOS_TYPE_FLOAT] = "FLOAT",
	[SOS_TYPE_DOUBLE] = "DOUBLE",
	[SOS_TYPE_LONG_DOUBLE] = "LONG_DOUBLE",
	[SOS_TYPE_TIMESTAMP] = "TIMESTAMP",
	[SOS_TYPE_OBJ] = "OBJ",
	[SOS_TYPE_BYTE_ARRAY] = "BYTE_ARRAY",
	[SOS_TYPE_INT32_ARRAY] = "INT32_ARRAY",
	[SOS_TYPE_INT64_ARRAY] = "INT64_ARRAY",
	[SOS_TYPE_UINT32_ARRAY] = "UINT32_ARRAY",
	[SOS_TYPE_UINT64_ARRAY] = "UINT64_ARRAY",
	[SOS_TYPE_FLOAT_ARRAY] = "FLOAT_ARRAY",
	[SOS_TYPE_DOUBLE_ARRAY] = "DOUBLE_ARRAY",
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = "LONG_DOUBLE_ARRAY",
	[SOS_TYPE_OBJ_ARRAY] = "OBJ_ARRAY"
};

const char *type_name(sos_type_t t)
{
	if (t <= SOS_TYPE_LAST)
		return type_names[t];
	return "corrupted!";
}

void sos_schema_print(sos_schema_t schema, FILE *fp)
{
	sos_attr_t attr;
	fprintf(fp, "schema :\n");
	fprintf(fp, "    name      : %s\n", schema->data->name);
	fprintf(fp, "    schema_sz : %ld\n", schema->data->schema_sz);
	fprintf(fp, "    obj_sz    : %ld\n", schema->data->obj_sz);
	fprintf(fp, "    id        : %d\n", schema->data->id);
	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		fprintf(fp, "    -attribute : %s\n", attr->data->name);
		fprintf(fp, "        type          : %s\n", type_name(attr->data->type));
		fprintf(fp, "        idx           : %d\n", attr->data->id);
		fprintf(fp, "        indexed       : %d\n", attr->data->indexed);
		fprintf(fp, "        offset        : %ld\n", attr->data->offset);
	}
}

int print_schema(struct rbn *n, void *fp_, int level)
{
	FILE *fp = fp_;
	sos_attr_t attr;

	sos_schema_t schema = container_of(n, struct sos_schema_s, name_rbn);
	sos_schema_print(schema, fp);

	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		if (attr->index) {
			ods_idx_info(attr->index, fp);
			ods_info(ods_idx_ods(attr->index), fp);
		}
	}
	return 0;
}

void sos_container_info(sos_t sos, FILE *fp)
{
	rbt_traverse(&sos->schema_name_rbt, print_schema, fp);
	ods_idx_info(sos->schema_idx, fp);
	ods_info(ods_idx_ods(sos->schema_idx), fp);
	ods_info(sos->obj_ods, fp);
	ods_info(sos->schema_ods, fp);
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
		free(attr->key_type);
		free(attr->idx_type);
		free(attr);
	}
	free(schema);
}

void free_sos(sos_t sos, sos_commit_t flags)
{
	struct rbn *rbn;
	sos_obj_t obj;

	assert(sos->ref_count == 0);

	/* There should be no objects on the active list */
	assert(LIST_EMPTY(&sos->obj_list));

	/* Iterate through the object free list and free all the objects */
	while (!LIST_EMPTY(&sos->obj_free_list)) {
		obj = LIST_FIRST(&sos->obj_free_list);
		LIST_REMOVE(obj, entry);
		free(obj);
	}

	/* Iterate through all the schema and free each one */
	while (NULL != (rbn = rbt_min(&sos->schema_name_rbt))) {
		rbt_del(&sos->schema_name_rbt, rbn);
		free_schema(container_of(rbn, struct sos_schema_s, name_rbn));
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

int schema_name_cmp(void *a, void *b)
{
	return strcmp((char *)a, (char *)b);
}

int schema_id_cmp(void *a, void *b)
{
	return *(uint32_t *)a - *(uint32_t *)b;
}

sos_t sos_container_open(const char *path, sos_perm_t o_perm)
{
	char tmp_path[PATH_MAX];
	sos_t sos;
	int rc;
	ods_iter_t iter;

	sos = calloc(1, sizeof(*sos));
	if (!sos) {
		errno = ENOMEM;
		return NULL;
	}

	pthread_mutex_init(&sos->lock, NULL);
	sos->ref_count = 1;
	LIST_INIT(&sos->obj_list);
	LIST_INIT(&sos->obj_free_list);

	sos->path = strdup(path);
	sos->o_perm = (ods_perm_t)o_perm;
	rbt_init(&sos->schema_name_rbt, schema_name_cmp);
	rbt_init(&sos->schema_id_rbt, schema_id_cmp);
	sos->schema_count = 0;

	/* Open the ODS containing the schema objects */
	sprintf(tmp_path, "%s/schemas", path);
	sos->schema_ods = ods_open(tmp_path, sos->o_perm);
	if (!sos->schema_ods)
		goto err;

	/* Open the index on the schema objects */
	sprintf(tmp_path, "%s/schema_idx", path);
	sos->schema_idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!sos->schema_idx)
		goto err;

	/* Create the ODS to contain the objects */
	sprintf(tmp_path, "%s/objects", path);
	sos->obj_ods = ods_open(tmp_path, sos->o_perm);
	if (!sos->obj_ods)
		goto err;

	/*
	 * Iterate through all the schemas and open/create the indices and
	 * repositories.
	 */
	iter = ods_iter_new(sos->schema_idx);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		ods_obj_t obj_ref = ods_iter_obj(iter);
		ods_obj_t schema_obj =
			ods_ref_as_obj(sos->schema_ods,
				       SOS_OBJ_REF(obj_ref)->obj_ref);
		ods_obj_put(obj_ref);
		sos_schema_t schema = init_schema(sos, schema_obj);
		if (!schema)
			goto err;
		sos->schema_count++;
		LIST_INSERT_HEAD(&sos->schema_list, schema, entry);
	}
	ods_iter_delete(iter);
	return sos;
 err:
	sos_container_put(sos);
	errno = ENOENT;
	return NULL;
}

int sos_container_stat(sos_t sos, struct stat *sb)
{
	return ods_stat(sos->obj_ods, sb);
}

int sos_container_extend(sos_t sos, size_t new_size)
{
	return ods_extend(sos->obj_ods, new_size);
}

void sos_container_close(sos_t sos, sos_commit_t flags)
{
	sos_container_put(sos);
}

sos_obj_t __sos_init_obj(sos_t sos, sos_schema_t schema, ods_obj_t ods_obj)
{
	sos_obj_t sos_obj;
	pthread_mutex_lock(&sos->lock);
	if (!LIST_EMPTY(&sos->obj_free_list)) {
		sos_obj = LIST_FIRST(&sos->obj_free_list);
		LIST_REMOVE(sos_obj, entry);
	} else
		sos_obj = malloc(sizeof *sos_obj);
	pthread_mutex_unlock(&sos->lock);
	if (!sos_obj)
		return NULL;
	SOS_OBJ(ods_obj)->schema = schema->data->id;
	sos_obj->sos = sos;
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

	if (!schema->sos)
		return NULL;
	ods_obj = __obj_new(schema->sos->obj_ods, schema->data->obj_sz, &schema->sos->lock);
	if (!ods_obj)
		goto err_0;
	sos_obj = __sos_init_obj(schema->sos, schema, ods_obj);
	if (!sos_obj)
		goto err_1;
	return sos_obj;
 err_1:
	ods_obj_delete(ods_obj);
	ods_obj_put(ods_obj);
 err_0:
	return NULL;
}

sos_ref_t sos_obj_ref(sos_obj_t obj)
{
	if (obj->obj)
		return ods_obj_ref(obj->obj);
	return 0;
}

sos_obj_t sos_obj_from_ref(sos_t sos, sos_ref_t ref)
{
	ods_obj_t ods_obj;
	if (0 == ref)
		return NULL;

	ods_obj = ods_ref_as_obj(sos->obj_ods, ref);
	if (!ods_obj)
		return NULL;

	/* Get the schema id from the SOS object */
	sos_obj_data_t sos_obj = ods_obj->as.ptr;
	sos_schema_t schema = sos_schema_by_id(sos, sos_obj->schema);
	if (!schema)
		return NULL;

	return __sos_init_obj(sos, schema, ods_obj);
}

sos_obj_t sos_obj_from_value(sos_t sos, sos_value_t ref_val)
{
	ods_ref_t ref;
	if (sos_attr_type(ref_val->attr) != SOS_TYPE_OBJ)
		return NULL;

	if (0 == ref_val->data->prim.ref_)
		return NULL;

	return sos_obj_from_ref(sos, ref_val->data->prim.ref_);
}

sos_schema_t sos_obj_schema(sos_obj_t obj)
{
	return sos_schema_get(obj->schema);
}

void sos_obj_delete(sos_obj_t obj)
{
	sos_attr_t attr;
	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		struct sos_value_s v_;
		sos_value_t value;
		if (!sos_attr_is_array(attr))
			continue;
		value = sos_value_init(&v_, obj, attr);
		if (!value)
			continue;
		ods_ref_delete(obj->sos->obj_ods, value->data->prim.ref_);
		sos_value_put(value);
	}
	ods_obj_delete(obj->obj);
	obj->obj = NULL;
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

size_t sos_schema_count(sos_t sos)
{
	return sos->schema_count;
}

sos_schema_t sos_schema_get(sos_schema_t schema)
{
	ods_atomic_inc(&schema->ref_count);
	return schema;
}

void sos_schema_put(sos_schema_t schema)
{
	/* Ignore for internal schemas */
	if (!schema || (schema->flags & SOS_SCHEMA_F_INTERNAL))
		return;
	if (!ods_atomic_dec(&schema->ref_count))
		free_schema(schema);
}

sos_obj_t sos_obj_get(sos_obj_t obj)
{
	ods_atomic_inc(&obj->ref_count);
	return obj;
}

void sos_obj_put(sos_obj_t obj)
{
	if (obj && !ods_atomic_dec(&obj->ref_count)) {
		sos_t sos = obj->sos;
		sos_schema_t schema = obj->schema;
		ods_obj_put(obj->obj);
		pthread_mutex_lock(&sos->lock);
		LIST_INSERT_HEAD(&sos->obj_free_list, obj, entry);
		pthread_mutex_unlock(&sos->lock);
		sos_schema_put(schema);
	}
}

int sos_obj_remove(sos_obj_t obj)
{
	sos_value_t value;
	sos_attr_t attr;
	size_t key_sz;
	int rc;
	ods_ref_t ref;
	sos_key_t key;

	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		sos_key_t mkey;
		struct sos_value_s v_;
		sos_value_t value;
		if (!attr->data->indexed)
			continue;
		assert(attr->index);
		value = sos_value_init(&v_, obj, attr);
		key_sz = sos_value_size(value);
		key = sos_key_new(key_sz);
		if (!key) {
			sos_value_put(value);
			return ENOMEM;
		}
		ods_key_set(key, sos_value_as_key(value), key_sz);
		rc = ods_idx_delete(attr->index, key, &ref);
		sos_key_put(key);
		sos_value_put(value);
		if (rc)
			return rc;
	}

	return 0;
}

int sos_obj_index(sos_obj_t obj)
{
	struct sos_value_s v_;
	sos_value_t value;
	sos_attr_t attr;
	size_t key_sz;
	SOS_KEY(key);
	int rc;

	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		if (!attr->data->indexed)
			continue;
		assert(attr->index);
		value = sos_value_init(&v_, obj, attr);
		key_sz = sos_value_size(value);
		ods_key_set(key, sos_value_as_key(value), key_sz);
		rc = ods_idx_insert(attr->index, key, ods_obj_ref(obj->obj));
		sos_value_put(value);
		if (rc)
			goto err;
	}
	return 0;
 err:
	ods_obj_put(key);
	return rc;
}

int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
				  const char *attr_name, const char *attr_value,
				  char **endptr)
{
	sos_value_t value;
	sos_attr_t attr;
	attr = sos_schema_attr_by_name(sos_obj->schema, attr_name);
	if (!attr)
		return ENOENT;

	return sos_obj_attr_from_str(sos_obj, attr, attr_value, endptr);
}

char *sos_obj_attr_by_name_to_str(sos_obj_t sos_obj,
				  const char *attr_name, char *str, size_t len)
{
	sos_value_t value;
	sos_attr_t attr;
	attr = sos_schema_attr_by_name(sos_obj->schema, attr_name);
	if (!attr)
		return NULL;

	return sos_obj_attr_to_str(sos_obj, attr, str, len);
}

