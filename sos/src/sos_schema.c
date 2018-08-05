/*
 * Copyright (c) 2016 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2016 Sandia Corporation. All rights reserved.
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

#include <sys/types.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <endian.h>
#include <sos/sos.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

static struct sos_schema_s *ischema_dir[SOS_TYPE_LAST+1];
static struct rbt ischema_rbt;

static uint32_t type_sizes[] = {
	[SOS_TYPE_INT16] = 2,
	[SOS_TYPE_INT32] = 4,
	[SOS_TYPE_INT64] = 8,
	[SOS_TYPE_UINT16] = 2,
	[SOS_TYPE_UINT32] = 4,
	[SOS_TYPE_UINT64] = 8,
	[SOS_TYPE_FLOAT] = 4,
	[SOS_TYPE_DOUBLE] = 8,
	[SOS_TYPE_LONG_DOUBLE] = 16,
	[SOS_TYPE_TIMESTAMP] = 8,
	[SOS_TYPE_OBJ] = 16,
	[SOS_TYPE_BYTE_ARRAY] = 16,
	[SOS_TYPE_CHAR_ARRAY] = 16,
	[SOS_TYPE_INT16_ARRAY] = 16,
	[SOS_TYPE_INT32_ARRAY] = 16,
	[SOS_TYPE_INT64_ARRAY] = 16,
	[SOS_TYPE_UINT16_ARRAY] = 16,
	[SOS_TYPE_UINT32_ARRAY] = 16,
	[SOS_TYPE_UINT64_ARRAY] = 16,
	[SOS_TYPE_FLOAT_ARRAY] = 16,
	[SOS_TYPE_DOUBLE_ARRAY] = 16,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = 16,
	[SOS_TYPE_OBJ_ARRAY] = 16,
};

/**
 * \page schema_overview Schemas
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
 * - sos_schema_by_name()    Find the schema with the specified name
 * - sos_schema_by_id()      Find the schema with the specified integer id
 * - sos_schema_delete()     Remove a schema from the container
 * - sos_schema_count()	     Returns the number of schema in the container
 * - sos_schema_name()	     Get the schema's name
 * - sos_schema_attr_count() Returns the number of attributes in the schema.
 * - sos_schema_attr_by_id() Returns the attribute by ordinal id
 * - sos_schema_attr_by_name() Returns the attribute with the specified name
 */

/** \defgroup schema_funcs Schema Functions
 * @{
 */
/**
 * \brief Create a schema
 *
 * A schema defines a SOS object. Every object in a SOS database is
 * associated with a schema via an internal schema_id.
 *
 * After a schema is created it must be associated with one or more
 * containers. See the sos_schema_add() function to add a schema to a
 * container so that objects of that type can subsequently be created
 * in the container. Once a schema has been added, it can be looked up
 * with the sos_schema_by_name() and sos_schema_by_id() functions.
 *
 * Objects are created with the sos_obj_new() function. This function
 * takes a schema-handle as it's argument. The schema-id is saved
 * internally with the object data. An object is therefore
 * self-describing.
 *
 * \param name	The name of the schema. This name must be unique
 * within the container.
 * \returns	A pointer to the new schema or a NULL pointer if there
 * is an error. The errno variable is set to provide detail on the error.
 */
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
		TAILQ_INIT(&schema->idx_attr_list);
		schema->ref_count = 1;
	}
	return schema;
}

void sos_schema_free(sos_schema_t schema)
{
	if (schema->sos)
		return;
	__sos_schema_free(schema);
}

void __sos_schema_free(sos_schema_t schema)
{
	/* Free all of our attributes and close it's indices */
	while (!TAILQ_EMPTY(&schema->attr_list)) {
		sos_attr_t attr = TAILQ_FIRST(&schema->attr_list);
		TAILQ_REMOVE(&schema->attr_list, attr, entry);
		if (attr->index)
			sos_index_close(attr->index, ODS_COMMIT_ASYNC);
		free(attr->key_type);
		free(attr->idx_type);
		if (attr->idx_args)
			free(attr->idx_args);
		free(attr);
	}
	/* Free the attribute dictionary */
	if (schema->dict)
		free(schema->dict);

	/* Drop our reference on the schema object */
	if (schema->schema_obj)
		ods_obj_put(schema->schema_obj);

	free(schema);
}

/**
 * \brief Create a schema from a SOS template
 *
 * This convenience function creates a SOS schema from a structure
 * that specifies the schema name and an array of attribute
 * definitions. For example:
 *
 *     struct sos_schema_template employee = {
 *         .name = "employee",
 *         .attrs = {
 *              {
 *                  .name = "First",
 *                  .TYPE = SOS_TYPE_BYTE_ARRAY
 *              },
 *              {
 *                   .name = "Last",
 *                   .type = SOS_TYPE_BYTE_ARRAY,
 *                   .indexed = 1
 *              },
 *              {
 *                   .name = "Salary",
 *                   .type = SOS_TYPE_FLOAT
 *              },
 *              { NULL } // terminates attribute list
 *         }
 *     };
 *     sos_schema_t schema = sos_schema_from_template(&employee);
 *
 * \param t   Pointer to a template structure
 * \retval sos_schema_t created from the structure definition.
 */
sos_schema_t sos_schema_from_template(sos_schema_template_t t)
{
	int i, rc;
	sos_schema_t schema = sos_schema_new(t->name);
	if (!schema)
		goto err;
	for (i = 0; 1; i++) {
		if (!t->attrs[i].name)
			break;
		rc = sos_schema_attr_add(schema,
					 t->attrs[i].name,
					 t->attrs[i].type,
					 t->attrs[i].size,
					 t->attrs[i].join_list);
		if (rc)
			goto err;
		if (t->attrs[i].indexed) {
			rc = sos_schema_index_add(schema, t->attrs[i].name);
			if (rc)
				goto err;
			const char *idx_type = t->attrs[i].idx_type;
			if (!idx_type)
				idx_type = "BXTREE";
			rc = sos_schema_index_modify
				(schema,
				 t->attrs[i].name,
				 idx_type,
				 t->attrs[i].key_type,
				 t->attrs[i].idx_args);
			if (rc)
				goto err;
		}
	}
	return schema;
 err:
	__sos_schema_free(schema);
	return NULL;
}

/**
 * \brief Return the number of attributes in the schema.
 *
 * This function returns the number of attributes in the schema. See
 * the sos_schema_attr_by_id() function for an example that iterates
 * through all attributes defined in the schema.
 *
 * \param schema The schema handle.
 * \retval The number of attributes in the schema.
 */
int sos_schema_attr_count(sos_schema_t schema)
{
	return schema->data->attr_cnt;
}

/**
 * \brief Returns the schema's name
 * \param schema The schema handle.
 * \returns The schema's name.
 */
const char *sos_schema_name(sos_schema_t schema)
{
	return schema->data->name;
}

/**
 * \brief Return the schema's unique id
 *
 * Every schema has a unique Id that is stored along with the object
 * in the database. This a llows the object's type to be identified.
 *
 * \param schema The schema handle.
 * \retval The schema Id
 */
int sos_schema_id(sos_schema_t schema)
{
	return schema->data->id;
}

/**
 * \brief Return the number of schema in container
 *
 * \param sos The container handle
 * \retval The number of schema in the container
 */
size_t sos_schema_count(sos_t sos)
{
	return sos->schema_count;
}

static const char *key_types[] = {
	[SOS_TYPE_INT16] = "INT16",
	[SOS_TYPE_INT32] = "INT32",
	[SOS_TYPE_INT64] = "INT64",
	[SOS_TYPE_UINT16] = "UINT16",
	[SOS_TYPE_UINT32] = "UINT32",
	[SOS_TYPE_UINT64] = "UINT64",
	[SOS_TYPE_FLOAT] = "FLOAT",
	[SOS_TYPE_DOUBLE] = "DOUBLE",
	[SOS_TYPE_LONG_DOUBLE] = "LONG_DOUBLE",
	[SOS_TYPE_TIMESTAMP] = "TIMESTAMP",
	[SOS_TYPE_OBJ] = "NONE",
	[SOS_TYPE_STRUCT] = "MEMCMP",
	[SOS_TYPE_JOIN] = "COMPOUND",
	[SOS_TYPE_BYTE_ARRAY] = "MEMCMP",
	[SOS_TYPE_CHAR_ARRAY] = "STRING",
	[SOS_TYPE_INT16_ARRAY] = "INT16_ARRAY",
	[SOS_TYPE_INT32_ARRAY] = "INT32_ARRAY",
	[SOS_TYPE_INT64_ARRAY] = "INT64_ARRAY",
	[SOS_TYPE_UINT16_ARRAY] = "UINT16_ARRAY",
	[SOS_TYPE_UINT32_ARRAY] = "UINT32_ARRAY",
	[SOS_TYPE_UINT64_ARRAY] = "UINT64_ARRAY",
	[SOS_TYPE_FLOAT_ARRAY] = "FLOAT_ARRAY",
	[SOS_TYPE_DOUBLE_ARRAY] = "DOUBLE_ARRAY",
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = "LONG_DOUBLE_ARRAY",
	[SOS_TYPE_OBJ_ARRAY] = "NONE",
};

static sos_attr_t attr_new(sos_schema_t schema, sos_type_t type, int size)
{
	sos_attr_t attr = calloc(1, sizeof *attr);
	if (!attr)
		goto out;
	attr->data = &attr->data_;
	if (size)
		attr->data->size = size;
	else
		attr->data->size = type_sizes[type];
	attr->schema = schema;
	attr->size_fn = __sos_attr_size_fn_for_type(type);
	attr->to_str_fn = __sos_attr_to_str_fn_for_type(type);
	attr->from_str_fn = __sos_attr_from_str_fn_for_type(type);
	attr->strlen_fn = __sos_attr_strlen_fn_for_type(type);
	attr->key_value_fn = __sos_attr_key_value_fn_for_type(type);
	attr->idx_type = strdup("BXTREE");
	attr->key_type = strdup(key_types[type]);
 out:
	return attr;
}

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

/**
 * \brief Add an attribute to a schema
 *
 * Add an attribute to a schema. The schema can only be modified if it
 * is not already a member of the container.
 *
 *    rc = sos_schema_attr_add(s, "an_int32_attr", SOS_TYPE_INT32);
 *    rc = sos_schema_attr_add(s, "a_float_attr", SOS_TYPE_FLOAT);
 *    rc = sos_schema_attr_add(s, "a_struct_attr", SOS_TYPE_STRUCT, 24);   // 24B untyped value
 *    rc = sos_schema_attr_add(s, "a_float_array", SOS_TYPE_FLOAT_ARRAY);  // Arrays have strict element type, but variable length
 *    char *join[] = {
 *          "an_int32_attr", "a_float_attr"
 *    };
 *    rc = sos_schema_attr_add(s, "a_join_key", SOS_TYPE_JOIN, 2, join);   // An attribute that is composed of two other attributes
 *
 * \param schema	The schema
 * \param name		The attribute name
 * \param type		The attribute type
 *
 * If the type is SOS_TYPE_STRUCT:
 * \param size          The size in bytes of the attribute.
 *
 * If the type is SOS_TYPE_JOIN:
 * \param count         The number of attributes in this key
 * \param cmp_fn        Name of the key comparison function. If NULL, memcmp is used.
 * \param a_1-N         Count additional arguments that specify the attribute names comprising the key
 *
 * \retval 0		Success
 * \retval ENOMEM	Insufficient resources
 * \retval EEXIST	An attribute with that name already exists
 * \retval EINUSE	The schema is already a member of a container
 * \retval EINVAL	A parameter was invalid
 */
int sos_schema_attr_add(sos_schema_t schema, const char *name, sos_type_t type, ...)
{
	sos_attr_t attr;
	va_list ap;
	size_t size;
	int i, join_count;
	char **join_attrs;
	sos_array_t ext_ptr = NULL;

	if (schema->schema_obj)
		return EBUSY;

	/* Search the schema to see if the name is already in use */
	attr = _attr_by_name(schema, name);
	if (attr)
		return EEXIST;

	if (type > SOS_TYPE_LAST)
		return EINVAL;

	switch (type) {
	case SOS_TYPE_STRUCT:
		va_start(ap, type);
		size = va_arg(ap, size_t);
		size = (size + 3) & ~3; /* round up to a multiple of 4B */
		if (!size)
			return EINVAL;
		break;
	case SOS_TYPE_JOIN:
		va_start(ap, type);
		join_count = va_arg(ap, size_t);
		join_attrs = va_arg(ap, char **);
		size = 0;	/* JOIN attrs consume no space in the object */
		ext_ptr = malloc(SOS_ARRAY_SIZE(join_count, uint32_t));
		if (!ext_ptr)
			return ENOMEM;
		ext_ptr->count = join_count;
		for (i = 0; i < join_count; i++) {
			int join_type;
			sos_attr_t join_attr = sos_schema_attr_by_name(schema, join_attrs[i]);
			if (!join_attr) {
				free(ext_ptr);
				return ENOENT;
			}
			join_type = sos_attr_type(join_attr);
			if (join_type == SOS_TYPE_JOIN) {
				/* Join types cannot be nested */
				free(ext_ptr);
				return EINVAL;
			}
			ext_ptr->data.uint32_[i] = sos_attr_id(join_attr);
		}
		break;
	default:
		size = 0;
		break;
	}
	attr = attr_new(schema, type, size);
	if (!attr) {
		if (ext_ptr)
			free(ext_ptr);
		return ENOMEM;
	}
	strcpy(attr->data->name, name);
	attr->data->type = type;
	attr->data->id = schema->data->attr_cnt++;
	attr->ext_ptr = ext_ptr;
	if (!TAILQ_EMPTY(&schema->attr_list)) {
		sos_attr_t prev = TAILQ_LAST(&schema->attr_list, sos_attr_list);
		attr->data->offset = prev->data->offset + prev->data->size;
	} else {
		attr->data->offset = sizeof(struct sos_obj_data_s);
	}

	uint32_t a_size = sos_attr_size(attr);
	if (a_size > schema->data->key_sz)
		schema->data->key_sz = a_size;
	schema->data->obj_sz = attr->data->offset + attr->data->size;

	/* Append new attribute to tail of list */
	TAILQ_INSERT_TAIL(&schema->attr_list, attr, entry);
	return 0;
}

/**
 * \brief Add an index to an attribute
 *
 * Marks an attribute as having a key-value index. The index is not
 * actually created until the schema containing the attribute is added
 * to the container.
 *
 * \param schema	The schema handle
 * \param attr_name	The attribute name
 * \retval 0		The index was succesfully added.
 * \retval ENOENT	The specified attribute does not exist.
 * \retval EINVAL	One or more parameters was invalid.
 */
int sos_schema_index_add(sos_schema_t schema, const char *attr_name)
{
	sos_attr_t attr;

	if (schema->schema_obj)
		return EBUSY;

	/* Find the attribute */
	attr = _attr_by_name(schema, attr_name);
	if (!attr)
		return ENOENT;

	/* Protect against adding it to the idx_list twice */
	if (!attr->data->indexed) {
		TAILQ_INSERT_TAIL(&schema->idx_attr_list, attr, idx_entry);
		attr->data->indexed = 1;
		return 0;
	}
	return EEXIST;
}

static void __toupper(char *s)
{
	while (s[0]) {
		*s = toupper(*s);
		s++;
	}
}

/**
 * \brief Modify the index for an attribute
 *
 * By default an attribute index is a modified form of a B+Tree that
 * efficiently handles duplicate keys. There are, however, other index
 * types as well as user-defined indices. The type of index is
 * specified as a string that identifies a shared library that
 * implements the necessary index strategy routines, e.g. insert,
 * delete, etc...
 *
 * For keys, the default key type is associated with the attribute's
 * data type, however, it is possible to implement user-defined
 * key types. These are useful for indexing complex data types that
 * are not understood as primitive types; for example a set of fields
 * that are represented as a UINT64.

 * \param schema	The schema handle.
 * \param name		The attribute name.
 * \param idx_type	The index type name. This parameter cannot be null.
 * \param key_type	The key type name. It may be null, in which
 *			case, the default key type for the attribute will be used.
 * \param ...		Some index types have additional parameters,
 *			for example, the BXTREE has an order parameter
 *			that specifies the number of entries in each BXTREE node.
 * \retval 0		The index was succesfully added.
 * \retval ENOENT	The specified attribute does not exist.
 * \retval EINVAL	One or more parameters was invalid.
 */
int sos_schema_index_modify(sos_schema_t schema, const char *name,
			    const char *idx_type, const char *key_type,
			    const char *idx_args)
{
	sos_attr_t attr;

	if (schema->schema_obj)
		return EBUSY;

	/* Find the attribute */
	attr = _attr_by_name(schema, name);
	if (!attr)
		return ENOENT;

	if (idx_type) {
		if (attr->idx_type)
			free(attr->idx_type);
		attr->idx_type = strdup(idx_type);
		__toupper(attr->idx_type);
	}
	if (key_type) {
		if (attr->key_type)
			free(attr->key_type);
		attr->key_type = strdup(key_type);
		__toupper(attr->key_type);
	}
	if (idx_args) {
		if (attr->idx_args) {
			free(attr->idx_args);
		}
		attr->idx_args = strdup(idx_args);
	} else {
		attr->idx_args = NULL;
	}
	return 0;
}

/**
 * \brief Find an attribute by name
 * \param schema	The schema handle
 * \param name		The attribute's name
 * \returns The attribute handle or NULL if the attribute was not found.
 */
sos_attr_t sos_schema_attr_by_name(sos_schema_t schema, const char *name)
{
	return _attr_by_name(schema, name);
}

/**
 * \brief Find an attribute by id
 *
 * This function is useful for iterating through all attributes in the
 * schema as shown in the following code fragment:
 *
 *     for (i = 0; i < sos_schema_attr_count(schema); i++) {
 *        sos_attr_t attr = sos_schema_attr_by_id(i);
 *        ... code to manipulate attribute ...
 *     }
 *
 * \param schema	The schema handle
 * \param attr_id	The attribute's ordinal id
 * \returns The attribute handle or NULL if the attribute was not found.
 */
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

/**
 * \brief Return the attribute's type
 * \returns The attribute type
 */
sos_type_t sos_attr_type(sos_attr_t attr)
{
	return attr->data->type;
}

/**
 * \brief Return the attribute's ordinal ID.
 *
 * \param attr	The attribute handle.
 * \returns The attribute id.
 */
int sos_attr_id(sos_attr_t attr)
{
	return attr->data->id;
}

/**
 * \brief Return the attribute's name
 * \returns The attribute name
 */
const char *sos_attr_name(sos_attr_t attr)
{
	return attr->data->name;
}

/**
 * \brief Return an array with the list of attributes in the join list
 *
 * The function will return NULL if the attribute is not of the type
 * SOS_TYPE_JOIN.
 *
 *    retval->count                    - The attribute count.
 *    retval->data.uint32_[0..count-1] - The id of each attribute
 *
 * \param attr The attribute handle
 * \returns A pointer to an array defining the join list.
 */
sos_array_t sos_attr_join_list(sos_attr_t attr)
{
	return attr->ext_ptr;
}

/**
 * \brief Return the schema of an attribute
 *
 * \param attr The attribute handle
 * \returns The schema handle
 */
sos_schema_t sos_attr_schema(sos_attr_t attr)
{
	return attr->schema;
}

sos_schema_t __sos_get_ischema(sos_type_t type)
{
	assert(type >= SOS_TYPE_OBJ && type <= SOS_TYPE_LAST);
	return ischema_dir[type];
}

/**
 * \brief Return !0 if the attribute is a reference
 */
int sos_attr_is_ref(sos_attr_t attr)
{
	return attr->data->type >= SOS_TYPE_OBJ;
}

/**
 * \brief Return !0 if the attribute is an array
 */
int sos_attr_is_array(sos_attr_t attr)
{
	return attr->data->type >= SOS_TYPE_ARRAY;
}

/**
 * \brief Return a pointer to the object's data
 *
 * This function returns a pointer to the object's internal data. The
 * application is responsible for understanding the internal
 * format. The application must call sos_obj_put() when finished with
 * the object to avoid a memory leak.
 *
 * This function is intended to be used when the schema of the object
 * is well known by the application. If the application is generic for
 * all objects, see the sos_value() functions.
 *
 * \param obj
 * \retval void * pointer to the objects's data
 */
void *sos_obj_ptr(sos_obj_t obj)
{
	sos_obj_data_t data = obj->obj->as.ptr;
	return data->data;
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
ods_obj_t __sos_obj_new(ods_t ods, size_t size, pthread_mutex_t *lock)
{
	ods_obj_t ods_obj;
	size_t extend_size = (size < SOS_ODS_EXTEND_SZ ? SOS_ODS_EXTEND_SZ : size << 4);
	ods_obj = ods_obj_alloc(ods, size);
	if (!ods_obj) {
		int rc = ods_extend(ods, extend_size);
		if (rc)
			goto err_0;
		ods_obj = ods_obj_alloc(ods, size);
	}
 err_0:
	return ods_obj;
}

sos_value_t sos_array_new(sos_value_t val, sos_attr_t attr, sos_obj_t obj, size_t count)
{
	size_t size;
	struct sos_array_s *array;
	ods_obj_t array_obj;
	sos_schema_t schema;
	if (!sos_attr_is_array(attr)) {
		errno = EINVAL;
		return NULL;
	}
	schema = __sos_get_ischema(attr->data->type);
	size = sizeof(struct sos_obj_data_s)
		+ sizeof(uint32_t) /* element count */
		+ (count * schema->data->el_sz); /* array elements */

	if (!obj) {
		array = calloc(1, size);
		if (!array)
			goto err;
		array->count = count;
		val->attr = attr;
		val->data = (sos_value_data_t)array;
		val->obj = NULL;
		return val;
	}
	array_obj = __sos_obj_new(ods_obj_ods(obj->obj), size, &obj->sos->lock);
	if (!array_obj)
		goto err;

	val->attr = attr;
	val->data = (sos_value_data_t)&obj->obj->as.bytes[attr->data->offset];

	array = (struct sos_array_s *)&SOS_OBJ(array_obj)->data[0];
	array->count = count;

	/* Free the old array contents if present */
	if (val->data->prim.ref_.ref.obj) {
		ods_t ods = __sos_ods_from_ref(obj->sos, val->data->prim.ref_.ref.ods);
		ods_ref_delete(ods, val->data->prim.ref_.ref.obj);
	}
	/* Update the array reference in the containing object */
	val->data->prim.ref_.ref.ods = obj->obj_ref.ref.ods;
	val->data->prim.ref_.ref.obj = ods_obj_ref(array_obj);

	/* Point the value at the new array */
	val->obj = __sos_init_obj(obj->sos, schema, array_obj, val->data->prim.ref_);
	if (!val->obj)
		goto err;
	val->data = (sos_value_data_t)&SOS_OBJ(array_obj)->data[0];
	return val;
 err:
	errno = ENOMEM;
	return NULL;
}

sos_obj_t sos_array_obj_new(sos_t sos, sos_type_t type, size_t count)
{
	ods_obj_t array_obj;
	sos_schema_t schema;
	sos_part_t part;
	size_t size;

	schema = __sos_get_ischema(type);
	if (!schema)
		return NULL;
	size = sizeof(struct sos_obj_data_s)
		+ sizeof(uint32_t) /* element count */
		+ (count * schema->data->el_sz); /* array elements */

	part = __sos_primary_obj_part(sos);
	array_obj = __sos_obj_new(part->obj_ods, size, &sos->lock);
	if (!array_obj)
		return NULL;

	struct sos_array_s *array = (struct sos_array_s *)&SOS_OBJ(array_obj)->data[0];
	array->count = count;
	union sos_obj_ref_s obj_ref = {
		.ref = {
			SOS_PART(part->part_obj)->part_id,
			ods_obj_ref(array_obj)
		}
	};
	return __sos_init_obj(sos, schema, array_obj, obj_ref);
}

/**
 * \brief Test if an attribute has an index.
 *
 * \param attr	The sos_attr_t handle
 * \returns !0 if the attribute has an index
 */
sos_index_t sos_attr_index(sos_attr_t attr)
{
	return attr->index;
}

/**
 * \brief Return the size of an attribute's data
 *
 * \param attr The sos_attr_t handle
 * \returns The size of the attribute's data in bytes
 */
size_t sos_attr_size(sos_attr_t attr)
{
	return attr->data->size;
}

/**
 * \brief Return !0 if the value is an array
 */
int sos_value_is_array(sos_value_t v)
{
	return v->attr->data->type >= SOS_TYPE_ARRAY;
}

/**
 * \brief Return the value's type
 * \returns The type
 */
sos_type_t sos_value_type(sos_value_t v)
{
	return v->attr->data->type;
}

/**
 * \brief Get the size of an attribute value
 *
 * \param value The value handle
 *
 * \returns The size of the attribute value in bytes
 */
size_t sos_value_size(sos_value_t value)
{
	return value->attr->size_fn(value);
}

void *sos_value_as_key(sos_value_t value)
{
	return value->attr->key_value_fn(value);
}

/**
 * \brief Return the length of the string required for the object value
 *
 * This function returns the size of the buffer required to hold the
 * object attribute value if formatted as a string. This function is
 * useful when allocating buffers used with the sos_obj_attr_to_str()
 * function.

 * \param obj The object handle
 * \param attr The attribute handle
 * \returns The size of the string in bytes.
 */
size_t sos_obj_attr_strlen(sos_obj_t obj, sos_attr_t attr)
{
	size_t size;
	struct sos_value_s v_;
	sos_value_t v = sos_value_init(&v_, obj, attr);
	if (!v)
		return 1;
	size = sos_value_strlen(v);
	sos_value_put(v);
	return size;
}

/**
 * \brief Format an object attribute as a string
 *
 * \param obj The object handle
 * \param attr The attribute handle
 * \param str Pointer to the string to receive the formatted value
 * \param len The size of the string in bytes.
 * \returns A pointer to the str argument or NULL if there was a
 *          formatting error.
 */
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

/**
 * \brief Set the object attribute from a string
 *
 * \param obj The object handle
 * \param attr The attribute handle
 * \param str The input string value to parse
 * \param endptr Receives the point in the str argumeent where parsing stopped.
 *               This parameter may be NULL.
 * \retval 0 The string was successfully parsed and the value set
 * \retval EINVAL The string was incorrectly formatted for this value
 *                type.
 */
int sos_obj_attr_from_str(sos_obj_t obj, sos_attr_t attr, const char *str, char **endptr)
{
	int rc;
	sos_value_t v;
	struct sos_value_s v_;
	size_t sz;

	if (!sos_attr_is_array(attr)) {
		v = sos_value_init(&v_, obj, attr);
		if (v) {
			rc = sos_value_from_str(v, str, endptr);
			sos_value_put(v);
		} else
			rc = EINVAL;
		return rc;
	}
	/*
	 * Char has no delimiters. All other types have delimeters in
	 * the input string. Delimters considered are [,:_]. Note that
	 * potential delimiters such as '-' and '.' will not work with
	 * numbers in the general case.
	 * TODO: Make the delimiters a container configuration option
	 */
	if (sos_attr_type(attr) == SOS_TYPE_CHAR_ARRAY) {
		sz = strlen(str) + 1;
	} else {
		const char *p = str;
		char *q;
		const char *delim = ",:_";
		for (sz = 1, q = strpbrk(p, delim);
		     q; q = strpbrk(p, delim)) {
			sz += 1;
			p = q + 1;
		}
	}
	rc = ENOMEM;
	v = sos_value_init(&v_, obj, attr);
	if (v && sos_array_count(v) < sz) {
		/* Too short, delete and re-alloc */
		sos_obj_delete(v->obj);
		sos_obj_put(v->obj);
		v = NULL;
	}
	if (!v) {
		/* The array has not been allocated yet, or was too short */
		v = sos_array_new(&v_, attr, obj, sz);
	}
	if (v) {
		rc = sos_value_from_str(v, str, endptr);
		sos_value_put(v);
	}
	return rc;
}

/**
 * \brief Create a copy of a schema
 *
 * Create a replica of a schema that is not associated with a
 * container. This function is useful for adding a schema from one
 * container to another container.
 *
 * \param schema
 * \retval A pointer to the new schema
 * \retval NULL Insufficient resources
 */
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
	TAILQ_INIT(&dup->idx_attr_list);
	dup->data = &dup->data_;
	*dup->data = *schema->data;
	dup->dict = calloc(dup->data->attr_cnt, sizeof(sos_attr_t));
	if (!dup->dict)
		goto err_0;
	idx = 0;
	TAILQ_FOREACH(src_attr, &schema->attr_list, entry) {
		attr = attr_new(dup, src_attr->data->type, src_attr->data->size);
		if (!attr)
			goto err_1;
		schema->dict[idx++] = attr;
		*attr->data = *src_attr->data;
		if (attr->data->indexed)
			TAILQ_INSERT_TAIL(&dup->idx_attr_list, attr, idx_entry);
		if (src_attr->ext_ptr) {
			size_t join_ext_sz = SOS_ARRAY_SIZE(src_attr->ext_ptr->count, uint32_t);
			sos_array_t join_ext = malloc(join_ext_sz);
			if (!join_ext)
				goto err_1;
			memcpy(join_ext, src_attr->ext_ptr, join_ext_sz);
			attr->ext_ptr = join_ext;
		}
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
		if (attr->ext_ptr)
			free(attr->ext_ptr);
		free(attr->key_type);
		free(attr->idx_type);
		free(attr);
	}
 err_0:
	free(dup);
	return NULL;
}

sos_schema_t __sos_schema_init(sos_t sos, ods_obj_t schema_obj)
{
	sos_attr_t attr;
	sos_schema_t schema;
	int idx;
	ods_obj_t ext_obj;
	sos_array_t ext;
	size_t ext_sz;

	if (!schema_obj)
		return NULL;
	schema = calloc(1, sizeof(*schema));
	if (!schema)
		return NULL;
	schema->ref_count = 1;
	TAILQ_INIT(&schema->attr_list);
	TAILQ_INIT(&schema->idx_attr_list);
	schema->schema_obj = schema_obj;
	schema->sos = sos;
	schema->data = schema_obj->as.ptr;
	schema->dict = calloc(schema->data->attr_cnt, sizeof(sos_attr_t));
	if (!schema->dict)
		goto err_0;
	for (idx = 0; idx < schema->data->attr_cnt; idx++) {
		attr = attr_new(schema, schema->data->attr_dict[idx].type,
				schema->data->attr_dict[idx].size);
		if (!attr)
			goto err_1;
		schema->dict[idx] = attr;
		attr->data = &schema->data->attr_dict[idx];
		if (attr->data->indexed)
			TAILQ_INSERT_TAIL(&schema->idx_attr_list, attr, idx_entry);
		if (attr->data->ext_ref) {
			ext_obj = ods_ref_as_obj(ods_obj_ods(schema_obj), attr->data->ext_ref);
			if (!ext_obj) {
				errno = EPROTO;
				goto err_1;
			}
			ext = ODS_PTR(sos_array_t, ext_obj);
			ext_sz = SOS_ARRAY_SIZE(ext->count, uint32_t);
			attr->ext_ptr = malloc(ext_sz);
			if (!attr->ext_ptr) {
				errno = ENOMEM;
				goto err_1;
			}
			/* Instantiate the array and place a memory copy in attr */
			memcpy(attr->ext_ptr, ext, ext_sz);
			ods_obj_put(ext_obj);
		}
		TAILQ_INSERT_TAIL(&schema->attr_list, attr, entry);
	}
	rbn_init(&schema->name_rbn, schema->data->name);
	rbt_ins(&sos->schema_name_rbt, &schema->name_rbn);
	rbn_init(&schema->id_rbn, &schema->data->id);
	rbt_ins(&sos->schema_id_rbt, &schema->id_rbn);
	sos->schema_count++;
	LIST_INSERT_HEAD(&sos->schema_list, schema, entry);
	return schema;
 err_1:
	free(schema->dict);
	while (!TAILQ_EMPTY(&schema->attr_list)) {
		attr = TAILQ_FIRST(&schema->attr_list);
		TAILQ_REMOVE(&schema->attr_list, attr, entry);
		if (attr->ext_ptr)
			free(attr->ext_ptr);
		free(attr->key_type);
		free(attr->idx_type);
		free(attr);
	}
	ods_obj_put(schema_obj);
 err_0:
	free(schema);
	return NULL;
}

int __sos_schema_open(sos_t sos, sos_schema_t schema)
{
	int rc;
	sos_attr_t attr;
	char idx_name[SOS_SCHEMA_NAME_LEN + SOS_ATTR_NAME_LEN + 2];

	TAILQ_FOREACH(attr, &schema->idx_attr_list, idx_entry) {
		assert(attr->data->indexed);
	retry:
		sprintf(idx_name, "%s_%s", schema->data->name, attr->data->name);
		attr->index = sos_index_open(sos, idx_name);
		if (!attr->index) {
			if (errno != ENOENT) {
				rc = errno;
				goto err;
			}
			rc = sos_index_new(sos, idx_name,
					   attr->idx_type, attr->key_type,
					   attr->idx_args);
			if (rc)
				goto err;
			goto retry;
		}
	}
	return 0;
 err:
	return rc;
}

static sos_schema_t __schema_by_name(sos_t sos, const char *name)
{
	ods_idx_data_t data;
	sos_obj_ref_t obj_ref;
	ods_obj_t schema_obj;
	sos_schema_t schema = NULL;
	SOS_KEY(key);

	ods_key_set(key, name, strlen(name)+1);
	if (0 == ods_idx_find(sos->schema_idx, key, &data)) {
		obj_ref.idx_data = data;
		schema_obj = ods_ref_as_obj(sos->schema_ods, obj_ref.ref.obj);
		schema = __sos_schema_init(sos, schema_obj);
	}
	return schema;
}

static sos_schema_t __sos_schema_by_name(sos_t sos, const char *name)
{
	sos_schema_t schema = NULL;
	struct rbt *tree;
	struct rbn *rbn;

	if (name[0] == '_' && name[1] == '_')
		tree = &ischema_rbt;
	else
		tree = &sos->schema_name_rbt;

	rbn = rbt_find(tree, (void *)name);
	if (!rbn) {
		/* Schema not in cache. Check persistent storage. */
		schema = __schema_by_name(sos, name);
		goto out;
	}
	schema = container_of(rbn, struct sos_schema_s, name_rbn);
out:
	return schema;
}

/**
 * \brief Find the schema with the specified name
 *
 * Find the schema with the specified name. The schema returned may be
 * used to create objects of this type in the container.
 *
 * \param sos	The container handle
 * \param name	The name of the schema
 * \returns A pointer to the schema or a NULL pointer if a schema with
 * that name does not exist in the container.
 */
sos_schema_t sos_schema_by_name(sos_t sos, const char *name)
{
	sos_schema_t schema;

	ods_lock(sos->schema_ods, 0, NULL);
	schema = __sos_schema_by_name(sos, name);
	ods_unlock(sos->schema_ods, 0);

	return schema;
}

/**
 * \brief Find a schema by id
 *
 * Each schema is assigned a unique identifier when it is created.
 * Find the schema with the specified identifier. The schema returned may be
 * used to create objects of this type in the container.
 *
 * \param sos	The container handle
 * \param id	The schema id
 * \retval A pointer to the schema
 * \retval NULL pointer if a schema with that name does not exist in the container
 */
sos_schema_t sos_schema_by_id(sos_t sos, uint32_t id)
{
	sos_schema_t schema;
	if (id < SOS_SCHEMA_FIRST_USER)
		return ischema_dir[id];
	struct rbn *rbn = rbt_find(&sos->schema_id_rbt, (void *)&id);
	if (!rbn)
		return NULL;
	schema = container_of(rbn, struct sos_schema_s, id_rbn);
	return schema;
}

/**
 * \brief Add a schema to a container
 *
 * A a new schema to the container. A schema with the same name must
 * not already exist in the container.
 *
 * \param sos	The container handle
 * \param schema Pointer to the new schema
 * \retval 0	The schema was successfully added
 * \retval EEXIST A schema with the same name already exists in the
 * container.
 * \retval ENOMEM Insufficient resources
 * \retval EINVAL An invalid parameter was specified.
 */
int sos_schema_add(sos_t sos, sos_schema_t schema)
{
	size_t size, key_len;
	ods_obj_t schema_obj;
	ods_key_t schema_key;
	sos_attr_t attr;
	int idx, rc;
	ods_obj_t udata;
	struct stat sb;

	/* See if this schema is already part of a container */
	if (schema->schema_obj)
		return EBUSY;

	ods_lock(sos->schema_ods, 0, NULL);

	/* Check to see if a schema by this name is already in the container */
	if (__sos_schema_by_name(sos, schema->data->name)) {
		rc = EEXIST;
		goto err_0;
	}

	udata = ods_get_user_data(sos->schema_ods);
	if (!udata) {
		rc= ENOMEM;
		goto err_0;
	}

	rc = ods_stat(sos->schema_ods, &sb);
	if (rc) {
		rc = errno;
		goto err_1;
	}

	/* Compute the size of the schema data */
	size = sizeof(struct sos_schema_data_s);
	size += schema->data->attr_cnt * sizeof(struct sos_attr_data_s);
	schema_obj = __sos_obj_new(sos->schema_ods, size, &sos->lock);
	if (!schema_obj) {
		rc = errno;
		goto err_1;
	}
	sos_obj_ref_t sos_obj_ref = {
		.ref = {
			0,
			ods_obj_ref(schema_obj)
		}
	};

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
	SOS_SCHEMA(schema_obj)->el_sz = schema->data->el_sz;
	SOS_SCHEMA(schema_obj)->attr_cnt = schema->data->attr_cnt;
	SOS_SCHEMA(schema_obj)->id =
		ods_atomic_inc(&SOS_SCHEMA_UDATA(udata)->last_schema_id);

	idx = 0;
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
		if (attr->ext_ptr) {
			int i;
			size_t ext_sz = SOS_ARRAY_SIZE(attr->ext_ptr->count, uint32_t);
			ods_obj_t ext_obj = ods_obj_alloc(ods_obj_ods(schema_obj), ext_sz);
			sos_array_t join = ODS_PTR(sos_array_t, ext_obj);
			join->count = attr->ext_ptr->count;
			for (i = 0; i < attr->ext_ptr->count; i++)
				join->data.uint32_[i] = attr->ext_ptr->data.uint32_[i];
			attr_data->ext_ref = ods_obj_ref(ext_obj);
			ods_obj_put(ext_obj);
		}
		attr->data = attr_data;
		if (attr->data->indexed && !attr->key_type) {
			rc = EINVAL;
			goto err_3;
		}
		schema->dict[idx++] = attr;
	}
	schema->data = schema_obj->as.ptr;
	rc = ods_idx_insert(sos->schema_idx, schema_key, sos_obj_ref.idx_data);
	if (rc)
		goto err_3;

	rbn_init(&schema->name_rbn, schema->data->name);
	rbt_ins(&sos->schema_name_rbt, &schema->name_rbn);

	rbn_init(&schema->id_rbn, &schema->data->id);
	rbt_ins(&sos->schema_id_rbt, &schema->id_rbn);

	sos->schema_count++;
	LIST_INSERT_HEAD(&sos->schema_list, schema, entry);

	ods_obj_put(schema_key);
	ods_obj_put(udata);
	ods_unlock(sos->schema_ods, 0);
	return __sos_schema_open(sos, schema);
 err_3:
	ods_obj_delete(schema_key);
	ods_obj_put(schema_key);
 err_2:
	ods_obj_delete(schema_obj);
	ods_obj_put(schema_obj);
 err_1:
	ods_obj_put(udata);
 err_0:
	ods_unlock(sos->schema_ods, 0);
	return rc;
}

/**
 * \brief Return a handle to the first schema in the container
 *
 * \param sos The SOS handle
 * \retval The schema pointer
 * \retval NULL If there are no schema defined in the container
 */
sos_schema_t sos_schema_first(sos_t sos)
{
	sos_schema_t first = LIST_FIRST(&sos->schema_list);
	if (first)
		return first;
	return NULL;
}

/**
 * \brief Return a handle to the next schema in the container
 *
 * \param schema The schema handle returned by sos_schema_first() or sos_schema_next()
 * \retval The schema pointer
 * \retval NULL If there are no more schema defined in the container
 */
sos_schema_t sos_schema_next(sos_schema_t schema)
{
	sos_schema_t next = LIST_NEXT(schema, entry);
	if (next)
		return next;
	return NULL;
}

/**
 * \brief Remove a schema from a container
 *
 * Remove the schema with the specified name from the container. If
 * there are any objects in the container that use the specified
 * schema, the schema cannot be deleted.
 *
 * \param sos	The container handle
 * \param name	The name of the container
 * \retval 0	The schema was successfully deleted
 * \retval EINUSE An objects exists in the container that uses the
 * specified schema
 * \retval ENOENT No schema was found with the specified name
 */
int sos_schema_delete(sos_t sos, const char *name)
{
	return ENOSYS;
}

static const char *type_names[] = {
	[SOS_TYPE_INT16] = "INT16",
	[SOS_TYPE_INT32] = "INT32",
	[SOS_TYPE_INT64] = "INT64",
	[SOS_TYPE_UINT16] = "UINT16",
	[SOS_TYPE_UINT32] = "UINT32",
	[SOS_TYPE_UINT64] = "UINT64",
	[SOS_TYPE_FLOAT] = "FLOAT",
	[SOS_TYPE_DOUBLE] = "DOUBLE",
	[SOS_TYPE_LONG_DOUBLE] = "LONG_DOUBLE",
	[SOS_TYPE_TIMESTAMP] = "TIMESTAMP",
	[SOS_TYPE_OBJ] = "OBJ",
	[SOS_TYPE_STRUCT] = "STRUCT",
	[SOS_TYPE_JOIN] = "JOIN",
	[SOS_TYPE_BYTE_ARRAY] = "BYTE_ARRAY",
	[SOS_TYPE_CHAR_ARRAY] = "CHAR_ARRAY",
	[SOS_TYPE_INT16_ARRAY] = "INT16_ARRAY",
	[SOS_TYPE_INT32_ARRAY] = "INT32_ARRAY",
	[SOS_TYPE_INT64_ARRAY] = "INT64_ARRAY",
	[SOS_TYPE_UINT16_ARRAY] = "UINT16_ARRAY",
	[SOS_TYPE_UINT32_ARRAY] = "UINT32_ARRAY",
	[SOS_TYPE_UINT64_ARRAY] = "UINT64_ARRAY",
	[SOS_TYPE_FLOAT_ARRAY] = "FLOAT_ARRAY",
	[SOS_TYPE_DOUBLE_ARRAY] = "DOUBLE_ARRAY",
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = "LONG_DOUBLE_ARRAY",
	[SOS_TYPE_OBJ_ARRAY] = "OBJ_ARRAY",
};

static const char *type_name(sos_type_t t)
{
	if (t <= SOS_TYPE_LAST)
		return type_names[t];
	return "corrupted!";
}

/**
 * \brief Print the schema to a File pointer
 *
 * This convenience function formats the schema definition in YAML and
 * writes it to the specified FILE pointer. For example:
 *
 *     sos_schema_t schema = sos_container_find(sos, "Sample");
 *     sos_schema_print(schema, stdout);
 *
 * \param schema The schema handle
 * \param fp The output FILE pointer
 */
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

/** @} */

sos_schema_t __sos_internal_schema_new(const char *name, uint32_t id,
				       sos_type_t el_type, size_t el_size)
{
	struct sos_schema_s *schema;
	schema = sos_schema_new(name);
	assert(schema);
	schema->flags = SOS_SCHEMA_F_INTERNAL;
	schema->ref_count = 1;
	schema->data->id = id;
	schema->data->el_sz = el_size;
	sos_schema_attr_add(schema, "count", SOS_TYPE_UINT32);
	sos_schema_attr_add(schema, "data", el_type);
	rbn_init(&schema->name_rbn, schema->data->name);
	rbt_ins(&ischema_rbt, &schema->name_rbn);
	rbn_init(&schema->id_rbn, &schema->data->id);
	ischema_dir[schema->data->id] = schema;
	return schema;
}

struct ischema_data {
	const char *name;
	int id;
	sos_type_t el_type;
	size_t el_size;
} ischema_data_[] = {
	{ "__BYTE_ARRAY_OBJ", SOS_ISCHEMA_BYTE_ARRAY,
	  SOS_TYPE_BYTE_ARRAY, sizeof(char) },

	{ "__CHAR_ARRAY_OBJ", SOS_ISCHEMA_CHAR_ARRAY,
	  SOS_TYPE_CHAR_ARRAY, sizeof(char) },

	{ "__INT16_ARRAY_OBJ", SOS_ISCHEMA_INT16_ARRAY,
	  SOS_TYPE_INT16_ARRAY, sizeof(int16_t) },

	{ "__UINT16_ARRAY_OBJ", SOS_ISCHEMA_UINT16_ARRAY,
	  SOS_TYPE_UINT16_ARRAY, sizeof(uint16_t) },

	{ "__INT32_ARRAY_OBJ", SOS_ISCHEMA_INT32_ARRAY,
	  SOS_TYPE_INT32_ARRAY, sizeof(int32_t) },

	{ "__UINT32_ARRAY_OBJ", SOS_ISCHEMA_UINT32_ARRAY,
	  SOS_TYPE_UINT32_ARRAY, sizeof(uint32_t) },

	{ "__INT64_ARRAY_OBJ", SOS_ISCHEMA_INT64_ARRAY,
	  SOS_TYPE_INT64_ARRAY, sizeof(int64_t) },

	{ "__UINT64_ARRAY_OBJ", SOS_ISCHEMA_UINT64_ARRAY,
	  SOS_TYPE_UINT64_ARRAY, sizeof(uint64_t) },

	{ "__FLOAT_ARRAY_OBJ", SOS_ISCHEMA_FLOAT_ARRAY,
	  SOS_TYPE_FLOAT_ARRAY, sizeof(float) },

	{ "__DOUBLE_ARRAY_OBJ", SOS_ISCHEMA_DOUBLE_ARRAY,
	  SOS_TYPE_DOUBLE_ARRAY, sizeof(double) },

	{ "__LONG_DOUBLE_ARRAY_OBJ", SOS_ISCHEMA_LONG_DOUBLE_ARRAY,
	  SOS_TYPE_LONG_DOUBLE_ARRAY, sizeof(long double) },

	{ "__OBJ_ARRAY_OBJ", SOS_ISCHEMA_OBJ_ARRAY,
	  SOS_TYPE_OBJ_ARRAY, sizeof(sos_obj_ref_t) },

	{ NULL, },
};

static void __attribute__ ((constructor)) sos_lib_init(void)
{
	struct ischema_data *id;
	rbt_init(&ischema_rbt, __sos_schema_name_cmp);
	for (id = &ischema_data_[0]; id->name; id++) {
		sos_schema_t schema =
			__sos_internal_schema_new(id->name, id->id,
						  id->el_type, id->el_size);
		assert(schema);
	}
}
