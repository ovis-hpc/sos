/*
 * Copyright (c) 2013 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2013 Sandia Corporation. All rights reserved.
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
typedef struct sos_schema_s *sos_schema_t;
typedef struct sos_obj_s *sos_obj_t;

#define SOS_CONTAINER_NAME_LEN  64
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

struct sos_array_s {
	uint32_t count;
	union sos_array_element_u data;
};

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
 * with the sos_schema_find() function.
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
sos_schema_t sos_schema_new(const char *name);

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
sos_schema_t sos_schema_dup(sos_schema_t schema);

/**
 * \brief Return number of schema in container
 *
 * \param sos The container handle
 * \retval The number of schema in the container
 */
size_t sos_schema_count(sos_t sos);

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
int sos_schema_add(sos_t sos, sos_schema_t schema);

/**
 * \brief Find a schema in a container.
 *
 * Find the schema with the specified name. The schema returned may be
 * used to create objects of this type in the container.
 *
 * \param sos	The container handle
 * \param name	The name of the schema
 * \returns A pointer to the schema or a NULL pointer if a schema with
 * that name does not exist in the container.
 */
sos_schema_t sos_schema_find(sos_t sos, const char *name);
sos_schema_t sos_schema_by_name(sos_t sos, const char *name);
sos_schema_t sos_schema_by_id(sos_t sos, const int id);

/**
 * \brief Print the schema to a File pointer
 *
 * This convenience function formats the schema definition in YAML an
 * writes it to the specified FILE pointer. For example:
 *
 *     sos_schema_t schema = sos_container_find(sos, "Sample");
 *     sos_schema_print(schema, stdout);
 *
 * \param schema The schema handle
 * \param fp A FILE pointer
 */
void sos_schema_print(sos_schema_t schema, FILE *fp);

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
int sos_schema_delete(sos_t sos, const char *name);

sos_schema_t sos_schema_first(sos_t sos);
sos_schema_t sos_schema_next(sos_schema_t schema);

/**
 * \brief Take a reference on a schema
 *
 * SOS schema are reference counted. This function takes a reference
 * on a SOS schema and returns a pointer to the same. The typical
 * calling sequence is:
 *
 *     sos_schema_t my_schema_ptr = sos_schema_get(schema);
 *
 *
 * This allows for the schema to be safely pointed to from multiple
 * places. The sos_schema_put() function is used to drop a reference on
 * the schema. For example:
 *
 *     sos_schema_put(my_schema_ptr);
 *     my_schema_ptr = NULL;
 *
 * \param schema	The SOS schema handle
 * \returns The schema handle
 */
sos_schema_t sos_schema_get(sos_schema_t schema);

/**
 * \brief Drop a reference on a schema
 *
 * The memory consumed by a schema is not released until all
 * references have been dropped.
 *
 * \param schema	The schema handle
 */
void sos_schema_put(sos_schema_t schema);

/**
 * \brief Returns the schema's name
 * \param schema The schema handle.
 * \returns The schema's name.
 */
const char *sos_schema_name(sos_schema_t schema);
int sos_schema_id(sos_schema_t schema);

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
int sos_schema_attr_count(sos_schema_t schema);

/**
 * \brief Add an attribute to a schema
 *
 * Add an attribute to a schema. The schema can only be modified if it
 * is not already a member of the container.
 *
 * \param schema	The schema
 * \param name		The attribute name
 * \param type		The attribute type
 * \retval 0		Success
 * \retval ENOMEM	Insufficient resources
 * \retval EEXIST	An attribute with that name already exists
 * \retval EINUSE	The schema is already a member of a container
 * \retval EINVAL	A parameter was invalid
 */
int sos_schema_attr_add(sos_schema_t schema, const char *name, sos_type_t type);

/**
 * \brief Add an index to an attribute
 *
 * Marks an attribute as having a key-value index. The index is not
 * actually created until the schema containing the attribute is added
 * to the container.
 *
 * \param schema	The schema handle
 * \param name		The attribute name
 * \retval 0		The index was succesfully added.
 * \retval ENOENT	The specified attribute does not exist.
 * \retval EINVAL	One or more parameters was invalid.
 */
int sos_schema_index_add(sos_schema_t schema, const char *name);

/**
 * \brief Configure the index for an attribute
 *
 * By default an attribute index is a BXTREE. A BXTREE is a modified
 * form of a B+Tree that efficiently handles duplicate keys. There
 * are, however, other index types as well as user-defined
 * indices. The type of index is specified as a string that identifies
 * a shared library that implements the necessary index strategy
 * routines, e.g. insert, delete, etc...
 *
 * For keys, the default key type is associated with the attribute
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
int sos_schema_index_cfg(sos_schema_t schema, const char *name,
			 const char *idx_type, const char *key_type, ...);

/**
 * \brief Find an attribute by name
 * \param schema	The schema handle
 * \param name		The attribute's name
 * \returns The attribute handle or NULL if the attribute was not found.
 */
sos_attr_t sos_schema_attr_by_name(sos_schema_t schema, const char *name);

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
sos_attr_t sos_schema_attr_by_id(sos_schema_t schema, int attr_id);

sos_attr_t sos_schema_attr_first(sos_schema_t schema);
sos_attr_t sos_schema_attr_last(sos_schema_t schema);
sos_attr_t sos_schema_attr_next(sos_attr_t attr);
sos_attr_t sos_schema_attr_prev(sos_attr_t attr);

/**
 * \brief Return the attribute's ordinal ID.
 *
 * \param attr	The attribute handle.
 * \returns The attribute id.
 */
int sos_attr_id(sos_attr_t attr);

/**
 * \brief Return the attribute's name
 * \returns The attribute name
 */
const char *sos_attr_name(sos_attr_t attr);

/**
 * \brief Return the attribute's type
 * \returns The attribute type
 */
sos_type_t sos_attr_type(sos_attr_t attr);

/**
 * \brief Test if an attribute has an index.
 *
 * \param attr	The sos_attr_t handle

 * \returns !0 if the attribute has an index
 */
int sos_attr_index(sos_attr_t attr);

size_t sos_attr_size(sos_attr_t attr);

/**
 * \brief Return the schema of an attribute
 *
 * \param attr The attribute handle
 * \returns The schema handle
 */
sos_schema_t sos_attr_schema(sos_attr_t attr);

/**
 * \brief Set an object attribute's value from a string
 *
 * This convenience function uses the attribute's string processing
 * functions to interpret a value specified as a character
 * string.
 *
 * For example:
 *
 *      sos_attr_t an_int = sos_schema_attr_by_name(schema, "my_int_attr");
 *      int rc = sos_set_attr_from_str(an_obj, an_int, "1234");
 *      if (!rc)
 *          printf("Success!!\n");
 *
 * \param sos_obj	The object handle.
 * \param attr		The attribute handle
 * \param attr_value	The attribute value
 * \param endptr Receives the point in the str argumeent where parsing stopped.
 *               This parameter may be NULL.
 * \retval 0 Success
 * \retval EINVAL The string format was invalid for the attribute type
 * \retval ENOSYS There is no string formatter for this attribute type
 */
int sos_set_attr_from_str(sos_obj_t sos_obj, sos_attr_t attr,
			  const char *attr_value, char **endptr);

/**
 * \brief Set an object attribute's value from a string
 *
 * This convenience function set's an object's attribute value specified as a
 * string. The attribute to set is specified by name.
 *
 * For example:
 *
 *     int rc = sos_set_attr_by_name_from_str(an_obj, "my_int_attr", "1234");
 *     if (!rc)
 *        printf("Success!!\n");
 *
 * See the sos_set_attr_from_str() function to set the value with a string if
 * the attribute handle is known.
 *
 * \param sos_obj	The object handle
 * \param attr_name	The attribute name
 * \param attr_value	The attribute value as a string
 * \param endptr Receives the point in the str argumeent where parsing stopped.
 *               This parameter may be NULL.
 * \retval 0 Success
 * \retval EINVAL The string format was invalid for the attribute type
 * \retval ENOSYS There is no string formatter for this attribute type
 */
int sos_set_attr_by_name_from_str(sos_obj_t sos_obj,
				  const char *attr_name, const char *attr_value,
				  char **endptr);

/** @} */
/** @} */

/** \defgroup container SOS Storage Containers
 * @{
 */

/**
 * \brief Create a Container
 *
 * Creates a SOS container. The o_flags and o_mode parameters accept
 * the same values and have the same meaning as the corresponding
 * parameters to the open() system call.
 *
 * Containers are logically maintained in a Unix filesystem
 * namespace. The specified path must be unique for the Container and
 * all sub-directories in the path up to, but not including the
 * basename() must exist.
 *
 * \param path		Pathname for the Container.
 * \param o_mode	The file mode for the Container.
 * \retval 0		The container was successfully created.
 * \retval EINVAL	A parameter was invalid
 * \retval EPERM	The user has insufficient permission
 * \retval EEXIST	A container already exists at the specified path
 */
int sos_container_new(const char *path, int o_mode);

typedef enum sos_perm_e {
	SOS_PERM_RO = 0,
	SOS_PERM_RW,
} sos_perm_t;
/**
 * \brief Open a Container
 *
 * Open a SOS container. If successfull, the <tt>c</tt> parameter will
 * contain a valid sos_t handle on exit.
 *
 * \param path		Pathname for the Container. See sos_container_new()
 * \param o_perm	The requested read/write permissions
 * \param pc		Pointer to a sos_t handle.
 * \retval 0		Success.
 * \retval EPERM	The user has insufficient privilege to open the container.
 * \retval ENOENT	The container does not exist
 */
int sos_container_open(const char *path, sos_perm_t o_perm, sos_t *pc);

/**
 * \brief Delete storage associated with a Container
 *
 * Removes all resources associated with the Container. The sos_t
 * handle must be provided (requiring an open) because it is necessary
 * to know the associated indexes in order to be able to know the
 * names of the associated files. sos_destroy will also close \c sos, as the
 * files should be closed before removed.
 *
 * \param c	The container handle
 * \retval 0	The container was deleted
 * \retval EPERM The user has insufficient privilege
 * \retval EINUSE The container is in-use by other clients
 */
int sos_container_delete(sos_t c);

typedef enum sos_commit_e {
	/** Returns immediately, the sync to storage will be completed
	 *  asynchronously */
	SOS_COMMIT_ASYNC,
	/** Does not return until the sync is complete */
	SOS_COMMIT_SYNC
} sos_commit_t;

/**
 * \brief Extend the size of a Container
 *
 * Expand the size of  Container's object store. This function cannot
 * be used to make the container smaller. See the
 * sos_container_truncate() function.
 *
 * \param sos	The container handle
 * \param new_size The desired size of the container
 * \retval 0 The container was successfully extended.
 * \retval ENOSPC There is insufficient storage to extend the container
 * \retval EINVAL The container is currently larger than the requested size
 */
int sos_container_extend(sos_t sos, size_t new_size);
int sos_container_stat(sos_t sos, struct stat *sb);

/**
 * \brief Close a Container
 *
 * This function commits the index changes to stable storage and
 * releases all in-memory resources associated with the container.
 *
 * If SOS_COMMIT_SYNC is specified in the flags parameter, the function
 * will wait until the changes are commited to stable stroage before
 * returning.
 *
 * \param c	The container handle
 * \param flags	The commit flags
 */
void sos_container_close(sos_t c, sos_commit_t flags);

/**
 * \brief Flush outstanding changes to persistent storage
 *
 * This function commits the index changes to stable storage. If
 * SOS_COMMIT_SYNC is specified in the flags parameter, the function
 * will wait until the changes are commited to stable stroage before
 * returning.
 *
 * \param c	Handle for the container
 * \param flags	The commit flags
 */
int sos_container_commit(sos_t c, sos_commit_t flags);

/**
 * \brief Print container information
 *
 * Prints information about the container to the specified FILE pointer.
 *
 * \param sos	The container handle
 * \param fp	The FILE pointer
 */
void sos_container_info(sos_t sos, FILE* fp);

/**
 * \brief Take a reference on a container
 *
 * SOS container are reference counted. This function takes a reference
 * on a SOS container and returns a pointer to the same. The typical
 * calling sequence is:
 *
 *     sos_t my_sos_ptr = sos_container_get(sos);
 *
 * This allows for the container to be safely pointed to from multiple
 * places. The sos_container_put() function is used to drop a reference on
 * the container. For example:
 *
 *     sos_container_put(my_sos_ptr);
 *     my_sos_ptr = NULL;
 *
 * \param sos	The SOS container handle
 * \retval The container handle
 */
sos_t sos_container_get(sos_t sos);

/**
 * \brief Drop a reference on a container
 *
 * The memory consumed by the container is not released until all
 * references have been dropped. This refers only to references in
 * main memory.
 *
 * \param sos	The container handle
 */
void sos_container_put(sos_t sos);

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
 * - sos_obj_delete()    Delete an object from the container
 * - sos_obj_get()	 Take a reference on an object
 * - sos_obj_put()	 Drop a reference on an object
 * - sos_obj_index()	 Add an object to it's indices
 * - sos_obj_remove()	 Remove an object from it's indices
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

/**
 * \brief Allocate an object from the SOS object store.
 *
 * This call will automatically extend the size of the backing store
 * to accomodate the new object. This call will fail if there is
 * insufficient disk space. Use the sos_obj_index() to add the object
 * to all indices defined by it's object class.
 *
 * See the sos_schema_find() function call for information on how to
 * obtain a schema handle.
 *
 * \param schema	The schema handle
 * \returns Pointer to the new object
 * \returns NULL if there is an error
 */
sos_obj_t sos_obj_new(sos_schema_t schema);

/**
 * \brief Release the storage consumed by the object in the SOS object store.
 *
 * Deletes the object and any arrays to which the object refers. It
 * does not delete an object that is referred to by this object,
 * i.e. SOS_TYPE_OBJ_REF attribute values.
 *
 * This function does not drop any references on the memory resources
 * for this object. Object references must still be dropped with the
 * sos_obj_put() function.
 *
 * \param obj	Pointer to the object
 */
void sos_obj_delete(sos_obj_t obj);

/**
 * \brief Take a reference on an object
 *
 * SOS objects are reference counted. This function takes a reference
 * on a SOS object and returns a pointer to the object. The typical
 * calling sequence is:
 *
 *     sos_obj_t my_obj_ptr = sos_obj_get(obj);
 *
 * This allows for the object to be safely pointed to from multiple
 * places. The sos_obj_put() function is used to drop a reference on
 * a SOS object. For example:
 *
 *     sos_obj_put(my_obj_ptr);
 *     my_obj_ptr = NULL;
 *
 * \param obj	The SOS object handle
 * \retval The object handle
 */
sos_obj_t sos_obj_get(sos_obj_t obj);

/**
 * \brief Drop a reference on an object
 *
 * SOS objects are reference counted. The memory consumed by the
 * object is not released until all references have been dropped. This
 * refers only to references in main memory. The object will continue
 * to exist in persistent storage. See the sos_obj_delete() function
 * for information on removing an object from persistent storage.
 *
 * \param obj	The object handle
 */
void sos_obj_put(sos_obj_t obj);

/**
 * \brief Add an object to it's indexes
 *
 * Add an object to all the indices defined in it's schema. This
 * function should only be called after all attributes that have
 * indexes have had their values set.
 *
  * \param obj	Handle for the object to add
 *
 * \retval 0	Success
 * \retval -1	An error occurred. Refer to errno for detail.
 */
int sos_obj_index(sos_obj_t obj);

/**
 * \brief Remove an object from the SOS
 *
 * This removes an object from all indexes of which it is a
 * member. The object itself is not destroyed. Use the
 * sos_obj_delete() function to release the storage consumed by the
 * object itself.
 *
 * \param obj	Handle for the object to remove
 *
 * \returns 0 on success.
 * \returns Error code on error.
 */
int sos_obj_remove(sos_obj_t obj);

/**
 * \brief Initialize a value with an object's attribute data
 *
 * Returns an object value handle for the specified attribute
 * name. If the <tt>attr_id</tt> parameter is non-null, the parameter
 * is filled in with associated attribute id.
 *
 * \param value Pointer to the value to be initialized
 * \param schema The schema handle
 * \param obj The object handle
 * \param name The attribute name
 * \param attr_id A pointer to the attribute id
 * \retval Pointer to the sos_value_t handle
 * \retval NULL if the specified attribute does not exist.
 */
sos_value_t sos_value_by_name(sos_value_t value, sos_schema_t schema, sos_obj_t obj,
			      const char *name, int *attr_id);

/**
 * \brief Initialize a value with an object's attribute data
 *
 * Returns the sos_value_t for the attribute with the specified
 * id.
 *
 * \param value Pointer to the value to be initialized
 * \param obj		The SOS object handle
 * \param attr_id	The Id for the attribute.
 * \retval Pointer to the sos_value_t handle
 * \retval NULL if the specified attribute does not exist.
 */
sos_value_t sos_value_by_id(sos_value_t value, sos_obj_t obj, int attr_id);

int sos_attr_is_ref(sos_attr_t attr);
int sos_attr_is_array(sos_attr_t attr);
size_t sos_array_count(sos_value_t val);
sos_value_t sos_array_new(sos_value_t val, sos_attr_t attr, sos_obj_t obj, size_t count);
sos_value_t sos_value_new();
void sos_value_free(sos_value_t v);

/**
 * \brief Initialize a value with an object's attribute data
 *
 * \param value Pointer to the value to be initialized
 * \param obj The object handle
 * \param attr The attribute handle
 * \retval The value handle
 */
sos_value_t sos_value_init(sos_value_t value, sos_obj_t obj, sos_attr_t attr);

sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr);

/**
 * \brief Compare two value
 *
 * Compares <tt>a</tt> and <tt>b</tt> and returns <0 if a < b, 0 if
 * a == b, and >0 if a > b
 *
 * \param a The lhs
 * \param b The rhs
 * \returns The result as described above
 */
int sos_value_cmp(sos_value_t a, sos_value_t b);

/**
 * \brief Get the size of an attribute value
 *
 * \param value The value handle
 *
 * \returns The size of the attribute value
 */
size_t sos_value_size(sos_value_t value);

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
const char *sos_obj_attr_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len);

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
int sos_obj_attr_from_str(sos_obj_t obj, sos_attr_t attr, const char *str, char **endptr);

/**
 * \brief Format a value as a string
 *
 * \param value The value handle
 * \param str Pointer to the string to receive the formatted value
 * \param len The size of the string in bytes.
 * \returns A pointer to the str argument or NULL if there was a
 *          formatting error.
 */
const char *sos_value_to_str(sos_value_t value, char *str, size_t len);

/**
 * \brief Set the value from a string
 *
 * \param value The value handle
 * \param str The input string value to parse
 * \param endptr Receives the point in the str argumeent where parsing stopped.
 *               This parameter may be NULL.
 * \retval 0 The string was successfully parsed and the value set
 * \retval EINVAL The string was incorrectly formatted for this value
 *                type.
 */
int sos_value_from_str(sos_value_t value, const char *str, char **endptr);

/** @} */

/** \defgroup keys SOS Keys
 * @{
 */
typedef struct ods_obj_s *sos_key_t;

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
 * these keys comes from memory. See the ods_key_alloc() function for
 * keys that are stored in the ODS.
 *
 * If the size of the key is known to be less than 254 bytes, the
 * ODS_KEY() macro is useful for defining an ODS key that is allocated
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

/**
 * \brief Set the value of a key
 *
 * Sets the value of the key to 'value'. The \c value parameter is of
 * type void* to make it convenient to use values of arbitrary
 * types. The minimum of 'sz' and the maximum key length is
 * copied. The number of bytes copied into the key is returned.
 *
 * \param key	The key
 * \param value	The value to set the key to
 * \param sz	The size of value in bytes
 * \returns The number of bytes copied
 */
size_t sos_key_set(sos_key_t key, void *value, size_t sz);

/**
 * \brief Set the value of a key from a string
 *
 * \param attr	The attribute handle
 * \param key	The key
 * \param str	Pointer to a string
 * \retval 0	if successful
 * \retval -1	if there was an error converting the string to a value
 */
int sos_key_from_str(sos_attr_t attr, sos_key_t key, const char *str);

/**
 * \brief Return a string representation of the key value
 *
 * \param attr	The attribute handle
 * \param key	The key
 * \return A const char * representation of the key value.
 */
const char *sos_key_to_str(sos_attr_t attr, sos_key_t key);

/**
 * \brief Compare two keys using the attribute index's compare function
 *
 * \param attr	The attribute handle
 * \param a	The first key
 * \param b	The second key
 * \return <0	a < b
 * \return 0	a == b
 * \return >0	a > b
 */
int sos_key_cmp(sos_attr_t attr, sos_key_t a, sos_key_t b);

/**
 * \brief Return the size of the attribute index's key
 *
 * Returns the native size of the attribute index's key values. If the
 * key value is variable size, this function returns -1. See the sos_key_len()
 * and sos_key_size() functions for the current size of the key's
 * value and the size of the key's buffer respectively.
 *
 * \return The native size of the attribute index's keys in bytes
 */
size_t sos_attr_key_size(sos_attr_t attr);

/**
 * \brief Return the maximum size of this key's value
 *
 * \returns The size in bytes of this key's value buffer.
 */
size_t sos_key_size(sos_key_t key);

/**
 * \brief Return the length of the key's value
 *
 * Returns the current size of the key's value.
 *
 * \param key	The key
 * \returns The size of the key in bytes
 */
size_t sos_key_len(sos_key_t key);

/**
 * \brief Return the value of a key
 *
 * \param key	The key
 * \returns Pointer to the value of the key
 */
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

/**
 * \brief Create a new SOS iterator
 *
 * \param attr The schema attribute handle
 *
 * \retval sos_iter_t For the specified key
 * \retval NULL       If there was an error creating the iterator. Note
 *		      that failure to find a matching object is not an
 *		      error.
 */
sos_iter_t sos_iter_new(sos_attr_t attr);

/**
 * \brief Release the resources associated with a SOS iterator
 *
 * \param iter	The iterator returned by \c sos_new_iter
 */
void sos_iter_free(sos_iter_t iter);

/**
 * \brief Return the iterator name.
 *
 * An iterator inherits it's name from the associated attribute
 *
 * \param iter  Handle for the iterator.
 *
 * \returns Pointer to the attribute name for the iterator.
 */
const char *sos_iter_name(sos_iter_t iter);

/**
 * \brief Return the attribute handle used to create the iterator
 *
 * \param iter The iterator handle
 * \returns The attribute handle
 */
sos_attr_t sos_iter_attr(sos_iter_t iter);

/**
 * \brief Compare iterator object's key with other key.
 *
 * This function compare the key of the object pointed by the iterator with the
 * other key. This is a convenience routine and is equivalent to the
 * following code sequence:
 *
 *     sos_key_t iter_key = sos_iter_key(iter);
 *     int rc = sos_key_cmp(attr, iter_key, other);
 *     sos_key_put(iter_key);
 *
 * \param iter	The iterator handle
 * \param other	The other key
 * \retval <0	iter < other
 * \retval 0	iter == other
 * \retval >0	iter > other
 */
int sos_iter_key_cmp(sos_iter_t iter, ods_key_t other);

/**
 * \brief Position the iterator at the specified key
 *
 * \param iter  Handle for the iterator.
 * \param key   The key for the iterator. The appropriate index will
 *		be searched to find the object that matches the key.
 *
 * \retval 0 Iterator is positioned at matching object.
 * \retval ENOENT No matching object was found.
 */
int sos_iter_find(sos_iter_t iter, ods_key_t key);

/**
 * \brief Position the iterator at the infimum of the specified key.
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \retval 0 if the iterator is positioned at the infinum
 * \retval ENOENT if the infimum does not exist
 */
int sos_iter_inf(sos_iter_t i, ods_key_t key);

/**
 * \brief Position the iterator at the supremum of the specified key
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \retval 0 The iterator is positioned at the supremum
 * \retval ENOENT No supremum exists
 */
int sos_iter_sup(sos_iter_t i, ods_key_t key);

typedef enum sos_iter_flags_e {
	SOS_ITER_F_ALL = ODS_ITER_F_ALL,
	/** The iterator will skip duplicate keys in the index */
	SOS_ITER_F_UNIQUE = ODS_ITER_F_UNIQUE,
	SOS_ITER_F_MASK = ODS_ITER_F_MASK
} sos_iter_flags_t;

/**
 * \brief Set iterator behavior flags
 *
 * \param i The iterator
 * \param flags The iterator flags
 * \retval 0 The flags were set successfully
 * \retval EINVAL The iterator or flags were invalid
 */
int sos_iter_flags_set(sos_iter_t i, sos_iter_flags_t flags);

/**
 * \brief Get the iterator behavior flags
 *
 * \param i The iterator
 * \retval The sos_iter_flags_t for the iterator
 */
sos_iter_flags_t sos_iter_flags_get(sos_iter_t i);

/**
 * \brief Return the number of positions in the iterator
 * \returns The cardinality of the iterator
 */
uint64_t sos_iter_card(sos_iter_t i);

/**
 * \brief Return the number of duplicates in the index
 * \returns The count of duplicates
 */
uint64_t sos_iter_dups(sos_iter_t i);

/**
 * \brief Returns the current iterator position
 *
 * \param i The iterator handle
 * \param pos The sos_pos_t that will receive the position value.
 * \returns The current iterator position or 0 if position is invalid
 */
int sos_iter_pos(sos_iter_t i, sos_pos_t pos);

/**
 * \brief Sets the current iterator position
 *
 * \param i The iterator handle
 * \param pos The iterator cursor position
 * \retval 0 Success
 * \retval ENOENT if the specified position is invalid
 */
int sos_iter_set(sos_iter_t i, const sos_pos_t pos);

/**
 * \brief Position the iterator at next object in the index
 *
 * Advance the iterator position to the next entry.
 *
 * \param iter The iterator handle
 *
 * \retval 0 The iterator is positioned at the next object in the index
 * \retval ENOENT No more entries in the index
 */
 int sos_iter_next(sos_iter_t iter);

/**
 * \brief Retrieve the next object from the iterator
 *
 * Advance the iterator position to the previous entry.
 *
 * \param i Iterator handle
 *
 * \returns 0  The iterator is positioned at the previous entry
 * \returns ENOENT If no more matching records were found.
 */
int sos_iter_prev(sos_iter_t i);

/**
 * Position the iterator at the first object.
 *
 * \param i	The iterator handle

 * \return 0 The iterator is positioned at the first object in the index
 * \return ENOENT The index is empty
 */
int sos_iter_begin(sos_iter_t i);

/**
 * Position the iterator at the last object in the index
 *
 * \param i The iterator handle
 * \return 0 The iterator is positioned at the last object in the index
 * \return ENOENT The index is empty
 */
int sos_iter_end(sos_iter_t i);

/**
 * \brief Return the key at the current iterator position
 *
 * Return the key associated with the current iterator position. This
 * key is persistent and reference counted. Use the sos_key_put()
 * function to drop the reference given by this function when finished
 * with the key.
 *
 * \param iter	The iterator handle
 * \return ods_key_t at the current position
 */
sos_key_t sos_iter_key(sos_iter_t iter);

/**
 * \brief Return the object reference of the current iterator position
 *
 * \param iter	The iterator handle
 * \return ods_ref_t at the current position
 */
sos_obj_t sos_iter_obj(sos_iter_t iter);

/**
 * \brief Remove object at the current iterator position
 *
 * After removal, the iterator points at the next object if it
 * exists. Otherwise, it points to the previous object.
 *
 * \param iter The iterator handle
 * \return 0 on success.
 * \return Error code on failure.
 */
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
#if 0
/** \defgroup mgmt SOS Data Management Services
 * @{
 */

/**
 * \brief Verify index of the given \c attr_id.
 *
 * \param sos The SOS handle.
 * \param attr_id The attribute ID
 *
 * \return 0 if OK.
 * \return -1 if Error.
 */
int sos_verify_index(sos_t sos, int attr_id);

/**
 * \brief Rebuild index of the given attribute \c attr_id.
 *
 * \param sos The SOS handle.
 * \param attr_id The attribute ID
 *
 * \return 0 on success.
 * \return -1 on failure.
 */
int sos_rebuild_index(sos_t sos, int attr_id);

/**
 * \brief Change SOS owner.
 *
 * \param sos SOS handle
 * \param owner Owner's UID
 * \param group Owner's GID
 *
 * \retval 0 success
 * \retval -1 failed
 */
int sos_chown(sos_t sos, uid_t owner, gid_t group);

/**
 * \brief Rotate store (similar to logrotate).
 *
 * Assuming that the current store path is SPATH, this function will rename the
 * current store to SPATH.1, and strip indices from it. The existing SPATH.1 ...
 * SPATH.(N-1) will be renamed to SPATH.2 ... SPATH.N respectively. The old
 * SPATH.N is removed by this process. Then, this function re-initialize the
 * store SPATH and returns. If \c N is 0, no existing rotated store will be
 * removed.
 *
 * If the application want to use a post-rotate hook mechanism, please see
 * sos_post_rotation() function.
 *
 * Please also note that if a rotation is a success, new SOS store handle is
 * returned and the existing store handle will be destroyed. All outstanding
 * object handle will be stale.
 *
 * \param sos SOS handle
 * \param N The number of backups allowed. 0 means no limit.
 *
 * \retval sos New SOS handle if rotation is a success.
 * \retval NULL If rotation failed, the input \c sos is left unchanged.
 */
sos_t sos_rotate(sos_t sos, int N);

/**
 * \brief Similar to sos_roate(), but keep indices.
 *
 * \param sos SOS handle
 * \param N The number of backups allowed. 0 means unlimited.
 *
 * \retval sos New SOS handle if rotation is a success.
 * \retval NULL If rotation failed, the input \c sos is left unchanged.
 */
sos_t sos_rotate_i(sos_t sos, int N);

/**
 * \brief Convenient post-rotation function call.
 *
 * This convenient function provide a post-rotation hook mechanism. Namely, the
 * function will execute the command/script in the given environment variable \c
 * env_var using ovis_execute(). SOS_PATH environment variable will be set to
 * the path of the given \c sos for the command/script execution. If \c env_var
 * is \c NULL, "SOS_POST_ROTATE" is used. If the environment variable does not
 * exists, the function returns \c EINVAL.
 *
 * Please note that this function does not wait the hook script to finish. It
 * just fork and return (the child process does the exec obviously).
 *
 * \param sos SOS handle.
 * \param env_var Environment variable pointing to the post-rotation hook
 *                script. If this is NULL, "SOS_POST_ROTATE" is used.
 *
 * \retval 0 OK
 * \retval error_code Error
 */
int sos_post_rotation(sos_t sos, const char *env_var);

/**
 * \brief Reinitialize SOS.
 *
 * Reinitializing SOS will destroy all objects and indices in the store. The
 * store will be resized to the given initial size. If the given initial size is
 * 0, the default value (::SOS_INITIAL_SIZE) will be used.
 *
 * \param sos the store handle to be reinitialized.
 * \param sz Size (in bytes). If 0, SOS_INITIAL_SIZE will be used.
 *
 * \retval sos the new SOS handle if reinitialization is a success.
 * \retval NULL if reinitialization is a failure.
 */
sos_t sos_reinit(sos_t sos, uint64_t sz);

/** @} */
#endif

#endif
