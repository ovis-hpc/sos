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

/**
 * \mainpage Scalable Object Store Documentation
 *
 * \section intro Introduction
 *
 * The Scalable Object Storage Service is a high performance storage
 * engine designed to store structured data to persistent media very
 * efficiently. The design criteria are that objects can be searched
 * for and iterated over for a set of pre-specified object
 * attributes. Iteration can be in both the forward and backward
 * direction and the object attribute key can be formatted by the user
 * or consist of the attribute value. This allows for indexes that
 * automatically bin data, for example, an index that takes a metric
 * value and stores it such that it's key consists of it's standard
 * deviation from the mean. Any number of objects can have the same
 * key.
 *
 * # SOS Storage
 *
 * - sos_open()    Create or re-open an object store.
 * - sos_close()   Close an object store and release it's in-core resources.
 * - sos_commit()  Commit all changes to the object store to persistent storage.
 *
 * # SOS Object Classes
 *
 * - sos_obj_new()	Create a new object
 * - sos_obj_add()	Add the object to the the associated indices.
 * - sos_obj_delete()	Remove an object from the object store.
 *
 */

typedef struct sos_container_s *sos_t;
typedef struct sos_attr_s *sos_attr_t;
typedef struct sos_schema_s *sos_schema_t;
typedef struct sos_obj_s *sos_obj_t;

#define SOS_ATTR_NAME_LEN	32
#define SOS_SCHEMA_NAME_LEN	32

/** \defgroup class SOS Object Classes
 * @{
 */

typedef enum sos_type_e {
	SOS_TYPE_INT32 = 0,
	SOS_TYPE_INT64,
	SOS_TYPE_UINT32,
	SOS_TYPE_UINT64,
	SOS_TYPE_FLOAT,
	SOS_TYPE_DOUBLE,
	SOS_TYPE_LONG_DOUBLE,
	SOS_TYPE_OBJ,
	SOS_TYPE_BYTE_ARRAY,
	SOS_TYPE_FIRST_ARRAY = SOS_TYPE_BYTE_ARRAY,
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
struct sos_array_s {
	uint32_t count;
	union {
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
};

typedef union sos_value_data_u {
	union sos_primary_u prim;
	struct sos_array_s array;
} *sos_value_data_t;

typedef struct sos_value_s {
	sos_obj_t obj;
	sos_value_data_t data;
} *sos_value_t;

#pragma pack()

char *sos_type_to_str(enum sos_type_e type);

/**
 * @}
 */

/** \defgroup schema SOS Schema
 * @{
 */
/**
 * \brief Create/Define Object Schemas
 */

/**
 * \brief Create a schema
 *
 * A schema defines a SOS object. Every object in a SOS database is
 * associated with a schema via an internal schema_id. This ID is used
 * internally to define how objects are indexed and accessed.
 *
 * After a schema is created it must be associated with a
 * container. See the sos_schema_add() function to add a schema to a
 * container so that objects of that type can subsequently be created
 * in the container. Once a schema has been added, it can be looked up
 * with the sos_schema_find() command. The schema is provided to the
 * sos_obj_new() function when new objects are created.
 *
 * \param name	The name of the schema. This name must be unique
 * within the container.
 * \returns	A pointer to the new schema or a NULL pointer if there
 * is an error.
 */
sos_schema_t sos_schema_new(const char *name);

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
int sos_schema_del(sos_t sos, const char *name);

/**
 * \brief Take a reference on a schema
 *
 * SOS schema are reference counted. This function takes a reference
 * on a SOS schema and returns a pointer to the same. The typical
 * calling sequence is:
 * <code>
 *    sos_schema_t my_schema_ptr = sos_schema_get(schema);
 * </code>
 *
 * This allows for the schema to be safely pointed to from multiple
 * places. The sos_schema_put() function is used to drop a reference on
 * the schema. For example:
 * <code>
 *    sos_schema_put(my_schema_ptr);
 *    my_schema_ptr = NULL;
 * </code>
 *
 * \param schema	The SOS schema handle
 * \retval The schema handle
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
void sos_schema_put(sos_schema_t schema);;

int sos_schema_attr_count(sos_schema_t schema);
const char *sos_schema_name(sos_schema_t schema);

/**
 * \brief Add an attribute to a schema
 *
 * Add an attribute to a schema. The schema can only be modified if it
 * is not already a member of the container.
 *
 * \param schema	The schema
 * \param name		The attribute name
 * \param type		The attribte type
 * \param initial_len	If the attribute type is an array, this specifies the
 *			initial array size. It may be 0.
 * \retval 0		Success
 * \retval ENOMEM	Insufficient resources
 * \retval EEXIST	An attribute with that name already exists
 * \retval EINUSE	The schema is already a member of a container
 * \retval EINVAL	A parameter was invalid
 */
int sos_attr_add(sos_schema_t schema, const char *name,
		 sos_type_t type, int initial_len);

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
int sos_index_add(sos_schema_t schema, const char *name);

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
int sos_index_cfg(sos_schema_t schema, const char *name,
		  const char *idx_type, const char *key_type, ...);




int sos_attr_id(sos_attr_t attr);

sos_attr_t sos_attr_by_name(sos_schema_t schema, const char *name);
sos_attr_t sos_attr_by_id(sos_schema_t schema, int attr_id);

const char *sos_attr_name(sos_attr_t attr);
sos_type_t sos_attr_type(sos_attr_t attr);

int sos_attr_from_str(sos_obj_t sos_obj, sos_attr_t attr, const char *attr_value);
int sos_attr_by_name_from_str(sos_schema_t schema, sos_obj_t sos_obj,
			      const char *attr_name, const char *attr_value);

/**
 * \brief Test if an attribute has an index.
 *
 * \param attr	The sos_attr_t handle

 * \returns !0 if the attribute has an index
 */
int sos_attr_index(sos_attr_t attr);

/**
 * @}
 */

/** \defgroup store SOS Storage
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
 * \retval 0 		The container was successfully created.
 * \retval EINVAL	A parameter was invalid
 * \retval EPERM	The user has insufficient permission
 * \retval EEXIST	A container already exists at the specified path
 */
int sos_container_new(const char *path, int o_mode);

/**
 * \brief Open a Container
 *
 * Open a SOS container. If successfull, the <tt>c<tt> parameter will
 * contain a valid sos_t handle on exit.
 *
 * \param path		Pathname for the Container. See sos_container_new()
 * \param o_flags	Permission flags, see the open() system call.
 * \param pc		Pointer to a sos_t handle.
 * \retval 0		Success.
 * \retval EPERM	The user has insufficient privilege to open the container.
 * \retval ENOENT	The container does not exist
 */
int sos_container_open(const char *path, int o_flags, sos_t *pc);

/**
 * \brief Delete storage associated with a Container
 *
 * Removes all resources associated with the Container. The sos_t
 * handle must be provided (requiring an open) because it is necessary
 * to know the associated indexes in order to be able to know the
 * names of the associated files. sos_destroy will also close \c sos, as the
 * files should be closed before removed.
 *
 * \param c 	The container handle
 * \retval 0	The container was deleted
 * \retval EPERM The user has insufficient privilege
 * \retval EINUSE The container is in-use by other clients
 */
int sos_container_del(sos_t c);

typedef enum sos_commit_e {
	SOS_COMMIT_ASYNC,
	SOS_COMMIT_SYNC
} sos_commit_t;

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
 * <code>
 *    sos_t my_sos_ptr = sos_container_get(sos);
 * </code>
 *
 * This allows for the container to be safely pointed to from multiple
 * places. The sos_container_put() function is used to drop a reference on
 * the container. For example:
 * <code>
 *    sos_container_put(my_sos_ptr);
 *    my_sos_ptr = NULL;
 * </code>
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

/**
 * @}
 */

/**
 * \defgroup objects SOS Objects
 * @{
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
 * insufficient disk space. Use the sos_obj_add() to add the object
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
 * \param sos	Handle for the SOS
 * \param obj	Pointer to the object
 */
void sos_obj_delete(sos_t sos, sos_obj_t obj);

/**
 * \brief Take a reference on an object
 *
 * SOS objects are reference counted. This function takes a reference
 * on a SOS object and returns a pointer to the object. The typical
 * calling sequence is:
 * <code>
 *    sos_obj_t my_obj_ptr = sos_obj_get(obj);
 * </code>
 *
 * This allows for the object to be safely pointed to from multiple
 * places. The sos_obj_put() function is used to drop a reference on
 * an SOS object. For example:
 * <code>
 *    sos_obj_put(my_obj_ptr);
 *    my_obj_ptr = NULL;
 * </code>
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
 * \param sos	The container handle
 * \param obj	Handle for the object to add
 *
 * \retval 0	Success
 * \retval -1	An error occurred. Refer to errno for detail.
 */
int sos_obj_index(sos_t s, sos_obj_t obj);

/**
 * \brief Remove an object from the SOS
 *
 * This removes an object from all indexes of which it is a
 * member. The object itself is not destroyed. Use the 
 * sos_obj_delete() function to release the storage consumed by the
 * object itself.
 *
 * \param sos	Handle for the SOS
 * \param obj	Handle for the object to remove
 *
 * \returns 0 on success.
 * \returns Error code on error.
 */
int sos_obj_remove(sos_t s, sos_obj_t obj);

/**
 * @}
 */

/**
 * \section value SOS Object Value Functions
 * @{
 */

/**
 * \brief Get the SOS value handle by name
 *
 */
sos_value_t sos_value_by_name(sos_schema_t schema, sos_obj_t obj,
			      const char *name, int *attr_id);


/**
 * \brief Get the SOS value handle by ID
 *
 * Returns the sos_value_t for the attribute with the specified
 * id.
 *
 * \param obj		The SOS object handle
 * \param attr_id	The Id for the attribute.
 * \retval Pointer to the sos_value_t handle
 * \retval NULL if the specified attribute does not exist.
 */
sos_value_t sos_value_by_id(sos_obj_t obj, int attr_id);

sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr);

/**
 * \brief Get the size of an attribute value
 *
 * \param attr	The attribute handle
 * \param value The value handle
 *
 * \returns The size of the attribute value
 */
size_t sos_value_size(sos_attr_t attr, sos_value_t value);

const char *sos_value_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len);
int sos_value_from_str(sos_obj_t obj, sos_attr_t attr, const char *str);

/**
 * @}
 */

/**
 * @}
 */

/**
 * \defgroup keys SOS Keys
 * @{
 */
typedef struct ods_obj_s *sos_key_t;

/**
 * \brief Create a memory key
 *
 * A key is just a an object with a set of convenience routines to
 * help with getting and setting it's value based on the key type used
 * on an index.
 *
 * A memory key is used to look up objects in a SOS attribute's
 * index. The storage for these keys comes from memory and is not
 * persistent.
 *
 * \param attr	The attribute handle
 * \param sz	The maximum size in bytes of the key value
 * \retval !0	sos_key_t pointer to the key
 * \retval 0	Insufficient resources
 */
sos_key_t sos_key_malloc(sos_attr_t attr, size_t sz);

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
 * \param idx	The index handle
 * \param key	The key
 * \param str	Pointer to a string
 * \retval 0	if successful
 * \retval -1	if there was an error converting the string to a value
 */
int sos_key_from_str(sos_attr_t attr, sos_key_t key, const char *str);

/**
 * \brief Return a string representation of the key value
 *
 * \param idx	The index handle
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

void *sos_value_as_key(sos_attr_t attr, sos_value_t value);
/**
 * @}
 */

/* @}
/**
 * \defgroup iter SOS Iterators
 * @{
 */
typedef struct sos_iter_s *sos_iter_t;

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
 * \brief Compare iterator object's key with other key.
 *
 * This function compare the key of the object pointed by the iterator with the
 * other key. This is a convenience routine and is equivalent to the
 * following code sequence:
 * <code>
 *    sos_key_t iter_key = sos_iter_key(iter);
 *    int rc = sos_key_cmp(attr, iter_key, other);
 *    sos_key_put(iter_key);
 * </code>
 *
 * \param iter	The iterator handle
 * \param other	The other key
 * \return <0	iter < other
 * \return 0	iter == other
 * \return >0	iter > other
 */
int sos_iter_key_cmp(sos_iter_t iter, ods_key_t other);

/**
 * \brief Position the iterator at the specified key
 *
 * \param iter  Handle for the iterator.
 * \param key   The key for the iterator. The appropriate index will
 *		be searched to find the object that matches the key.
 *
 * \returns 0 Iterator is positioned at matching object.
 * \returns ENOENT No matching object was found.
 */
int sos_iter_seek(sos_iter_t iter, ods_key_t key);

/**
 * \brief Position the iterator at the infimum of the specified key.
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \returns 0 if the iterator is positioned at the infinum
 * \returns ENOENT if the infimum does not exist
 */
int sos_iter_seek_inf(sos_iter_t i, ods_key_t key);

/**
 * \brief Position the iterator at the supremum of the specified key
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \return 0 The iterator is positioned at the supremum
 * \return ENOENT No supremum exists
 */
int sos_iter_seek_sup(sos_iter_t i, ods_key_t key);

/**
 * \brief Position the iterator at next object in the index
 *
 * \param i The iterator handle
 *
 * \return 0 The iterator is positioned at the next object in the index
 * \return ENOENT No more entries in the index
 */
int sos_iter_next(sos_iter_t iter);

/**
 * \brief Retrieve the next object from the iterator
 *
 * \param i Iterator handle
 *
 * \returns 0  The iterator is positioned at the previous
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

/**
 * \section mgmt SOS Data Management Services
 *
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

/**
 * @}
 */

#endif
