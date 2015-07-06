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
 * This convenience function set's an object's attribute value specified as a
 * string. The attribute to set is specified by name.
 *
 * For example:
 *
 *     int rc = sos_obj_attr_by_name_from_str(an_obj, "my_int_attr", "1234");
 *     if (!rc)
 *        printf("Success!!\n");
 *
 * See the sos_obj_attr_from_str() function to set the value with a string if
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

/**
 * \brief Allocate an object from the SOS object store.
 *
 * This call will automatically extend the size of the backing store
 * to accomodate the new object. This call will fail if there is
 * insufficient disk space. Use the sos_obj_index() to add the object
 * to all indices defined by it's object class.
 *
 * See the sos_schema_by_name() function call for information on how to
 * obtain a schema handle.
 *
 * \param schema	The schema handle
 * \returns Pointer to the new object
 * \returns NULL if there is an error
 */
sos_obj_t sos_obj_new(sos_schema_t schema);

/**
 * \brief Returns an Object's schema
 * \param obj The object handle
 * \retval The schema used to create the object.
 */
sos_schema_t sos_obj_schema(sos_obj_t obj);

typedef ods_ref_t sos_ref_t;
sos_ref_t sos_obj_ref(sos_obj_t obj);

/**
 * \brief  Return the object associated with the value
 *
 * This function will return a sos_obj_t for the object that is referred
 * to by ref_val. Use the function sos_obj_from_ref() to obtain an
 * object from a raw sos_ref_t value.
 *
 * \param sos The container handle
 * \param ref_val A value handle to an attribute of type SOS_TYPE_OBJ
 * \retval The object to which the reference refers.
 * \retval NULL The reference did not point to a well formed object, or the schema
 *              in the object header was not part of the container.
 */
sos_obj_t sos_obj_from_value(sos_t sos, sos_value_t ref_val);

/**
 * \brief  Return the object associated with the reference
 *
 * This function will return a sos_obj_t for the object that is referred
 * to by 'ref'. Use the function sos_obj_from_value() to obtain an
 * object from a sos_value_t value.
 *
 * \param sos The container handle
 * \param ref The object reference
 * \retval The object to which the reference refers.
 * \retval NULL The reference did not point to a well formed object, or the schema
 *              in the object header was not part of the container.
 */
sos_obj_t sos_obj_from_ref(sos_t sos, sos_ref_t ref);

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
static inline void *sos_array_ptr(sos_value_t val) {
	return val->data->array.data.byte_;
}
sos_value_t sos_value_new();
void sos_value_free(sos_value_t v);

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
void *sos_obj_ptr(sos_obj_t obj);

/**
 * \brief Initialize a value with an object's attribute data
 *
 * \param value Pointer to the value to be initialized
 * \param obj The object handle
 * \param attr The attribute handle
 * \retval The value handle
 */
sos_value_t sos_value_init(sos_value_t value, sos_obj_t obj, sos_attr_t attr);
#define SOS_VALUE(_name_)				\
	struct sos_value_s  _name_ ## __, *_name_ = &_name_ ## __;

/**
 * \brief Return a value for the specified object's attribute.
 *
 * This function returns a sos_value_t for an object's attribute.
 * The reference on this value should be dropped with sos_value_put()
 * when the application is finished with the value.
 *
 * \param obj The object handle
 * \param attr The attribute handle
 * \returns The value of the object's attribute.
 */
sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr);

/**
 * \brief Drop a reference on a value
 *
 * \param value The value handle.
 */
void sos_value_put(sos_value_t value);

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
 * \brief Set an object value from a buffer
 *
 * Set the value from an untyped void buffer. If the buflen is too
 * large to fit, only sos_value_size() bytes will be written.
 *
 * \param value  The value handle
 * \param buf    The buffer containing the data
 * \param buflen The number of bytes to write from the buffer
 * \retval The number of bytes written
 */
size_t sos_value_memset(sos_value_t value, void *buf, size_t buflen);

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
char *sos_obj_attr_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len);

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
int sos_key_from_str(sos_attr_t attr, sos_key_t key, const char *str);
const char *sos_key_to_str(sos_attr_t attr, sos_key_t key);
int sos_key_cmp(sos_attr_t attr, sos_key_t a, sos_key_t b);

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
int sos_iter_key_cmp(sos_iter_t iter, sos_key_t other);

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
int sos_iter_find(sos_iter_t iter, sos_key_t key);

/**
 * \brief Position the iterator at the infimum of the specified key.
 *
 * Position the iterator at the object whose key is the greatest
 * lower bound of the specified key.
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \retval 0 if the iterator is positioned at the infinum
 * \retval ENOENT if the infimum does not exist
 */
int sos_iter_inf(sos_iter_t i, sos_key_t key);

/**
 * \brief Position the iterator at the supremum of the specified key
 *
 * Position the iterator at the object whose key is the least
 * upper bound of the specified key.
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \retval 0 The iterator is positioned at the supremum
 * \retval ENOENT No supremum exists
 */
int sos_iter_sup(sos_iter_t i, sos_key_t key);

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
 * \return sos_key_t at the current position
 */
sos_key_t sos_iter_key(sos_iter_t iter);

/**
 * \brief Return the object at the current iterator position
 *
 * \param iter	The iterator handle
 * \return ods_obj_t at the current position
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
