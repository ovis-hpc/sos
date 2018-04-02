/*
 * Copyright (c) 2013-2017 Open Grid Computing, Inc. All rights reserved.
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
 *      Neither the name of Open Grid Computing nor the names of any
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *      Modified source versions must be plainly marked as such, and
 *      must not be misrepresented as being the original software.
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

#ifndef _ODS_IDX_H_
#define _ODS_IDX_H_
#include <stdint.h>
#include <stdio.h>
#include <ods/ods.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ods_idx *ods_idx_t;
typedef struct ods_iter *ods_iter_t;
#define ODS_POS_DATA_LEN	sizeof(ods_ref_t)
typedef struct ods_pos_s {
	union {
		ods_ref_t ref;
		unsigned char data[ODS_POS_DATA_LEN];
	};
} *ods_pos_t;

#define ODS_IDX_SIGNATURE	"ODSIDX00"

#define ODS_IDX_DATA_LEN 16
typedef struct ods_idx_data_s {
	union {
		unsigned char bytes[ODS_IDX_DATA_LEN];
		uint64_t uint64_[ODS_IDX_DATA_LEN/sizeof(uint64_t)];
	};
} ods_idx_data_t;

/**
 * \brief Create an index
 *
 * An index implements a persistent key/value store in an ODS data
 * store. The key is a generic ods_key_t and the value is 16B of
 * arbitrary data represented by the ods_idx_data_t type. In order to
 * support a common usage model in which the value data is used to
 * refer to an ODS object in an ODS Object Store, the union includes
 * an ods_idx_ref_s structure that contains <tt>user_data</tt> and
 * an <tt>obj_ref</tt> fields. The <tt>obj_ref</tt> field is an
 * ods_ref_t which can be transformed back and forth between
 * references and memory pointers  using the ods_ods_ref_to_ptr() and
 * ods_obj_ptr_to_ref() functions. The <tt>user_data</tt> field can be
 * used for whatever purpose the application chooses. It should be
 * noted that the value field has no functional implication in the
 * index itself, it can be used in whatever way the application sees
 * fit.
 *
 * The 'type' parameter is the name of the type of index,
 * e.g. "BPTREE" or "RADIXTREE". This parameter identifies the
 * organization of the index and indirectly specifies the name of a
 * shared library that is loaded and implements the generic index
 * interface. There are convenience macros defined for the following
 * index types:
 *
 * - ODS_IDX_BPTREE Implements a B+ Tree. The order of the tree is
 *   specifed in an 'order' parameter that follows the 'type'
 *   parameter. This parameter is only consulted if O_CREAT was
 *   specified and the index does not already exist.
 * - ODS_IDX_RBTREE Implements a Red-Black Tree.
 * - ODS_IDX_RADIXTREE Implements a Radix Tree.
 *
 * An index supports a untyped key that is an arbitrarily long
 * array of bytes. The application must specify the name of a 'key'
 * comparison function that compares two keys and returns the relative
 * order or equivalence of the keys. See the ods_key_t documentation
 * for more information. The 'key' parameter indirectly specifies the
 * name of a shared library that implements the key comparator
 * function. There are convenience macros defined for the predefined
 * key types. See the ods_key_t documention for the names and behavior
 * of these comparators.
 *
 * \param path	The path to the ODS store
 * \param mode	The file mode permission flags
 * \param type	The type of the index
 * \param key	The type of the key
 * \param args  Optional(can be NULL) comma separated attr=value
 *              arguments to index provider
 * \return 0	The index was successfully create. Use ods_idx_open()
 *		to access the index.
 * \return !0	An errno indicating the reason for failure.
 */
#define ODS_IDX_ARGS_LEN 256
int ods_idx_create(const char *path, int mode,
		   const char *type, const char *key,
		   const char *args);

/**
 * \brief destroy an index
 *
 * \param path The path to the ODS
 */
int ods_idx_destroy(const char *path);

/**
 * \brief Open an index
 *
 * An index implements a persistent key/value store in an ODS data
 * store.
 *
 * \param path	The path to the ODS store
 * \param o_perm The requested read/write permissions
 * \retval !0	The ods_idx_t handle for the index
 * \retval 0	The index file could not be opened. See the
 *		errno for the reason.
 */
ods_idx_t ods_idx_open(const char *path, ods_perm_t o_perm);

/**
 * \brief Set run-time index option
 *
 * Set options that affect the behavior of the index API.  Run-time
 * options must be set one at a time after each call to
 * ods_idx_open(), i.e. options are not persistent.  Some options have
 * additional arguments.
 *
 * Set the ODS_IDX_OPT_MP_UNSAFE option to indicate that the
 * application is responsible for ensuring that the index is
 * multi-process safe (see the ods_idx_lock() function). By default,
 * indices are MP Safe.
 *
 * \param idx The index handle
 * \param opt The option id
 * \param ... Some options have additional arguments
 * \retval 0 The option was set successfully
 * \retval EINVAL An invalid runtime option was specified
 */
#define ODS_IDX_OPT_MP_UNSAFE	1 /*! Index not multi-process safe */
#define ODS_IDX_OPT_VISIT_ASYNC	2 /*! Make ods_idx_visit() asynchronous */
typedef uint32_t ods_idx_rt_opts_t;
int ods_idx_rt_opts_set(ods_idx_t idx, ods_idx_rt_opts_t opt, ...);

/**
 * \brief Return the index run-time property flags
 * \param idx The index handle
 * \returns A bitmask of runtime options that were set
 */
ods_idx_rt_opts_t ods_idx_rt_opts_get(ods_idx_t idx);

/**
 * \brief Lock the index
 *
 * Application call to explicitly lock an Index. This call should only
 * be used if ODS_IDX_F_MP_SAFE has been set to False (0). If called
 * and ODS_IDX_F_MP_SAFE is True (1), calls to ods_idx_ functions will
 * deadlock.
 *
 * \param idx The index handle
 * \param wait A timeout interval
 * \retval 0 The lock is held
 * \retval ETIMEDOUT The timeout period expired
 * \retval EINVAL One or more of the arguments are invalid
 */
int ods_idx_lock(ods_idx_t idx, struct timespec *wait);

/**
 * \brief Unlock the index
 *
 * Application call to explicitly unlock an Index. This call should only
 * be used if ODS_IDX_F_MP_SAFE has been set to False (0).
 *
 * \param idx The index handle
 */
void ods_idx_unlock(ods_idx_t idx);

/**
 * \brief Close an index
 *
 * Close the ODS index store. If flags includes ODS_COMMIT_SYNC,
 * the function will not return until the index contents are commited
 * to stable storage. After this function returns, the idx handle is
 * no longer valid.
 *
 * This function protects against closing a NULL idx handle.
 *
 * \param idx	The index handle
 * \param flags	If flags includdes ODS_COMMIT_SYNC flag, the
 *		function will not return until all content is commited
 *		to stable storage.
 */
void ods_idx_close(ods_idx_t idx, int flags);

/**
 * \brief Return the ODS handle used by the index
 *
 * \param idx	The index handle
 * \returns ods	The ODS handle
 */
ods_t ods_idx_ods(ods_idx_t idx);

/**
 * \brief Commit the index to stable storage
 *
 * Commit all index updates to stable storage. If flags includes
 * ODS_COMMIT_SYNC, the function will not return until all changes
 * are commited to stable storage.
 *
 * This function protects against commiting a NULL idx handle.
 *
 * \param idx	The index handle
 * \param flags	If flags & ODS_COMMIT_SYNC is not zero, the
 *		function will not return until all changes are commited
 *		to stable storage.
 * \retval 0		Changes are commited
 * \retval EINPROGRESS	The changes are being written to storage
 * \retval EBADF	The specified idx handle is invalid
 */
void ods_idx_commit(ods_idx_t idx, int flags);

/**
 * \brief Implements the structure of a key
 *
 * A key is a counted array of bytes with the following format:
 * <code>
 * struct ods_key_value {
 *     uint16_t len;
 *     unsigned char value[0];
 * };
 * </code>
 * The key is simply a special format for the value portion of an ODS
 * object. A key can also be used as an argument to any function that
 * otherwise takes an ods_obj_t.
 *
 * Keys are ordered. A 'comparator' abstraction is used to define the
 * order and equivalence of two keys. A comparator is defined by a
 * name that indirectly identifies a shared library that implements
 * the comparator function. There are a set of pre-defined comparator
 * functions for which there are convenience macros to specify their
 * names as follows:
 *
 * - ODS_KEY_STRING The key is a string. The strncmp function is used
 *    to compare the two keys. If the lengths of the two keys is
 *    not equal, but they are lexically equal, the function returns the
 *    difference in length between the two keys.
 * - ODS_KEY_INT32 The key is a 32b signed integer; the comparator returns
 *    key_a - key_b.
 * - ODS_KEY_UINT32 The key is a 32b unsigned int; the comparator returns
 *    key_a - key_b.
 * - ODS_KEY_INT64 The key is a 32b signed long; the comparator returns
 *    key_a - key_b.
 * - ODS_KEY_UINT64 The key is a 64b unsigned long; the comparator returns
 *    key_a - key_b.
 *
 * These macros are used as parameters to the ods_idx_create()
 * function when the index is created.
 */
#pragma pack(2)
typedef struct ods_key_value_s {
	uint16_t len;
	unsigned char value[0];
} *ods_key_value_t;
#pragma pack()
typedef ods_obj_t ods_key_t;

/**
 * A compound key is comprised of multiple values as follows:
 * primary_key, secondary_key, tertiary_key, etc... Each key has a
 * prefix that includes a type and (if needed) a length.
 */
#pragma pack(1)
typedef struct ods_key_comp_val_str_s {
	/**
	 * The number of bytes in the \c str field below
	 */
	uint16_t len;
	/**
	 * The string/byte data. This array is not NULL terminated.
	 */
	char str[0];
} ods_key_comp_val_str_t;

typedef union ods_key_comp_val_u {
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
	struct ods_timeval_s tv_;
	ods_key_comp_val_str_t str;
} ods_key_comp_val_t;

typedef struct ods_key_comp_s {
	/**
	 * The type of this key component. By convention, these values
	 * mirror the sos_type_e enumeration
	 */
	uint16_t type;
	/**
	 * The value data for this key component
	 */
	ods_key_comp_val_t value;
} *ods_key_comp_t;

typedef struct ods_comp_key_s {	/* overloads the ods_key_value_t */
	/**
	 * The length of the entire key in bytes
	 */
	uint16_t len;
	/**
	 * An array of key components
	 */
	struct ods_key_comp_s value[0];	/* array of key components */
} *ods_comp_key_t;
#pragma pack()

/**
 * \brief Create an key in the ODS store
 *
 * A key is a counted byte array with a set of convenience routines to
 * help with getting and setting it's value based on the key type used
 * on an index.
 *
 * Keys allocated with this function come from the ODS store. See the
 * ods_key_malloc() function for keys allocated in memory
 *
 * \param idx	The index handle in which the key is being allocated
 * \param sz	The maximum size in bytes of the key value
 * \retval !0	ods_key_t pointer to the key
 * \retval 0	Insufficient resources
 */
ods_key_t _ods_key_alloc(ods_idx_t idx, size_t sz, const char *func, int line);
#define ods_key_alloc(idx, sz) _ods_key_alloc(idx, sz, __func__, __LINE__)

/**
 * \brief Copy the contents of one key to another
 *
 * \param dst The destination key
 * \param src The source key
 */
void ods_key_copy(ods_key_t dst, ods_key_t src);

/**
 * \brief Define an ODS stack key
 *
 * A key that is up to 256 bytes in length that is allocated on the
 * current stack frame. If your application uses keys that are greater
 * than this length, use the ods_key_malloc() function or redefine the
 * SOS_STACK_KEY_SIZE macro and recompile your application.
 *
 * Do not use the ods_obj_put() function to release keys of this type,
 * they will be automatically destroyed when the containing function
 * returns.
 *
 * \param _name_	The variable name to use to refer to the key.
 */
#define ODS_STACK_KEY_SIZE 256
#define ODS_KEY(_name_)					\
	struct ods_key_value_s  ## _name_ {		\
		uint16_t len;				\
		unsigned char value[ODS_STACK_KEY_SIZE];\
	} _name_ ## _ ## data;				\
	ODS_OBJ(_name_, &_name_ ## _ ## data, 256);	\

/**
 * \brief Create a memory key
 *
 * A key is just a an object with a set of convenience routines to
 * help with getting and setting it's value based on the key type used
 * on an index.
 *
 * A memory key is used to look up an object key in the ODS. The
 * storage for these keys comes from non-persistent memory. See the
 * ods_key_alloc() function for keys that are stored in the ODS.
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
extern ods_key_t _ods_key_malloc(size_t sz, const char *func, int line);
#define ods_key_malloc(sz) _ods_key_malloc(sz, __func__, __LINE__)

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
size_t ods_key_set(ods_key_t key, const void *value, size_t sz);

/**
 * \brief Return the value of a key
 *
 * \param key	The key
 * \returns Pointer to the value of the key
 */
static inline ods_key_value_t ods_key_value(ods_key_t key) { return key->as.ptr; }

/**
 * \brief Set the value of a key from a string
 *
 * \param idx	The index handle
 * \param key	The key
 * \param str	Pointer to a string
 * \retval 0	if successful
 * \retval -1	if there was an error converting the string to a value
 */
int ods_key_from_str(ods_idx_t idx, ods_key_t key, const char *str);

/**
 * \brief Return a string representation of the key value
 *
 * The string buffer passed in is assumed to be at least
 * ods_idx_key_str_size() bytes long. If the ods_idx_key_str_size()
 * function returns 0, then the key is variable size. In this
 * case, the <tt>buf</tt> parameter is ignored and ods_key_to_str()
 * function will return a buffer that it allocates internally and must
 * be freed by the caller.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param buf   The string buffer to contain the value
 * \param len   The size of the buffer in bytes
 * \return A const char * representation of the key value.
 */
const char *ods_key_to_str(ods_idx_t idx, ods_key_t key, char *buf, size_t len);

/**
 * \brief Compare two keys using the index's compare function
 *
 * \param idx	The index handle
 * \param a	The first key
 * \param b	The second key
 * \return <0	a < b
 * \return 0	a == b
 * \return >0	a > b
 */
int64_t ods_key_cmp(ods_idx_t idx, ods_key_t a, ods_key_t b);

/**
 * \brief Return the size of the index's key
 *
 * Returns the native size of the index's key values. If the key value
 * is variable size, this function returns -1. See the ods_key_len()
 * and ods_key_size() functions for the current size of the key's
 * value and the size of the key's buffer respectively.
 *
 * \return The native size of the index's keys in bytes
 */
size_t ods_idx_key_size(ods_idx_t idx);

/**
 * \brief Return the size of the key as a string
 *
 * Returns the size of the string buffer (including the terminating '\0')
 * required to contain the key when formatted as a character string.
 *
 * \param idx
 * \retval Buffer size
 */
size_t ods_idx_key_str_size(ods_idx_t idx, ods_key_t key);

/**
 * \brief Return the max size of this key's value
 *
 * \returns The size in bytes of this key's value buffer.
 */
size_t ods_key_size(ods_key_t key);

/**
 * \brief Return the length of the key's value
 *
 * Returns the current size of the key's value.
 *
 * \param key	The key
 * \return The size of the key in bytes
 */
size_t ods_key_len(ods_key_t key);

/**
 * \brief Compare two keys
 *
 * Compare two keys and return:
 *
 *    < 0 if key a is less than b,
 *      0 if key a == key b
 *    > 0 if key a is greater than key b
 *
 * Note that while a difference in the length means !=, we leave it to
 * the comparator to decide how that affects order.
 *
 * \param a	The first key
 * \param len_a	The length of the key a
 * \param b	The second key
 * \param len_b	The length of the key b
 */
typedef int64_t (*ods_idx_compare_fn_t)(ods_key_t a, ods_key_t b);

/**
 * \brief Insert a key and associated value into the index
 *
 * Insert an key and it's associated value into the index. The 'data'
 * is the value associated with the key and there is no special
 * processing or consideration given to its value other than the value
 * cannot 0. The 'key' parameter is duplicated on entry and stored in
 * the index in its ODS store. The user can reuse the key or destroy
 * it, as the key in the index has the same value but occupies
 * different storage.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param data	The key value
 *
 * \retval 0		Success
 * \retval ENOMEM	Insuffient resources
 * \retval EINVAL	The idx specified is invalid or the obj
 *			specified is 0
 */
int ods_idx_insert(ods_idx_t idx, ods_key_t key, ods_idx_data_t data);

/**
 * \brief Locate the key position and call the callback function
 *
 * Locate the position in the index corresponding to the specified
 * key. If the key is present in the index, this is the first entry
 * with this key. If the key is not present, this is the position in
 * the index wwhere the key would be located. When the callback
 * function is called, the index lock is held.
 *
 * The callback function is called with the user provided context and
 * the reference associated with the key if the key was found. If the
 * reference is NULL, they key was not found.
 *
 * This function provides an efficient means to implement functions
 * like insert-if-not-found without traversing the index twice,
 * i.e. once to determine if the key was found and a second time to
 * insert the new key. It also avoids a second lock to lock around a
 * find, and subsequent insert.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param cb_fn	The callback function to call for each matching key
 * \param ctxt	A user supplied context pointer that will be passed to
 *              the callback function
 *
 * \retval 0		Success
 * \retval ENOMEM	Insuffient resources
 * \retval EINVAL	The idx specified is invalid or the obj
 *			specified is 0
 */
typedef enum ods_visit_action {
	ODS_VISIT_ADD = 1,	/*! Add the key and set it's data to idx_data */
	ODS_VISIT_DEL,		/*! Delete the key */
	ODS_VISIT_UPD,		/*! Update the index data for key */
	ODS_VISIT_NOP		/*! Do nothing */
} ods_visit_action_t;
typedef ods_visit_action_t (*ods_visit_cb_fn_t)(ods_idx_t idx,
						ods_key_t key, ods_idx_data_t *idx_data,
						int found,
						void *arg);
int ods_idx_visit(ods_idx_t idx, ods_key_t key, ods_visit_cb_fn_t cb_fn, void *arg);

/**
 * \brief Return the index data and key value for the minimum key
 *
 * If the <tt>key</tt> parameter is not NULL, it will be set to point
 * to the ods_key_t corresponding to the minimum key value in the
 * index.
 *
 * If the <tt>idx_data</tt> parameter is not NULL, the contents of the
 * structure pointed to by ods_idx_data_t will be set to the data
 * associated with the key. If there are duplicates in the index,
 * which data value duplicate returned is undefined.
 *
 * \retval ENOENT The index is empty
 * \retval 0 The key and idx_data parameters contain the min key and data
 *
 */
int ods_idx_min(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *idx_data);

/**
 * \brief Return the index data and key value for the maximum key
 *
 * If the <tt>key</tt> parameter is not NULL, it will be set to point
 * to the ods_key_t corresponding to the maximum key value in the
 * index.
 *
 * If the <tt>idx_data</tt> parameter is not NULL, the contents of the
 * structure pointed to by ods_idx_data_t will be set to the data
 * associated with the key. If there are duplicates in the index,
 * which data value duplicate returned is undefined.
 *
 * \retval ENOENT The index is empty
 * \retval 0 The key and idx_data parameters contain the min key and data
 *
 */
int ods_idx_max(ods_idx_t idx, ods_key_t *key, ods_idx_data_t *idx_data);

/**
 * \brief Update the data value associated with a key in the index
 *
 * Update a key and it's associated value into the index. The 'data'
 * is the value associated with the key and there is no special
 * processing or consideration given to its value other than the value
 * cannot 0. The 'key' parameter is duplicated on entry and stored in
 * the index in its ODS store. The user can reuse the key or destroy
 * it, as the key in the index has the same value but occupies
 * different storage.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param data	The object reference
 *
 * \retval 0		Success
 * \retval ENOMEM	Insuffient resources
 * \retval EINVAL	The idx specified is invalid or the obj
 *			specified is 0
 */
int ods_idx_update(ods_idx_t idx, ods_key_t key, ods_idx_data_t data);

/**
 * \brief Delete a key and associated value from the index
 *
 * Delete a key and it's associated value from the index. The
 * resources associated with the 'key' are released. The 'data' is
 * the return value.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param data	Pointer to the data
 *
 * \retval 0	The key was found and removed
 * \retval ENOENT	The key was not found
 */
int ods_idx_delete(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data);

/**
 * \brief Find the specified key
 *
 * Search the index for the specified key and return the associated
 * ods_idx_data_t. If there are duplicate keys in the index, the first
 * object value the matching key is returned. See the
 * ods_iter_find_first() and ods_iter_find_last() functions for
 * information on managing duplicates.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param data	Pointer to the data
 * \retval 0	The key was found
 * \retval ENOENT	The key was not found
 */
int ods_idx_find(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data);

/**
 * \brief Find the least upper bound of the specified key
 *
 * Search the index for the least upper bound of the specified key.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param data	Pointer to the data
 *
 * \retval 0	The LUB was found and placed in ref
 * \retval ENOENT	There is no least-upper-bound
 */
int ods_idx_find_lub(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data);

/**
 * \brief Find the greatest lower bound of the specified key
 *
 * Search the index for the greatest lower bound of the specified key.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param data	Pointer to the data
 *
 * \retval 0	The GLB was placed in \c ref
 * \retval ENOENT	There is no least-upper-bound
 */
int ods_idx_find_glb(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data);


typedef struct ods_idx_stat_s {
	uint64_t cardinality;
	uint64_t duplicates;
	uint64_t size;
} *ods_idx_stat_t;

/**
 * \brief Get index statistics
 *
 * Queries index statistics and returns them in the provided
 * statistics buffer.
 *
 * \param idx The index handle
 * \param sb Pointer to the idx_stat buffer
 * \retval 0 Success
 * \retval EINVAL the idx handle is invalid
 */
int ods_idx_stat(ods_idx_t idx, ods_idx_stat_t sb);

/**
 * \brief Create an iterator
 *
 * An iterator logically represents a series over an index. It
 * maintains a current position that refers to a key/value or 0 if the
 * index is empty.
 *
 * Iterators are associated with a single index and cannot be moved
 * from one index to another.
 *
 * The current cursor position can be moved forward (lexically greater
 * keys) with the ods_iter_next() function and back (lexically lesser
 * keys) with the ods_iter_prev() function.
 *
 * \retval !0 A new iterator handle
 * \retval 0 Insufficient resources
 */
ods_iter_t ods_iter_new(ods_idx_t idx);

/**
 * \brief Return the index associated with the iterator
 * \param iter The iterator handle
 * \retval ods_idx_t used to create the iterator
 */
ods_idx_t ods_iter_idx(ods_iter_t iter);

/**
 * \brief Remove the index entry at the current iterator position
 *
 * Remove the object at the current iterator position. At exit, the
 * iterator will have been advanced to the next entry if present.
 *
 * \param iter The iterator handle
 * \param data Pointer to the index data for the entry that was removed
 * \retval 0 Success
 * \retval ENOENT The entry no longer exists
 */
int ods_iter_entry_delete(ods_iter_t iter, ods_idx_data_t *data);

/**
 * \brief Destroy an iterator
 *
 * Release the resources associated with the iterator. This function
 * has no impact on the index itself.
 *
 * \param iter The iterator handle
 */
void ods_iter_delete(ods_iter_t iter);

/**
 * \brief Position an iterator at the specified key.
 *
 * Positions the iterator's cursor to the specified key. If there are
 * duplicates in the index, the cursor will be positioned at the first
 * value with the specified key. See the ods_iter_find_last()
 * function for position the iterator at the last instance of the
 * specified key.
 *
 * \param iter	The iterator handle
 * \param key	The key for the search
 * \returns 0 Success
 * \returns ENOENT The key was not found
 */
int ods_iter_find(ods_iter_t iter, ods_key_t key);

/**
 * \brief Position an iterator at the first instance of the specified key.
 *
 * Positions the iterator's cursor at the first value with the specified
 * key. If there are duplicate keys, this function ensures that the
 * cursor is positioned at the first instance.
 *
 * \param iter	The iterator handle
 * \param key	The key for the search
 * \returns 0 Success
 * \returns ENOENT The key was not found
 */
int ods_iter_find_first(ods_iter_t iter, ods_key_t key);

/**
 * \brief Position an iterator at the last instance of the specified key.
 *
 * Positions the iterator's cursor at the last value for the specified
 * key. This function is useful when there are duplicate keys in the
 * index and the application wishes to iterate backwards.
 *
 * Use the \c ods_iter_key() an \c ods_iter_data() functions to retrieve
 * the key and associated value.
 *
 * \param iter	The iterator handle
 * \param key	The key for the search
 * \returns 0 Success
 * \returns ENOENT The key was not found
 */
int ods_iter_find_last(ods_iter_t iter, ods_key_t key);

/**
 * \brief Set the iterator cursor position
 *
 * \param iter The iterator handle
 * \param pos  The desired iterator position
 * \retval 0   Success
 * \retval EINVAL The specified position is invalid
 */
int ods_iter_pos_set(ods_iter_t iter, const ods_pos_t pos);

/**
 * \brief Get the current iterator position
 *
 * \param iter The iterator handle
 * \param pos The ods_pos_t structure that will receive the iterator position
 * \retval 0 Success
 * \retval ENOENT The iterator cursor is not at a valid position.
 */
int ods_iter_pos_get(ods_iter_t iter, ods_pos_t pos);

/**
 * \brief Indicate that this iterator position will no longer be used
 *
 * Iterator positions are saved for a period of time by the index. If the
 * application is finished with the iterator position, this function can
 * be used to indicate to the index that the position will no longer be used.
 *
 * \param iter The iterator handle
 * \param pos The iterator position
 * \retval 0 Success
 * \retval ENOENT The iterator position is invalid
 */
int ods_iter_pos_put(ods_iter_t iter, ods_pos_t pos);

/**
 * \brief Position an iterator at the least-upper-bound of the \c key.
 *
 * Use the \c ods_iter_key() an \c ods_iter_data() functions to retrieve
 * the key and associated value.
 *
 * \retval 0 Success
 * \retval ENOENT There is no least-upper-bound
 */
int ods_iter_find_lub(ods_iter_t iter, ods_key_t key);

/**
 * \brief Position an iterator at the greatest-lower-bound of the \c key.
 *
 * Use the \c ods_iter_key() an \c ods_iter_data() functions to retrieve
 * the key and associated value.
 *
 * \retval 0 Success
 * \retval ENOENT There is no greatest-lower-bound
 */
int ods_iter_find_glb(ods_iter_t iter, ods_key_t key);

typedef enum ods_iter_flags_e {
	ODS_ITER_F_ALL = 0,
	/** The iterator will skip duplicate keys in the index */
	ODS_ITER_F_UNIQUE = 1,
	ODS_ITER_F_GLB_LAST_DUP = 2,
	ODS_ITER_F_LUB_LAST_DUP = 4,
	ODS_ITER_F_MASK = 0x7
} ods_iter_flags_t;

/**
 * \brief Set iterator behavior flags
 *
 * \param i The iterator
 * \param flags The iterator flags
 * \retval 0 The flags were set successfully
 * \retval EINVAL The iterator or flags were invalid
 */
int ods_iter_flags_set(ods_iter_t i, ods_iter_flags_t flags);

/**
 * \brief Get the iterator behavior flags
 *
 * \param i The iterator
 * \retval The ods_iter_flags_t for the iterator
 */
ods_iter_flags_t ods_iter_flags_get(ods_iter_t i);

/**
 * \brief Position the iterator cursor at the first key in the index
 *
 * Positions the cursor at the first key in the index. Calling
 * ods_iter_prev() will return ENOENT.
 *
 * \param iter	The iterator handle.
 * \retval 0 Success
 * \retval ENOENT The index is empty
 */
int ods_iter_begin(ods_iter_t iter);

/**
 * \brief Position the iterator at the last key in the index
 *
 * Positions the cursor at the last key in the index. Calling
 * ods_iter_next() will return ENOENT.
 *
 * Use the ods_iter_key() an ods_iter_data() functions to retrieve
 * the object reference and associated key.
 *
 * \param iter	The iterator handle.
 * \retval 0 Success
 * \retval ENOENT The index is empty
 */
int ods_iter_end(ods_iter_t iter);

/**
 * \brief Move the iterator cursor to the next key in the index
 *
 * Use the ods_iter_key() an ods_iter_data() functions to retrieve
 * the object reference and associated key.
 *
 * \param iter	The iterator handle
 * \retval 0 Success
 * \retval ENOENT At the end of the index or the index is empty
 */
int ods_iter_next(ods_iter_t iter);

/**
 * \brief Move the iterator cursor to the previous key in the index
 *
 * Use the ods_iter_key() an ods_iter_data() functions to retrieve
 * the key and associated data.
 *
 * \param iter	The iterator handle
 * \retval 0 Success
 * \retval ENOENT Already at the beginning of the index or the index is empty
 */
int ods_iter_prev(ods_iter_t iter);

/**
 * \brief Returns the key associated with current cursor position
 *
 * \param iter	The iterator handle
 * \return !0	Pointer to the key
 * \return 0	The cursor is not positioned at an object
 */
ods_key_t ods_iter_key(ods_iter_t iter);

/**
 * \brief Returns the data associated with current cursor position
 *
 * \param iter	The iterator handle
 * \return !0	The object reference
 * \return 0	The cursor is not positioned at an object
 */
ods_idx_data_t ods_iter_data(ods_iter_t iter);

/**
 * \brief Print a textual representation of the index
 *
 * Print a textual representation of the index to the specified FILE
 * pointer. This function is intended for debug purposes when writing
 * new index plugins.
 *
 * \param idx	The index handle
 * \param fp	The file pointer
 */
void ods_idx_print(ods_idx_t idx, FILE* fp);
/*
 * \brief Print Index information
 *
 * Print information about the index
 *
 * \param idx	The index handle
 * \param file	The file pointer
 */
void ods_idx_info(ods_idx_t idx, FILE* fp);
#ifdef __cplusplus
}
#endif

#endif
