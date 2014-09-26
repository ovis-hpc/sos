/*
 * Copyright (c) 2013 Open Grid Computing, Inc. All rights reserved.
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
#include "ods.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ods_idx *ods_idx_t;
typedef struct ods_iter *ods_iter_t;

#define ODS_IDX_SIGNATURE	"ODSIDX00"
#define ODS_IDX_BPTREE		"BPTREE"
#define ODS_IDX_RADIXTREE	"RADIXTREE"
#define ODS_IDX_RBTREE		"RBTREE"

/**
 * \brief Create an object index
 *
 * An index implements a persistent key/value store in an ODS data
 * store. The key is a generic ods_key_t and the value is an
 * ods_ref_t in the associated ODS object store. An ODS ods_ref_t can
 * be transformed back and forth between references and memory
 * pointers using the ods_ods_ref_to_ptr() and ods_obj_ptr_to_ref()
 * functions respectively.
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
 * \param ... Optional index type specific parameters
 *
 * \return 0	The index was successfully create. Use ods_idx_open()
 *		to access the index.
 * \return !0	An errno indicating the reason for failure.
 */
int ods_idx_create(const char *path, int mode,
		   const char *type, const char *key,
		   ...);

/**
 * \brief Open an object index
 *
 * An index implements a persistent key/value store in an ODS data
 * store. The key is a generic ods_key_t and the value is an
 * ods_ref_t in the associated ODS object store. An ODS ods_ref_t can
 * be transformed back and forth between references and memory
 * pointers using the ods_ods_ref_to_ptr() and ods_obj_ptr_to_ref()
 * functions respectively.
 *
 * \param path	The path to the ODS store
 * \retval !0	The ods_idx_t handle for the index
 * \retval 0	The index file could not be opened. See the
 *		errno for the reason.
 */
ods_idx_t ods_idx_open(const char *path);

/**
 * \brief Close an object index
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
 * \brief Implements the structure of an object key
 *
 * An object key is a counted array of bytes with the following format:
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
 * \brief Create an key in the ODS store
 *
 * A key is just a an object with a set of convenience routines to
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
enum ods_key_type { 
	ODS_KEY_PERSISTENT,
	ODS_KEY_MEMORY
} ods_key_type_t;
ods_key_t _ods_key_alloc(ods_idx_t idx, size_t sz);
#define ods_key_alloc(idx, sz) ({		\
	ods_key_t k = _ods_key_alloc(idx, sz);	\
	if (k) {				\
		k->alloc_line = __LINE__;	\
		k->alloc_func = __func__;	\
	}					\
	k;					\
})

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
 * \param idx	The index handle in which the key is being allocated
 * \param sz	The maximum size in bytes of the key value
 * \retval !0	ods_key_t pointer to the key
 * \retval 0	Insufficient resources
 */
ods_key_t _ods_key_malloc(ods_idx_t idx, size_t sz);
#define ods_key_malloc(idx, sz) ({		\
	ods_key_t k = _ods_key_malloc(idx, sz);	\
	if (k) {				\
		k->alloc_line = __LINE__;	\
		k->alloc_func = __func__;	\
	}					\
	k;					\
})

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
size_t ods_key_set(ods_key_t key, void *value, size_t sz);

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
 * \param idx	The index handle
 * \param key	The key
 * \return A const char * representation of the key value.
 */
const char *ods_key_to_str(ods_idx_t idx, ods_key_t key);

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
int ods_key_cmp(ods_idx_t idx, ods_key_t a, ods_key_t b);

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
typedef int (*ods_idx_compare_fn_t)(ods_key_t a, ods_key_t b);

/**
 * \brief Insert an object and associated key into the index
 *
 * Insert an object and it's associated key into the index. The 'obj'
 * is the value associated with the key and there is no special
 * processing or consideration given to its value other than the value
 * cannot 0. The 'key' parameter is duplicated on entry and stored in
 * the index in its ODS store. The user can reuse the key or destroy
 * it, as the key in the index has the same value but occupies
 * different storage.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param obj	The object reference
 *
 * \retval 0		Success
 * \retval ENOMEM	Insuffient resources
 * \retval EINVAL	The idx specified is invalid or the obj
 *			specified is 0
 */
int ods_idx_insert(ods_idx_t idx, ods_key_t key, ods_ref_t obj);

/**
 * \brief Update the object associated with a key in the index
 *
 * Update an object and it's associated key into the index. The 'obj'
 * is the value associated with the key and there is no special
 * processing or consideration given to its value other than the value
 * cannot 0. The 'key' parameter is duplicated on entry and stored in
 * the index in its ODS store. The user can reuse the key or destroy
 * it, as the key in the index has the same value but occupies
 * different storage.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param obj	The object reference
 *
 * \retval 0		Success
 * \retval ENOMEM	Insuffient resources
 * \retval EINVAL	The idx specified is invalid or the obj
 *			specified is 0
 */
int ods_idx_update(ods_idx_t idx, ods_key_t key, ods_ref_t obj);

/**
 * \brief Delete an object and associated key from the index
 *
 * Delete an object and it's associated key from the index. The
 * resources associated with the 'key' are released. The 'obj' is
 * the return value.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param ref	Pointer to the reference
 *
 * \retval 0	The key was found and removed
 * \retval ENOENT	The key was not found
 */
int ods_idx_delete(ods_idx_t idx, ods_key_t key, ods_ref_t *ref);

/**
 * \brief Find the object associated with the specified key
 *
 * Search the index for the specified key and return the associated
 * ods_ref_t.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param ref	Pointer to the reference
 * \retval 0	The key was found
 * \retval ENOENT	The key was not found
 */
int ods_idx_find(ods_idx_t idx, ods_key_t key, ods_ref_t *ref);

/**
 * \brief Find the least upper bound of the specified key
 *
 * Search the index for the least upper bound of the specified key.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param ref	Pointer to the reference
 *
 * \retval 0	The LUB was found and placed in ref
 * \retval ENOENT	There is no least-upper-bound
 */
int ods_idx_find_lub(ods_idx_t idx, ods_key_t key, ods_ref_t *ref);

/**
 * \brief Find the greatest lower bound of the specified key
 *
 * Search the index for the greatest lower bound of the specified key.
 *
 * \param idx	The index handle
 * \param key	The key
 * \param ref	Pointer to the reference
 *
 * \retval 0	The GLB was placed in \c ref
 * \retval ENOENT	There is no least-upper-bound
 */
int ods_idx_find_glb(ods_idx_t idx, ods_key_t key, ods_ref_t *ref);

/**
 * \brief Create an iterator
 *
 * An iterator logically represents a series over an index. It
 * maintains a current position that refers to an object or 0 if the
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
 * \brief Destroy an iterator
 *
 * Release the resources associated with the iterator. This function
 * has no impact on the index itself.
 *
 * \param iter	The iterator handle
 */
void ods_iter_delete(ods_iter_t iter);

/**
 * \brief Position an iterator at the specified key.
 *
 * Positions the iterator's cursor to the object with the specified
 * key.
 *
 * Use the \c ods_iter_key() an \c ods_iter_ref() functions to retrieve
 * the object reference and associated key.
 *
 * \param iter	The iterator handle
 * \param key	The key for the search
 * \retval ENOENT The specified key was not found in the index
 * \retval 0	The iterator position now points to the object associated
 *		with the specified key
 */
int ods_iter_find(ods_iter_t iter, ods_key_t key);

/**
 * \brief Position an iterator at the least-upper-bound of the \c key.
 *
 * Use the \c ods_iter_key() an \c ods_iter_ref() functions to retrieve
 * the object reference and associated key.
 *
 * \retval ENOENT if there is no least-upper-bound record.
 * \retval 0 if there the iterator successfully positioned at the
 *		least-upper-bound record.
 *
 */
int ods_iter_find_lub(ods_iter_t iter, ods_key_t key);

/**
 * \brief Position an iterator at the greatest-lower-bound of the \c key.
 *
 * Use the \c ods_iter_key() an \c ods_iter_ref() functions to retrieve
 * the object reference and associated key.
 *
 * \retval ENOENT if there is no greatest-lower-bound record.
 * \retval 0 if there the iterator successfully positioned at the
 *		least-upper-bound record.
 *
 */
int ods_iter_find_glb(ods_iter_t iter, ods_key_t key);

/**
 * \brief Position the iterator cursor at the first object in the index
 *
 * Positions the cursor at the first object in the index. Calling
 * ods_iter_prev() will return ENOENT.
 *
 * Use the ods_iter_key() an ods_iter_ref() functions to retrieve
 * the object reference and associated key.
 *
 * \param iter	The iterator handle.
 * \return ENOENT The index is empty
 * \return 0 The cursor is positioned at the first object in the index
 */
int ods_iter_begin(ods_iter_t iter);

/**
 * \brief Position the iterator cursor at the last object in the index
 *
 * Positions the cursor at the last object in the index. Calling
 * ods_iter_next() will return ENOENT.
 *
 * Use the ods_iter_key() an ods_iter_ref() functions to retrieve
 * the object reference and associated key.
 *
 * \param iter	The iterator handle.
 * \retval ENOENT The index is empty
 * \retval 0 The cursor is positioned at the last object in the index
 */
int ods_iter_end(ods_iter_t iter);

/**
 * \brief Move the iterator cursor to the next object in the index
 *
 * Use the ods_iter_key() an ods_iter_ref() functions to retrieve
 * the object reference and associated key.
 *
 * \param iter	The iterator handle
 * \return ENOENT The cursor is at the end of the index
 * \return 0	The cursor is positioned at the next object
 */
int ods_iter_next(ods_iter_t iter);

/**
 * \brief Move the iterator cursor to the previous object in the index
 *
 * Use the ods_iter_key() an ods_iter_ref() functions to retrieve
 * the object reference and associated key.
 *
 * \param iter	The iterator handle
 * \return ENOENT The cursor is at the beginning of the index
 * \return 0	The cursor is positioned at the next object
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
 * \brief Returns the object reference associated with current cursor position
 *
 * \param iter	The iterator handle
 * \return !0	The object reference
 * \return 0	The cursor is not positioned at an object
 */
ods_ref_t ods_iter_ref(ods_iter_t iter);

#ifdef __cplusplus
}
#endif

#endif
