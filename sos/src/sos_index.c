/* -*- c-basic-offset : 8 -*-
 * Copyright (c) 2015 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2015 Sandia Corporation. All rights reserved.
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
/**
 * \page indices Indices
 * \section indices Indices
 *
 * An Index keeps a named, ordered collection of Key/Value references to
 * Objects. The Key is define by the type sos_key_t and the value is
 * defined by the type sos_obj_ref_t. The Value is a Reference to an
 * Object.
 *
 * Some Indices are associated with an Attribute of a Schema. This is
 * a convenience since internal to SOS, all of these collections are
 * implemented as an Index. Whether a Schema attribute has an Index is
 * specified when the Schema is created.
 *
 * Other Indices are created directly with the sos_index_new()
 * function. These Indices are primarily used when a complex key is
 * required that is based on the value of more than a single
 * Attribute. For example, if a use requires an ordering by Job +
 * Time, a JobTime Index may be created where the key is the
 * concatenation of the Job Id and the Unix Timestamp.
 *
 * The functions for managing Indices include the following:
 *
 * - sos_index_new() Create a new index
 * - sos_index_open() Open an existing index
 * - sos_index_insert() Insert an Object into an Index
 * - sos_index_find() Find an object in the index with the specified key
 * - sos_index_find_inf() Find the object inferior (i.e. greatest lower bound) to the specified key
 * - sos_index_find_sup() Find the object superior (i.e. least upper bound) orf the specified key
 * - sos_index_commit() Commit index changes to stable storage
 * - sos_index_close() Close the index
 * - sos_index_key_size() Return the size of a key on this index
 * - sos_index_key_new() Create a key for this index
 * - sos_index_key_from_str() Assign a value to a key from a string
 * - sos_index_key_to_str() Return a formated string representation of a key
 * - sos_index_key_cmp() Compare two keys
 * - sos_index_print() Print an internal representation of the index
 * - sos_index_update() Update a key and associated data
 * - sos_container_index_list() Print a list of indices defined on the container
 * - sos_container_index_iter_new() Create a container index iterator
 * - sos_container_index_iter_free() Destroy a container index iterator
 * - sos_container_index_iter_first() Return the first index in the container
 * - sos_container_index_iter_next() Return the next index on the iterator
 */
#define _GNU_SOURCE
#include <sys/types.h>
#include <stdlib.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include <sys/queue.h>
#include "sos_priv.h"

static sos_index_t __sos_index_alloc(sos_t sos, const char *name)
{
	sos_index_t index = calloc(1, sizeof *index);
	if (!index)
		return NULL;
	index->sos = sos;
	strcpy(index->name, name);

	return index;
}

/**
 * \brief Create a new Index
 *
 * \param sos The container handle
 * \param name The unique index name
 * \param idx_type The index type, e.g. "BXTREE"
 * \param key_type The key type, e.g. "UINT64"
 * \param idx_args The index type specific arguments, e.g. "ORDER=5" for a B+Tree
 * \retval 0 Success
 * \retval !0 A Unix error code
 */
int sos_index_new(sos_t sos, const char *name,
		  const char *idx_type, const char *key_type,
		  const char *idx_args)
{
	int rc;
	sos_obj_ref_t idx_ref;
	ods_obj_t idx_obj;
	SOS_KEY(idx_key);

	size_t len = strlen(name);
	if (len >= SOS_INDEX_NAME_LEN)
		return EINVAL;

	len = strlen(idx_type);
	if (len >= SOS_INDEX_TYPE_LEN)
		return EINVAL;

	len = strlen(key_type);
	if (len >= SOS_INDEX_KEY_TYPE_LEN)
		return EINVAL;

	if (!idx_args)
		idx_args = "";
	len = strlen(idx_args);
	if (len >= SOS_INDEX_ARGS_LEN)
		return EINVAL;

	ods_lock(sos->idx_ods, 0, NULL);
	ods_key_set(idx_key, name, strlen(name)+1);
	rc = ods_idx_find(sos->idx_idx, idx_key, &idx_ref.idx_data);
	if (!rc) {
		rc = EEXIST;
		goto out;
	}
 retry:
	idx_obj = ods_obj_alloc(sos->idx_ods, sizeof(struct sos_idx_data_s));
	if (!idx_obj) {
		rc = ods_extend(sos->idx_ods, ods_size(sos->idx_ods) * 2);
		if (!rc)
			goto retry;
		rc = ENOMEM;
		goto out;
	}
	SOS_IDX(idx_obj)->ref_count = 1;
	SOS_IDX(idx_obj)->mode = 0664;
	strcpy(SOS_IDX(idx_obj)->name, name);
	strcpy(SOS_IDX(idx_obj)->idx_type, idx_type);
	strcpy(SOS_IDX(idx_obj)->key_type, key_type);
	strcpy(SOS_IDX(idx_obj)->args, idx_args);
	ods_obj_update(idx_obj);
	uuid_clear(idx_ref.ref.part_uuid);
	idx_ref.ref.obj = ods_obj_ref(idx_obj);
	rc = ods_idx_insert(sos->idx_idx, idx_key, idx_ref.idx_data);
	if (rc)
		goto err_0;
	ods_obj_put(idx_obj);
 out:
	ods_unlock(sos->idx_ods, 0);
	ods_commit(sos->idx_ods, ODS_COMMIT_SYNC);
	ods_idx_commit(sos->idx_idx, ODS_COMMIT_SYNC);
	return rc;
 err_0:
	ods_obj_delete(idx_obj);
	ods_obj_put(idx_obj);
	ods_unlock(sos->idx_ods, 0);
	return rc;
}

ods_idx_t _open_or_creat(sos_t sos, char *path,
			sos_perm_t be,
			uint32_t mode,
			const char *idx_type,
			const char *key_type,
			const char *args)
{
	int rc;
	ods_idx_t idx = ods_idx_open(path, sos->o_perm);
	if (idx)
		return idx;
	/* Attempt to create if it does not exist. */
	if (errno != ENOENT)
		return NULL;
	rc = ods_idx_create(path, sos->o_perm | be, mode, idx_type, key_type, args);
	if (!rc)
		return ods_idx_open(path, sos->o_perm);
	return NULL;
}

int __sos_index_reopen(sos_index_t index)
{
	ods_idx_ref_t idx_ref;
	sos_part_t part;
	char tmp_path[PATH_MAX];
	sos_part_iter_t iter;
	ods_idx_t idx;
	ods_idx_ref_t ods_idx_ref;
	int rc = 0;

	/* Close all the existing ODS */
	while (!LIST_EMPTY(&index->active_idx_list)) {
		idx_ref = LIST_FIRST(&index->active_idx_list);
		LIST_REMOVE(idx_ref, entry);
		ods_idx_close(idx_ref->idx, ODS_COMMIT_ASYNC);
		sos_part_put(idx_ref->part);
		free(idx_ref);
	}
	/* Iterate and reopen for each partition */
	iter = sos_part_iter_new(index->sos);
	if (!iter)
		return errno;
	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {
		if (sos_part_state(part) == SOS_PART_STATE_OFFLINE) {
			sos_part_put(part);
			continue;
		}
		sprintf(tmp_path, "%s/%s_idx", sos_part_path(part), index->name);
		idx = _open_or_creat(index->sos, tmp_path,
				     sos_part_be_get(part),
				     SOS_IDX(index->idx_obj)->mode,
				     SOS_IDX(index->idx_obj)->idx_type,
				     SOS_IDX(index->idx_obj)->key_type,
				     SOS_IDX(index->idx_obj)->args);
		if (!idx) {
			rc = errno;
			sos_error("The index %s could not be "
				  "opened or created.\n",
				  tmp_path);
			sos_part_put(part);
			goto err_0;
		}
		ods_idx_ref = malloc(sizeof *ods_idx_ref);
		if (!ods_idx_ref) {
			rc = ENOMEM;
			goto err_1;
		}
		ods_idx_ref->idx = idx;
		ods_idx_ref->part = part;
		LIST_INSERT_HEAD(&index->active_idx_list, ods_idx_ref, entry);
	}
	sos_part_iter_free(iter);
	index->part_gn = index->sos->part_gn;
	return 0;
 err_1:
	ods_idx_close(idx, ODS_COMMIT_SYNC);
 err_0:
	while (!LIST_EMPTY(&index->active_idx_list)) {
		ods_idx_ref = LIST_FIRST(&index->active_idx_list);
		LIST_REMOVE(ods_idx_ref, entry);
		ods_idx_close(ods_idx_ref->idx, ODS_COMMIT_ASYNC);
		sos_part_put(ods_idx_ref->part);
	}
	sos_part_iter_free(iter);
	return rc;
}

/**
 * \brief Open an Index
 *
 * \param sos The container handle
 * \param name The unique index name
 * \retval 0 Success
 * \retval !0 A Unix error code
 */
sos_index_t sos_index_open(sos_t sos, const char *name)
{
	size_t name_len;
	char tmp_path[PATH_MAX];
	sos_obj_ref_t idx_ref;
	ods_obj_t idx_obj = NULL;
	SOS_KEY(idx_key);
	ods_idx_t idx;
	sos_index_t index;
	sos_part_iter_t iter = NULL;
	int rc;
	sos_part_t part;
	ods_idx_ref_t ods_idx_ref;

	name_len = strlen(name);
	if (name_len >= SOS_INDEX_NAME_LEN) {
		errno = EINVAL;
		goto err_0;
	}

	index = __sos_index_alloc(sos, name);
	if (!index)
		goto err_1;

	ods_key_set(idx_key, name, strlen(name)+1);
	ods_lock(sos->idx_ods, 0, NULL);
	rc = ods_idx_find(sos->idx_idx, idx_key, &idx_ref.idx_data);
	if (rc) {
		errno = rc;
		goto err_2;
	}

	idx_obj = ods_ref_as_obj(sos->idx_ods, idx_ref.ref.obj);
	if (!idx_obj) {
		errno = EINVAL;
		goto err_2;
	}

	iter = sos_part_iter_new(sos);
	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {
		if (sos_part_state(part) == SOS_PART_STATE_OFFLINE) {
			sos_part_put(part);
			continue;
		}
		sprintf(tmp_path, "%s/%s_idx", sos_part_path(part), name);
		idx = _open_or_creat(sos, tmp_path,
				sos_part_be_get(part),
				SOS_IDX(idx_obj)->mode,
				SOS_IDX(idx_obj)->idx_type,
				SOS_IDX(idx_obj)->key_type,
				SOS_IDX(idx_obj)->args);
		if (!idx) {
			sos_part_put(part);
			goto err_3;
		}
		ods_idx_ref = malloc(sizeof *ods_idx_ref);
		if (!ods_idx_ref) {
			errno = ENOMEM;
			goto err_4;
		}
		ods_idx_ref->idx = idx;
		ods_idx_ref->part = part;
		LIST_INSERT_HEAD(&index->active_idx_list, ods_idx_ref, entry);
	}
	index->idx_obj = idx_obj;
	index->part_gn = index->sos->part_gn;
	ods_unlock(sos->idx_ods, 0);
	sos_part_iter_free(iter);
	return index;
 err_4:
	ods_idx_close(idx, ODS_COMMIT_SYNC);
 err_3:
	ods_obj_put(idx_obj);
 err_2:
	ods_unlock(sos->idx_ods, 0);
 err_1:
	while (!LIST_EMPTY(&index->active_idx_list)) {
		ods_idx_ref = LIST_FIRST(&index->active_idx_list);
		LIST_REMOVE(ods_idx_ref, entry);
		ods_idx_close(ods_idx_ref->idx, ODS_COMMIT_ASYNC);
		sos_part_put(ods_idx_ref->part);
	}
	free(index);
 err_0:
	sos_part_iter_free(iter);
	return NULL;
}


int sos_index_rt_opt_set(sos_index_t idx, sos_index_rt_opt_t opt, ...)
{
	ods_idx_ref_t iref;
	int rc = 0;
	va_list ap;
	LIST_FOREACH(iref, &idx->active_idx_list, entry) {
		va_start(ap, opt);
		rc = ods_idx_rt_opts_set_va(iref->idx, opt, ap);
		va_end(ap);
	}
	return rc;
}

struct sos_visit_cb_ctxt_s {
	sos_index_t index;
	sos_visit_cb_fn_t cb_fn;
	void *arg;
};

static ods_visit_action_t visit_cb(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data, int found, void *arg)
{
	struct sos_visit_cb_ctxt_s *visit_arg = arg;
	int rc = (ods_visit_action_t)visit_arg->cb_fn(visit_arg->index,
						      key, (sos_idx_data_t *)data,
						      found, visit_arg->arg);
	free(visit_arg);
	return rc;
}
/**
 * \brief Visit the key entry in the index
 *
 * Find the key in the index and call the visit callback function with
 * the index, the key, a 1 or 0 to indicate if the key was found, a
 * pointer to a sos_idx_data_t structure, and the ctxt passed to
 * sos_index_update(). If the found parameter to the cb_fn() is 1,
 * the the idx_data structure contains the index data for the key. If
 * the found parameter is 0, the contents of this structure are
 * undefined.
 *
 * The cb_fn() must return one of the following value:
 *
 * - SOS_VISIT_ADD - Add the key and set the key data to the contents
 *                   of the idx_data structure. This return value is
 *                   ignored if found is true and sos_index_visit()
 *                   will return EINVAL.
 * - SOS_VISIT_DEL - Delete the key from the index. If found is false,
 *                   this value is ignored and sos_index_visit()
 *                   will return EINVAL.
 * - SOS_VISIT_UPD - Update the index data at this key with the
 *                   contents of the idx_data structure. If found
 *                   is false, this value is ignored and
 *                   sos_index_visit() will return EINVAL.
 * - SOS_VISIT_NOP - Do nothing
 */
int sos_index_visit(sos_index_t index, sos_key_t key, sos_visit_cb_fn_t cb_fn, void *arg)
{
	int rc = 0;
	ods_idx_ref_t iref;
	LIST_FOREACH(iref, &index->active_idx_list, entry) {
		struct sos_visit_cb_ctxt_s *ctxt = malloc(sizeof *ctxt);
		ctxt->index = index;
		ctxt->cb_fn = cb_fn;
		ctxt->arg = arg;
		int irc = ods_idx_visit(iref->idx, key, visit_cb, ctxt);
		if (irc != SOS_VISIT_NOP) {
			rc = irc;
			break;
		}
		if (irc && !rc)
			rc = irc;
	}
	return rc;
}
/**
 * \brief Add an object to an index
 *
 * \param index The index handle
 * \param key The key
 * \param obj The object to which the key will refer
 * \retval 0 Success
 * \retval EINVAL The object or index is invalid
 * \retval ENOSPC The index for the object partition is missing
 * \retval !0 A Unix error code
 */
int sos_index_insert(sos_index_t index, sos_key_t key, sos_obj_t obj)
{
	int rc;
	if (NULL == obj->obj->ods) {
		rc = sos_obj_commit(obj);
		if (rc)
			return rc;
	}
	if (!ods_ref_valid(obj->obj->ods, obj->obj_ref.ref.obj))
		return EINVAL;
	if (index->part_gn != index->sos->part_gn) {
		int rc = __sos_index_reopen(index);
		if (rc)
			return rc;
	}
	sos_obj_commit(obj);
	ods_obj_update(obj->obj);
	ods_idx_t idx = __sos_idx_find(index, obj);
	if (idx)
		return ods_idx_insert(idx, key, obj->obj_ref.idx_data);
	return ENOSPC;
}

/**
 * \brief Remove a key from an index
 *
 * Remove a key/value from the index. Note that the function takes an
 * object as a paramter. This is necessary to discriminate when
 * multiple objects are referred to by the same key.
 *
 * \param index The index handle
 * \param key The key
 * \param obj The specific object to which the key will refer
 * \retval 0 Success
 * \retval !0 A Unix error code
 */
int sos_index_remove(sos_index_t index, sos_key_t key, sos_obj_t obj)
{
	ods_idx_data_t data;
	data = obj->obj_ref.idx_data;
	ods_idx_t idx = __sos_idx_find(index, obj);
	if (!idx)
		return ENOENT;
	return ods_idx_delete(idx, key, &data);
}

/**
 * \brief Find an object in the index by its key
 *
 * \param index The index handle
 * \param key The key
 * \retval !NULL The object associated with the key
 * \retval NULL The object was not found
 */
sos_obj_t sos_index_find(sos_index_t index, sos_key_t key)
{
	int rc;
	sos_obj_ref_t obj_ref;
	sos_obj_t obj;
	ods_idx_ref_t idx_ref;
	LIST_FOREACH(idx_ref, &index->active_idx_list, entry) {
		rc = ods_idx_find(idx_ref->idx, key, &obj_ref.idx_data);
		if (rc)
			continue;
		obj = sos_ref_as_obj(index->sos, obj_ref);
		if (!obj)
			continue;
		return obj;
	}
	errno = ENOENT;
	return NULL;
}

/**
 * \brief Find a key in the index and return the ref
 *
 * \param index The index handle
 * \param key The key
 * \param ref Pointer to sos_obj_ref_t
 * \retval 0 The key was found and ref contains the data
 * \retval ENOENT The key was not found
 */
int sos_index_find_ref(sos_index_t index, sos_key_t key, sos_obj_ref_t *ref)
{
	int rc = ENOENT;
	ods_idx_ref_t idx_ref;
	LIST_FOREACH(idx_ref, &index->active_idx_list, entry) {
		rc = ods_idx_find(idx_ref->idx, key, &ref->idx_data);
		if (!rc)
			return rc;
	}
	return rc;
}

/**
 * \brief Find a key in the index for the attribute and return the associated object
 *
 * \param index The index handle
 * \param key The key
 * \returns The sos_obj_t handle for the object
 */
sos_obj_t sos_obj_find(sos_attr_t attr, sos_key_t key)
{
	sos_obj_t obj;
	ods_idx_ref_t idx_ref;
	sos_index_t index = sos_attr_index(attr);
	if (!index)
		return NULL;
	LIST_FOREACH(idx_ref, &index->active_idx_list, entry) {
		obj = sos_index_find(index, key);
		if (obj)
			return obj;
	}
	return NULL;
}

/**
 * \brief Find the minima in the index and return the associated object
 *
 * \param index The index handle
 * \param pkey If not NULL, a pointer to a sos_key_t to receive the min key value
 * \returns The sos_obj_t handle for the object
 */
sos_obj_t sos_index_find_min(sos_index_t index, sos_key_t *pkey)
{
	sos_obj_t obj;
	sos_key_t key, min_key = NULL;
	sos_obj_ref_t obj_ref;
	sos_obj_ref_t min_obj_ref;
	ods_idx_ref_t idx_ref;
	int rc;

	LIST_FOREACH(idx_ref, &index->active_idx_list, entry) {
		rc = ods_idx_min(idx_ref->idx, &key, &obj_ref.idx_data);
		if (rc)
			continue;
		if (min_key) {
			rc = sos_index_key_cmp(index, key, min_key);
			if (rc < 0) {
				/* New min */
				sos_key_put(min_key);
				min_key = key;
				min_obj_ref = obj_ref;
			} else if (key) {
				sos_key_put(key);
			}
		} else {
			min_key = key;
			min_obj_ref = obj_ref;
		}
	}
	*pkey = min_key;
	obj = sos_ref_as_obj(index->sos, min_obj_ref);
	if (!obj)
		errno = ENOMEM;
	return obj;
}

/**
 * \brief Find the maxima in the index and return the associated object
 *
 * \param index The index handle
 * \param pkey If not NULL, a pointer to a sos_key_t to receive the max key value
 * \returns The sos_obj_t handle for the object
 */
sos_obj_t sos_index_find_max(sos_index_t index, sos_key_t *pkey)
{
	sos_obj_t obj;
	sos_key_t key, max_key = NULL;
	sos_obj_ref_t obj_ref;
	sos_obj_ref_t max_obj_ref;
	ods_idx_ref_t idx_ref;
	int rc;

	LIST_FOREACH(idx_ref, &index->active_idx_list, entry) {
		rc = ods_idx_max(idx_ref->idx, &key, &obj_ref.idx_data);
		if (rc)
			continue;
		if (max_key) {
			rc = sos_index_key_cmp(index, key, max_key);
			if (rc > 0) {
				/* New max */
				sos_key_put(max_key);
				max_key = key;
				max_obj_ref = obj_ref;
			} else if (key) {
				sos_key_put(key);
			}
		} else {
			max_key = key;
			max_obj_ref = obj_ref;
		}
	}
	*pkey = max_key;
	obj = sos_ref_as_obj(index->sos, max_obj_ref);
	if (!obj)
		errno = ENOMEM;
	return obj;
}

/**
 * \brief Find the first object that its key attribute is greater than or equal
 *        to the given key.
 *
 * \param index The index handle
 * \param key   The key
 * \retval obj  The object associated with the key
 * \retval NULL The object was not found
 */
sos_obj_t sos_index_find_ge(sos_index_t index, sos_key_t key)
{
	sos_obj_t obj;
	sos_key_t min_key = NULL;
	sos_obj_ref_t obj_ref;
	sos_obj_ref_t min_obj_ref;
	ods_idx_ref_t idx_ref;
	int rc;

	LIST_FOREACH(idx_ref, &index->active_idx_list, entry) {
		rc = ods_idx_find_lub(idx_ref->idx, key, &obj_ref.idx_data);
		if (rc)
			continue;
		if (min_key) {
			rc = sos_index_key_cmp(index, key, min_key);
			if (rc < 0) {
				/* New min */
				sos_key_put(min_key);
				min_key = key;
				min_obj_ref = obj_ref;
			} else if (key) {
				sos_key_put(key);
			}
		} else {
			min_key = key;
			min_obj_ref = obj_ref;
		}
	}
	obj = sos_ref_as_obj(index->sos, min_obj_ref);
	if (!obj)
		errno = ENOMEM;
	return obj;
}

/**
 * \brief Find the first object that its key attribute is less than or equal
 *        to the given key.
 *
 * \param index The index handle
 * \param key   The key
 * \retval obj  The object associated with the key
 * \retval NULL The object was not found
 */
sos_obj_t sos_index_find_le(sos_index_t index, sos_key_t key)
{
	sos_obj_t obj;
	sos_key_t max_key = NULL;
	sos_obj_ref_t obj_ref;
	sos_obj_ref_t max_obj_ref;
	ods_idx_ref_t idx_ref;
	int rc;

	LIST_FOREACH(idx_ref, &index->active_idx_list, entry) {
		rc = ods_idx_find_glb(idx_ref->idx, key, &obj_ref.idx_data);
		if (rc)
			continue;
		if (max_key) {
			rc = sos_index_key_cmp(index, key, max_key);
			if (rc > 0) {
				/* New max */
				sos_key_put(max_key);
				max_key = key;
				max_obj_ref = obj_ref;
			} else if (key) {
				sos_key_put(key);
			}
		} else {
			max_key = key;
			max_obj_ref = obj_ref;
		}
	}
	obj = sos_ref_as_obj(index->sos, max_obj_ref);
	if (!obj)
		errno = ENOMEM;
	return obj;
}

/**
 * \brief Find the supremum (least upper bound) of the specified key
 *
 * \note This is the same as \c sos_index_find_ge().
 *
 * \param index The index handle
 * \param key The key
 * \retval !NULL The object associated with the key
 * \retval NULL The object was not found
 */
sos_obj_t sos_index_find_sup(sos_index_t index, sos_key_t key)
{
	return sos_index_find_ge(index, key);
}

/**
 * \brief Find the infimum (greatest lower bound) of the specified key
 *
 * \note This is the same as \c sos_index_find_le().
 *
 * \param index The index handle
 * \param key The key
 * \retval !NULL The object associated with the key
 * \retval NULL The object was not found
 */
sos_obj_t sos_index_find_inf(sos_index_t index, sos_key_t key)
{
	return sos_index_find_le(index, key);
}

/**
 * @brief Close the index and commit changes to storage
 *
 * @param index The index handle
 * @param flags The commit flags: SOS_COMMIT_SYNC, SOS_COMMIT_ASYNC
 * @retval 0 Success
 * @retval !0 A unix errno
 */
int sos_index_close(sos_index_t index, sos_commit_t flags)
{
	ods_idx_ref_t idx_ref;
	if (!index)
		return EINVAL;
	while (!LIST_EMPTY(&index->active_idx_list)) {
		idx_ref = LIST_FIRST(&index->active_idx_list);
		LIST_REMOVE(idx_ref, entry);
		ods_idx_close(idx_ref->idx, ODS_COMMIT_ASYNC);
		sos_part_put(idx_ref->part);
		free(idx_ref);
	}
	ods_obj_put(index->idx_obj);
	free(index);
	return 0;
}

/**
 * \brief Return the size of the index's key
 *
 * Returns the native size of the index's key values. If the key value
 * is variable size, this function returns -1. See the sos_key_len()
 * and sos_key_size() functions for the current size of the key's
 * value and the size of the key's buffer respectively.
 *
 * \return The native size of the index's keys in bytes
 */
size_t sos_index_key_size(sos_index_t index)
{
	ods_idx_ref_t iref = LIST_FIRST(&index->active_idx_list);
	return ods_idx_key_size(iref->idx);
}

/**
 * \brief Create a key based on the specified index or size
 *
 * Create a new SOS key for the specified index. If the size
 * is specified, the index parameter is ignored and the key is
 * based on the specified size.
 *
 * \param index The index handle
 * \param size The desired key size
 * \retval A pointer to the new key or NULL if there is an error
 */
sos_key_t sos_index_key_new(sos_index_t index, size_t size)
{
	if (!size)
		return sos_key_new(sos_index_key_size(index));
	return sos_key_new(size);
}

/**
 * \brief Set the value of a key from a string
 *
 * \param index	The index handle
 * \param key	The key
 * \param str	Pointer to a string
 * \retval 0	if successful
 * \retval -1	if there was an error converting the string to a value
 */
int sos_index_key_from_str(sos_index_t index, sos_key_t key, const char *str)
{
	ods_idx_ref_t iref = LIST_FIRST(&index->active_idx_list);
	return ods_key_from_str(iref->idx, key, str);
}

/**
 * \brief Return a string representation of the key value
 *
 * \param index	The index handle
 * \param key	The key
 * \return A const char * representation of the key value.
 */
const char *sos_index_key_to_str(sos_index_t index, sos_key_t key)
{
	ods_idx_ref_t iref = LIST_FIRST(&index->active_idx_list);
	size_t keylen = ods_idx_key_str_size(iref->idx, key);
	char *keystr = malloc(keylen);
	return ods_key_to_str(iref->idx, key, keystr, keylen);
}

/**
 * \brief Compare two keys using the index's compare function
 *
 * \param index	The index handle
 * \param a	The first key
 * \param b	The second key
 * \return <0	a < b
 * \return 0	a == b
 * \return >0	a > b
 */
int64_t sos_index_key_cmp(sos_index_t index, sos_key_t a, sos_key_t b)
{
	ods_idx_ref_t iref = LIST_FIRST(&index->active_idx_list);
	return ods_key_cmp(iref->idx, a, b);
}

void sos_index_print(sos_index_t index, FILE *fp)
{
	ods_idx_ref_t iref;
	LIST_FOREACH(iref, &index->active_idx_list, entry)
		ods_idx_print(iref->idx, (fp ? fp : stdout));
}

/**
 * @brief Verify the internal consistency of an index
 *
 * @param index The index handle
 * @param fp A FILE pointer into which error information is reported.
 *           If this is NULL, the index errors will not be printed, but
 *           the status of the index is returned.
 * @param verbose -  0: errors are not printed
 *                -  1: Names of partition containing corrupted indices
 *                     are printed
 *                - >1: Corruption errors are also printed.
 *
 * @return 0 The index is consistent
 * @return -1 The index in one or more partitions is corrupted
 */
int sos_index_verify(sos_index_t index, FILE *fp, int verbose)
{
	int rc = 0;
	ods_idx_ref_t iref;
	if (!verbose)
		fp = NULL;
	LIST_FOREACH(iref, &index->active_idx_list, entry) {
		if (ods_idx_verify(iref->idx, fp)) {
			if (fp && verbose)
				printf("Partition Name: %s, Path %s\n",
				       sos_part_name(iref->part),
				       sos_part_path(iref->part));
			rc = -1;
		}
	}
	return rc;
}

struct sos_container_index_iter_s {
	sos_t sos;
	ods_iter_t iter;
};

sos_container_index_iter_t sos_container_index_iter_new(sos_t sos)
{
	sos_container_index_iter_t iter = calloc(1, sizeof *iter);
	iter->sos = sos;
	iter->iter = ods_iter_new(sos->idx_idx);
	if (!iter->iter) {
		free(iter);
		return NULL;
	}
	return iter;
}

void sos_container_index_iter_free(sos_container_index_iter_t iter)
{
	if (iter->iter)
		ods_iter_delete(iter->iter);

	free(iter);
}

sos_index_t sos_container_index_iter_first(sos_container_index_iter_t iter)
{
	sos_obj_ref_t idx_ref;
	sos_index_t idx;
	int rc = ods_iter_begin(iter->iter);
	if (rc)
		return NULL;
	idx_ref.idx_data = ods_iter_data(iter->iter);
	ods_obj_t idx_obj = ods_ref_as_obj(iter->sos->idx_ods, idx_ref.ref.obj);
	idx = sos_index_open(iter->sos, SOS_IDX(idx_obj)->name);
	ods_obj_put(idx_obj);
	return idx;
}

sos_index_t sos_container_index_iter_next(sos_container_index_iter_t iter)
{
	sos_obj_ref_t idx_ref;
	sos_index_t idx;
	int rc;
 next:
	rc = ods_iter_next(iter->iter);
	if (rc)
		return NULL;
	idx_ref.idx_data = ods_iter_data(iter->iter);
	ods_obj_t idx_obj = ods_ref_as_obj(iter->sos->idx_ods, idx_ref.ref.obj);
	idx = sos_index_open(iter->sos, SOS_IDX(idx_obj)->name);
	ods_obj_put(idx_obj);
	if (!idx)
		goto next;
	return idx;
}

void sos_container_index_list(sos_t sos, FILE *fp)
{
	sos_obj_ref_t idx_ref;
	int rc;
	ods_iter_t iter = ods_iter_new(sos->idx_idx);
	if (!fp)
		fp = stdout;
	fprintf(fp, "%-20s %-20s %-20s %-20s\n",
		"Name", "Index Type", "Key Type", "Index Args");
	fprintf(fp,
		"-------------------- -------------------- -------------------- "
		"--------------------\n");
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		idx_ref.idx_data = ods_iter_data(iter);
		ods_obj_t idx_obj = ods_ref_as_obj(sos->idx_ods, idx_ref.ref.obj);
		fprintf(fp, "%-20s %-20s %-20s %-20s\n",
			SOS_IDX(idx_obj)->name,
			SOS_IDX(idx_obj)->idx_type,
			SOS_IDX(idx_obj)->key_type,
			SOS_IDX(idx_obj)->args
			);
		ods_obj_put(idx_obj);
	}
	ods_iter_delete(iter);
}

/**
 * \brief Commit an Index's data to stable storage
 *
 * \param index The Index handle
 * \param flags The commit flags
 * \retval 0 Success
 * \retval !0 A Unix error code
 */
int sos_index_commit(sos_index_t index, sos_commit_t flags)
{
	ods_idx_ref_t iref;
	LIST_FOREACH(iref, &index->active_idx_list, entry) {
		ods_idx_commit(iref->idx, flags);
	}
	return 0;
}

/**
 * \brief Return the size in bytes of the Index
 *
 * \param index The index handle
 * \retval off_t in bytes of the Index
 */
off_t sos_index_size(sos_index_t index)
{
	ods_idx_ref_t iref;
	off_t size = 0;
	LIST_FOREACH(iref, &index->active_idx_list, entry) {
		struct ods_idx_stat_s isb;
		ods_idx_stat(iref->idx, &isb);
		size += isb.size;
	}
	return size;
}

/**
 * \brief Return the index's name
 * \param index The index handle
 * \retval The index name
 */
const char *sos_index_name(sos_index_t index)
{
	return index->name;
}

/**
 * \brief Return the statistics of the Index
 *
 * \param index The index handle
 * \retval 0 Success
 * \retval !0 A Unix errno
 */
int sos_index_stat(sos_index_t index, sos_index_stat_t sb)
{
	ods_idx_ref_t iref;
	int rc = 0;
	if (!index || !sb)
		return EINVAL;
	memset(sb, 0, sizeof(*sb));
	LIST_FOREACH(iref, &index->active_idx_list, entry) {
		struct ods_idx_stat_s osb;
		rc = ods_idx_stat(iref->idx, &osb);
		if (rc)
			return rc;
		sb->cardinality += osb.cardinality;
		sb->duplicates += osb.duplicates;
		sb->size += osb.size;
	}
	return 0;
}

