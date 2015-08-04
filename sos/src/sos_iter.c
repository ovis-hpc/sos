/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2014 Sandia Corporation. All rights reserved.
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

sos_iter_t sos_index_iter_new(sos_index_t index)
{
	sos_iter_t i;

	/* Find first active object partition and matching index */
	sos_idx_part_t idx_part = NULL;
	sos_obj_part_t obj_part = __sos_active_obj_part(index->sos);
	if (obj_part)
		idx_part = __sos_matching_idx_part(index, obj_part);
	if (!idx_part)
		return NULL;

	i = calloc(1, sizeof *i);
	if (!i)
		goto err;

	i->index = index;
	i->obj_part = obj_part;
	i->idx_part = idx_part;
	i->iter = ods_iter_new(idx_part->index);
	if (!i->iter)
		goto err;
	ods_iter_begin(i->iter);
	return i;
 err:
	if (i)
		free(i);
	return NULL;
}

/**
 * \brief Create a new SOS iterator
 *
 * Create an iterator on the specified attribute. If there is no index
 * defined on the iterator, the function will fail.
 *
 * \param attr The schema attribute handle
 *
 * \retval sos_iter_t for the specified attribute
 * \retval NULL       If there was an error creating the iterator. Note
 *		      that failure to find a matching object is not an
 *		      error.
 */
sos_iter_t sos_attr_iter_new(sos_attr_t attr)
{
	sos_iter_t i;

	if (!sos_attr_index(attr))
		return NULL;

	assert(attr->index);
	return sos_index_iter_new(attr->index);
}

/**
 * \brief Set iterator behavior flags
 *
 * \param i The iterator
 * \param flags The iterator flags
 * \retval 0 The flags were set successfully
 * \retval EINVAL The iterator or flags were invalid
 */
int sos_iter_flags_set(sos_iter_t iter, sos_iter_flags_t flags)
{
	return ods_iter_flags_set(iter->iter, flags);
}

/**
 * \brief Get the iterator behavior flags
 *
 * \param iter The iterator
 * \retval The sos_iter_flags_t for the iterator
 */
sos_iter_flags_t sos_iter_flags_get(sos_iter_t iter)
{
	return (sos_iter_flags_t)ods_iter_flags_get(iter->iter);
}

/**
 * \brief Return the number of positions in the iterator
 * \param iter The iterator handle
 * \returns The cardinality of the iterator
 */
uint64_t sos_iter_card(sos_iter_t iter)
{
	struct ods_idx_stat_s sb;
	int rc = ods_idx_stat(ods_iter_idx(iter->iter), &sb);
	if (rc)
		return 0;
	return sb.cardinality;
}

/**
 * \brief Return the number of duplicates in the index
 * \returns The count of duplicates
 */
uint64_t sos_iter_dups(sos_iter_t iter)
{
	struct ods_idx_stat_s sb;
	int rc = ods_idx_stat(ods_iter_idx(iter->iter), &sb);
	if (rc)
		return 0;
	return sb.duplicates;
}

/**
 * \brief Returns the current iterator position
 *
 * \param i The iterator handle
 * \param pos The sos_pos_t that will receive the position value.
 * \returns The current iterator position or 0 if position is invalid
 */
int sos_iter_pos(sos_iter_t iter, sos_pos_t pos)
{
	return ods_iter_pos(iter->iter, (ods_pos_t)pos);
}

/**
 * \brief Sets the current iterator position
 *
 * \param i The iterator handle
 * \param pos The iterator cursor position
 * \retval 0 Success
 * \retval ENOENT if the specified position is invalid
 */
int sos_iter_set(sos_iter_t iter, const sos_pos_t pos)
{
	return ods_iter_set(iter->iter, (ods_pos_t)pos);
}

/**
 * \brief Release the resources associated with a SOS iterator
 *
 * \param iter	The iterator returned by \c sos_new_iter
 */
void sos_iter_free(sos_iter_t iter)
{
	ods_iter_delete(iter->iter);
	free(iter);
}

/**
 * \brief Return the object at the current iterator position
 *
 * \param iter	The iterator handle
 * \return ods_obj_t at the current position
 */
sos_obj_t sos_iter_obj(sos_iter_t i)
{
	ods_ref_t ods_ref = ods_iter_ref(i->iter);
	if (!ods_ref)
		return NULL;
	ods_obj_t obj = ods_ref_as_obj(i->obj_part->obj_ods, ods_ref);
	if (!obj)
		return NULL;
	sos_schema_t schema = sos_schema_by_id(i->index->sos, SOS_OBJ(obj)->schema);
	return __sos_init_obj(i->index->sos,
			      schema,
			      obj,
			      i->obj_part);
}

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
int sos_iter_obj_remove(sos_iter_t iter)
{
	return ENOSYS;
}

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
int sos_iter_next(sos_iter_t i)
{
	return ods_iter_next(i->iter);
}

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
int sos_iter_prev(sos_iter_t i)
{
	return ods_iter_prev(i->iter);
}

/**
 * Position the iterator at the first object.
 *
 * \param i	The iterator handle

 * \return 0 The iterator is positioned at the first object in the index
 * \return ENOENT The index is empty
 */
int sos_iter_begin(sos_iter_t i)
{
	/* TODO clean if restarting */

	/* Get first partition */
	return ods_iter_begin(i->iter);
}

/**
 * Position the iterator at the last object in the index
 *
 * \param i The iterator handle
 * \return 0 The iterator is positioned at the last object in the index
 * \return ENOENT The index is empty
 */
int sos_iter_end(sos_iter_t i)
{
	return ods_iter_end(i->iter);
}

/**
 * \brief Position the iterator at the supremum of the specified key
 *
 * Position the iterator at the object whose key is the least
 * upper bound of the specified key.
 *
 * If the supremum is a duplicate key, the cursor is positioned at
 * the first instance of the key.
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \retval 0 The iterator is positioned at the supremum
 * \retval ENOENT No supremum exists
 */
int sos_iter_sup(sos_iter_t i, sos_key_t key)
{
	return ods_iter_find_lub(i->iter, key);
}

/**
 * \brief Position the iterator at the infinum of the specified key.
 *
 * Position the iterator at the object whose key is the greatest
 * lower bound of the specified key.
 *
 * If the infininum is a duplicate key, the cursor is positioned at
 * the first instance of the key.
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \retval 0 if the iterator is positioned at the infinum
 * \retval ENOENT if the infinum does not exist
 */
int sos_iter_inf(sos_iter_t i, sos_key_t key)
{
	return ods_iter_find_glb(i->iter, key);
}

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
int sos_iter_key_cmp(sos_iter_t iter, sos_key_t key)
{
	int rc;
	ods_key_t iter_key = ods_iter_key(iter->iter);
	rc = ods_key_cmp(iter->idx_part->index, iter_key, key);
	ods_obj_put(iter_key);
	return rc;
}

/**
 * \brief Position the iterator at the specified key
 *
 * If the index contains duplicate keys, the iterator will be
 * positioned at the first instance of the specified key.
 *
 * \param iter  Handle for the iterator.
 * \param key   The key for the iterator. The appropriate index will
 *		be searched to find the object that matches the key.
 *
 * \retval 0 Iterator is positioned at matching object.
 * \retval ENOENT No matching object was found.
 */
int sos_iter_find(sos_iter_t iter, sos_key_t key)
{
	return ods_iter_find(iter->iter, key);
}

/**
 * \brief Position the iterator at the first instance of the specified key
 *
 * \param iter  Handle for the iterator.
 * \param key   The key for the iterator. The appropriate index will
 *		be searched to find the object that matches the key.
 *
 * \retval 0 Iterator is positioned at matching object.
 * \retval ENOENT No matching object was found.
 */
int sos_iter_find_first(sos_iter_t iter, sos_key_t key)
{
	return ods_iter_find_first(iter->iter, key);
}

/**
 * \brief Position the iterator at the last instance of the specified key
 *
 * \param iter  Handle for the iterator.
 * \param key   The key for the iterator. The appropriate index will
 *		be searched to find the object that matches the key.
 *
 * \retval 0 Iterator is positioned at matching object.
 * \retval ENOENT No matching object was found.
 */
int sos_iter_find_last(sos_iter_t iter, sos_key_t key)
{
	return ods_iter_find_last(iter->iter, key);
}

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
sos_key_t sos_iter_key(sos_iter_t iter)
{
	return ods_iter_key(iter->iter);
}
static int lt_fn(sos_value_t obj_value, sos_value_t cond_value)
{
	int rc = sos_value_cmp(obj_value, cond_value);
	return (rc < 0);
}

static int le_fn(sos_value_t obj_value, sos_value_t cond_value)
{
	int rc = sos_value_cmp(obj_value, cond_value);
	return (rc <= 0);
}

static int eq_fn(sos_value_t obj_value, sos_value_t cond_value)
{
	int rc = sos_value_cmp(obj_value, cond_value);
	return (rc == 0);
}

static int ne_fn(sos_value_t obj_value, sos_value_t cond_value)
{
	int rc = sos_value_cmp(obj_value, cond_value);
	return (rc != 0);
}

static int ge_fn(sos_value_t obj_value, sos_value_t cond_value)
{
	int rc = sos_value_cmp(obj_value, cond_value);
	return (rc >= 0);
}

static int gt_fn(sos_value_t obj_value, sos_value_t cond_value)
{
	int rc = sos_value_cmp(obj_value, cond_value);
	return (rc > 0);
}

sos_filter_fn_t fn_table[] = {
	[SOS_COND_LT] = lt_fn,
	[SOS_COND_LE] = le_fn,
	[SOS_COND_EQ] = eq_fn,
	[SOS_COND_GE] = ge_fn,
	[SOS_COND_GT] = gt_fn,
	[SOS_COND_NE] = ne_fn,
};

sos_filter_t sos_filter_new(sos_iter_t iter)
{
	sos_filter_t f = calloc(1, sizeof *f);
	if (f)
		TAILQ_INIT(&f->cond_list);
	f->iter = iter;
	return f;
}

void sos_filter_free(sos_filter_t f)
{
	sos_filter_cond_t cond;
	while (!TAILQ_EMPTY(&f->cond_list)) {
		cond = TAILQ_FIRST(&f->cond_list);
		TAILQ_REMOVE(&f->cond_list, cond, entry);
		free(cond);
	}
	free(f);
}

int sos_filter_flags_set(sos_filter_t f, sos_iter_flags_t flags)
{
	return sos_iter_flags_set(f->iter, flags);
}

int sos_filter_cond_add(sos_filter_t f,
			sos_attr_t attr, enum sos_cond_e cond_e, sos_value_t value)
{
	sos_filter_cond_t cond = malloc(sizeof *cond);
	if (!cond)
		return ENOMEM;
	cond->attr = attr;
	cond->cmp_fn = fn_table[cond_e];
	cond->value = value;
	cond->cond = cond_e;
	TAILQ_INSERT_TAIL(&f->cond_list, cond, entry);
	return 0;
}

sos_filter_cond_t sos_filter_eval(sos_obj_t obj, sos_filter_t filt)
{
	sos_filter_cond_t cond;
	TAILQ_FOREACH(cond, &filt->cond_list, entry) {
		struct sos_value_s v_;
		sos_value_t obj_value = sos_value_init(&v_, obj, cond->attr);
		int rc = cond->cmp_fn(obj_value, cond->value);
		sos_value_put(obj_value);
		if (!rc)
			return cond;
	}
	return NULL;
}

static sos_obj_t next_match(sos_filter_t filt)
{
	int rc;
	do {
		sos_obj_t obj = sos_iter_obj(filt->iter);
		sos_filter_cond_t cond = sos_filter_eval(obj, filt);
		if (cond) {
			sos_obj_put(obj);
			if (cond->attr->index == filt->iter->index
			    && cond->cond <= SOS_COND_EQ)
				/* On ordered index and the condition
				 * requires a value <= */
				break;
			rc = sos_iter_next(filt->iter);
		} else
			return obj;
	} while (rc == 0);
	return NULL;
}

static sos_obj_t prev_match(sos_filter_t filt)
{
	int rc;
	do {
		sos_obj_t obj = sos_iter_obj(filt->iter);
		sos_filter_cond_t cond = sos_filter_eval(obj, filt);
		if (cond) {
			sos_obj_put(obj);
			if (cond->attr->index == filt->iter->index
			    && cond->cond >= SOS_COND_EQ)
				/* On ordered index and the condition
				 * requires a value >= */
				break;
			rc = sos_iter_prev(filt->iter);
		} else
			return obj;
	} while (rc == 0);
	return NULL;
}

sos_obj_t sos_filter_begin(sos_filter_t filt)
{
	sos_filter_cond_t cond;
	sos_obj_t obj;
	int rc;
	SOS_KEY(key);

	/* Find the filter attribute condition with the smallest cardinality */
	rc = sos_iter_begin(filt->iter);
	TAILQ_FOREACH(cond, &filt->cond_list, entry) {
		if (cond->attr->index != filt->iter->index)
			continue;
		/* NB: this check presumes the index is in
		 * increasing order. For all the built-in
		 * types, this is the case, however, if the
		 * user builds their own types/comparators,
		 * this check is invalid */
		if (cond->cond < SOS_COND_EQ)
			continue;

		sos_key_set(key, sos_value_as_key(cond->value),
			    sos_value_size(cond->value));
		rc = sos_iter_sup(filt->iter, key);
	}
	if (!rc)
		return next_match(filt);
	return NULL;
}

sos_obj_t sos_filter_next(sos_filter_t filt)
{
	int rc = sos_iter_next(filt->iter);
	if (!rc)
		return next_match(filt);
	return NULL;
}

int sos_filter_set(sos_filter_t filt, const sos_pos_t pos)
{
	return sos_iter_set(filt->iter, pos);
}

int sos_filter_pos(sos_filter_t filt, sos_pos_t pos)
{
	return sos_iter_pos(filt->iter, pos);
}

sos_obj_t sos_filter_prev(sos_filter_t filt)
{
	int rc = sos_iter_prev(filt->iter);
	if (!rc)
		return prev_match(filt);
	return NULL;
}

sos_obj_t sos_filter_end(sos_filter_t filt)
{
	sos_filter_cond_t cond;
	sos_obj_t obj;
	int rc;

	rc = sos_iter_end(filt->iter);
	SOS_KEY(key);
	TAILQ_FOREACH(cond, &filt->cond_list, entry) {
		if (cond->attr->index != filt->iter->index)
			continue;

		/* NB: this check presumes the index is in
		 * increasing order. For all the built-in
		 * types, this is the case, however, if the
		 * user builds their own types/comparators,
		 * this check is invalid */
		if (cond->cond >= SOS_COND_EQ)
			continue;

		sos_key_set(key, sos_value_as_key(cond->value),
			    sos_value_size(cond->value));
		rc = sos_iter_inf(filt->iter, key);
	}
	if (!rc)
		return prev_match(filt);
	return NULL;
}

sos_obj_t sos_filter_obj(sos_filter_t filt)
{
	return sos_iter_obj(filt->iter);
}
