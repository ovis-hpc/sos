/*
 * Copyright (c) 2020 Open Grid Computing, Inc. All rights reserved.
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
#include <assert.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <errno.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

static int __attr_join_idx(sos_attr_t filt_attr, sos_attr_t attr);

static int __sos_filter_key_set(sos_filter_t filt, sos_key_t key,
				int min_not_max, int last_match);
static sos_filter_cond_t __sos_find_filter_condition(sos_filter_t filt,
						     int attr_id);
static sos_obj_t next_match(sos_filter_t filt);
static sos_obj_t prev_match(sos_filter_t filt);

#if 0
static int __iter_rbn_printer(struct ods_rbn *rbn, void *-arg, int level)
{
	ods_key_t key = rbn->key;
	printf("%p %*c%-2d: %d\n", rbn, 80 - (level * 6), (rbn->color?'B':'R'),
	       level, key->as.key->uint32_[0]);
	return 0;
}
#endif

static int64_t __iter_key_cmp(void *a, const void *b, void *arg)
{
	return sos_index_key_cmp((sos_index_t)arg, a, (void *)b);
}

/**
 * \brief Create a SOS iterator from an index
 *
 * Create an iterator on the specified index.
 *
 * \param index The index handle
 *
 * \retval sos_iter_t for the specified index
 * \retval NULL       If there was an error creating the iterator.
 */
sos_iter_t sos_index_iter_new(sos_index_t index)
{
	sos_iter_t i;
	ods_iter_ref_t iter_ref;
	ods_idx_ref_t ods_idx_ref;

	i = malloc(sizeof *i);
	if (!i)
		return NULL;
	i->attr = NULL;
	i->index = index;
	ods_rbt_init(&i->rbt, __iter_key_cmp, index);
	LIST_INIT(&i->iter_list);
	LIST_FOREACH(ods_idx_ref, &index->active_idx_list, entry) {
		iter_ref = malloc(sizeof *iter_ref);
		if (!iter_ref)
			goto err;
		iter_ref->iter = ods_iter_new(ods_idx_ref->idx);
		if (!iter_ref->iter)
			goto err;
		LIST_INSERT_HEAD(&i->iter_list, iter_ref, entry);
	}
	return i;
 err:
	while (!LIST_EMPTY(&i->iter_list)) {
		iter_ref = LIST_FIRST(&i->iter_list);
		LIST_REMOVE(iter_ref, entry);
		if (iter_ref->iter)
			ods_iter_delete(iter_ref->iter);
		free(iter_ref);
	}
	free(i);
	return NULL;
}

/**
 * \brief Create a SOS iterator from an attribute index
 *
 * Create an iterator on the specified attribute. If there is no index
 * defined on the iterator, the function will fail.
 *
 * \param attr The schema attribute handle
 *
 * \retval sos_iter_t for the specified attribute
 * \retval NULL       The attribute is not indexed
 */
sos_iter_t sos_attr_iter_new(sos_attr_t attr)
{
	sos_iter_t iter;
	sos_index_t index = sos_attr_index(attr);

	if (!index) {
		errno = EINVAL;
		return NULL;
	}

	iter = sos_index_iter_new(index);
	if (iter)
		iter->attr = attr;

	return iter;
}

/**
 * \brief Return the attribute associated with the iterator
 *
 * \param iter The iterator handle
 * \returns A pointer to the attribute or NULL if the iterator is not
 *          associated with a schema attribute.
 */
sos_attr_t sos_iter_attr(sos_iter_t iter)
{
	return iter->attr;
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
	int rc;
	ods_iter_ref_t iter_ref;
	LIST_FOREACH(iter_ref, &iter->iter_list, entry) {
		rc = ods_iter_flags_set(iter_ref->iter, flags);
	}
	return rc;
}

/**
 * \brief Get the iterator behavior flags
 *
 * \param iter The iterator
 * \retval The sos_iter_flags_t for the iterator
 */
sos_iter_flags_t sos_iter_flags_get(sos_iter_t iter)
{
	ods_iter_ref_t iter_ref = LIST_FIRST(&iter->iter_list);
	return (sos_iter_flags_t)ods_iter_flags_get(iter_ref->iter);
}

/**
 * \brief Return the number of positions in the iterator
 * \param iter The iterator handle
 * \returns The cardinality of the iterator
 */
uint64_t sos_iter_card(sos_iter_t iter)
{
	int rc;
	uint64_t cardinality = 0;
	ods_iter_ref_t iter_ref;
	struct ods_idx_stat_s sb;

	LIST_FOREACH(iter_ref, &iter->iter_list, entry) {
		rc = ods_idx_stat(ods_iter_idx(iter_ref->iter), &sb);
		if (!rc)
			cardinality += sb.cardinality;
	}
	return cardinality;
}

/**
 * \brief Return the number of duplicates in the index
 * \returns The count of duplicates
 */
uint64_t sos_iter_dups(sos_iter_t iter)
{
	struct ods_idx_stat_s sb;
	uint64_t dups = 0;
	ods_iter_ref_t iter_ref;

	LIST_FOREACH(iter_ref, &iter->iter_list, entry) {
		int rc = ods_idx_stat(ods_iter_idx(iter_ref->iter), &sb);
		if (!rc)
			dups = sb.duplicates;
	}
	return dups;
}

static void __sos_reset_iter(sos_iter_t i)
{
	/* Remove all entries from the tree */
	while (!ods_rbt_empty(&i->rbt)) {
		struct ods_rbn *rbn = ods_rbt_min(&i->rbt);
		ods_iter_obj_ref_t ref =
			container_of(rbn, struct ods_iter_obj_ref_s, rbn);
		if (ref->key)
			sos_key_put(ref->key);
		ods_rbt_del(&i->rbt, rbn);
		free(ref);
	}
	i->pos = NULL;
}

/**
 * \brief Release the resources associated with a SOS iterator
 *
 * \param iter	The iterator returned by \c sos_new_iter
 */
void sos_iter_free(sos_iter_t iter)
{
	ods_iter_ref_t iter_ref;
	__sos_reset_iter(iter);
	while (!LIST_EMPTY(&iter->iter_list)) {
		iter_ref = LIST_FIRST(&iter->iter_list);
		LIST_REMOVE(iter_ref, entry);
		ods_iter_delete(iter_ref->iter);
		free(iter_ref);
	}
	free(iter);
}

/**
 * \brief Return the object at the current iterator position
 *
 * \param iter	The iterator handle
 * \return sos_obj_t at the current position
 */
sos_obj_t sos_iter_obj(sos_iter_t i)
{
	sos_obj_t obj;
	if (!i->pos) {
		errno = ENOENT;
		return NULL;
	}
	obj = sos_ref_as_obj(i->index->sos, i->pos->ref);
	if (!obj)
		errno = EINVAL;
	return obj;
}

/**
 * \brief Return the object reference at the current iterator position
 *
 * \param iter	The iterator handle
 * \return sos_obj_ref_t at the current position
 */
sos_obj_ref_t sos_iter_ref(sos_iter_t i)
{
	return i->pos->ref;
}

/**
 * \brief Remove the index entry at the current iterator position
 *
 * Removes the index entry at the current cursor position.
 * After removal, the iterator points at the next entry if it
 * exists, or empty if the tail was deleted.
 *
 * \param iter The iterator handle
 * \return 0 on success.
 * \return Error code on failure.
 */
int sos_iter_entry_remove(sos_iter_t iter)
{
	ods_idx_data_t data = iter->pos->ref.idx_data;
	int rc = ods_iter_entry_delete(iter->pos->iter, &data);
	ods_rbt_del(&iter->rbt, &iter->pos->rbn);
	memset(&iter->pos, 0, sizeof iter->pos);
	return rc;
}

static int __sos_iter_obj_ref_new(sos_iter_t sos_iter, ods_iter_t ods_iter,
									struct ods_rbn **rbn)
{
	ods_key_t key = ods_iter_key(ods_iter);
	/*
	 * If the iterator has the unique flag set, check if the object
	 * is already in the tree.
	 */
	if (ods_iter_flags_get(ods_iter) & ODS_ITER_F_UNIQUE) {
		struct ods_rbn *rbn = ods_rbt_find(&sos_iter->rbt, key);
		if (rbn) {
			sos_key_put(key);
			return EEXIST;
		}
	}

	ods_iter_obj_ref_t new_ref = malloc(sizeof *new_ref);
	if (!new_ref)
		return ENOMEM;

	new_ref->iter = ods_iter;
	new_ref->ref.idx_data = ods_iter_data(ods_iter);
	new_ref->key = key;
	ods_rbn_init(&new_ref->rbn, new_ref->key);
	ods_rbt_ins(&sos_iter->rbt, &new_ref->rbn);
	if (rbn)
		*rbn = &new_ref->rbn;
#if 0
	ods_rbt_verify(&sos_iter->rbt);
	printf("----------\n");
	ods_rbt_print(&sos_iter->rbt, __iter_rbn_printer, NULL);
#endif
	return 0;
}

static int __sos_iter_pos_iter(sos_iter_t sos_iter, ods_iter_t ods_iter)
{
	ods_iter_obj_ref_t iter_obj_ref;

	/* Make the current position that of an ods_iter */
	iter_obj_ref = malloc(sizeof *iter_obj_ref);
	if (!iter_obj_ref)
		return errno;
	iter_obj_ref->iter = ods_iter;
	iter_obj_ref->ref.idx_data = ods_iter_data(ods_iter);
	iter_obj_ref->key = ods_iter_key(ods_iter);
	sos_iter->pos = iter_obj_ref;
	ods_rbn_init(&iter_obj_ref->rbn, iter_obj_ref->key);
	ods_rbt_ins(&sos_iter->rbt, &iter_obj_ref->rbn);
#if 0
	ods_rbt_verify(&sos_iter->rbt);
	printf("----------\n");
	ods_rbt_print(&sos_iter->rbt, __iter_rbn_printer, NULL);
#endif
	return 0;
}

static int __sos_iter_pos_max(sos_iter_t sos_iter)
{
	ods_iter_obj_ref_t iter_obj_ref;
	struct ods_rbn *rbn;
	int rc;

	/* Make the current position the max in the RBT */
	rbn = ods_rbt_max(&sos_iter->rbt);
	if (rbn) {
		iter_obj_ref = container_of(rbn, struct ods_iter_obj_ref_s, rbn);
		sos_iter->pos = iter_obj_ref;
		rc = 0;
	} else {
		sos_iter->pos = NULL;
		rc = ENOENT;
	}
	return rc;
}

static int __sos_iter_pos_min(sos_iter_t sos_iter)
{
	ods_iter_obj_ref_t iter_obj_ref;
	struct ods_rbn *rbn;
	int rc;

	/* Make the current position the min in the RBT */
	rbn = ods_rbt_min(&sos_iter->rbt);
	if (rbn) {
		iter_obj_ref = container_of(rbn, struct ods_iter_obj_ref_s, rbn);
		sos_iter->pos = iter_obj_ref;
		rc = 0;
	} else {
		sos_iter->pos = NULL;
		rc = ENOENT;
	}
	return rc;
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
	/* Make the current position the min in the RBT */
	if (NULL == i->pos)
		return __sos_iter_pos_min(i);

	struct ods_rbn *new_rbn;
	ods_iter_obj_ref_t new_ref;
	struct ods_rbn *next_rbn = ods_rbn_succ(&i->pos->rbn);
	ods_iter_t ods_iter = i->pos->iter;
	int rc;

	/*
	 * Remove the current position from the tree
	 */
	ods_rbt_del(&i->rbt, &i->pos->rbn);
	sos_key_put(i->pos->key);
	free(i->pos);
	i->pos = NULL;
#if 0
	printf("----------\n");
	ods_rbt_print(&i->rbt, __iter_rbn_printer, NULL);
#endif
skip:
	rc = ods_iter_next(ods_iter);
	if (rc) {
		if (!next_rbn)
			return ENOENT;
		i->pos = container_of(next_rbn, struct ods_iter_obj_ref_s, rbn);
		return 0;
	}

	/* Add the new ref to the tree */
	rc = __sos_iter_obj_ref_new(i, ods_iter, &new_rbn);
	if (rc == EEXIST)
		goto skip;
	if (rc)
		return rc;
	assert(new_rbn);

	if (!next_rbn) {
		i->pos = container_of(new_rbn, struct ods_iter_obj_ref_s, rbn);
		return 0;
	}

	/*
	 * There is a successor in the tree, and the iterator has another
	 * entry. Make the new position the lesser of the next_rbn and the
	 * new rbn
	 */
	rc = ods_rbn_cmp(&i->rbt, next_rbn, new_rbn);
	if (rc < 0) {
		new_ref = container_of(next_rbn,
					struct ods_iter_obj_ref_s, rbn);
	} else if (rc < 0) {
		new_ref = container_of(new_rbn,
					struct ods_iter_obj_ref_s, rbn);
	} else {
		/*
		 * The pos needs to be set to the predecessor which could be new_rbn
		 * or the next_rbn
		 */
		if (new_rbn == ods_rbn_pred(next_rbn))
			new_ref = container_of(new_rbn,
						struct ods_iter_obj_ref_s, rbn);
		else
			new_ref = container_of(next_rbn,
						struct ods_iter_obj_ref_s, rbn);
	}
	i->pos = new_ref;
	return 0;
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
		/* Make the current position the min in the RBT */
	if (NULL == i->pos)
		return __sos_iter_pos_max(i);

	struct ods_rbn *new_rbn;
	ods_iter_obj_ref_t new_ref;
	struct ods_rbn *prev_rbn = ods_rbn_pred(&i->pos->rbn);
	ods_iter_t ods_iter = i->pos->iter;
	int rc;

	/*
	 * Remove the current position from the tree
	 */
	ods_rbt_del(&i->rbt, &i->pos->rbn);
	sos_key_put(i->pos->key);
	free(i->pos);
	i->pos = NULL;
#if 0
	printf("----------\n");
	ods_rbt_print(&i->rbt, __iter_rbn_printer, NULL);
#endif
skip:
	rc = ods_iter_prev(ods_iter);
	if (rc) {
		if (!prev_rbn)
			return ENOENT;
		i->pos = container_of(prev_rbn, struct ods_iter_obj_ref_s, rbn);
		return 0;
	}

	/* Add the new ref to the tree */
	rc = __sos_iter_obj_ref_new(i, ods_iter, &new_rbn);
	if (rc == EEXIST)
		goto skip;
	if (rc)
		return rc;
	assert(new_rbn);

	if (!prev_rbn) {
		i->pos = container_of(new_rbn, struct ods_iter_obj_ref_s, rbn);
		return 0;
	}

	/*
	 * There is a predecessor in the tree, and the iterator has another
	 * entry. Make the new position the greater of the prev_rbn and the
	 * new rbn
	 */
	rc = ods_rbn_cmp(&i->rbt, prev_rbn, new_rbn);
	if (rc > 0) {
		new_ref = container_of(prev_rbn,
					struct ods_iter_obj_ref_s, rbn);
	} else if (rc < 0) {
		new_ref = container_of(new_rbn,
					struct ods_iter_obj_ref_s, rbn);
	} else {
		/*
		 * The pos needs to be set to the successor which could be new_rbn
		 * or prev_rbn
		 */
		if (new_rbn == ods_rbn_succ(prev_rbn))
			new_ref = container_of(new_rbn,
						struct ods_iter_obj_ref_s, rbn);
		else
			new_ref = container_of(prev_rbn,
						struct ods_iter_obj_ref_s, rbn);
	}
	i->pos = new_ref;
	return 0;
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
	ods_iter_ref_t iter_ref;
	int rc;

	__sos_reset_iter(i);
	LIST_FOREACH(iter_ref, &i->iter_list, entry) {
		rc = ods_iter_begin(iter_ref->iter);
		if (rc)
			continue;
	skip:
		rc = __sos_iter_obj_ref_new(i, iter_ref->iter, NULL);
		if (rc == EEXIST) {
			rc = ods_iter_next(iter_ref->iter);
			if (!rc)
				goto skip;
			else
				continue;
		}
		if (rc)
			return rc;
	}

	/* Make the current position the min in the RBT */
	return __sos_iter_pos_min(i);
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
	ods_iter_ref_t iter_ref;
	int rc;

	__sos_reset_iter(i);
	LIST_FOREACH(iter_ref, &i->iter_list, entry) {
		rc = ods_iter_end(iter_ref->iter);
		if (rc)
			continue;
	skip:
		rc = __sos_iter_obj_ref_new(i, iter_ref->iter, NULL);
		if (rc == EEXIST) {
			rc = ods_iter_prev(iter_ref->iter);
			if (!rc)
				goto skip;
			else
				continue;
		}
		if (rc)
			return rc;
	}
	/* Make the current position the max in the RBT */
	return __sos_iter_pos_max(i);
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
 * This behavior can be changed using the sos_iter_flags_set()
 * function to set the SOS_ITER_F_SUP_LAST_DUP option. This will cause
 * this function to place the iterator position at the last
 * duplicate. Note that this _may_ break the axiom that INF(set) <=
 * SUP(set)
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \retval 0 The iterator is positioned at the supremum
 * \retval ENOENT No supremum exists
 */
int sos_iter_sup(sos_iter_t i, sos_key_t key)
{
	ods_iter_ref_t iter_ref;
	int rc;

	__sos_reset_iter(i);
	LIST_FOREACH(iter_ref, &i->iter_list, entry) {
		rc = ods_iter_find_lub(iter_ref->iter, key);
		if (rc)
			continue;
	skip:
		rc = __sos_iter_obj_ref_new(i, iter_ref->iter, NULL);
		if (rc == EEXIST) {
			rc = ods_iter_next(iter_ref->iter);
			if (!rc)
				goto skip;
			else
				continue;
		}
		if (rc)
			return rc;
	}
	/* Return the least of the lower bounds */
	rc = __sos_iter_pos_min(i);
	return rc;
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
 * This behavior can be changed using the sos_iter_flags_set()
 * function to set the SOS_ITER_F_INF_LAST_DUP option. This will cause
 * this function to place the iterator position at the last
 * duplicate. Note that this _may_ break the axiom that INF(set) <=
 * SUP(set)
 *
 * \param i Pointer to the iterator
 * \param key The key.
 *
 * \retval 0 if the iterator is positioned at the infinum
 * \retval ENOENT if the infinum does not exist
 */
int sos_iter_inf(sos_iter_t i, sos_key_t key)
{
	ods_iter_ref_t iter_ref;
	int rc;

	__sos_reset_iter(i);
	LIST_FOREACH(iter_ref, &i->iter_list, entry) {
		rc = ods_iter_find_glb(iter_ref->iter, key);
		if (rc)
			continue;
	skip:
		rc = __sos_iter_obj_ref_new(i, iter_ref->iter, NULL);
		if (rc == EEXIST) {
			/*
			 * If the iterator is marked UNIQUE, then we don't allow
			 * duplicates in the tree. Back the iterator up to the
			 * previous entry. If we are iterating backwards (i.e. prev)
			 * the iter_key < inf_key will be present as expected. If we
			 * are iterate forward, then these entries need to be
			 * discarded.
			 */
			rc = ods_iter_prev(iter_ref->iter);
			if (!rc)
				goto skip;
			else
				continue;
		}
		if (rc)
			return rc;
	}

	/* Return the greatest of the lower bounds */
	return __sos_iter_pos_max(i);
}

/**
 * \brief Compare iterator object's key with other key.
 *
 * This function compare the key of the object pointed by the iterator with the
 * other key. This is a convenience routine and is equivalent to the
 * following code sequence:
 *
 *     sos_key_t iter_key = sos_iter_key(iter);
 *     int64_t rc = sos_key_cmp(attr, iter_key, other);
 *     sos_key_put(iter_key);
 *
 * \param iter	The iterator handle
 * \param other	The other key
 * \retval <0	iter < other
 * \retval 0	iter == other
 * \retval >0	iter > other
 */
int64_t sos_iter_key_cmp(sos_iter_t iter, sos_key_t key)
{
	int64_t rc;
	ods_key_t iter_key = ods_iter_key(iter->pos->iter);
	rc = ods_key_cmp(iter->index->primary_idx, iter_key, key);
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
	int rc = ENOENT;
	ods_iter_ref_t iter_ref;

	__sos_reset_iter(iter);
	LIST_FOREACH(iter_ref, &iter->iter_list, entry) {
		rc = ods_iter_find(iter_ref->iter, key);
		if (!rc) {
			/* Set the current iterator position */
			rc = __sos_iter_pos_iter(iter, iter_ref->iter);
			if (rc)
				continue;
			break;
		}
	}

	return rc;
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
	int rc;
	ods_iter_ref_t iter_ref;

	__sos_reset_iter(iter);
	LIST_FOREACH(iter_ref, &iter->iter_list, entry) {
		rc = ods_iter_find_first(iter_ref->iter, key);
		if (!rc) {
			/* Set the current iterator position */
			rc = __sos_iter_pos_iter(iter, iter_ref->iter);
			if (rc)
				continue;
			break;
		}
	}

	return rc;
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
	int rc;
	ods_iter_ref_t iter_ref;

	__sos_reset_iter(iter);
	LIST_FOREACH(iter_ref, &iter->iter_list, entry) {
		rc = ods_iter_find_last(iter_ref->iter, key);
		if (!rc) {
			/* Set the current iterator position */
			rc = __sos_iter_pos_iter(iter, iter_ref->iter);
			if (rc)
				continue;
			break;
		}
	}

	return rc;
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
	if (iter->pos)
		return ods_iter_key(iter->pos->iter);
	return NULL;
}

static int lt_fn(sos_value_t obj_value, sos_value_t cond_value, int *ret)
{
	int rc = *ret = sos_value_cmp(obj_value, cond_value);
	return (rc < 0);
}

static int le_fn(sos_value_t obj_value, sos_value_t cond_value, int *ret)
{
	int rc = *ret = sos_value_cmp(obj_value, cond_value);
	return (rc <= 0);
}

static int eq_fn(sos_value_t obj_value, sos_value_t cond_value, int *ret)
{
	int rc = *ret = sos_value_cmp(obj_value, cond_value);
	return (rc == 0);
}

static int ne_fn(sos_value_t obj_value, sos_value_t cond_value, int *ret)
{
	int rc = *ret = sos_value_cmp(obj_value, cond_value);
	return (rc != 0);
}

static int ge_fn(sos_value_t obj_value, sos_value_t cond_value, int *ret)
{
	int rc = *ret = sos_value_cmp(obj_value, cond_value);
	return (rc >= 0);
}

static int gt_fn(sos_value_t obj_value, sos_value_t cond_value, int *ret)
{
	int rc = *ret = sos_value_cmp(obj_value, cond_value);
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

/**
 * \brief allocate a Sos Filter
 *
 * This function inherits the iterator reference from the caller.
 *
 * \param iter The iterator handle.
 * \returns A new filter object or NULL if there is an error
 */
sos_filter_t sos_filter_new(sos_iter_t iter)
{
	sos_filter_t f = calloc(1, sizeof *f);
	if (f) {
		TAILQ_INIT(&f->cond_list);
		f->iter = iter;
		f->last_match = ODS_OBJ_INIT(f->last_match_obj,
					     &f->last_match_key_data,
					     SOS_STACK_KEY_SIZE);
	}
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
	sos_iter_free(f->iter);
	free(f);
}

int sos_filter_flags_set(sos_filter_t f, sos_iter_flags_t flags)
{
	return sos_iter_flags_set(f->iter, flags);
}

sos_iter_flags_t sos_filter_flags_get(sos_filter_t f)
{
	return sos_iter_flags_get(f->iter);
}

static int __attr_join_idx(sos_attr_t filt_attr, sos_attr_t attr)
{
	int idx;
	int attr_id;
	sos_array_t attr_ids;

	if (sos_attr_type(filt_attr) != SOS_TYPE_JOIN)
		return -1;

	/*
	 * If the filter iterator attribute is a JOIN,
	 * check if the condition attribute is a member
	 */
	attr_id = sos_attr_id(attr);
	attr_ids = sos_attr_join_list(filt_attr);
	for (idx = 0; idx < attr_ids->count; idx++) {
		if (attr_ids->data.uint32_[idx] == attr_id)
			return idx;
	}
	return -1;
}

static void
__insert_filter_cond_fwd(sos_attr_t filt_attr, struct sos_cond_list *head,
			 struct sos_filter_cond_s *new_cond)
{
	int filt_attr_id, new_attr_id, new_join_idx;
	struct sos_filter_cond_s *cond;

	if (TAILQ_EMPTY(head)) {
		TAILQ_INSERT_TAIL(head, new_cond, entry);
		return;
	}

	filt_attr_id = sos_attr_id(filt_attr);
	new_attr_id = sos_attr_id(new_cond->attr);
	new_join_idx = __attr_join_idx(filt_attr, new_cond->attr);

	TAILQ_FOREACH(cond, head, entry) {
		if (new_join_idx >= 0) {
			int cond_join_idx = __attr_join_idx(filt_attr, cond->attr);
			/* New condition is in the iterators join attr */
			if (cond_join_idx < 0) {
				/* cond not in join_attr, new_cond takes prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_join_idx < cond_join_idx) {
				/* cond join index greater, new_cond takes prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_join_idx > cond_join_idx) {
				/* cond's join idx is before new_cond in
				 * the key, but there may be other conds
				 * that also take precedence */
				continue;
			} else if (new_cond->cond > cond->cond) {
				/* same join attr, condition takes prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_cond->cond < cond->cond) {
				TAILQ_INSERT_AFTER(head, cond, new_cond, entry);
				return;
			} else {
				/* Found duplicate condition, remove it */
				sos_value_put(new_cond->value);
				free(new_cond);
				return;
			}
		} else if (filt_attr_id == new_attr_id) {
			/* new cond is on iterator attribute */
			if (sos_attr_id(cond->attr) != filt_attr_id) {
				/* cond is not on iterator attribute, new_cond takes prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_cond->cond > cond->cond) {
				/* cond is also on iterator attr, comparator defines prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_cond->cond < cond->cond) {
				TAILQ_INSERT_AFTER(head, cond, new_cond, entry);
				return;
			} else {
				/* Found duplicate condition, remove it */
				sos_value_put(new_cond->value);
				free(new_cond);
				return;
			}
		} else if (new_attr_id == sos_attr_id(cond->attr)) {
			/* Neiter condition is on filter iter, condition takes precedence */
			if (new_cond->cond > cond->cond) {
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_cond->cond < cond->cond) {
				TAILQ_INSERT_AFTER(head, cond, new_cond, entry);
				return;
			} else {
				/* Found duplicate condition, remove it */
				sos_value_put(new_cond->value);
				free(new_cond);
				return;
			}
		}
		/*
		 * New attribute doesn't match the condition and is
		 * not on the iterator attribute, keep going to see if
		 * we find ourselves.
		 */
	}
	/* No other rule using this attribute, append to tail */
	TAILQ_INSERT_TAIL(head, new_cond, entry);
}

static void
__insert_filter_cond_bkwd(sos_attr_t filt_attr, struct sos_cond_list *head,
			  struct sos_filter_cond_s *new_cond)
{
	int filt_attr_id, new_attr_id, new_join_idx;
	struct sos_filter_cond_s *cond;

	if (TAILQ_EMPTY(head)) {
		TAILQ_INSERT_TAIL(head, new_cond, entry);
		return;
	}

	filt_attr_id = sos_attr_id(filt_attr);
	new_attr_id = sos_attr_id(new_cond->attr);
	new_join_idx = __attr_join_idx(filt_attr, new_cond->attr);

	TAILQ_FOREACH(cond, head, entry) {
		if (new_join_idx >= 0) {
			int cond_join_idx = __attr_join_idx(filt_attr, cond->attr);
			/* New condition is in the iterators join attr */
			if (cond_join_idx < 0) {
				/* cond not in join_attr, new_cond takes prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_join_idx < cond_join_idx) {
				/* cond join index greater, new_cond takes prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_join_idx > cond_join_idx) {
				/* cond's join idx is before new_cond in
				 * the key, but there may be other conds
				 * that also take precedence */
				continue;
			} else if (new_cond->cond < cond->cond) {
				/* same join attr, condition takes prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_cond->cond > cond->cond) {
				TAILQ_INSERT_AFTER(head, cond, new_cond, entry);
				return;
			} else {
				/* Found duplicate condition, remove it */
				sos_value_put(new_cond->value);
				free(new_cond);
				return;
			}
		} else if (filt_attr_id == new_attr_id) {
			/* new cond is on iterator attribute */
			if (sos_attr_id(cond->attr) != filt_attr_id) {
				/* cond is not on iterator attribute, new_cond takes prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_cond->cond < cond->cond) {
				/* cond is also on iterator attr, comparator defines prec. */
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_cond->cond > cond->cond) {
				TAILQ_INSERT_AFTER(head, cond, new_cond, entry);
				return;
			} else {
				/* Found duplicate condition, remove it */
				sos_value_put(new_cond->value);
				free(new_cond);
				return;
			}
		} else if (new_attr_id == sos_attr_id(cond->attr)) {
			/* Neiter condition is on filter iter, condition takes precedence */
			if (new_cond->cond < cond->cond) {
				TAILQ_INSERT_BEFORE(cond, new_cond, entry);
				return;
			} else if (new_cond->cond > cond->cond) {
				TAILQ_INSERT_AFTER(head, cond, new_cond, entry);
				return;
			} else {
				/* Found duplicate condition, remove it */
				sos_value_put(new_cond->value);
				free(new_cond);
				return;
			}
		}
		/*
		 * New attribute doesn't match the condition and is
		 * not on the iterator attribute, keep going to see if
		 * we find ourselves.
		 */
	}
	/* No other rule using this attribute, append to tail */
	TAILQ_INSERT_TAIL(head, new_cond, entry);
}

/*
 * Sort the filter conditions
 */
static void __sort_filter_conds_fwd(sos_filter_t f)
{
	sos_attr_t filt_attr = sos_iter_attr(f->iter);
	struct sos_cond_list cond_list;
	struct sos_filter_cond_s *cond;

	TAILQ_INIT(&cond_list);
	while (!TAILQ_EMPTY(&f->cond_list)) {
		cond = TAILQ_FIRST(&f->cond_list);
		TAILQ_REMOVE(&f->cond_list, cond, entry);
		__insert_filter_cond_fwd(filt_attr, &cond_list, cond);
	}
	while (!TAILQ_EMPTY(&cond_list)) {
		cond = TAILQ_FIRST(&cond_list);
		TAILQ_REMOVE(&cond_list, cond, entry);
		TAILQ_INSERT_TAIL(&f->cond_list, cond, entry);
	}
}

static void __sort_filter_conds_bkwd(sos_filter_t f)
{
	sos_attr_t filt_attr = sos_iter_attr(f->iter);
	struct sos_cond_list cond_list;
	struct sos_filter_cond_s *cond;

	TAILQ_INIT(&cond_list);
	while (!TAILQ_EMPTY(&f->cond_list)) {
		cond = TAILQ_FIRST(&f->cond_list);
		TAILQ_REMOVE(&f->cond_list, cond, entry);
		__insert_filter_cond_bkwd(filt_attr, &cond_list, cond);
	}
	while (!TAILQ_EMPTY(&cond_list)) {
		cond = TAILQ_FIRST(&cond_list);
		TAILQ_REMOVE(&cond_list, cond, entry);
		TAILQ_INSERT_TAIL(&f->cond_list, cond, entry);
	}
}

/**
 * \brief Add a filter condition to the filter
 *
 * The filter conditions affect which objects are returned by
 * sos_filter_begin(), sos_filter_next(), etc...
 *
 * Logically, all filter conditions are ANDed together to get a
 * TRUE/FALSE answer when evaluating an object. If all filter
 * conditions match, the sos_filter_xxx() iterator functions will
 * return the object, otherwise, the next object in the index will be
 * evaluated until a match is found or all objects in the index are
 * exhausted.
 *
 * \param filt    The filter handle returned by sos_filter_new()
 * \param attr    The object attribute that will be evaluated by this condition
 * \param cond_e  One of the sos_cond_e comparison conditions
 * \param value   The value used in the expression "object-attribute-value cond_e value"
 * \retval 0      The condition was added successfully
 * \retval ENOMEM There was insufficient memory to allocate the filter condition
 */

int sos_filter_cond_add(sos_filter_t filt,
			sos_attr_t attr, enum sos_cond_e cond_e, sos_value_t value)
{
	sos_filter_cond_t cond = calloc(1, sizeof *cond);
	if (!cond)
		return ENOMEM;
	cond->attr = attr;
	cond->cmp_fn = fn_table[cond_e];
	cond->value = sos_value_copy(&cond->value_, value);
	cond->cond = cond_e;
	TAILQ_INSERT_TAIL(&filt->cond_list, cond, entry);
	return 0;
}

static sos_filter_cond_t sos_filter_eval(sos_obj_t obj, sos_filter_t filt)
{
	sos_filter_cond_t cond;
	struct sos_value_s v_;
	sos_value_t obj_value;
	int rc;
	TAILQ_FOREACH(cond, &filt->cond_list, entry) {
		obj_value = sos_value_init(&v_, obj, cond->attr);
		rc = cond->cmp_fn(obj_value, cond->value, &cond->ret);
		sos_value_put(obj_value);
		if (!rc)
			return cond;
	}
	sos_key_t key = sos_iter_key(filt->iter);
	if (key) {
		sos_key_copy(filt->last_match, key);
		sos_key_put(key);
	}
	return NULL;
}

/*
 * Searches for the next object that matches all of the conditions in
 * filt->cond_list. To avoid testing every object, conditions are sorted
 * and objects that can't match are skipped by building a key and searching
 * on the index.
 *
 * This table specifies how the key is constructed given the 1st
 * failing condition on an object.
 *
 * +==================+==================+==================+==================+
 * | obj-value ? key  |         <        |        ==        |         >        |
 * +==================I==================+==================+==================+
 * | condition   <    I         X        |    seek(max)     |    seek(max)     |
 * +------------------I------------------+------------------+------------------+
 * |             <=   I         X        |        X         |    seek(max)     |
 * +------------------I------------------+------------------+------------------+
 * |             ==   I seek(cond->value)|        X         |    seek(max)     |
 * +------------------I------------------+------------------+------------------+
 * |             >=   I seek(cond->value)|        X         |         X        |
 * +------------------I------------------+------------------+------------------+
 * |             >    I seek(cond->value)|      next        |         X        |
 * +------------------+------------------+------------------+------------------+
 *
 * The 'X' means "can't happen". The seek(min) sets the join key in
 * question to the minimum value for the value-type of the condition
 * attribute.
 *
 * seek(cond->value) means set the key to the value tested in the condition.
 */
static sos_obj_t next_match(sos_filter_t filt)
{
	SOS_KEY(key);
	int rc, i, join_idx;
	sos_obj_t obj;
	sos_filter_cond_t cond;
	sos_array_t attr_ids;
	struct sos_value_s v_;
	sos_value_t obj_value;
	ods_comp_key_t comp_key;
	ods_key_comp_t key_comp;
	size_t comp_len;
	sos_filter_cond_t join_cond;
	ods_ref_t last_ref = 0;

	filt->miss_cnt = 0;
	do {
		obj_value = NULL;
		obj = sos_iter_obj(filt->iter);
		if (!obj)
			break;
		cond = sos_filter_eval(obj, filt);
		if (!cond) {
			return obj;
		}
		filt->miss_cnt += 1;
		/*
		 * One or more conditions failed, determine if there
		 * can be any subsequent key that matches all
		 * conditions given the index ordering.
		 */
		if (cond->cond == SOS_COND_NE)
			/* No ordering optimizations for NE */
			goto next;

		join_idx = __attr_join_idx(sos_iter_attr(filt->iter), cond->attr);
		if (join_idx < 0) {
			/*
			 * The filter key is not a join or the
			 * condition attribute is not in the join key
			 */
			if (cond->attr != filt->iter->attr)
				/*
				 * The filter index is not on
				 * condition attribute nothing can be
				 * assumed about the ordering
				 */
				goto next;
			if (cond->cond < SOS_COND_GE)
				/*
				 * The condition requires <=, this
				 * attribute is the key and the
				 * comparison failed. There can be no
				 * more matches
				 */
				break;
			/*
			 * Missing optimization to skip to 1st
			 * possibly matching key
			 */
			goto next;
		}

		if (sos_attr_is_array(cond->attr))
			goto next;

		/* Key = { k[0], k[1], ... k[join_idx], ..., k[N] }
		 *                              ^
		 *                              |
		 * Failing Condition :----------+
		 *
		 */

		if (join_idx == 0 || cond == TAILQ_FIRST(&filt->cond_list)) {
			if (join_idx)
				/* 1st condition skips join prefix, we know nothing */
				goto next;
			/*
			 * The failing condition was <, <= or ==. If the match
			 * was >, then there can not possibly be any more
			 * matches for this condition past this point in the
			 * index.
			 */
			if (cond->cond <= SOS_COND_GE) {
				if (cond->ret >= 0)
					break;

				/* Cond is <, <=, and key is smaller
				 * than value, keep searching.
				 */
			}
			/* Cond is >=, >, keep searching */
		}
		/*
		 * Construct a key putting max in the component
		 * key position associated with the failing condition
		 * and search for the least upper bound (i.e. next)
		 */
		comp_key = (ods_comp_key_t)ods_key_value(key);
		comp_key->len = 0;
		key_comp = comp_key->value;
		attr_ids = sos_attr_join_list(sos_iter_attr(filt->iter));

		for (i = 0; i < attr_ids->count; i++) {
			int attr_id = attr_ids->data.uint32_[i];
			obj_value = sos_value_by_id(&v_, obj, attr_id);
			join_cond = __sos_find_filter_condition(filt, attr_id);
			if (i < join_idx) {
				if (!join_cond)
					goto next;
				key_comp = __sos_set_key_comp(key_comp, obj_value, &comp_len);
			} else if (i == join_idx) {
				switch (cond->cond) {
				case SOS_COND_LT:
				case SOS_COND_LE:
					if (__sos_value_is_max(obj_value))
						goto next;
					key_comp = __sos_set_key_comp_to_max(key_comp, obj_value->attr, &comp_len);
					break;
				case SOS_COND_EQ:
					if (cond->ret < 0) {
						key_comp = __sos_set_key_comp(key_comp, cond->value, &comp_len);
					} else {
						if (__sos_value_is_max(obj_value))
							goto next;
						key_comp = __sos_set_key_comp_to_max(key_comp, obj_value->attr, &comp_len);
					}
					break;
				case SOS_COND_GE:
					key_comp = __sos_set_key_comp(key_comp, cond->value, &comp_len);
					break;
				case SOS_COND_GT:
					if (cond->ret < 0)
						key_comp = __sos_set_key_comp(key_comp, cond->value, &comp_len);
					else
						goto next;
					break;
				case SOS_COND_NE:
					goto next;
				}
			} else {
				if (sos_attr_is_array(obj_value->attr))
					goto next;
				key_comp = __sos_set_key_comp_to_min(key_comp, obj_value->attr, &comp_len);
			}
			sos_value_put(obj_value);
			obj_value = NULL;
			comp_key->len += comp_len;
		}
		rc = sos_iter_sup(filt->iter, key);
		if (rc)
			break;
		if (last_ref == obj->obj->ref)
			goto out;
		last_ref = obj->obj->ref;
		sos_obj_put(obj);
		continue;
	next:
		if (obj_value) {
			sos_value_put(obj_value);
			obj_value = NULL;
		}
		rc = sos_iter_next(filt->iter);
		if (!rc)
			sos_obj_put(obj);
	} while (rc == 0);
 out:
	sos_obj_put(obj);
	return NULL;
}

/*
 * Searches for the previous object that matches all of the conditions in
 * filt->cond_list. To avoid testing every object, conditions are sorted
 * and objects that can't match are skipped by building a key and searching
 * on the index.
 *
 * This table specifies how the key is constructed given the 1st
 * failing condition on an object.
 *
 * +==================+==================+==================+==================+
 * | obj-value ? key  |         <        |        ==        |         >        |
 * +==================I==================+==================+==================+
 * | condition   <    I         X        |       prev       | seek(cond->value)|
 * +------------------I------------------+------------------+------------------+
 * |             <=   I         X        |        X         | seek(cond->value)|
 * +------------------I------------------+------------------+------------------+
 * |             ==   I    seek(min)     |        X         | seek(cond->value)|
 * +------------------I------------------+------------------+------------------+
 * |             >=   I    seek(min)     |        X         |         X        |
 * +------------------I------------------+------------------+------------------+
 * |             >    I    seek(min)     |    seek(min)     |         X        |
 * +------------------+------------------+------------------+------------------+
 *
 * The 'X' means "can't happen". The seek(min) sets the join key in
 * question to the minimum value for the value-type of the condition
 * attribute.
 *
 * Seek(cond->value) means set the key to the value tested in the condition.
 */
static sos_obj_t prev_match(sos_filter_t filt)
{
	SOS_KEY(key);
	int rc, i, join_idx;
	sos_obj_t obj;
	sos_filter_cond_t cond;
	sos_array_t attr_ids;
	struct sos_value_s v_;
	sos_value_t obj_value;
	ods_comp_key_t comp_key;
	ods_key_comp_t key_comp;
	size_t comp_len;
	sos_filter_cond_t join_cond;
	ods_ref_t last_ref = 0;
	do {
		obj_value = NULL;
		obj = sos_iter_obj(filt->iter);
		if (!obj)
			break;
		cond = sos_filter_eval(obj, filt);
		if (!cond) {
			return obj;
		}
		/*
		 * One or more conditions failed, determine if there
		 * can be any subsequent key that matches all
		 * conditions given the index ordering.
		 */
		if (cond->cond == SOS_COND_NE)
			/* No ordering optimizations for NE */
			goto prev;

		join_idx = __attr_join_idx(sos_iter_attr(filt->iter), cond->attr);
		if (join_idx < 0) {
			/*
			 * The filter key is not a join or the
			 * condition attribute is not in the join key
			 */
			if (cond->attr != filt->iter->attr)
				/*
				 * The filter index is not on
				 * condition attribute nothing can be
				 * assumed about the ordering
				 */
				goto prev;
			if (cond->cond > SOS_COND_EQ)
				/*
				 * The condition requires >=, this
				 * attribute is the key and the
				 * comparison failed. There can be no
				 * more matches
				 */
				break;
			goto prev;
		}

		if (join_idx == 0 || cond == TAILQ_FIRST(&filt->cond_list)) {
			if (join_idx)
				/* 1st condition skips join prefix, we know nothing */
				goto prev;
			/*
			 * The failing condition was ==, >=, or >. If
			 * the match was <, then there can not
			 * possibly be any more matches for this
			 * condition prior to this point in the index.
			 */
			if (cond->cond >= SOS_COND_EQ) {
				if (cond->ret <= 0)
					break;

				/* Cond is ==, >=,  or > and key is greater
				 * than value, keep searching.
				 */
				assert(0);
			}
			/* Cond is ==, <=, or < keep searching */
			goto prev;
		}

		if (sos_attr_is_array(cond->attr))
			goto prev;

		comp_key = (ods_comp_key_t)ods_key_value(key);
		comp_key->len = 0;
		key_comp = comp_key->value;
		attr_ids = sos_attr_join_list(sos_iter_attr(filt->iter));

		for (i = 0; i < attr_ids->count; i++) {
			int attr_id = attr_ids->data.uint32_[i];
			obj_value = sos_value_by_id(&v_, obj, attr_id);
			join_cond = __sos_find_filter_condition(filt, attr_id);
			if (i < join_idx) {
				if (!join_cond)
					goto prev;
				key_comp = __sos_set_key_comp(key_comp, obj_value, &comp_len);
			} else if (i == join_idx) {
				switch (cond->cond) {
				case SOS_COND_LT:
				case SOS_COND_LE:
					if (cond->ret > 0)
						key_comp = __sos_set_key_comp(key_comp, cond->value, &comp_len);
					else
						goto prev;
					break;
				case SOS_COND_EQ:
					if (cond->ret < 0) {
						key_comp = __sos_set_key_comp(key_comp, cond->value, &comp_len);
					} else {
						if (__sos_value_is_min(obj_value))
							goto prev;
						key_comp = __sos_set_key_comp_to_min(key_comp, obj_value->attr, &comp_len);
					}
					break;
				case SOS_COND_GE:
				case SOS_COND_GT:
					if (__sos_value_is_min(obj_value))
						goto prev;
					key_comp = __sos_set_key_comp_to_min(key_comp, obj_value->attr, &comp_len);
					break;
				case SOS_COND_NE:
					goto prev;
				}
			} else {
				if (sos_attr_is_array(obj_value->attr))
					goto prev;
				key_comp = __sos_set_key_comp_to_max(key_comp, obj_value->attr, &comp_len);
			}
			sos_value_put(obj_value);
			obj_value = NULL;
			comp_key->len += comp_len;
		}
		rc = sos_iter_inf(filt->iter, key);
		if (rc)
			break;
		if (last_ref == obj->obj->ref)
			goto out;
		last_ref = obj->obj->ref;
		sos_obj_put(obj);
		continue;
	prev:
		if (obj_value) {
			sos_value_put(obj_value);
			obj_value = NULL;
		}
		rc = sos_iter_prev(filt->iter);
		if (!rc)
			sos_obj_put(obj);
	} while (rc == 0);

 out:
	sos_obj_put(obj);
	return NULL;
}

static sos_filter_cond_t __sos_find_filter_condition(sos_filter_t filt, int attr_id)
{
	sos_filter_cond_t cond;
	TAILQ_FOREACH(cond, &filt->cond_list, entry) {
		if (attr_id == sos_attr_id(cond->attr))
			return cond;
	}
	return NULL;
}

static int __sos_filter_key_set(sos_filter_t filt, sos_key_t key, int min_not_max, int last_match)
{
	sos_filter_cond_t cond;
	int join_idx;
	sos_attr_t filt_attr = sos_iter_attr(filt->iter);
	int filt_attr_id = sos_attr_id(filt_attr);
	int search = 0;
	sos_array_t attr_ids = sos_attr_join_list(filt_attr);

	if (last_match) {
		sos_key_copy(key, filt->last_match);
		return ESRCH;
	}

	if (sos_attr_type(filt_attr) != SOS_TYPE_JOIN) {
		/* Find the first condition that matches the filter attr */
		cond = __sos_find_filter_condition(filt, filt_attr_id);
		if (!cond)
			goto out;
		sos_key_set(key, sos_value_as_key(cond->value),
			    sos_value_size(cond->value));
		if (cond->cond != SOS_COND_NE && (cond->cond >= SOS_COND_EQ))
			search = ESRCH;
	} else {
		ods_comp_key_t comp_key;
		ods_key_comp_t key_comp;
		comp_key = (ods_comp_key_t)ods_key_value(key);
		key_comp = comp_key->value;
		comp_key->len = 0;
		for (join_idx = 0; join_idx < attr_ids->count; join_idx++) {
			int join_attr_id = attr_ids->data.uint32_[join_idx];
			size_t comp_len;
			/* Search the condition list for this attribute */
			cond = __sos_find_filter_condition(filt, join_attr_id);

			if (cond) {
				if (sos_attr_is_array(cond->attr)) {
					key_comp = __sos_set_key_comp(key_comp, cond->value, &comp_len);
				} else if (min_not_max) {
					switch (cond->cond) {
					case SOS_COND_LT:
					case SOS_COND_NE:
					case SOS_COND_LE:
						key_comp = __sos_set_key_comp_to_min(key_comp, cond->attr, &comp_len);
						search = ESRCH;
						break;
					case SOS_COND_EQ:
					case SOS_COND_GE:
					case SOS_COND_GT:
						key_comp = __sos_set_key_comp(key_comp, cond->value, &comp_len);
						search = ESRCH;
						break;
					}
				} else {
					switch (cond->cond) {
					case SOS_COND_LT:
					case SOS_COND_LE:
					case SOS_COND_EQ:
						key_comp = __sos_set_key_comp(key_comp, cond->value, &comp_len);
						search = ESRCH;
						break;
					case SOS_COND_NE:
					case SOS_COND_GE:
					case SOS_COND_GT:
						key_comp = __sos_set_key_comp_to_max(key_comp, cond->attr, &comp_len);
						search = ESRCH;
						break;
					}
				}
				comp_key->len += comp_len;
			} else {
				sos_attr_t attr;
				/*
				 * If there is no condition on the prefix,
				 * don't bother with a search
				 */
				if (!join_idx)
					goto out;
				attr = sos_schema_attr_by_id(sos_attr_schema(filt_attr),
							     join_attr_id);
				if (sos_attr_is_array(attr) || sos_attr_is_ref(attr)) {
					/* There is no condition for this key component and the
					 * attribute has a variable length. The key order after the
					 * previous components will be determined by length.
					 */
					goto out;
				}
				if (min_not_max) {
					key_comp = __sos_set_key_comp_to_min(key_comp, attr, &comp_len);
				} else {
					key_comp = __sos_set_key_comp_to_max(key_comp, attr, &comp_len);
				}
				search = ESRCH;
				comp_key->len += comp_len;
			}
		}
	}
 out:
	return search;
}

/**
 * \brief Return the miss-compare count
 *
 * A miss-compare is an object on the iterator that was skipped due to
 * a failure to match all conditions on the filter. This value can be
 * useful when tuning queries for performance.
 *
 * \returns The miss count
 */
int sos_filter_miss_count(sos_filter_t filt)
{
	return filt->miss_cnt;
}


/**
 * \brief Return the first matching object.
 *
 * \param filt The filter handle.
 * \retval !NULL Pointer to the matching sos_obj_t.
 * \retval NULL  No object's matched all of the filter conditions.
 */
sos_obj_t sos_filter_begin(sos_filter_t filt)
{
	int rc;
	SOS_KEY(key);

	__sort_filter_conds_fwd(filt);
	rc = __sos_filter_key_set(filt, key, 1, 0);
	switch (rc) {
	case 0:
		rc = sos_iter_begin(filt->iter);
		break;
	default:
		rc = sos_iter_sup(filt->iter, key);
		break;
	}
	if (!rc)
		return next_match(filt);
	return NULL;
}

/**
 * \brief Return the next matching object.
 *
 * \param filt The filter handle.
 * \retval !NULL Pointer to the matching sos_obj_t.
 * \retval NULL  No object's matched all of the filter conditions.
 */
sos_obj_t sos_filter_next(sos_filter_t filt)
{
	if (0 == sos_iter_next(filt->iter))
		return next_match(filt);
	/*
	 * There are no more entries in the iterator. Position the cursor
	 * at the last object so that sos_filter_next will return any
	 * appended data.
	 */
	sos_iter_end(filt->iter);
	return NULL;
}

sos_obj_t sos_filter_prev(sos_filter_t filt)
{
	if (0 == sos_iter_prev(filt->iter))
		return prev_match(filt);
	/*
	 * There are no more entries in the iterator. Position the cursor
	 * at the 1st object so that sos_filter_prev will pick up any data
	 * added before this entry
	 */
	sos_iter_begin(filt->iter);
	return NULL;
}

sos_obj_t sos_filter_end(sos_filter_t filt)
{
	int rc;
	SOS_KEY(key);

	__sort_filter_conds_bkwd(filt);
	rc = __sos_filter_key_set(filt, key, 0, 0);
	switch (rc) {
	case 0:
		rc = sos_iter_end(filt->iter);
		break;
	default:
		rc = sos_iter_inf(filt->iter, key);
		break;
	}
	if (!rc)
		return prev_match(filt);
	return NULL;
}

sos_obj_t sos_filter_obj(sos_filter_t filt)
{
	return sos_iter_obj(filt->iter);
}
