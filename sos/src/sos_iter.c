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

sos_iter_t sos_iter_new(sos_attr_t attr)
{
	sos_iter_t i;

	if (!sos_attr_index(attr))
		return NULL;

	/* Find first active object partition and matching index */
	sos_idx_part_t idx_part = NULL;
	sos_obj_part_t obj_part = __sos_active_obj_part(attr->schema->sos);
	if (obj_part)
		idx_part = __sos_matching_idx_part(attr, obj_part);
	if (!idx_part)
		return NULL;

	i = calloc(1, sizeof *i);
	if (!i)
		goto err;

	sos_schema_get(attr->schema);
	i->attr = attr;
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

int sos_iter_flags_set(sos_iter_t iter, sos_iter_flags_t flags)
{
	return ods_iter_flags_set(iter->iter, flags);
}

sos_iter_flags_t sos_iter_flags_get(sos_iter_t iter)
{
	return (sos_iter_flags_t)ods_iter_flags_get(iter->iter);
}

uint64_t sos_iter_card(sos_iter_t iter)
{
	struct ods_idx_stat_s sb;
	int rc = ods_idx_stat(ods_iter_idx(iter->iter), &sb);
	if (rc)
		return 0;
	return sb.cardinality;
}

uint64_t sos_iter_dups(sos_iter_t iter)
{
	struct ods_idx_stat_s sb;
	int rc = ods_idx_stat(ods_iter_idx(iter->iter), &sb);
	if (rc)
		return 0;
	return sb.duplicates;
}

int sos_iter_pos(sos_iter_t iter, sos_pos_t pos)
{
	return ods_iter_pos(iter->iter, (ods_pos_t)pos);
}

int sos_iter_set(sos_iter_t iter, const sos_pos_t pos)
{
	return ods_iter_set(iter->iter, (ods_pos_t)pos);
}

void sos_iter_free(sos_iter_t iter)
{
	sos_schema_put(iter->attr->schema);
	ods_iter_delete(iter->iter);
	free(iter);
}

sos_obj_t sos_iter_obj(sos_iter_t i)
{
	ods_ref_t ods_ref = ods_iter_ref(i->iter);
	if (!ods_ref)
		return NULL;
	return __sos_init_obj(i->attr->schema->sos,
			      i->attr->schema,
			      ods_ref_as_obj(i->obj_part->obj_ods, ods_ref),
			      i->obj_part);
}

int sos_iter_obj_remove(sos_iter_t iter)
{
	return ENOSYS;
}

const char *sos_iter_name(sos_iter_t i)
{
	return sos_attr_name(i->attr);
}

sos_attr_t sos_iter_attr(sos_iter_t i)
{
	return i->attr;
}

int sos_iter_next(sos_iter_t i)
{
	return ods_iter_next(i->iter);
}

int sos_iter_prev(sos_iter_t i)
{
	return ods_iter_prev(i->iter);
}

int sos_iter_begin(sos_iter_t i)
{
	return ods_iter_begin(i->iter);
}

int sos_iter_end(sos_iter_t i)
{
	return ods_iter_end(i->iter);
}

int sos_iter_sup(sos_iter_t i, sos_key_t key)
{
	return ods_iter_find_lub(i->iter, key);
}

int sos_iter_inf(sos_iter_t i, sos_key_t key)
{
	return ods_iter_find_glb(i->iter, key);
}

int sos_iter_key_cmp(sos_iter_t iter, sos_key_t key)
{
	int rc;
	ods_key_t iter_key = ods_iter_key(iter->iter);
	rc = ods_key_cmp(iter->idx_part->index, iter_key, key);
	ods_obj_put(iter_key);
	return rc;
}

int sos_iter_find(sos_iter_t i, sos_key_t key)
{
	return ods_iter_find(i->iter, key);
}

sos_key_t sos_iter_key(sos_iter_t i)
{
	return ods_iter_key(i->iter);
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
			if (cond->attr == filt->iter->attr
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
			if (cond->attr == filt->iter->attr
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
		if (cond->attr != filt->iter->attr)
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
		if (cond->attr != filt->iter->attr)
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
