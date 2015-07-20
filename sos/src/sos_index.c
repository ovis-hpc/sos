/*
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

#include <sys/types.h>
#include <stdlib.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <sos/sos.h>
#include "sos_priv.h"

static sos_index_t __sos_index_alloc(sos_t sos)
{
	sos_index_t index = calloc(1, sizeof *index);
	if (!index)
		return NULL;
	index->sos = sos;
	TAILQ_INIT(&index->idx_list);
	return index;
}

int sos_index_new(sos_t sos, const char *name,
		  const char *idx_type, const char *key_type,
		  ...)
{
	char tmp_path[PATH_MAX];
	sos_obj_part_t obj_part = __sos_primary_obj_part(sos);
	sprintf(tmp_path, "%s/%s/%s_idx",
		sos->path, SOS_PART(obj_part->part_obj)->name, name);

	int rc = ods_idx_create(tmp_path, sos->o_mode,
				idx_type, key_type, 5);
	return rc;
}

sos_index_t sos_index_open(sos_t sos, const char *name)
{
	char tmp_path[PATH_MAX];
	sos_obj_part_t obj_part = __sos_primary_obj_part(sos);
	sos_index_t index = __sos_index_alloc(sos);
	if (!index)
		goto err_0;

	sprintf(tmp_path, "%s/%s/%s_idx",
		sos->path, SOS_PART(obj_part->part_obj)->name, name);

	sos_idx_part_t part = calloc(1, sizeof *part);
	if (!part)
		goto err_1;
	part->part_obj = ods_obj_get(obj_part->part_obj);
	part->index = ods_idx_open(tmp_path, sos->o_perm);
	if (part->index)
		TAILQ_INSERT_TAIL(&index->idx_list, part, entry);
	else
		goto err_2;
	return index;
 err_2:
	ods_obj_put(part->part_obj);
	free(part);
 err_1:
	free(index);
 err_0:
	return NULL;
}

int sos_index_modify(sos_t sos, const char *name,
		     const char *idx_type, const char *key_type, ...);
int sos_index_insert(sos_index_t index, sos_key_t key, sos_obj_t obj)
{
	sos_idx_part_t part = __sos_matching_idx_part(index, obj->part);
	return ods_idx_insert(part->index, key, ods_obj_ref(obj->obj));
}

sos_obj_t sos_index_find(sos_index_t index, sos_key_t key)
{
	ods_ref_t ref;
	sos_obj_part_t obj_part = __sos_primary_obj_part(index->sos);
	sos_idx_part_t idx_part = __sos_matching_idx_part(index, obj_part);
	int rc = ods_idx_find(idx_part->index, key, &ref);
	if (rc) {
		errno = ENOENT;
		return NULL;
	}
	ods_obj_t obj = ods_ref_as_obj(obj_part->obj_ods, ref);
	if (!obj)
		return NULL;
	uint64_t schema_id = SOS_OBJ(obj)->schema;
	sos_schema_t schema = sos_schema_by_id(index->sos, schema_id);
	return __sos_init_obj(index->sos, schema, obj, obj_part);
}

int sos_index_close(sos_index_t index, sos_commit_t flags)
{
	sos_idx_part_t part;
	if (!index)
		return EINVAL;
	TAILQ_FOREACH(part, &index->idx_list, entry)
		ods_idx_close(part->index, ODS_COMMIT_ASYNC);
	free(index->key_type);
	free(index->idx_type);
	free(index);
	return 0;
}

