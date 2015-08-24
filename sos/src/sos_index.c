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
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

static sos_index_t __sos_index_alloc(sos_t sos)
{
	sos_index_t index = calloc(1, sizeof *index);
	if (!index)
		return NULL;
	index->sos = sos;
	return index;
}

int sos_index_new(sos_t sos, const char *name,
		  const char *idx_type, const char *key_type,
		  const char *idx_args)
{
	int rc;
	char tmp_path[PATH_MAX];
	sos_obj_ref_t idx_ref;
	ods_obj_t idx_obj;
	SOS_KEY(idx_key);

	size_t len = strlen(name);
	if (len >= SOS_IDX_NAME_LEN)
		return EINVAL;

	len = strlen(idx_type);
	if (len >= SOS_IDX_TYPE_LEN)
		return EINVAL;

	len = strlen(key_type);
	if (len >= SOS_IDX_KEY_TYPE_LEN)
		return EINVAL;

	if (!idx_args)
		idx_args = "";
	len = strlen(idx_args);
	if (len >= SOS_IDX_ARGS_LEN)
		return EINVAL;

	ods_spin_lock(&sos->idx_lock, -1);
	ods_key_set(idx_key, name, strlen(name)+1);
	rc = ods_idx_find(sos->idx_idx, idx_key, &idx_ref.idx_data);
	if (!rc) {
		rc = EEXIST;
		goto out;
	}
	idx_obj = ods_obj_alloc(sos->idx_ods, sizeof(struct sos_idx_data_s));
	if (!idx_obj) {
		rc = ENOMEM;
		goto out;
	}
	SOS_IDX(idx_obj)->ref_count = 1;
	SOS_IDX(idx_obj)->mode = 0664;
	strcpy(SOS_IDX(idx_obj)->name, name);
	strcpy(SOS_IDX(idx_obj)->idx_type, idx_type);
	strcpy(SOS_IDX(idx_obj)->key_type, key_type);
	strcpy(SOS_IDX(idx_obj)->args, idx_args);

	idx_ref.ref.ods = 0;
	idx_ref.ref.obj = ods_obj_ref(idx_obj);
	rc = ods_idx_insert(sos->idx_idx, idx_key, idx_ref.idx_data);
	if (rc)
		goto err_0;

	sprintf(tmp_path, "%s/%s_idx", sos->path, name);
	rc = ods_idx_create(tmp_path, sos->o_mode, idx_type, key_type, NULL);
	if (rc)
		goto err_1;
 out:
	ods_spin_unlock(&sos->idx_lock);
	return rc;
 err_1:
	ods_idx_delete(sos->idx_idx, idx_key, &idx_ref.idx_data);
 err_0:
	ods_obj_delete(idx_obj);
	ods_obj_put(idx_obj);
	ods_spin_unlock(&sos->idx_lock);
	return rc;
}

sos_index_t sos_index_open(sos_t sos, const char *name)
{
	size_t name_len;
	char tmp_path[PATH_MAX];
	sos_obj_ref_t idx_ref;
	ods_obj_t idx_obj = NULL;
	SOS_KEY(idx_key);
	sos_index_t index;
	int rc;

	name_len = strlen(name);
	if (name_len >= SOS_IDX_NAME_LEN) {
		errno = EINVAL;
		goto err_0;
	}

	index = __sos_index_alloc(sos);
	if (!index)
		goto err_1;

	ods_key_set(idx_key, name, strlen(name)+1);
	ods_spin_lock(&sos->idx_lock, -1);
	rc = ods_idx_find(sos->idx_idx, idx_key, &idx_ref.idx_data);
	if (rc)
		goto err_2;

	sprintf(tmp_path, "%s/%s_idx", sos->path, name);
 retry:
	index->idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!index->idx) {
		if (idx_obj)
			/* Already failed once, err out */
			goto err_3;
		/* Attempt to create it */
		idx_obj = ods_ref_as_obj(sos->idx_ods, idx_ref.ref.obj);
		if (!idx_obj) {
			rc = EINVAL;
			goto err_2;
		}
		rc = ods_idx_create(SOS_IDX(idx_obj)->name,
				    SOS_IDX(idx_obj)->mode,
				    SOS_IDX(idx_obj)->idx_type,
				    SOS_IDX(idx_obj)->key_type,
				    SOS_IDX(idx_obj)->args);
		if (!rc)
			goto retry;
		goto err_3;
	}
	ods_obj_put(idx_obj);
	ods_spin_unlock(&sos->idx_lock);
	return index;
 err_3:
	ods_obj_put(idx_obj);
 err_2:
	ods_spin_unlock(&sos->idx_lock);
 err_1:
	free(index);
 err_0:
	return NULL;
}

int sos_index_insert(sos_index_t index, sos_key_t key, sos_obj_t obj)
{
	return ods_idx_insert(index->idx, key, obj->obj_ref.idx_data);
}

int sos_index_obj_remove(sos_index_t index, sos_key_t key, sos_obj_t obj)
{
	sos_obj_ref_t idx_ref;
	ods_obj_t ref_obj;
	int rc = ods_idx_delete(index->idx, key, &idx_ref.idx_data);
	if (rc)
		return rc;
	ref_obj = ods_ref_as_obj(ods_idx_ods(index->idx), idx_ref.ref.obj);
	if (!ref_obj)
		return EINVAL;
	ods_ref_delete(ods_idx_ods(index->idx), idx_ref.ref.obj);
	return 0;
}

sos_obj_t sos_index_find(sos_index_t index, sos_key_t key)
{
	sos_obj_ref_t idx_ref;
	sos_obj_t obj;
	int rc = ods_idx_find(index->idx, key, &idx_ref.idx_data);
	if (rc) {
		errno = ENOENT;
		return NULL;
	}
	obj = sos_ref_as_obj(index->sos, idx_ref);
	if (!obj)
		errno = EINVAL;
	return obj;
}

sos_obj_t sos_index_find_sup(sos_index_t index, sos_key_t key)
{
	sos_obj_ref_t idx_ref;
	sos_obj_t obj;
	int rc = ods_idx_find_lub(index->idx, key, &idx_ref.idx_data);
	if (rc) {
		errno = ENOENT;
		return NULL;
	}
	obj = sos_ref_as_obj(index->sos, idx_ref);
	if (!obj)
		errno = EINVAL;
	return obj;
}

sos_obj_t sos_index_find_inf(sos_index_t index, sos_key_t key)
{
	sos_obj_ref_t idx_ref;
	sos_obj_t obj;
	int rc = ods_idx_find_glb(index->idx, key, &idx_ref.idx_data);
	if (rc) {
		errno = ENOENT;
		return NULL;
	}
	obj = sos_ref_as_obj(index->sos, idx_ref);
	if (!obj)
		errno = EINVAL;
	return obj;
}

int sos_index_close(sos_index_t index, sos_commit_t flags)
{
	if (!index)
		return EINVAL;
	ods_idx_close(index->idx, ODS_COMMIT_ASYNC);
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
	return ods_idx_key_size(index->idx);
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
	return ods_key_from_str(index->idx, key, str);
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
	char *keystr = malloc(ods_idx_key_str_size(index->idx));
	return ods_key_to_str(index->idx, key, keystr);
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
int sos_index_key_cmp(sos_index_t index, sos_key_t a, sos_key_t b)
{
	return ods_key_cmp(index->idx, a, b);
}

void sos_index_print(sos_index_t index, FILE *fp)
{
	ods_idx_print(index->idx, fp);
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
	int rc = ods_iter_next(iter->iter);
	if (rc)
		return NULL;
	idx_ref.idx_data = ods_iter_data(iter->iter);
	ods_obj_t idx_obj = ods_ref_as_obj(iter->sos->idx_ods, idx_ref.ref.obj);
	idx = sos_index_open(iter->sos, SOS_IDX(idx_obj)->name);
	ods_obj_put(idx_obj);
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

