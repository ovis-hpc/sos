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

/* Create a new iterator */
sos_iter_t sos_iter_new(sos_attr_t attr)
{
	sos_iter_t i;

	if (!sos_attr_index(attr))
		return NULL;

	i = calloc(1, sizeof *i);
	if (!i)
		goto err;

	sos_schema_get(attr->schema);
	i->attr = attr;
	i->iter = ods_iter_new(attr->index);
	if (!i->iter)
		goto err;
	ods_iter_begin(i->iter);
	return i;
 err:
	if (i)
		free(i);
	return NULL;
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
	return __sos_init_obj(i->attr->schema,
			      ods_ref_as_obj(i->attr->schema->sos->obj_ods, ods_ref));
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

int sos_iter_seek_sup(sos_iter_t i, sos_key_t key)
{
	return ods_iter_find_lub(i->iter, key);
}

int sos_iter_seek_inf(sos_iter_t i, sos_key_t key)
{
	return ods_iter_find_glb(i->iter, key);
}

int sos_iter_key_cmp(sos_iter_t iter, sos_key_t key)
{
	int rc;
	ods_key_t iter_key = ods_iter_key(iter->iter);
	rc = ods_key_cmp(iter->attr->index, iter_key, key);
	ods_obj_put(iter_key);
	return rc;
}

int sos_iter_seek(sos_iter_t i, sos_key_t key)
{
	return ods_iter_find(i->iter, key);
}

sos_key_t sos_iter_key(sos_iter_t i)
{
	return ods_iter_key(i->iter);
}
