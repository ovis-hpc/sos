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

#include <sos/sos.h>
#include "sos_priv.h"

typedef int (*cmp_fn_t)(sos_value_t a, sos_value_t b);

static int INT32_cmp(sos_value_t a, sos_value_t b)
{
	return a->data->prim.int32_ - b->data->prim.int32_;
}

static int UINT32_cmp(sos_value_t a, sos_value_t b)
{
	return a->data->prim.uint32_ - b->data->prim.uint32_;
}

static int INT64_cmp(sos_value_t a, sos_value_t b)
{
	return a->data->prim.int64_ - b->data->prim.int64_;
}

static int UINT64_cmp(sos_value_t a, sos_value_t b)
{
	return a->data->prim.uint64_ - b->data->prim.uint64_;
}

static int FLOAT_cmp(sos_value_t a, sos_value_t b)
{
	float res = a->data->prim.float_ - b->data->prim.float_;
	if (res > 0)
		return 1;
	else if (res < 0)
		return -1;
	return 0;
}

static int DOUBLE_cmp(sos_value_t a, sos_value_t b)
{
	double res = a->data->prim.double_ - b->data->prim.double_;
	if (res > 0)
		return 1;
	else if (res < 0)
		return -1;
	return 0;
}

static int LONG_DOUBLE_cmp(sos_value_t a, sos_value_t b)
{
	long double res = a->data->prim.long_double_ - b->data->prim.long_double_;
	if (res > 0)
		return 1;
	else if (res < 0)
		return -1;
	return 0;
}

static int TIMESTAMP_cmp(sos_value_t a, sos_value_t b)
{
	if (a->data->prim.timestamp_.time > b->data->prim.timestamp_.time)
		return 1;
	else if (a->data->prim.timestamp_.time < b->data->prim.timestamp_.time)
		return -1;
	return 0;
}

static cmp_fn_t cmp_fn_table[] = {
	[SOS_TYPE_INT32] = INT32_cmp,
	[SOS_TYPE_INT64] = INT64_cmp,
	[SOS_TYPE_UINT32] = UINT32_cmp,
	[SOS_TYPE_UINT64] = UINT64_cmp,
	[SOS_TYPE_FLOAT] = FLOAT_cmp,
	[SOS_TYPE_DOUBLE] = DOUBLE_cmp,
	[SOS_TYPE_LONG_DOUBLE] = LONG_DOUBLE_cmp,
	[SOS_TYPE_TIMESTAMP] = TIMESTAMP_cmp,
};

int sos_value_cmp(sos_value_t a, sos_value_t b)
{
	if (a->attr->data->type < sizeof(cmp_fn_table)/sizeof(cmp_fn_table[0]))
		return cmp_fn_table[a->attr->data->type](a, b);
	return a == b;
}
