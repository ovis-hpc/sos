/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2014 Sandia Corporation. All rights reserved.
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
#define _GNU_SOURCE
#include <errno.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <sos/sos.h>
#include "sos_priv.h"

static size_t int32_size_fn(sos_value_t value)
{
	return sizeof(int32_t);
}

static size_t int64_size_fn(sos_value_t value)
{
	return sizeof(int64_t);
}

static size_t float_size_fn(sos_value_t value)
{
	return sizeof(float);
}

static size_t double_size_fn(sos_value_t value)
{
	return sizeof(double);
}

static size_t long_double_size_fn(sos_value_t value)
{
	return sizeof(long double);
}

static size_t timestamp_size_fn(sos_value_t value)
{
	return sizeof(uint64_t);
}

static size_t obj_size_fn(sos_value_t value)
{
	return sizeof(ods_ref_t);
}

/*
 * NB: Arrays are variable size. The size isn't known until the
 * application sets the size with the sos_value_array_size() function.
 */
static size_t byte_array_size_fn(sos_value_t value)
{
	return value->data->array.count;
}

static size_t int32_array_size_fn(sos_value_t value)
{
	return value->data->array.count * sizeof(int32_t);
}

static size_t int64_array_size_fn(sos_value_t value)
{
	return value->data->array.count * sizeof(int64_t);
}

static size_t float_array_size_fn(sos_value_t value)
{
	return value->data->array.count * sizeof(float);
}

static size_t double_array_size_fn(sos_value_t value)
{
	return value->data->array.count * sizeof(double);
}

static size_t long_double_array_size_fn(sos_value_t value)
{
	return value->data->array.count * sizeof(long double);
}

static size_t obj_array_size_fn(sos_value_t value)
{
	return value->data->array.count * sizeof(ods_ref_t);
}

sos_value_size_fn_t __sos_attr_size_fn_for_type(sos_type_t type)
{
	switch (type) {
	case SOS_TYPE_INT32:
		return int32_size_fn;
	case SOS_TYPE_INT64:
		return int64_size_fn;
	case SOS_TYPE_UINT32:
		return int32_size_fn;
	case SOS_TYPE_UINT64:
		return int64_size_fn;
	case SOS_TYPE_FLOAT:
		return float_size_fn;
	case SOS_TYPE_DOUBLE:
		return double_size_fn;
	case SOS_TYPE_LONG_DOUBLE:
		return long_double_size_fn;
	case SOS_TYPE_TIMESTAMP:
		return timestamp_size_fn;
	case SOS_TYPE_OBJ:
		return obj_size_fn;
	case SOS_TYPE_BYTE_ARRAY:
		return byte_array_size_fn;
	case SOS_TYPE_INT32_ARRAY:
		return int32_array_size_fn;
	case SOS_TYPE_INT64_ARRAY:
		return int64_array_size_fn;
	case SOS_TYPE_UINT32_ARRAY:
		return int32_array_size_fn;
	case SOS_TYPE_UINT64_ARRAY:
		return int64_array_size_fn;
	case SOS_TYPE_FLOAT_ARRAY:
		return float_array_size_fn;
	case SOS_TYPE_DOUBLE_ARRAY:
		return double_array_size_fn;
	case SOS_TYPE_LONG_DOUBLE_ARRAY:
		return long_double_array_size_fn;
	case SOS_TYPE_OBJ_ARRAY:
		return obj_array_size_fn;
	}
	return NULL;
}

static char *int32_to_str_fn(sos_value_t v, char *str, size_t len)
{
	snprintf(str, len, "%"PRId32"", v->data->prim.int32_);
	return str;
}

static char *int64_to_str_fn(sos_value_t v, char *str, size_t len)
{
	snprintf(str, len, "%"PRId64, v->data->prim.int64_);
	return str;
}

static char *uint32_to_str_fn(sos_value_t v, char *str, size_t len)
{
	snprintf(str, len, "%"PRIu32, v->data->prim.uint32_);
	return str;
}

static char *uint64_to_str_fn(sos_value_t v, char *str, size_t len)
{
	snprintf(str, len, "%"PRIu64, v->data->prim.uint64_);
	return str;
}

static char *float_to_str_fn(sos_value_t v, char *str, size_t len)
{
	snprintf(str, len, "%f", v->data->prim.float_);
	return str;
}

static char *double_to_str_fn(sos_value_t v, char *str, size_t len)
{
	snprintf(str, len, "%f", v->data->prim.double_);
	return str;
}

static char *long_double_to_str_fn(sos_value_t v, char *str, size_t len)
{
	snprintf(str, len, "%Lf", v->data->prim.long_double_);
	return str;
}

static char *timestamp_to_str_fn(sos_value_t v, char *str, size_t len)
{
	struct tm tm_;
	struct tm *tm = &tm_;
	size_t sz;
	time_t ts;

	memset(&tm_, 0, sizeof(tm_));
	/* NB: time_t is 8B on some machines and 4B on others. This is
	 * _not_ a 64b/32b issue, it's an OS choice and is not part of
	 * the C standards. Therefore fine.secs must be assigned to a
	 * time_t var and then passed to localtime to ensure
	 * correctness
	 */
	ts = (time_t)v->data->prim.timestamp_.fine.secs;
	tm = localtime_r(&ts, tm);
	sz = strftime(str, len, "%Y/%m/%d %H:%M:%S", tm);
	if (sz < len)
		snprintf(&str[sz], len - sz, ".%d",
			 v->data->prim.timestamp_.fine.usecs);
	return str;
}

static char *obj_to_str_fn(sos_value_t v, char *str, size_t len)
{
	sos_t sos = (v->obj ? v->obj->sos : NULL);
	sos_obj_t obj = (sos ? sos_obj_from_value(sos, v) : NULL);
	str[0] = '\0';
	snprintf(str, len, "%s@%lx",
		 (obj ? sos_schema_name(obj->schema) : "???"),
		 v->data->prim.ref_);
	if (obj)
		sos_obj_put(obj);
	return str;
}

static char *byte_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	if (!v)
		return "";
	memcpy(str, v->data->array.data.byte_, v->data->array.count);
	return str;
}

static char *int32_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	char *p = str;
	size_t count;
	int i;
	if (!v)
		return "";
	for (i = 0; i < v->data->array.count; i++) {
		if (i) {
			count = snprintf(p, len, ",");
			p++; len--;
		}
		count = snprintf(p, len, "%d", v->data->array.data.int32_[i]);
		p += count; len -= count;
	}
	return str;
}

static char *int64_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	char *p = str;
	size_t count;
	int i;

	for (i = 0; i < v->data->array.count; i++) {
		if (i) {
			count = snprintf(p, len, ",");
			p++; len--;
		}
		count = snprintf(p, len, "%"PRId64"", v->data->array.data.int64_[i]);
		p += count; len -= count;
	}
	return str;
}

static char *float_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	return "not supported";
}

static char *double_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	return "not supported";
}

static char *long_double_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	return "not supported";
}

static char *obj_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	return "not supported";
}

sos_value_to_str_fn_t __sos_attr_to_str_fn_for_type(sos_type_t type)
{
	switch (type) {
	case SOS_TYPE_INT32:
		return int32_to_str_fn;
	case SOS_TYPE_INT64:
		return int64_to_str_fn;
	case SOS_TYPE_UINT32:
		return int32_to_str_fn;
	case SOS_TYPE_UINT64:
		return int64_to_str_fn;
	case SOS_TYPE_FLOAT:
		return float_to_str_fn;
	case SOS_TYPE_DOUBLE:
		return double_to_str_fn;
	case SOS_TYPE_LONG_DOUBLE:
		return long_double_to_str_fn;
	case SOS_TYPE_TIMESTAMP:
		return timestamp_to_str_fn;
	case SOS_TYPE_OBJ:
		return obj_to_str_fn;
	case SOS_TYPE_BYTE_ARRAY:
		return byte_array_to_str_fn;
	case SOS_TYPE_INT32_ARRAY:
		return int32_array_to_str_fn;
	case SOS_TYPE_INT64_ARRAY:
		return int64_array_to_str_fn;
	case SOS_TYPE_UINT32_ARRAY:
		return int32_array_to_str_fn;
	case SOS_TYPE_UINT64_ARRAY:
		return int64_array_to_str_fn;
	case SOS_TYPE_FLOAT_ARRAY:
		return float_array_to_str_fn;
	case SOS_TYPE_DOUBLE_ARRAY:
		return double_array_to_str_fn;
	case SOS_TYPE_LONG_DOUBLE_ARRAY:
		return long_double_array_to_str_fn;
	case SOS_TYPE_OBJ_ARRAY:
		return obj_array_to_str_fn;
	}
	return NULL;
}

int int32_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.int32_ = strtol(value, endptr, 0);
	return 0;
}

int int64_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.int64_ = strtol(value, endptr, 0);
	return 0;
}

int uint32_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.uint32_ = strtoul(value, endptr, 0);
	return 0;
}

int uint64_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.uint64_ = strtoul(value, endptr, 0);
	return 0;
}

int float_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.float_ = strtof(value, endptr);
	return 0;
}

int double_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.double_ = strtod(value, endptr);
	return 0;
}

int long_double_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.long_double_ = strtold(value, endptr);
	return 0;
}

int timestamp_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	uint32_t secs, usecs;
	char *s;
	int rc;
	struct tm tm;
	memset(&tm, 0, sizeof(tm));
	if (!strchr(value, '/')) {
		if (strchr(value, '.')) {
			/* Treat as a float, time since epoch + microseconds */
			double ts = strtod(value, endptr);
			v->data->prim.timestamp_.fine.secs = (int)ts;
			v->data->prim.timestamp_.fine.usecs = 
				(uint32_t)((double)(ts - (int)ts) * 1.0e6);
		} else {
			/* Treat this like an epoch timestamp */
			v->data->prim.timestamp_.fine.secs = strtoul(value, endptr, 0);
			v->data->prim.timestamp_.fine.usecs = 0;
		}
	} else {
		/* Treat this like a formatted data/time string */
		s = strptime(value, "%Y/%m/%d %H:%M:%S", &tm);
		if (!s)
			return EINVAL;
		usecs = strtoul(s, endptr, 0);
		v->data->prim.timestamp_.fine.secs = mktime(&tm);
		v->data->prim.timestamp_.fine.usecs = usecs;
	}
	return 0;
}

int obj_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return ENOSYS;
}

int byte_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	strncpy(v->data->array.data.byte_, value, v->data->array.count);
	if (endptr)
		*endptr = (char *)(value + v->data->array.count);
	return 0;
}

int int32_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	char *saveptr;
	char *tok;
	size_t count;
	int i;
	char *str = strdup(value);
	if (!str)
		return ENOMEM;
	for (i = 0, tok = strtok_r(str, ",", &saveptr);
	     tok && i < v->data->array.count;
	     i++, tok = strtok_r(NULL, ",", &saveptr)) {
		v->data->array.data.int32_[i] = strtol(tok, endptr, 0);
	}
	free(str);
	return 0;
}

int int64_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return ENOSYS;
}

int uint32_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return ENOSYS;
}

int uint64_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return ENOSYS;
}

int float_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return ENOSYS;
}

int double_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return ENOSYS;
}

int long_double_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return ENOSYS;
}

int obj_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return ENOSYS;
}


sos_value_from_str_fn_t __sos_attr_from_str_fn_for_type(sos_type_t type)
{
	switch (type) {
	case SOS_TYPE_INT32:
		return int32_from_str_fn;
	case SOS_TYPE_INT64:
		return int64_from_str_fn;
	case SOS_TYPE_UINT32:
		return int32_from_str_fn;
	case SOS_TYPE_UINT64:
		return int64_from_str_fn;
	case SOS_TYPE_FLOAT:
		return float_from_str_fn;
	case SOS_TYPE_DOUBLE:
		return double_from_str_fn;
	case SOS_TYPE_LONG_DOUBLE:
		return long_double_from_str_fn;
	case SOS_TYPE_TIMESTAMP:
		return timestamp_from_str_fn;
	case SOS_TYPE_OBJ:
		return obj_from_str_fn;
	case SOS_TYPE_BYTE_ARRAY:
		return byte_array_from_str_fn;
	case SOS_TYPE_INT32_ARRAY:
		return int32_array_from_str_fn;
	case SOS_TYPE_INT64_ARRAY:
		return int64_array_from_str_fn;
	case SOS_TYPE_UINT32_ARRAY:
		return int32_array_from_str_fn;
	case SOS_TYPE_UINT64_ARRAY:
		return int64_array_from_str_fn;
	case SOS_TYPE_FLOAT_ARRAY:
		return float_array_from_str_fn;
	case SOS_TYPE_DOUBLE_ARRAY:
		return double_array_from_str_fn;
	case SOS_TYPE_LONG_DOUBLE_ARRAY:
		return long_double_array_from_str_fn;
	case SOS_TYPE_OBJ_ARRAY:
		return obj_array_from_str_fn;
	}
	return NULL;
}

/* Key Value Functions */

static void *int32_key_value_fn(sos_value_t val)
{
	return &val->data->prim.int32_;
}

static void *int64_key_value_fn(sos_value_t val)
{
	return &val->data->prim.int64_;
}

static void *uint32_key_value_fn(sos_value_t val)
{
	return &val->data->prim.uint32_;
}

static void *uint64_key_value_fn(sos_value_t val)
{
	return &val->data->prim.uint64_;
}

static void *float_key_value_fn(sos_value_t val)
{
	return &val->data->prim.float_;
}

static void *double_key_value_fn(sos_value_t val)
{
	return &val->data->prim.double_;
}

static void *long_double_key_value_fn(sos_value_t val)
{
	return &val->data->prim.long_double_;
}

static void *timestamp_key_value_fn(sos_value_t val)
{
	return &val->data->prim.timestamp_;
}

static void *obj_key_value_fn(sos_value_t val)
{
	assert(0); return NULL;
}

static void *byte_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.byte_;
}

static void *int32_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.int32_;
}

static void *int64_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.int64_;
}

static void *float_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.float_;
}

static void *double_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.double_;
}

static void *long_double_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.long_double_;
}

static void *obj_array_key_value_fn(sos_value_t val)
{
	assert(0); return NULL;
}

sos_value_key_value_fn_t __sos_attr_key_value_fn_for_type(sos_type_t type)
{
	switch (type) {
	case SOS_TYPE_INT32:
		return int32_key_value_fn;
	case SOS_TYPE_INT64:
		return int64_key_value_fn;
	case SOS_TYPE_UINT32:
		return int32_key_value_fn;
	case SOS_TYPE_UINT64:
		return int64_key_value_fn;
	case SOS_TYPE_FLOAT:
		return float_key_value_fn;
	case SOS_TYPE_DOUBLE:
		return double_key_value_fn;
	case SOS_TYPE_LONG_DOUBLE:
		return long_double_key_value_fn;
	case SOS_TYPE_TIMESTAMP:
		return timestamp_key_value_fn;
	case SOS_TYPE_OBJ:
		return obj_key_value_fn;
	case SOS_TYPE_BYTE_ARRAY:
		return byte_array_key_value_fn;
	case SOS_TYPE_INT32_ARRAY:
		return int32_array_key_value_fn;
	case SOS_TYPE_INT64_ARRAY:
		return int64_array_key_value_fn;
	case SOS_TYPE_UINT32_ARRAY:
		return int32_array_key_value_fn;
	case SOS_TYPE_UINT64_ARRAY:
		return int64_array_key_value_fn;
	case SOS_TYPE_FLOAT_ARRAY:
		return float_array_key_value_fn;
	case SOS_TYPE_DOUBLE_ARRAY:
		return double_array_key_value_fn;
	case SOS_TYPE_LONG_DOUBLE_ARRAY:
		return long_double_array_key_value_fn;
	case SOS_TYPE_OBJ_ARRAY:
		return obj_array_key_value_fn;
	}
	return NULL;
}

