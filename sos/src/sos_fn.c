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
#include <errno.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sos/sos.h>
#include "sos_priv.h"

static size_t int32_size_fn(sos_attr_t attr, sos_value_t value)
{
	return sizeof(int32_t);
}

static size_t int64_size_fn(sos_attr_t attr, sos_value_t value)
{
	return sizeof(int64_t);
}

static size_t float_size_fn(sos_attr_t attr, sos_value_t value)
{
	return sizeof(float);
}

static size_t double_size_fn(sos_attr_t attr, sos_value_t value)
{
	return sizeof(double);
}

static size_t long_double_size_fn(sos_attr_t attr, sos_value_t value)
{
	return sizeof(long double);
}

static size_t obj_size_fn(sos_attr_t attr, sos_value_t value)
{
	assert(0);
	return ENOSYS;
}

static size_t byte_array_size_fn(sos_attr_t attr, sos_value_t value)
{
	return attr->data->initial_count;
}

static size_t int32_array_size_fn(sos_attr_t attr, sos_value_t value)
{
	return attr->data->initial_count * sizeof(int32_t);
}

static size_t int64_array_size_fn(sos_attr_t attr, sos_value_t value)
{
	return attr->data->initial_count * sizeof(int64_t);
}

static size_t float_array_size_fn(sos_attr_t attr, sos_value_t value)
{
	return attr->data->initial_count * sizeof(float);
}

static size_t double_array_size_fn(sos_attr_t attr, sos_value_t value)
{
	return attr->data->initial_count * sizeof(double);
}

static size_t long_double_array_size_fn(sos_attr_t attr, sos_value_t value)
{
	return attr->data->initial_count * sizeof(long double);
}

static size_t obj_array_size_fn(sos_attr_t attr, sos_value_t value)
{
	assert(0);
	return ENOSYS;
}

sos_attr_size_fn_t __sos_attr_size_fn_for_type(sos_type_t type)
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

static char *int32_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	sos_value_t v = sos_value(obj, attr);
	snprintf(str, len, "%"PRId32"", v->data->prim.int32_);
	sos_value_put(v);
	return str;
}

static char *int64_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	sos_value_t v = sos_value(obj, attr);
	snprintf(str, len, "%"PRId64, v->data->prim.int64_);
	sos_value_put(v);
	return str;
}

static char *uint32_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	sos_value_t v = sos_value(obj, attr);
	snprintf(str, len, "%"PRIu32, v->data->prim.uint32_);
	sos_value_put(v);
	return str;
}

static char *uint64_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	sos_value_t v = sos_value(obj, attr);
	snprintf(str, len, "%"PRIu64, v->data->prim.uint64_);
	sos_value_put(v);
	return str;
}

static char *float_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	sos_value_t v = sos_value(obj, attr);
	snprintf(str, len, "%f", v->data->prim.float_);
	sos_value_put(v);
	return str;
}

static char *double_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	sos_value_t v = sos_value(obj, attr);
	snprintf(str, len, "%f", v->data->prim.double_);
	sos_value_put(v);
	return str;
}

static char *long_double_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	sos_value_t v = sos_value(obj, attr);
	snprintf(str, len, "%Lf", v->data->prim.long_double_);
	sos_value_put(v);
	return str;
}

static char *obj_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	return "not supported";
}

static char *byte_array_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return "";
	memcpy(str, v->data->array.byte_, v->data->array.count);
	sos_value_put(v);
	return str;
}

static char *int32_array_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	char *p = str;
	size_t count;
	int i;
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return "";
	for (i = 0; i < v->data->array.count; i++) {
		if (i) {
			count = snprintf(p, len, ",");
			p++; len--;
		}
		count = snprintf(p, len, "%d", v->data->array.int32_[i]);
		p += count; len -= count;
	}
	sos_value_put(v);
	return str;
}

static char *int64_array_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	char *p = str;
	size_t count;
	int i;
	sos_value_t v = sos_value(obj, attr);

	for (i = 0; i < v->data->array.count; i++) {
		if (i) {
			count = snprintf(p, len, ",");
			p++; len--;
		}
		count = snprintf(p, len, "%"PRId64"", v->data->array.int64_[i]);
		p += count; len -= count;
	}
	sos_value_put(v);
	return str;
}

static char *float_array_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	return "not supported";
}

static char *double_array_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	return "not supported";
}

static char *long_double_array_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	return "not supported";
}

static char *obj_array_to_str_fn(sos_attr_t attr, sos_obj_t obj, char *str, size_t len)
{
	return "not supported";
}

sos_attr_to_str_fn_t __sos_attr_to_str_fn_for_type(sos_type_t type)
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

int int32_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return errno;
	v->data->prim.int32_ = strtol(value, NULL, 0);
	sos_value_put(v);
	return 0;
}

int int64_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return errno;
	v->data->prim.int64_ = strtol(value, NULL, 0);
	sos_value_put(v);
	return 0;
}

int uint32_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return errno;
	v->data->prim.uint32_ = strtoul(value, NULL, 0);
	sos_value_put(v);
	return 0;
}

int uint64_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return errno;
	v->data->prim.uint64_ = strtoul(value, NULL, 0);
	sos_value_put(v);
	return 0;
}

int float_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return errno;
	v->data->prim.float_ = strtof(value, NULL);
	sos_value_put(v);
	return 0;
}

int double_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return errno;
	v->data->prim.double_ = strtod(value, NULL);
	sos_value_put(v);
	return 0;
}

int long_double_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return errno;
	v->data->prim.long_double_ = strtold(value, NULL);
	sos_value_put(v);
	return 0;
}

int obj_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	return ENOSYS;
}

int byte_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	sos_value_t v = sos_value(obj, attr);
	if (!v)
		return errno;
	strncpy(v->data->array.byte_, value, v->data->array.count);
	sos_value_put(v);
	return 0;
}

int int32_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	char *saveptr;
	char *tok;
	size_t count;
	int i;
	char *str = strdup(value);
	if (!str)
		return ENOMEM;
	sos_value_t v = sos_value(obj, attr);
	for (i = 0, tok = strtok_r(str, ",", &saveptr);
	     tok && i < v->data->array.count;
	     i++, tok = strtok_r(NULL, ",", &saveptr)) {
		v->data->array.int32_[i] = strtol(tok, NULL, 0);
	}
	sos_value_put(v);
	free(str);
	return 0;
}

int int64_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
}

int uint32_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
}

int uint64_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
}

int float_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
}

int double_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
}

int long_double_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
}

int obj_array_from_str_fn(struct sos_attr_s *attr, sos_obj_t obj, const char *value)
{
	return ENOSYS;
}


sos_attr_from_str_fn_t __sos_attr_from_str_fn_for_type(sos_type_t type)
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

static void *int32_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return &val->data->prim.int32_;
}

static void *int64_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return &val->data->prim.int64_;
}

static void *uint32_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return &val->data->prim.uint32_;
}

static void *uint64_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return &val->data->prim.uint64_;
}

static void *float_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return &val->data->prim.float_;
}

static void *double_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return &val->data->prim.double_;
}

static void *long_double_key_value_fn(sos_attr_t attr, sos_value_t val)
{
}

static void *obj_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	assert(0); return NULL;
}

static void *byte_array_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return val->data->array.byte_;
}

static void *int32_array_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return val->data->array.int32_;
}

static void *int64_array_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return val->data->array.int64_;
}

static void *float_array_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return val->data->array.float_;
}

static void *double_array_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return val->data->array.double_;
}

static void *long_double_array_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	return val->data->array.long_double_;
}

static void *obj_array_key_value_fn(sos_attr_t attr, sos_value_t val)
{
	assert(0); return NULL;
}

sos_attr_key_value_fn_t __sos_attr_key_value_fn_for_type(sos_type_t type)
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

