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

static size_t int16_size_fn(sos_value_t value)
{
	return sizeof(int16_t);
}

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

static size_t struct_size_fn(sos_value_t value)
{
	return value->attr->data->size;
}

/*
 * NB: Arrays are variable size. The size isn't known until the
 * application sets the size with the sos_value_array_size() function.
 */
static size_t byte_array_size_fn(sos_value_t value)
{
	return value->data->array.count;
}

static size_t char_array_size_fn(sos_value_t value)
{
	return value->data->array.count;
}

static size_t int16_array_size_fn(sos_value_t value)
{
	return value->data->array.count * sizeof(int16_t);
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

static size_t join_size_fn(sos_value_t value)
{
	/* join value is a byte array */
	return value->data->array.count;
}

static sos_value_size_fn_t __attr_size_fn_for_type[] = {
	[SOS_TYPE_INT16] = int16_size_fn,
	[SOS_TYPE_INT32] = int32_size_fn,
	[SOS_TYPE_INT64] = int64_size_fn,
	[SOS_TYPE_UINT16] = int16_size_fn,
	[SOS_TYPE_UINT32] = int32_size_fn,
	[SOS_TYPE_UINT64] = int64_size_fn,
	[SOS_TYPE_FLOAT] = float_size_fn,
	[SOS_TYPE_DOUBLE] = double_size_fn,
	[SOS_TYPE_LONG_DOUBLE] = long_double_size_fn,
	[SOS_TYPE_TIMESTAMP] = timestamp_size_fn,
	[SOS_TYPE_OBJ] = obj_size_fn,
	[SOS_TYPE_STRUCT] = struct_size_fn,
	[SOS_TYPE_JOIN] = join_size_fn,
	[SOS_TYPE_BYTE_ARRAY] = byte_array_size_fn,
	[SOS_TYPE_CHAR_ARRAY] = char_array_size_fn,
	[SOS_TYPE_INT16_ARRAY] = int16_array_size_fn,
	[SOS_TYPE_INT32_ARRAY] = int32_array_size_fn,
	[SOS_TYPE_INT64_ARRAY] = int64_array_size_fn,
	[SOS_TYPE_UINT16_ARRAY] = int16_array_size_fn,
	[SOS_TYPE_UINT32_ARRAY] = int32_array_size_fn,
	[SOS_TYPE_UINT64_ARRAY] = int64_array_size_fn,
	[SOS_TYPE_FLOAT_ARRAY] = float_array_size_fn,
	[SOS_TYPE_DOUBLE_ARRAY] = double_array_size_fn,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = long_double_array_size_fn,
	[SOS_TYPE_OBJ_ARRAY] = obj_array_size_fn,
};

sos_value_size_fn_t __sos_attr_size_fn_for_type(sos_type_t type)
{
	if (type >= SOS_TYPE_FIRST && type <= SOS_TYPE_LAST)
		return __attr_size_fn_for_type[type];
	return NULL;
}


#define PRIMITIVE_STRLEN_FN(_name_, _fmt_, _member_)			\
	static size_t _name_ ## _strlen_fn(sos_value_t v)		\
	{								\
		return snprintf(NULL, 0, _fmt_, v->data->prim. _member_); \
	}

PRIMITIVE_STRLEN_FN(int16, "%hd", int16_)
PRIMITIVE_STRLEN_FN(int32, "%"PRId32"", int32_)
PRIMITIVE_STRLEN_FN(int64, "%"PRId64"", int64_)
PRIMITIVE_STRLEN_FN(uint16, "%hu", uint16_)
PRIMITIVE_STRLEN_FN(uint32, "%"PRIu32"", uint32_)
PRIMITIVE_STRLEN_FN(uint64, "%"PRIu64"", uint64_)
PRIMITIVE_STRLEN_FN(float, "%f", float_)
PRIMITIVE_STRLEN_FN(double, "%lf", double_)
PRIMITIVE_STRLEN_FN(long_double, "%Lf", long_double_)

static size_t timestamp_strlen_fn(sos_value_t v)
{
	double t= (double)v->data->prim.timestamp_.fine.secs +
		((double)v->data->prim.timestamp_.fine.usecs) / 1.0e6;
	return snprintf(NULL, 0, "%.6f", t) + 1;
}

static size_t obj_strlen_fn(sos_value_t v)
{
	return snprintf(NULL, 0, "%lu@%lx",
			v->data->prim.ref_.ref.ods,
			v->data->prim.ref_.ref.obj) + 1;
}

/* Value is formatted as %02X%02X...%02X\0 */
static size_t struct_strlen_fn(sos_value_t v)
{
	return (v->attr->data->size * 2) + 1;
}

/* Value is formatted as %02X:%02X:...%02X\0 */
static size_t byte_array_strlen_fn(sos_value_t v)
{
	return v ? v->data->array.count * 3 : 1;
}

static size_t join_strlen_fn(sos_value_t v)
{
	return byte_array_strlen_fn(v);
}

static size_t char_array_strlen_fn(sos_value_t v)
{
	return v ? v->data->array.count + 1 : 1;
}

#define PRIMITIVE_ARRAY_STRLEN_FN(_name_, _fmt_, _member_)		\
	static size_t _name_ ## _array_strlen_fn(sos_value_t v)		\
	{								\
		size_t i, count = 1;					\
		if (!v)							\
			goto out;					\
		for (i = 0; i < v->data->array.count; i++) {		\
			if (i)						\
				count += snprintf(NULL, 0, ",");	\
			count += snprintf(NULL, 0, _fmt_, v->data->array.data. _member_ [i]); \
		}							\
	out:								\
		return count;						\
	}


PRIMITIVE_ARRAY_STRLEN_FN(int16, "%hd", int16_)
PRIMITIVE_ARRAY_STRLEN_FN(uint16, "0x%04hx", uint16_)

PRIMITIVE_ARRAY_STRLEN_FN(int32, "%"PRId32"", int32_)
PRIMITIVE_ARRAY_STRLEN_FN(uint32, "%"PRIu32"", uint32_)

PRIMITIVE_ARRAY_STRLEN_FN(int64, "%"PRId64"", int64_)
PRIMITIVE_ARRAY_STRLEN_FN(uint64, "%"PRIu64"", uint64_)

PRIMITIVE_ARRAY_STRLEN_FN(float, "%f", double_)
PRIMITIVE_ARRAY_STRLEN_FN(double, "%lf", double_)
PRIMITIVE_ARRAY_STRLEN_FN(long_double, "%Lf", long_double_)

static size_t obj_array_strlen_fn(sos_value_t v)
{
	size_t i, count = 1;
	if (!v)
		goto out;
	for (i = 0; i < v->data->array.count; i++) {
		if (i)
			count += snprintf(NULL, 0, ",");
		count += snprintf(NULL, 0, "%lu@%lx",
				  v->data->array.data.ref_[i].ref.ods,
				  v->data->array.data.ref_[i].ref.obj);
	}
 out:
	return count;
}

static sos_value_strlen_fn_t __attr_strlen_fn_for_type[] = {
	[SOS_TYPE_INT16] = int16_strlen_fn,
	[SOS_TYPE_INT32] = int32_strlen_fn,
	[SOS_TYPE_INT64] = int64_strlen_fn,
	[SOS_TYPE_UINT16] = uint16_strlen_fn,
	[SOS_TYPE_UINT32] = uint32_strlen_fn,
	[SOS_TYPE_UINT64] = uint64_strlen_fn,
	[SOS_TYPE_FLOAT] = float_strlen_fn,
	[SOS_TYPE_DOUBLE] = double_strlen_fn,
	[SOS_TYPE_LONG_DOUBLE] = long_double_strlen_fn,
	[SOS_TYPE_TIMESTAMP] = timestamp_strlen_fn,
	[SOS_TYPE_OBJ] = obj_strlen_fn,
	[SOS_TYPE_STRUCT] = struct_strlen_fn,
	[SOS_TYPE_JOIN] = join_strlen_fn,
	[SOS_TYPE_BYTE_ARRAY] = byte_array_strlen_fn,
	[SOS_TYPE_CHAR_ARRAY] = char_array_strlen_fn,
	[SOS_TYPE_INT16_ARRAY] = int16_array_strlen_fn,
	[SOS_TYPE_INT32_ARRAY] = int32_array_strlen_fn,
	[SOS_TYPE_INT64_ARRAY] = int64_array_strlen_fn,
	[SOS_TYPE_UINT16_ARRAY] = uint16_array_strlen_fn,
	[SOS_TYPE_UINT32_ARRAY] = uint32_array_strlen_fn,
	[SOS_TYPE_UINT64_ARRAY] = uint64_array_strlen_fn,
	[SOS_TYPE_FLOAT_ARRAY] = float_array_strlen_fn,
	[SOS_TYPE_DOUBLE_ARRAY] = double_array_strlen_fn,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = long_double_array_strlen_fn,
	[SOS_TYPE_OBJ_ARRAY] = obj_array_strlen_fn,
};

sos_value_strlen_fn_t __sos_attr_strlen_fn_for_type(sos_type_t type)
{
	if (type >= SOS_TYPE_FIRST && type <= SOS_TYPE_LAST)
		return __attr_strlen_fn_for_type[type];
	return NULL;
}

#define PRIMITIVE_TO_STR_FN(_name_, _fmt_, _member_) \
	static char *_name_ ## _to_str_fn(sos_value_t v, char *str, size_t len) \
	{ \
		snprintf(str, len, _fmt_, v->data->prim. _member_); \
		return str; \
	}

PRIMITIVE_TO_STR_FN(int16, "%hd", int16_)
PRIMITIVE_TO_STR_FN(int32, "%"PRId32"", int32_)
PRIMITIVE_TO_STR_FN(int64, "%"PRId64"", int64_)

PRIMITIVE_TO_STR_FN(uint16, "%hu", uint16_)
PRIMITIVE_TO_STR_FN(uint32, "%"PRIu32"", uint32_)
PRIMITIVE_TO_STR_FN(uint64, "%"PRIu64"", uint64_)

PRIMITIVE_TO_STR_FN(float, "%f", float_)
PRIMITIVE_TO_STR_FN(double, "%lf", double_)
PRIMITIVE_TO_STR_FN(long_double, "%Lf", long_double_)

static char *timestamp_to_str_fn(sos_value_t v, char *str, size_t len)
{
	double t= (double)v->data->prim.timestamp_.fine.secs +
		((double)v->data->prim.timestamp_.fine.usecs) / 1.0e6;
	snprintf(str, len, "%.6f", t);
	return str;
}

static char *obj_to_str_fn(sos_value_t v, char *str, size_t len)
{
	str[0] = '\0';
	snprintf(str, len, "%lu@%lx",
		 v->data->prim.ref_.ref.ods, v->data->prim.ref_.ref.obj);
	return str;
}

static char *struct_to_str_fn(sos_value_t v, char *str, size_t len)
{
	int i, res_cnt;
	char *p = str;
	if (!v)
		return "";
	for (i = 0; i < v->attr->data->size && len > 0; i++) {
		res_cnt = snprintf(p, len, "%02X", v->data->struc.byte_[i]);
		if (res_cnt > len)
			break;
		p += res_cnt;
		len -= res_cnt;
	}
	return str;
}

static char *byte_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	int i, res_cnt;
	char *fmt;
	char *p = str;
	if (!v)
		return "";
	for (i = 0; i < v->data->array.count && len > 0; i++) {
		if (p == str)
			fmt = "0x%02x";
		else
			fmt = ",0x%02x";
		res_cnt = snprintf(p, len, fmt, v->data->array.data.byte_[i]);
		if (res_cnt > len)
			break;
		p += res_cnt;
		len -= res_cnt;
	}
	return str;
}

static char *join_to_str_fn(sos_value_t v, char *str, size_t len)
{
	return byte_array_to_str_fn(v, str, len);
}

static char *char_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	if (!v)
		return "";
	strncpy(str, (char *)v->data->array.data.char_, v->data->array.count);
	str[v->data->array.count] = '\0';
	return str;
}

#define PRIMITIVE_ARRAY_TO_STR_FN(_name_, _fmt_, _member_)		\
	static char * _name_ ## _array_to_str_fn(sos_value_t v, char *str, size_t len) \
	{								\
		char *p = str;						\
		size_t count;						\
		int i;							\
		if (!v)							\
			return "";					\
		for (i = 0; i < v->data->array.count && len > 0; i++) { \
			if (i) {					\
				count = snprintf(p, len, ",");		\
				p++; len--;				\
				if (!len)				\
					break;				\
			}						\
			count = snprintf(p, len, _fmt_, v->data->array.data. _member_ [i]); \
			if (count > len)				\
				break;					\
			p += count; len -= count;			\
		}							\
		return str;						\
	}


PRIMITIVE_ARRAY_TO_STR_FN(int16, "%hd", int16_)
PRIMITIVE_ARRAY_TO_STR_FN(int32, "%"PRId32"", int32_)
PRIMITIVE_ARRAY_TO_STR_FN(int64, "%"PRId64"", int64_)

PRIMITIVE_ARRAY_TO_STR_FN(uint16, "%hu", uint16_)
PRIMITIVE_ARRAY_TO_STR_FN(uint32, "%"PRIu32"", uint32_)
PRIMITIVE_ARRAY_TO_STR_FN(uint64, "%"PRIu64"", uint64_)

PRIMITIVE_ARRAY_TO_STR_FN(float, "%f", float_)
PRIMITIVE_ARRAY_TO_STR_FN(double, "%lf", double_)
PRIMITIVE_ARRAY_TO_STR_FN(long_double, "%Lf", long_double_)

static char *obj_array_to_str_fn(sos_value_t v, char *str, size_t len)
{
	char *p = str;
	size_t count;
	int i;

	for (i = 0; i < v->data->array.count && len > 0; i++) {
		if (i) {
			count = snprintf(p, len, ",");
			p++; len--;
			if (!len)
				break;
		}
		count = snprintf(p, len, "%lu@%lx",
				 v->data->array.data.ref_[i].ref.ods,
				 v->data->array.data.ref_[i].ref.obj);
		if (count > len)
			break;
		p += count; len -= count;
	}
	return str;
}

static sos_value_to_str_fn_t __attr_to_str_fn_for_type[] = {
	[SOS_TYPE_INT16] = int16_to_str_fn,
	[SOS_TYPE_INT32] = int32_to_str_fn,
	[SOS_TYPE_INT64] = int64_to_str_fn,
	[SOS_TYPE_UINT16] = uint16_to_str_fn,
	[SOS_TYPE_UINT32] = uint32_to_str_fn,
	[SOS_TYPE_UINT64] = uint64_to_str_fn,
	[SOS_TYPE_FLOAT] = float_to_str_fn,
	[SOS_TYPE_DOUBLE] = double_to_str_fn,
	[SOS_TYPE_LONG_DOUBLE] = long_double_to_str_fn,
	[SOS_TYPE_TIMESTAMP] = timestamp_to_str_fn,
	[SOS_TYPE_OBJ] = obj_to_str_fn,
	[SOS_TYPE_STRUCT] = struct_to_str_fn,
	[SOS_TYPE_JOIN] = join_to_str_fn,
	[SOS_TYPE_BYTE_ARRAY] = byte_array_to_str_fn,
	[SOS_TYPE_CHAR_ARRAY] = char_array_to_str_fn,
	[SOS_TYPE_INT16_ARRAY] = int16_array_to_str_fn,
	[SOS_TYPE_INT32_ARRAY] = int32_array_to_str_fn,
	[SOS_TYPE_INT64_ARRAY] = int64_array_to_str_fn,
	[SOS_TYPE_UINT16_ARRAY] = uint16_array_to_str_fn,
	[SOS_TYPE_UINT32_ARRAY] = uint32_array_to_str_fn,
	[SOS_TYPE_UINT64_ARRAY] = uint64_array_to_str_fn,
	[SOS_TYPE_FLOAT_ARRAY] = float_array_to_str_fn,
	[SOS_TYPE_DOUBLE_ARRAY] = double_array_to_str_fn,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = long_double_array_to_str_fn,
	[SOS_TYPE_OBJ_ARRAY] = obj_array_to_str_fn,
};

sos_value_to_str_fn_t __sos_attr_to_str_fn_for_type(sos_type_t type)
{
	if (type >= SOS_TYPE_FIRST && type <= SOS_TYPE_LAST)
		return __attr_to_str_fn_for_type[type];
	return NULL;
}

int int16_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.int16_ = strtol(value, endptr, 0);
	return 0;
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

int uint16_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	v->data->prim.uint16_ = strtoul(value, endptr, 0);
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
	uint32_t usecs;
	char *s;
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

static int obj_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	ods_ref_t obj_ref;
	ods_ref_t part_ref;
	int cnt;
	int match = sscanf(value, "%ld@%lx%n", &part_ref, &obj_ref, &cnt);
	if (match < 2)
		return EINVAL;
	if (endptr)
		*endptr = (char *)(value + cnt);
	v->data->prim.ref_.ref.ods = part_ref;
	v->data->prim.ref_.ref.obj = obj_ref;
	return 0;
}

static int struct_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	int i, cnt, match;
	const char *str;
	char c;

	for (i = 0, str = value; i < v->attr->data->size && *str != '\0';
	     i++, str += cnt) {
		match = sscanf(str, "%02hhx%n", &c, &cnt);
		if (match < 1)
			return EINVAL;
		v->data->struc.byte_[i] = c;
	}
	if (endptr)
		*endptr = (char *)str;
	return 0;
}

static int byte_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	int i, cnt, match;
	const char *str;
	char c;

	for (i = 0, str = value; i < v->data->array.count && *str != '\0';
	     i++, str += cnt) {
		match = sscanf(str, "%hhx%n", &c, &cnt);
		if (match < 1)
			return EINVAL;
		v->data->array.data.byte_[i] = c;
		if (str[cnt] != '\0')
			cnt ++;	/* skip delimiter */
	}
	if (endptr)
		*endptr = (char *)str;
	return 0;
}

static int join_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	return byte_array_from_str_fn(v, value, endptr);
}

static int char_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	/* If there is an object associated with this value, honor
	 * that array length, otherwise, alloate memory sufficient to
	 * encode the array */
	if (!v->obj) {
		size_t sz, cnt = strlen(value);
		sz = cnt + sizeof(v->data_.array);
		if (sz > sizeof(v->data_)) {
			v->data = malloc(sz);
			if (!v->data)
				return ENOMEM;
		} else {
			v->data = &v->data_;
		}
		v->data->array.count = cnt;
	}
	strncpy(v->data->array.data.char_, value, v->data->array.count);
	if (endptr)
		*endptr = (char *)(value + strlen(value));
	return 0;
}

#define PRIMITIVE_ARRAY_FROM_STR_FN(_name_, _fmt_, _member_)		\
	static int _name_ ## _array_from_str_fn(sos_value_t v, const char *value, char **endptr) \
	{								\
		int i, cnt, match;					\
		const char *str;					\
									\
		if (!v->obj) {						\
			size_t sz, cnt;					\
			const char *p = value;				\
			char *q;					\
			const char *delim = ",:_";			\
			for (cnt = 1, q = strpbrk(p, delim);		\
			     q; q = strpbrk(p, delim)) {		\
				cnt += 1;				\
				p = q + 1;				\
			}						\
			sz = cnt * sizeof(v->data->array.data._member_[0]) \
				+ sizeof(v->data_.array);		\
			if (sz > sizeof(v->data_)) {			\
				v->data = malloc(sz);			\
				if (!v->data)				\
					return ENOMEM;			\
			} else {					\
				v->data = &v->data_;			\
			}						\
			v->data->array.count = cnt;			\
		}							\
									\
		for (i = 0, str = value; i < v->data->array.count && *str != '\0'; \
		     i++, str += cnt) {					\
			match = sscanf(str, _fmt_ "%n",			\
				       &v->data->array.data. _member_ [i], \
				       &cnt);				\
			if (match < 1) {				\
				return EINVAL;				\
			}						\
			if (str[cnt] != '\0')				\
				cnt ++;	/* skip delimiter */		\
		}							\
		if (endptr)						\
			*endptr = (char *)str;				\
		return 0;						\
	}

PRIMITIVE_ARRAY_FROM_STR_FN(int16, "%hd", int16_)
PRIMITIVE_ARRAY_FROM_STR_FN(int32, "%"PRId32"", int32_)
PRIMITIVE_ARRAY_FROM_STR_FN(int64, "%"PRId64"", int64_)

PRIMITIVE_ARRAY_FROM_STR_FN(uint16, "%hu", uint16_)
PRIMITIVE_ARRAY_FROM_STR_FN(uint32, "%"PRIu32"", uint32_)
PRIMITIVE_ARRAY_FROM_STR_FN(uint64, "%"PRIu64"", uint64_)

PRIMITIVE_ARRAY_FROM_STR_FN(float, "%f", float_)
PRIMITIVE_ARRAY_FROM_STR_FN(double, "%lf", double_)
PRIMITIVE_ARRAY_FROM_STR_FN(long_double, "%Lf", long_double_)

static int obj_array_from_str_fn(sos_value_t v, const char *value, char **endptr)
{
	int i, cnt, match;
	const char *str;
	ods_ref_t obj_ref;
	ods_ref_t part_ref;

	for (i = 0, str = value; i < v->data->array.count && *str != '\0';
	     i++, str += cnt) {
		match = sscanf(str, "%lu@%lx%n", &part_ref, &obj_ref, &cnt);
		if (match < 2)
			return EINVAL;
		v->data->array.data.ref_[i].ref.obj = obj_ref;
		v->data->array.data.ref_[i].ref.ods = part_ref;
		if (str[cnt] != '\0')
			cnt ++;	/* skip delimiter */
	}
	if (endptr)
		*endptr = (char *)str;
	return 0;
}

static sos_value_from_str_fn_t __from_str_fn_t[] = {
	[SOS_TYPE_INT16] = int16_from_str_fn,
	[SOS_TYPE_INT32] = int32_from_str_fn,
	[SOS_TYPE_INT64] = int64_from_str_fn,
	[SOS_TYPE_UINT16] = int16_from_str_fn,
	[SOS_TYPE_UINT32] = int32_from_str_fn,
	[SOS_TYPE_UINT64] = int64_from_str_fn,
	[SOS_TYPE_FLOAT] = float_from_str_fn,
	[SOS_TYPE_DOUBLE] = double_from_str_fn,
	[SOS_TYPE_LONG_DOUBLE] = long_double_from_str_fn,
	[SOS_TYPE_TIMESTAMP] = timestamp_from_str_fn,
	[SOS_TYPE_OBJ] = obj_from_str_fn,
	[SOS_TYPE_STRUCT] = struct_from_str_fn,
	[SOS_TYPE_JOIN] = join_from_str_fn,
	[SOS_TYPE_BYTE_ARRAY] = byte_array_from_str_fn,
	[SOS_TYPE_CHAR_ARRAY] = char_array_from_str_fn,
	[SOS_TYPE_INT16_ARRAY] = int16_array_from_str_fn,
	[SOS_TYPE_INT32_ARRAY] = int32_array_from_str_fn,
	[SOS_TYPE_INT64_ARRAY] = int64_array_from_str_fn,
	[SOS_TYPE_UINT16_ARRAY] = uint16_array_from_str_fn,
	[SOS_TYPE_UINT32_ARRAY] = uint32_array_from_str_fn,
	[SOS_TYPE_UINT64_ARRAY] = uint64_array_from_str_fn,
	[SOS_TYPE_FLOAT_ARRAY] = float_array_from_str_fn,
	[SOS_TYPE_DOUBLE_ARRAY] = double_array_from_str_fn,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = long_double_array_from_str_fn,
	[SOS_TYPE_OBJ_ARRAY] = obj_array_from_str_fn,
};

sos_value_from_str_fn_t __sos_attr_from_str_fn_for_type(sos_type_t type)
{
	if (type >= SOS_TYPE_FIRST && type <= SOS_TYPE_LAST)
		return __from_str_fn_t[type];
	return NULL;
}

/* Key Value Functions */

static void *int16_key_value_fn(sos_value_t val)
{
	return &val->data->prim.int16_;
}

static void *int32_key_value_fn(sos_value_t val)
{
	return &val->data->prim.int32_;
}

static void *int64_key_value_fn(sos_value_t val)
{
	return &val->data->prim.int64_;
}

static void *uint16_key_value_fn(sos_value_t val)
{
	return &val->data->prim.uint16_;
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

static void *struct_key_value_fn(sos_value_t val)
{
	return val->data->struc.byte_;
}

static void *join_key_value_fn(sos_value_t val)
{
	return val->data->join.data.byte_;
}

static void *byte_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.byte_;
}

static void *char_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.char_;
}

static void *int16_array_key_value_fn(sos_value_t val)
{
	return val->data->array.data.int16_;
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

static sos_value_key_value_fn_t __key_value_fn_t[] = {
	[SOS_TYPE_INT16] = int16_key_value_fn,
	[SOS_TYPE_INT32] = int32_key_value_fn,
	[SOS_TYPE_INT64] = int64_key_value_fn,
	[SOS_TYPE_UINT16] = uint16_key_value_fn,
	[SOS_TYPE_UINT32] = uint32_key_value_fn,
	[SOS_TYPE_UINT64] = uint64_key_value_fn,
	[SOS_TYPE_FLOAT] = float_key_value_fn,
	[SOS_TYPE_DOUBLE] = double_key_value_fn,
	[SOS_TYPE_LONG_DOUBLE] = long_double_key_value_fn,
	[SOS_TYPE_TIMESTAMP] = timestamp_key_value_fn,
	[SOS_TYPE_OBJ] = obj_key_value_fn,
	[SOS_TYPE_STRUCT] = struct_key_value_fn,
	[SOS_TYPE_JOIN] = join_key_value_fn,
	[SOS_TYPE_BYTE_ARRAY] = byte_array_key_value_fn,
	[SOS_TYPE_CHAR_ARRAY] = char_array_key_value_fn,
	[SOS_TYPE_INT16_ARRAY] = int16_array_key_value_fn,
	[SOS_TYPE_INT32_ARRAY] = int32_array_key_value_fn,
	[SOS_TYPE_INT64_ARRAY] = int64_array_key_value_fn,
	[SOS_TYPE_UINT16_ARRAY] = int16_array_key_value_fn,
	[SOS_TYPE_UINT32_ARRAY] = int32_array_key_value_fn,
	[SOS_TYPE_UINT64_ARRAY] = int64_array_key_value_fn,
	[SOS_TYPE_FLOAT_ARRAY] = float_array_key_value_fn,
	[SOS_TYPE_DOUBLE_ARRAY] = double_array_key_value_fn,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = long_double_array_key_value_fn,
	[SOS_TYPE_OBJ_ARRAY] = obj_array_key_value_fn,
};

sos_value_key_value_fn_t __sos_attr_key_value_fn_for_type(sos_type_t type)
{
	if (type >= SOS_TYPE_FIRST && type <= SOS_TYPE_LAST)
		return __key_value_fn_t[type];
	return NULL;
}
