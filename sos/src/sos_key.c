/*
 * Copyright (c) 2014-2017 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2014-2015 Sandia Corporation. All rights reserved.
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
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <float.h>
#include <errno.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

/** \addtogroup keys
 * @{
 */

/**
 * \brief Set the value of a key
 *
 * Sets the value of the key to 'value'. The \c value parameter is of
 * type void* to make it convenient to use values of arbitrary
 * types. The minimum of 'sz' and the maximum key length is
 * copied. The number of bytes copied into the key is returned.
 *
 * \param key	The key
 * \param value	The value to set the key to
 * \param sz	The size of value in bytes
 * \returns The number of bytes copied
 */
size_t sos_key_set(sos_key_t key, void *value, size_t sz)
{
	return ods_key_set(key, value, sz);
}

void sos_key_put(sos_key_t key)
{
	ods_obj_put(key);
}

int sos_key_copy(sos_key_t dst, sos_key_t src)
{
	return sos_key_set(dst, sos_key_value(src), sos_key_len(src));
}

/**
 * \brief Create a memory key
 *
 * A key is just a an object with a set of convenience routines to
 * help with getting and setting its value based on the key type used
 * on an index.
 *
 * A memory key is used to look up objects in the ODS. The storage for
 * these keys comes from memory. See the sos_key_new() function for
 * keys that are stored in the Container.
 *
 * If the size of the key is known to be less than 254 bytes, the
 * SOS_KEY() macro is useful for defining a SOS key that is allocated
 * on the stack and is automatically destroyed when the containing
 * function returns.
 *
 * \param sz	The maximum size in bytes of the key value
 * \retval !0	ods_key_t pointer to the key
 * \retval 0	Insufficient resources
 */
sos_key_t sos_key_new(size_t sz)
{
	ods_key_t k = ods_key_malloc(sz);
	return k;
}

sos_key_t __sos_key_maybe_new(sos_key_t key, int required_size)
{
	if (key && (ods_obj_size(key) >= required_size + sizeof(struct ods_key_value_s)))
		return key;

	return sos_key_new(required_size + sizeof(struct ods_key_value_s));
}

/**
 * \brief Create a key for an attribute and values
 *
 * Create a key of sufficient size given the provided schema attribute
 * and values. If the \c key parameter is not NULL, it will be used
 * unless the key is not large enough. If this is the case, the
 * function will return NULL, and errno will be set to ETOOBIG.
 *
 * If \c key is NULL, a key of sufficient size will be allocated.
 *
 * If the attribute type is an array, the associated value provided in
 * the arguement list must be followed by the length of the array.
 * For example:
 *
 * int int_array[] = { 1, 2, 3, 4 };
 * sos_key_t key = sos_key_for_attr(NULL, array_attr, 4, int_array);
 *
 * \param key A key to use or NULL if one is to be allocated.
 * \param attr The schema attribute describing the value of the key
 * \param va_list An argument list of values
 * \retval !NULL A pointer to the key
 * \retval NULL Refer to \c errno for the reason for failure
 */
sos_key_t sos_key_for_attr(sos_key_t key, sos_attr_t attr, ...)
{
	va_list ap;
	va_start(ap, attr);
	int len;
	char *data;
	sos_key_t key_;

	switch (sos_attr_type(attr)) {
	case SOS_TYPE_JOIN:
		len = sos_key_join_size_va(attr, ap);
		va_end(ap);
		va_start(ap, attr);
		key_ = __sos_key_maybe_new(key, len);
		if (key_) {
			int rc = sos_key_join_va(key_, attr, ap);
			if (rc) {
				if (key_ != key)
					sos_key_put(key);
				errno = rc;
				key = NULL;
			} else {
				key = key_;
			}
		}
		break;
	case SOS_TYPE_UINT16:
	case SOS_TYPE_INT16:
		key = __sos_key_maybe_new(key, sizeof(uint16_t));
		if (key) {
			uint16_t u16 = va_arg(ap, unsigned int);
			sos_key_set(key, &u16, sizeof(uint16_t));
		}
		break;
	case SOS_TYPE_FLOAT:
	case SOS_TYPE_UINT32:
	case SOS_TYPE_INT32:
		key = __sos_key_maybe_new(key, sizeof(uint32_t));
		if (key) {
			uint32_t u32 = va_arg(ap, uint32_t);
			sos_key_set(key, &u32, sizeof(uint32_t));
		}
		break;
	case SOS_TYPE_TIMESTAMP:
		key = __sos_key_maybe_new(key, sizeof(union sos_timestamp_u));
		if (key) {
			union sos_timestamp_u ts = va_arg(ap, union sos_timestamp_u);
			sos_key_set(key, &ts, sizeof(ts));
		}
		break;
	case SOS_TYPE_DOUBLE:
	case SOS_TYPE_INT64:
	case SOS_TYPE_UINT64:
		key = __sos_key_maybe_new(key, sizeof(uint64_t));
		if (key) {
			uint64_t u64 = va_arg(ap, uint64_t);
			sos_key_set(key, &u64, sizeof(uint64_t));
		}
		break;
	case SOS_TYPE_LONG_DOUBLE:
		key = __sos_key_maybe_new(key, sizeof(long double));
		if (key) {
			long double ld = va_arg(ap, long double);
			sos_key_set(key, &ld, sizeof(long double));
		}
		break;
	case SOS_TYPE_STRUCT:
		key = __sos_key_maybe_new(key, sos_attr_size(attr));
		if (key) {
			data = va_arg(ap, char *);
			sos_key_set(key, data, sos_attr_size(attr));
		}
		break;
	case SOS_TYPE_BYTE_ARRAY:
	case SOS_TYPE_CHAR_ARRAY:
		data = va_arg(ap, char *);
		len = va_arg(ap, int);
		key = __sos_key_maybe_new(key, len);
		if (key)
			sos_key_set(key, data, len);
		break;
	case SOS_TYPE_UINT16_ARRAY:
	case SOS_TYPE_INT16_ARRAY:
		data = va_arg(ap, char *);
		len = va_arg(ap, int) * sizeof(uint16_t);
		key = __sos_key_maybe_new(key, len);
		if (key)
			sos_key_set(key, data, len);
		break;
	case SOS_TYPE_FLOAT_ARRAY:
	case SOS_TYPE_UINT32_ARRAY:
	case SOS_TYPE_INT32_ARRAY:
		data = va_arg(ap, char *);
		len = va_arg(ap, int) * sizeof(uint32_t);
		key = __sos_key_maybe_new(key, len);
		if (key)
			sos_key_set(key, data, len);
		break;
	case SOS_TYPE_DOUBLE_ARRAY:
	case SOS_TYPE_INT64_ARRAY:
	case SOS_TYPE_UINT64_ARRAY:
		data = va_arg(ap, char *);
		len = va_arg(ap, int) * sizeof(uint64_t);
		key = __sos_key_maybe_new(key, len);
		if (key)
			sos_key_set(key, data, len);
		break;
	case SOS_TYPE_LONG_DOUBLE_ARRAY:
		data = va_arg(ap, char *);
		len = va_arg(ap, int) * sizeof(long double);
		key = __sos_key_maybe_new(key, len);
		if (key)
			sos_key_set(key, data, len);
		break;
	case SOS_TYPE_OBJ:
	case SOS_TYPE_OBJ_ARRAY:
	default:
		errno = EINVAL;
		key = NULL;
	}
	return key;
}


/**
 * \brief Create a key based on the specified attribute or size
 *
 * Create a new SOS key based on the specified attribute. If the size
 * is specified, the attribute parameter is ignored and the key is
 * based on the specified size.
 *
 * \param attr The attribute handle
 * \param size The desired key size
 * \retval A pointer to the new key or NULL if there is an error
 */
sos_key_t sos_attr_key_new(sos_attr_t attr, size_t size)
{
	sos_index_t index = sos_attr_index(attr);
	if (!index)
		return NULL;
	return sos_index_key_new(index, size);
}


/**
 * \brief Set the value of a key from a string
 *
 * \param attr	The attribute handle
 * \param key	The key
 * \param str	Pointer to a string
 * \retval 0	if successful
 * \retval -1	if there was an error converting the string to a value
 */
int sos_attr_key_from_str(sos_attr_t attr, sos_key_t key, const char *str)
{
	sos_index_t index = sos_attr_index(attr);
	if (!index)
		return -1;
	return sos_index_key_from_str(index, key, str);
}

/**
 * \brief Return a string representation of the key value
 *
 * \param attr	The attribute handle
 * \param key	The key
 * \return A const char * representation of the key value.
 */
const char *sos_attr_key_to_str(sos_attr_t attr, sos_key_t key)
{
	sos_index_t index = sos_attr_index(attr);
	if (!index)
		return NULL;
	return sos_index_key_to_str(index, key);
}

/**
 * \brief Compare two keys using the attribute index's compare function
 *
 * \param attr	The attribute handle
 * \param a	The first key
 * \param b	The second key
 * \return <0	a < b
 * \return 0	a == b
 * \return >0	a > b
 */
int sos_attr_key_cmp(sos_attr_t attr, sos_key_t a, sos_key_t b)
{
	sos_index_t index = sos_attr_index(attr);
	if (!index)
		return 0;
	return sos_index_key_cmp(index, a, b);
}

/**
 * \brief Return the size of the attribute index's key
 *
 * Returns the native size of the attribute index's key values. If the
 * key value is variable size, this function returns -1. See the sos_key_len()
 * and sos_key_size() functions for the current size of the key's
 * value and the size of the key's buffer respectively.
 *
 * \return The native size of the attribute index's keys in bytes
 */
size_t sos_attr_key_size(sos_attr_t attr)
{
	sos_array_t join_ids;
	sos_schema_t schema;
	size_t size;
	int i;
	sos_index_t index;

	switch (sos_attr_type(attr)) {
	case SOS_TYPE_JOIN:
		schema = sos_attr_schema(attr);
		join_ids = sos_attr_join_list(attr);
		for (size = i = 0; i < join_ids->count; i++) {
			sos_attr_t ja =
				sos_schema_attr_by_id(schema, join_ids->data.uint32_[i]);
			size += sos_attr_key_size(ja);
		}
		return size;
	default:
		index = sos_attr_index(attr);
		if (index)
			return sos_index_key_size(index);
		return sos_attr_size(attr);
	}
}

/**
 * \brief Return the maximum size of this key's value
 *
 * \returns The size in bytes of this key's value buffer.
 */
size_t sos_key_size(sos_key_t key)
{
	return ods_key_size(key);
}

/**
 * \brief Return the length of the key's value
 *
 * Returns the current size of the key's value.
 *
 * \param key	The key
 * \returns The size of the key in bytes
 */
size_t sos_key_len(sos_key_t key)
{
	ods_key_value_t kv = ods_key_value(key);
	return kv->len;
}

/**
 * \brief Return the value of a key
 *
 * \param key	The key
 * \returns Pointer to the value of the key
 */
unsigned char* sos_key_value(sos_key_t key)
{
	ods_key_value_t kv = ods_key_value(key);
	return kv->value;
}

/**
 * \brief Return the value of a key as a character string
 *
 * Format a key based on the provided <tt>fmt</tt> string. See the man
 * page for the printf() function for the format of this string. The
 * <tt>el_sz</tt> parameter indicates the size of each component of
 * the key. The <tt>sep</tt> parameter is placed between each
 * component in the formatted output.
 *
 * For example, to format the key as hex bytes:
 *
 *     char *hex_str = sos_key_to_str(key, "%02X", ":", 1);
 *     printf("%s\n", hex_str);
 *
 * The returned string should be freed with the free() function when
 * bo long needed.
 *
 * \param key The key handle
 * \retval The string
 */
char *sos_key_to_str(sos_key_t key, const char *fmt, const char *sep, size_t el_sz)
{
	ods_key_value_t kv = ods_key_value(key);
	size_t cnt = kv->len / el_sz;
	size_t res_cnt = 0;
	char *res_str, *str;
	size_t alloc_cnt;
	size_t sep_len = strlen(sep);
	int i;

	/* Get the size of one element. */
	res_cnt = snprintf(NULL, 0, fmt, kv->value);

	/*
	 * NB: this calculation assumes that all elements have the same
	 * formatted size 8-P
	 */
	alloc_cnt = (res_cnt * cnt) + (sep_len * cnt) + 2;
	res_str = malloc(alloc_cnt);
	if (!res_str)
		return NULL;

	unsigned char *p = kv->value;
	for (str = res_str, i = 0; i < cnt; i++) {
		if (i) {
			strcat(str, sep);
			str += sep_len;
			alloc_cnt -= sep_len;
		}
		switch (el_sz) {
		case 2:
			res_cnt = snprintf(str, alloc_cnt, fmt, *(short *)p);
			break;
		case 4:
			res_cnt = snprintf(str, alloc_cnt, fmt, *(uint32_t *)p);
			break;
		case 8:
			res_cnt = snprintf(str, alloc_cnt, fmt, *(uint64_t *)p);
			break;
		case 1:
		default:
			res_cnt = snprintf(str, alloc_cnt, fmt, *p);
		}
		if (res_cnt > alloc_cnt)
			/*  Ran out of memory. Return our best effort */
			break;
		str += res_cnt;
		alloc_cnt -= res_cnt;
		p += el_sz;
	}
	return res_str;
}

static uint64_t comp_type_size[] = {
	/* Primitive type sizes include the uint16_t type field. */
	[SOS_TYPE_LONG_DOUBLE] = sizeof(long double) + sizeof(uint16_t),
	[SOS_TYPE_DOUBLE] = sizeof(double) + sizeof(uint16_t),
	[SOS_TYPE_FLOAT] = sizeof(float) + sizeof(uint16_t),
	[SOS_TYPE_UINT64] = sizeof(uint64_t) + sizeof(uint16_t),
	[SOS_TYPE_TIMESTAMP] = sizeof(union sos_timestamp_u) + sizeof(uint16_t),
	[SOS_TYPE_INT64] = sizeof(int64_t) + sizeof(uint16_t),
	[SOS_TYPE_UINT32] = sizeof(uint32_t) + sizeof(uint16_t),
	[SOS_TYPE_INT32] = sizeof(int32_t) + sizeof(uint16_t),
	[SOS_TYPE_UINT16] = sizeof(uint16_t) + sizeof(uint16_t),
	[SOS_TYPE_INT16] = sizeof(int16_t) + sizeof(uint16_t),

	/* Array types do not include the uint16_t type field because
	 * this value is multipied by the array length. */
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = sizeof(long double),
	[SOS_TYPE_DOUBLE_ARRAY] = sizeof(double),
	[SOS_TYPE_FLOAT_ARRAY] = sizeof(float),
	[SOS_TYPE_UINT64_ARRAY] = sizeof(uint64_t),
	[SOS_TYPE_INT64_ARRAY] = sizeof(int64_t),
	[SOS_TYPE_UINT32_ARRAY] = sizeof(uint32_t),
	[SOS_TYPE_INT32_ARRAY] = sizeof(int32_t),
	[SOS_TYPE_UINT16_ARRAY] = sizeof(uint16_t),
	[SOS_TYPE_INT16_ARRAY] = sizeof(int16_t),
	[SOS_TYPE_CHAR_ARRAY] = sizeof(char),
	[SOS_TYPE_BYTE_ARRAY] = sizeof(unsigned char),
};

ods_key_comp_t __sos_next_key_comp(ods_key_comp_t comp)
{
	off_t koff;
	if (comp->type <= SOS_TYPE_TIMESTAMP) {
		koff = comp_type_size[comp->type];
	} else {
		koff = comp->value.str.len + sizeof(uint16_t) + sizeof(uint16_t);
	}
	assert(koff != 0);
	return (ods_key_comp_t)&((char *)comp)[koff];
}

int __sos_value_is_min(sos_value_t v)
{
	switch (sos_value_type(v)) {
	case SOS_TYPE_TIMESTAMP:
		return (v->data->prim.timestamp_.tv.tv_sec == 0
			&&
			v->data->prim.timestamp_.tv.tv_usec == 0);
	case SOS_TYPE_UINT64:
		return (v->data->prim.uint64_ == 0);
	case SOS_TYPE_INT64:
		return (v->data->prim.int64_ == LONG_MIN);
	case SOS_TYPE_UINT32:
		return (v->data->prim.uint32_ == 0);
	case SOS_TYPE_INT32:
		return (v->data->prim.int32_ == INT_MIN);
	case SOS_TYPE_DOUBLE:
		return (v->data->prim.double_ == DBL_MIN);
	case SOS_TYPE_FLOAT:
		return (v->data->prim.float_ == FLT_MIN);
	case SOS_TYPE_INT16:
		return (v->data->prim.int16_ == SHRT_MIN);
	case SOS_TYPE_UINT16:
		return (v->data->prim.uint16_ == 0);
	case SOS_TYPE_LONG_DOUBLE:
		sos_error("Unsupported type in sos_key_join\n");
		break;
	default:
		assert(0 == "unsupported type for function");
		break;
	}
	return 0;
}

ods_key_comp_t __sos_set_key_comp_to_min(ods_key_comp_t comp, sos_attr_t a, size_t *comp_len)
{
	comp->type = sos_attr_type(a);
	switch (comp->type) {
	case SOS_TYPE_TIMESTAMP:
		comp->value.tv_.tv_sec = 0;
		comp->value.tv_.tv_usec = 0;
		*comp_len = sizeof(comp->value.tv_) + sizeof(uint16_t);
		break;
	case SOS_TYPE_UINT64:
		comp->value.uint64_ = 0;
		*comp_len = sizeof(uint64_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_INT64:
		comp->value.int64_ = LONG_MIN;
		*comp_len = sizeof(int64_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_UINT32:
		comp->value.uint32_ = 0;
		*comp_len = sizeof(uint32_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_INT32:
		comp->value.int32_ = INT_MIN;
		*comp_len = sizeof(int32_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_DOUBLE:
		comp->value.double_ = DBL_MIN;
		*comp_len = sizeof(double) + sizeof(uint16_t);
		break;
	case SOS_TYPE_FLOAT:
		comp->value.float_ = FLT_MIN;
		*comp_len = sizeof(float) + sizeof(uint16_t);
		break;
	case SOS_TYPE_INT16:
		comp->value.int16_ = SHRT_MIN;
		*comp_len = sizeof(int16_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_UINT16:
		comp->value.uint16_ = 0;
		*comp_len = sizeof(uint16_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_LONG_DOUBLE:
		sos_error("Unsupported type in sos_key_join\n");
		break;
	default:
		assert(0 == "unsupported type for function");
		break;
	}
	return __sos_next_key_comp(comp);
}

int __sos_value_is_max(sos_value_t v)
{
	switch (sos_value_type(v)) {
	case SOS_TYPE_TIMESTAMP:
		return (v->data->prim.timestamp_.tv.tv_sec == UINT_MAX
			&&
			v->data->prim.timestamp_.tv.tv_usec == UINT_MAX);
	case SOS_TYPE_UINT64:
		return (v->data->prim.uint64_ == ULONG_MAX);
	case SOS_TYPE_INT64:
		return (v->data->prim.int64_ == LONG_MAX);
	case SOS_TYPE_UINT32:
		return (v->data->prim.uint32_ == UINT_MAX);
	case SOS_TYPE_INT32:
		return (v->data->prim.int32_ == INT_MAX);
	case SOS_TYPE_DOUBLE:
		return (v->data->prim.double_ == DBL_MAX);
	case SOS_TYPE_FLOAT:
		return (v->data->prim.float_ == FLT_MAX);
	case SOS_TYPE_INT16:
		return (v->data->prim.int16_ == SHRT_MAX);
	case SOS_TYPE_UINT16:
		return (v->data->prim.uint16_ == USHRT_MAX);
	case SOS_TYPE_LONG_DOUBLE:
		sos_error("Unsupported type in sos_key_join\n");
		break;
	default:
		assert(0 == "unsupported type for function");
		break;
	}
	return 0;
}

ods_key_comp_t __sos_set_key_comp_to_max(ods_key_comp_t comp, sos_attr_t a, size_t *comp_len)
{
	comp->type = sos_attr_type(a);
	switch (comp->type) {
	case SOS_TYPE_TIMESTAMP:
		comp->value.tv_.tv_sec = UINT_MAX;
		comp->value.tv_.tv_usec = UINT_MAX;
		*comp_len = sizeof(comp->value.tv_) + sizeof(uint16_t);
		break;
	case SOS_TYPE_UINT64:
		comp->value.uint64_ = ULONG_MAX;
		*comp_len = sizeof(uint64_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_INT64:
		comp->value.int64_ = LONG_MAX;
		*comp_len = sizeof(int64_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_UINT32:
		comp->value.uint32_ = UINT_MAX;
		*comp_len = sizeof(uint32_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_INT32:
		comp->value.int32_ = INT_MAX;
		*comp_len = sizeof(int32_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_DOUBLE:
		comp->value.double_ = DBL_MAX;
		*comp_len = sizeof(double) + sizeof(uint16_t);
		break;
	case SOS_TYPE_FLOAT:
		comp->value.float_ = FLT_MAX;
		*comp_len = sizeof(float) + sizeof(uint16_t);
		break;
	case SOS_TYPE_INT16:
		comp->value.int16_ = SHRT_MAX;
		*comp_len = sizeof(int16_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_UINT16:
		comp->value.uint16_ = USHRT_MAX;
		*comp_len = sizeof(uint16_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_LONG_DOUBLE:
		sos_error("Unsupported type in sos_key_join\n");
		break;
	default:
		assert(0 == "unsupported type for function");
		break;
	}
	return __sos_next_key_comp(comp);
}

ods_key_comp_t __sos_set_key_comp(ods_key_comp_t comp, sos_value_t v, size_t *comp_len)
{
	size_t sz;

	comp->type = sos_value_type(v);
	switch (comp->type) {
	case SOS_TYPE_TIMESTAMP:
	case SOS_TYPE_UINT64:
	case SOS_TYPE_INT64:
	case SOS_TYPE_DOUBLE:
		comp->value.uint64_ = v->data->prim.uint64_;
		*comp_len = sizeof(uint64_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_INT32:
	case SOS_TYPE_UINT32:
	case SOS_TYPE_FLOAT:
		comp->value.uint32_ = v->data->prim.uint32_;
		*comp_len = sizeof(uint32_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_INT16:
	case SOS_TYPE_UINT16:
		comp->value.uint16_ = v->data->prim.uint16_;
		*comp_len = sizeof(uint16_t) + sizeof(uint16_t);
		break;
	case SOS_TYPE_LONG_DOUBLE:
		sos_error("Unsupported type in sos_key_join\n");
		break;
	case SOS_TYPE_STRUCT:
		sz = sos_value_size(v);
		memcpy(comp->value.str.str, v->data->prim.struc_, sz);
		comp->value.str.len = sz;
		*comp_len = sz + sizeof(comp->value.str.len) + sizeof(uint16_t);
		break;
	default:
		sz = sos_value_size(v);
		memcpy(comp->value.str.str, sos_array_data(v, char_), sz);
		comp->value.str.len = sz;
		*comp_len = sz + sizeof(comp->value.str.len) + sizeof(uint16_t);
		break;
	}
	return __sos_next_key_comp(comp);
}

/**
 * \brief Builds a join key from the attributes provided va_list
 *
 * This function populates the \c key with the formatted values in the
 * argument list. The \c join_attr is treated like a format
 * specificiation. Each attribute type in the join attribute list is
 * consulted to interpret the arguments following \c join_attr. Array
 * attribute types have two values in the call list, the first is a
 * size_t length, and the second is a pointer to the array containing
 * the values.
 *
 * The key must be of sufficient size to contain the formatted key
 * value. See the sos_key_join_size() function for determining the
 * size of a join key given a join attribute and a set of attribute
 * values.
 *
 * \param key The key handle.
 * \param join_attr The attribute handle.
 * \param ... The attribute values.
 *
 * \returns 0 on success or an errno on error.
 */
int sos_key_join(sos_key_t key, sos_attr_t join_attr, ...)
{
	int rc;
	va_list ap;
	va_start(ap, join_attr);
	rc = sos_key_join_va(key, join_attr, ap);
	va_end(ap);
	return rc;
}

/**
 * \brief Builds a join key from the attributes in the argument list
 *
 * See sos_key_join() for more information.
 *
 * \param key The key handle.
 * \param join_attr The join attribute handle.
 * \param ap The va_list argument pointer.
 * \returns 0 on success or an errno on error.
 */
int sos_key_join_va(sos_key_t key, sos_attr_t join_attr, va_list ap)
{
	unsigned char *src;
	ods_comp_key_t comp_key = (ods_comp_key_t)ods_key_value(key);
	union sos_timestamp_u ts;
	ods_key_comp_t comp;
	size_t src_len;
	int attr_id, idx;
	sos_attr_t attr;
	sos_array_t join_ids = sos_attr_join_list(join_attr);
	if (!join_ids)
		/* This is not a SOS_TYPE_JOIN attribute */
		return EINVAL;

	comp = comp_key->value;
	comp_key->len = 0;

	for (idx = 0; idx < join_ids->count; idx++) {
		attr_id = join_ids->data.uint32_[idx];
		attr = sos_schema_attr_by_id(sos_attr_schema(join_attr),
					     attr_id);
		if (!attr) {
			sos_error("Join attr_id %d in attribute %s does "
				  "not exist.\n", attr_id,
				  sos_attr_name(join_attr));
			return E2BIG;
		}
		comp->type = sos_attr_type(attr);
		switch (comp->type) {
		case SOS_TYPE_TIMESTAMP:
			ts = va_arg(ap, union sos_timestamp_u);
			comp->value.tv_.tv_sec = ts.tv.tv_sec;
			comp->value.tv_.tv_usec = ts.tv.tv_usec;
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT64:
		case SOS_TYPE_DOUBLE:
			comp->value.uint64_ = va_arg(ap, uint64_t);
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_INT32:
		case SOS_TYPE_UINT32:
		case SOS_TYPE_FLOAT:
			comp->value.uint32_ = va_arg(ap, uint32_t);
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_INT16:
		case SOS_TYPE_UINT16:
			comp->value.uint16_ = (short)va_arg(ap, int);
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_LONG_DOUBLE:
			comp->value.long_double_ = va_arg(ap, long double);
			comp_key->len += comp_type_size[comp->type];
			break;
		default:
			src_len = va_arg(ap, size_t) * comp_type_size[comp->type];
			src = va_arg(ap, unsigned char *);
			memcpy(comp->value.str.str, src, src_len);
			comp->value.str.len = src_len;
			comp_key->len += src_len + sizeof(uint16_t) + sizeof(uint16_t);
			break;
		}
		comp = __sos_next_key_comp(comp);
	}
	assert(comp_key->len == ((unsigned long)comp - (unsigned long)comp_key->value));
	return 0;
}

int sos_key_join_size(sos_attr_t join_attr, ...)
{
	va_list ap;
	int rc;
	va_start(ap, join_attr);
	rc = sos_key_join_size_va(join_attr, ap);
	va_end(ap);
	return rc;
}

int sos_key_join_size_va(sos_attr_t join_attr, va_list ap)
{
	int size = 0;
	size_t src_len;
	sos_type_t type;
	int attr_id, idx;
	sos_attr_t attr;
	sos_array_t join_ids = sos_attr_join_list(join_attr);
	if (!join_ids)
		/* This is not a SOS_TYPE_JOIN attribute */
		return -EINVAL;

	size = 0;
	for (idx = 0; idx < join_ids->count; idx++) {
		attr_id = join_ids->data.uint32_[idx];
		attr = sos_schema_attr_by_id(sos_attr_schema(join_attr), attr_id);
		if (!attr) {
			sos_error("Join attr_id %d in attribute %s does "
				  "not exist.\n", attr_id,
				  sos_attr_name(join_attr));
			return -EINVAL;
		}
		type = sos_attr_type(attr);
		switch (type) {
		case SOS_TYPE_TIMESTAMP:
			(void)va_arg(ap, union sos_timestamp_u);
			size += comp_type_size[type];
			break;
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT64:
		case SOS_TYPE_DOUBLE:
			(void)va_arg(ap, uint64_t);
			size += comp_type_size[type];
			break;
		case SOS_TYPE_INT32:
		case SOS_TYPE_UINT32:
		case SOS_TYPE_FLOAT:
			(void)va_arg(ap, uint32_t);
			size += comp_type_size[type];
			break;
		case SOS_TYPE_INT16:
		case SOS_TYPE_UINT16:
			(void)va_arg(ap, int);
			size += comp_type_size[type];
			break;
		case SOS_TYPE_LONG_DOUBLE:
			(void)va_arg(ap, long double);
			size += comp_type_size[type];
			break;
		case SOS_TYPE_STRUCT:
			(void)va_arg(ap, unsigned char *);
			src_len = sos_attr_size(attr);
			size += src_len + sizeof(uint16_t) + sizeof(uint16_t);
			break;
		default:
			src_len = va_arg(ap, size_t) * comp_type_size[type];
			(void)va_arg(ap, unsigned char *);
			size += src_len + sizeof(uint16_t) + sizeof(uint16_t);
			break;
		case SOS_TYPE_OBJ:
		case SOS_TYPE_JOIN:
		case SOS_TYPE_OBJ_ARRAY:
			assert(0 == "Invalid type in JOIN Key.");
		}
	}
	return size;
}

int sos_key_split(sos_key_t key, sos_attr_t join_attr, ...)
{
	va_list ap;
	union sos_timestamp_u *ts;
	uint64_t *p64;
	uint32_t *p32;
	uint16_t *p16;
	long double *pld;
	unsigned char *dst;
	ods_comp_key_t comp_key = (ods_comp_key_t)ods_key_value(key);
	ods_key_comp_t comp;
	int attr_id, idx;
	sos_attr_t attr;
	sos_array_t join_ids = sos_attr_join_list(join_attr);
	if (!join_ids)
		/* This is not a SOS_TYPE_JOIN attribute */
		return EINVAL;

	va_start(ap, join_attr);

	comp = comp_key->value;

	for (idx = 0; idx < join_ids->count; idx++) {
		attr_id = join_ids->data.uint32_[idx];
		attr = sos_schema_attr_by_id(sos_attr_schema(join_attr),
					     attr_id);
		if (!attr) {
			sos_error("Join attr_id %d in attribute %s does "
				  "not exist.\n", attr_id,
				  sos_attr_name(join_attr));
			return E2BIG;
		}
		switch (sos_attr_type(attr)) {
		case SOS_TYPE_TIMESTAMP:
			ts = va_arg(ap, union sos_timestamp_u *);
			ts->tv.tv_sec = comp->value.tv_.tv_sec;
			ts->tv.tv_usec = comp->value.tv_.tv_usec;
			break;
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT64:
		case SOS_TYPE_DOUBLE:
			p64 = va_arg(ap, uint64_t *);
			*p64 = comp->value.uint64_;
			break;
		case SOS_TYPE_INT32:
		case SOS_TYPE_UINT32:
		case SOS_TYPE_FLOAT:
			p32 = va_arg(ap, uint32_t *);
			*p32 = comp->value.uint32_;
			break;
		case SOS_TYPE_INT16:
		case SOS_TYPE_UINT16:
			p16 = va_arg(ap, uint16_t *);
			*p16 = comp->value.uint16_;
			break;
		case SOS_TYPE_LONG_DOUBLE:
			pld = va_arg(ap, long double *);
			*pld = comp->value.long_double_;
			break;
		default:
			dst = va_arg(ap, unsigned char *);
			memcpy(dst, comp->value.str.str, comp->value.str.len);
			break;
		}
		comp = __sos_next_key_comp(comp);
	}
	va_end(ap);
	return 0;
}

/**
 * \brief Set the value of a component key
 *
 * Assign the value of a component key from an array of
 * sos_key_comp_spec_t. The caller can either specify the key as input
 * (in which case it must be of sufficient size to contain the
 * resulting key), or NULL, in which case a key will be allocated.
 *
 * Example:
 *
 *    SOS_VALUE(a);
 *    SOS_VALUE(b);
 *    struct sos_key_comp_spec spec[] = {
 *       { .type = SOS_KEY_UINT32, .value = a ),
 *       { .type = SOS_KEY_UINT32, .value = b )
 *    };
 *    SOS_KEY(the_key);
 *    a->data.prim.uint32_ = 1234;
 *    a->data.prim.uint32_ = 4568;;
 *
 *    rc = sos_comp_key_set(the_key, 2, spec);
 *
 * \param key	An optional key to contain the value. If NULL, a key will be allocated.
 * \param len	The number of elements in the key_spec array.
 * \param key_spec An array of key component specifications
 * \returns 0 Success
 * \returns EINVAL An invalid component value was specified.
 */
int sos_comp_key_set(sos_key_t key, size_t len, sos_comp_key_spec_t key_spec)
{
	ods_comp_key_t comp_key = (ods_comp_key_t)ods_key_value(key);
	ods_key_comp_t comp;
	sos_comp_key_spec_t spec;
	int idx;

	comp = comp_key->value;
	comp_key->len = 0;

	for (idx = 0; idx < len; idx++) {
		spec = &key_spec[idx];
		comp->type = spec->type;
		switch (comp->type) {
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT64:
		case SOS_TYPE_DOUBLE:
			comp->value.uint64_ = spec->data->prim.uint64_;
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_TIMESTAMP:
			comp->value.tv_.tv_sec = spec->data->prim.timestamp_.tv.tv_sec;
			comp->value.tv_.tv_usec = spec->data->prim.timestamp_.tv.tv_usec;
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_INT32:
		case SOS_TYPE_UINT32:
		case SOS_TYPE_FLOAT:
			comp->value.uint32_ = spec->data->prim.uint32_;
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_INT16:
		case SOS_TYPE_UINT16:
			comp->value.uint16_ = spec->data->prim.uint16_;
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_LONG_DOUBLE:
			comp->value.long_double_ = spec->data->prim.long_double_;
			comp_key->len += comp_type_size[comp->type];
			break;
		case SOS_TYPE_BYTE_ARRAY:
		case SOS_TYPE_CHAR_ARRAY:
		case SOS_TYPE_INT16_ARRAY:
		case SOS_TYPE_INT32_ARRAY:
		case SOS_TYPE_INT64_ARRAY:
		case SOS_TYPE_UINT16_ARRAY:
		case SOS_TYPE_UINT32_ARRAY:
		case SOS_TYPE_UINT64_ARRAY:
		case SOS_TYPE_FLOAT_ARRAY:
		case SOS_TYPE_DOUBLE_ARRAY:
		case SOS_TYPE_LONG_DOUBLE_ARRAY:
			memcpy(comp->value.str.str, spec->data->array.data.byte_,
			       spec->data->array.count * comp_type_size[comp->type]);
			comp->value.str.len = spec->data->array.count;
			comp_key->len +=
				(comp->value.str.len  * comp_type_size[comp->type])
				+ sizeof(uint16_t) + sizeof(uint16_t);
			break;
		default:
			return EINVAL;
		}
		comp = __sos_next_key_comp(comp);
	}
	assert(comp_key->len == ((unsigned long)comp - (unsigned long)comp_key->value));
	return 0;
}

size_t sos_comp_key_size(size_t len, sos_comp_key_spec_t key_spec)
{
	sos_comp_key_spec_t spec;
	int idx;
	size_t key_len = 0;

	for (idx = 0; idx < len; idx++) {
		spec = &key_spec[idx];
		if (spec->type < SOS_TYPE_BYTE_ARRAY) {
			key_len += comp_type_size[spec->type];
		} else {
			key_len +=
				(spec->data->array.count * comp_type_size[spec->type])
				+ sizeof(uint16_t) + sizeof(uint16_t);
		}
	}
	return key_len;
}

/**
 * \brief Split a key into its component parts.
 *
 * This function splits the input key into its component parts and
 * returns the result in the \c key_spec array parameter. The length
 * of this array is returned in the \c len parameter. On input \c len
 * is the number of elements in the \c key_spec parameter array. If
 * the array passed is too small to contain the result, ETOOBIG is
 * returned, and \c len is set to the required size of the \c key_spec
 * array.
 *
 * \param key The key to decompose
 * \param len Pointer to a size_t to receive the number of entries consumed in key_spec
 * \param key_spec An array of *len length to receive the result.
 * \retval 0 Success
 * \retval !0 An errno reason
 */
sos_comp_key_spec_t sos_comp_key_get(sos_key_t key, size_t *len)
{
	sos_comp_key_spec_t key_values;
	ods_comp_key_t comp_key = (ods_comp_key_t)ods_key_value(key);
	ods_key_comp_t comp;
	sos_comp_key_spec_t spec;
	int offset, idx;

	comp = comp_key->value;
	for (offset = 0, idx = 0; offset < comp_key->len; idx++) {
		comp = __sos_next_key_comp(comp);
		offset = (((unsigned long)comp - (unsigned long)comp_key->value));
	}
	*len = idx;

	key_values = malloc(idx * sizeof(struct sos_comp_key_spec));
	if (!key_values)
		goto out;

	comp = comp_key->value;
	for (offset = 0, idx = 0; offset < comp_key->len; idx++) {
		spec = &key_values[idx];
		spec->type = comp->type;
		switch (comp->type) {
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT64:
		case SOS_TYPE_DOUBLE:
			spec->data = sos_value_data_new(spec->type, 0);
			spec->data->prim.uint64_ = comp->value.uint64_;
			break;
		case SOS_TYPE_INT32:
		case SOS_TYPE_UINT32:
		case SOS_TYPE_FLOAT:
			spec->data = sos_value_data_new(spec->type, 0);
			spec->data->prim.uint32_ = comp->value.uint32_;
			break;
		case SOS_TYPE_INT16:
		case SOS_TYPE_UINT16:
			spec->data = sos_value_data_new(spec->type, 0);
			spec->data->prim.uint16_ = comp->value.uint16_;
			break;
		case SOS_TYPE_LONG_DOUBLE:
			spec->data = sos_value_data_new(spec->type, 0);
			spec->data->prim.long_double_ = comp->value.long_double_;
			break;
		case SOS_TYPE_TIMESTAMP:
			spec->data = sos_value_data_new(spec->type, 0);
			spec->data->prim.timestamp_.tv = comp->value.tv_;
			break;
		default:
			spec->data = sos_value_data_new(spec->type, comp->value.str.len);
			memcpy(spec->data->array.data.byte_,
			       comp->value.str.str, comp->value.str.len);
			spec->data->array.count =
				comp->value.str.len / comp_type_size[comp->type];
			break;
		}
		comp = __sos_next_key_comp(comp);
		offset = (((unsigned long)comp - (unsigned long)comp_key->value));
	}
	*len = idx;
 out:
	return key_values;
}


/** @} */
