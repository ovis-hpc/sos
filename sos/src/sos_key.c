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

/**
 * \brief Create a memory key
 *
 * A key is just a an object with a set of convenience routines to
 * help with getting and setting it's value based on the key type used
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
	return sos_index_key_new(attr->index, size);
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
	return sos_index_key_from_str(attr->index, key, str);
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
	return sos_index_key_to_str(attr->index, key);
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
	return sos_index_key_cmp(attr->index, a, b);
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
	return sos_index_key_size(attr->index);
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

int __sos_key_join(sos_key_t key, sos_attr_t join_attr, int join_idx, sos_value_t value)
{
	uint64_t u64;
	uint32_t u32;
	uint16_t u16;
	ods_key_value_t kv;
	unsigned char *dst;
	int attr_id, idx;
	sos_attr_t attr;
	sos_array_t join_ids = sos_attr_join_list(join_attr);
	if (!join_ids)
		/* This is not a SOS_TYPE_JOIN attribute */
		return EINVAL;

	if (join_idx >= join_ids->count)
		/* The specified join index is invalid */
		return EINVAL;

	kv = key->as.ptr;
	kv->len = sos_attr_size(join_attr); /* join attr is a struct */
	dst = kv->value;

	for (idx = 0; idx <= join_idx; idx++) {
		attr_id = join_ids->data.uint32_[idx];
		attr = sos_schema_attr_by_id(sos_attr_schema(join_attr), attr_id);
		if (!attr)
			/* The join id in the join_attr is invalid. This is probably corruption */
			return E2BIG;
		if (idx != join_idx) {
			dst += sos_attr_key_size(attr);
			continue;
		}
		switch (sos_attr_type(attr)) {
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT64:
		case SOS_TYPE_DOUBLE:
			u64 = htobe64(value->data->prim.uint64_);
			memcpy(dst, &u64, sizeof(u64));
			break;
		case SOS_TYPE_INT32:
		case SOS_TYPE_UINT32:
		case SOS_TYPE_FLOAT:
			u32 = htobe32(value->data->prim.uint32_);
			memcpy(dst, &u32, sizeof(u32));
			break;
		case SOS_TYPE_INT16:
		case SOS_TYPE_UINT16:
			u16 = htobe16(value->data->prim.uint16_);
			memcpy(dst, &u16, sizeof(u16));
			break;
		default:
			return E2BIG;
		}
	}
	return 0;
}

/** @} */
