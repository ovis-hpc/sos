/*
 * Copyright (c) 2018 Open Grid Computing, Inc. All rights reserved.
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
#include <string.h>
#include <ods/ods_idx.h>
#include <assert.h>
#include "../../sos/include/sos/sos.h"
#include "ods_idx_priv.h"

static const char *get_type(void)
{
	return "COMPOUND";
}

static const char *get_doc(void)
{
	return  "The key is n set of concatenated keys. Each key,\n"
		"its type and length are encoded in the key data.\n";
}

#define min(a, b) (a < b ? a : b)

static int64_t cmp(ods_key_t a, ods_key_t b)
{
	ods_comp_key_t ck_a = (ods_comp_key_t)ods_key_value(a);
	ods_comp_key_t ck_b = (ods_comp_key_t)ods_key_value(b);
	ods_key_comp_t comp_a;
	ods_key_comp_t comp_b;
	off_t koff;
	int64_t res = 0;
	int cmp_len = ck_a->len;
	if (ck_b->len < cmp_len)
		cmp_len = ck_b->len;

	for (koff = 0; koff < cmp_len;) {
		comp_a = (ods_key_comp_t)&((char *)ck_a->value)[koff];
		comp_b = (ods_key_comp_t)&((char *)ck_b->value)[koff];
#ifdef ODS_DEBUG
		assert(comp_a->type == comp_b->type);
#endif
		switch (comp_a->type) {
		case SOS_TYPE_DOUBLE:
			if (comp_a->value.double_ < comp_b->value.double_)
				return -1;
			if (comp_a->value.double_ > comp_b->value.double_)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.double_);
			break;
		case SOS_TYPE_TIMESTAMP:
			if (comp_a->value.tv_.tv_sec < comp_b->value.tv_.tv_sec)
				return -1;
			if (comp_a->value.tv_.tv_sec > comp_b->value.tv_.tv_sec)
				return 1;
			if (comp_a->value.tv_.tv_usec < comp_b->value.tv_.tv_usec)
				return -1;
			if (comp_a->value.tv_.tv_usec > comp_b->value.tv_.tv_usec)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.tv_);
			break;
		case SOS_TYPE_UINT64:
			if (comp_a->value.uint64_ < comp_b->value.uint64_)
				return -1;
			if (comp_a->value.uint64_ > comp_b->value.uint64_)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.uint64_);
			break;
		case SOS_TYPE_INT64:
			if (comp_a->value.int64_ < comp_b->value.int64_)
				return -1;
			if (comp_a->value.int64_ > comp_b->value.int64_)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.int64_);
			break;
		case SOS_TYPE_FLOAT:
			if (comp_a->value.float_ < comp_b->value.float_)
				return -1;
			if (comp_a->value.float_ > comp_b->value.float_)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.float_);
			break;
		case SOS_TYPE_UINT32:
			if (comp_a->value.uint32_ < comp_b->value.uint32_)
				return -1;
			if (comp_a->value.uint32_ > comp_b->value.uint32_)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.uint32_);
			break;
		case SOS_TYPE_INT32:
			if (comp_a->value.int32_ < comp_b->value.int32_)
				return -1;
			if (comp_a->value.int32_ > comp_b->value.int32_)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.int32_);
			break;
		case SOS_TYPE_UINT16:
			if (comp_a->value.uint16_ < comp_b->value.uint16_)
				return -1;
			if (comp_a->value.uint16_ > comp_b->value.uint16_)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.uint16_);
			break;
		case SOS_TYPE_INT16:
			if (comp_a->value.int16_ < comp_b->value.int16_)
				return -1;
			if (comp_a->value.int16_ > comp_b->value.int16_)
				return 1;
			koff += sizeof(uint16_t) + sizeof(comp_a->value.int16_);
			break;
		case SOS_TYPE_STRUCT:
		case SOS_TYPE_BYTE_ARRAY:
			res = min(comp_a->value.str.len, comp_b->value.str.len);
			res = memcmp(comp_a->value.str.str, comp_b->value.str.str, res);
			if (res == 0)
				res = comp_a->value.str.len - comp_b->value.str.len;
			if (res)
				return res;
			/* NB: if we get here we know the value and length of the two values are identical */
			koff += sizeof(uint16_t) + sizeof(comp_a->value.str.len) + comp_a->value.str.len;
			break;
		case SOS_TYPE_CHAR_ARRAY:
			res = min(comp_a->value.str.len, comp_b->value.str.len);
			res = strncmp(comp_a->value.str.str, comp_b->value.str.str, res);
			if (res == 0)
				res = comp_a->value.str.len - comp_b->value.str.len;
			if (res)
				return res;
			/* NB: if we get here we know the value and length of the two strings are identical */
			koff += sizeof(uint16_t) + sizeof(comp_a->value.str.len) + comp_a->value.str.len;
			break;
		case SOS_TYPE_LONG_DOUBLE:
		case SOS_TYPE_INT16_ARRAY:
		case SOS_TYPE_INT32_ARRAY:
		case SOS_TYPE_INT64_ARRAY:
		case SOS_TYPE_UINT16_ARRAY:
		case SOS_TYPE_UINT32_ARRAY:
		case SOS_TYPE_UINT64_ARRAY:
		case SOS_TYPE_FLOAT_ARRAY:
		case SOS_TYPE_DOUBLE_ARRAY:
		case SOS_TYPE_LONG_DOUBLE_ARRAY:
		default:
#if ODS_DEBUG
			assert(0 == "unsupported compound key component");
#endif
			res = -1;
			break;
		}
	}
	if (res == 0)
		return ck_a->len - ck_b->len;
	return res;
}

static const char *to_str(ods_key_t key, char *str, size_t len)
{
	ods_key_value_t kv = ods_key_value(key);
	int i, cnt;
	char *s = str;
	for (i = 0; i < kv->len; i++) {
		cnt = snprintf(s, len, "%02hhX", kv->value[i]);
		s += cnt; len -= cnt;
	}
	return str;
}

static int from_str(ods_key_t key, const char *str)
{
	ods_key_value_t kv = ods_key_value(key);
	size_t cnt;
	kv->len = 0;
	do {
		uint8_t b;
		cnt = sscanf(str, "%02hhX", &b);
		if (cnt > 0) {
			kv->value[kv->len] = b;
			kv->len++;
		}
		str += 2;
	} while (cnt > 0);
	return 0;
}

static size_t size(void)
{
	return -1; /* means variable length */
}

static size_t str_size(ods_key_t key)
{
	ods_key_value_t kv = key->as.ptr;
	return (kv->len * 2) + 2;
}

static struct ods_idx_comparator key_comparator = {
	get_type,
	get_doc,
	to_str,
	from_str,
	size,
	str_size,
	cmp
};

struct ods_idx_comparator *get(void)
{
	return &key_comparator;
}

