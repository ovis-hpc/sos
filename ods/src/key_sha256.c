/*
 * Copyright (c) 2013 Open Grid Computing, Inc. All rights reserved.
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
#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <ods/ods_idx.h>
#include "ods_idx_priv.h"

static const char *get_type(void)
{
	return "SHA256";
}

static const char *get_doc(void)
{
	return  "ODS_KEY_SHA256: The key is a 256b SHA.\n"
		"                The comparator returns -1,1,0 if a <,>,= b respectively.\n";
}

static int64_t sha256_comparator(ods_key_t a, ods_key_t b)
{
	ods_key_value_t av = ods_key_value(a);
	ods_key_value_t bv = ods_key_value(b);
	assert(av->len == 32);
	assert(bv->len == 32);
	return memcmp(av->value, bv->value, 32);
}

static const char *to_str(ods_key_t key, char *sbuf, size_t len)
{
	ods_key_value_t kv = ods_key_value(key);
	int i;
	char *s = sbuf;
	for (i = 0; i < 32; i++) {
		int cnt = snprintf(s, len, "%02X", kv->value[i]);
		s += cnt; len -= cnt;
	}
	return sbuf;
}

static uint8_t char_to_num(const char c)
{
	if (isdigit(c))
		return (c - '0');
	return (toupper(c) - 'A') + 10;
}

static int from_str(ods_key_t key, const char *str)
{
	ods_key_value_t kv = ods_key_value(key);
	int i;
	uint8_t b;
	for (i = 0; i < 32; i++) {
		b = char_to_num(str[0]) << 4;
		b |= char_to_num(str[1]);
		str += 2;
		kv->value[i] = b;
	}
	return 0;
}

static size_t size(void)
{
	return 32;
}

static size_t str_size(ods_key_t key)
{
	return 128;
}

static struct ods_idx_comparator key_comparator = {
	get_type,
	get_doc,
	to_str,
	from_str,
	size,
	str_size,
	sha256_comparator
};

struct ods_idx_comparator *get(void)
{
	return &key_comparator;
}

