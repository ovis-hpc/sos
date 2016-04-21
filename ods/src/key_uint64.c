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
#include <ods/ods_idx.h>
#include "ods_idx_priv.h"

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

static const char *get_type(void)
{
	return "UINT64";
}

static const char *get_doc(void)
{
	return  "ODS_KEY_UINT64: The key is an unsigned 64b long.\n"
		"                The comparator returns -1,1,0 if a <,>,= b respectively.\n";
}

static int uint64_comparator(ods_key_t a, ods_key_t b)
{
	ods_key_value_t av = ods_key_value(a);
	ods_key_value_t bv = ods_key_value(b);
	uint64_t aa = *(uint64_t*)av->value;
	uint64_t bb = *(uint64_t*)bv->value;
	assert(av->len == 8);
	assert(bv->len == 8);
	if (aa < bb)
		return -1;
	if (aa > bb)
		return 1;

	return 0;
}

static const char *to_str(ods_key_t key, char *buf, size_t len)
{
	ods_key_value_t kv = ods_key_value(key);
	snprintf(buf, len, "0x%"PRIx64"", *(uint64_t *)kv->value);
	return buf;
}

static int from_str(ods_key_t key, const char *str)
{
	ods_key_value_t kv = ods_key_value(key);
	uint64_t v;
	errno = 0;
	v = strtoull(str, NULL, 0);
	if (errno)
		return -1;
	memcpy(kv->value, &v, 8);
	kv->len = 8;
	return 0;
}

static size_t size(void)
{
	return sizeof(uint64_t);
}

static size_t str_size(ods_key_t key)
{
	return 32;
}

static struct ods_idx_comparator key_comparator = {
	get_type,
	get_doc,
	to_str,
	from_str,
	size,
	str_size,
	uint64_comparator
};

struct ods_idx_comparator *get(void)
{
	return &key_comparator;
}

