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
	return "FLOAT_ARRAY";
}

static const char *get_doc(void)
{
	return  "FLOAT_ARRAY: The key is an array of float.\n";
}

static int64_t comparator(ods_key_t a, ods_key_t b)
{
	ods_key_value_t kva = a->as.key;
	ods_key_value_t kvb = b->as.key;
	int i;
	size_t count = kva->len;
	for (i = 0; count; i++) {
		if (kva->float_[i] < kvb->float_[i])
			return -1;
		if (kva->float_[i] > kvb->float_[i])
			return 1;
		count -= sizeof(kva->float_[i]);
	}
	return kva->len - kvb->len;
}

static const char *to_str(ods_key_t key, char *buf, size_t len)
{
	ods_key_value_t kv = ods_key_value(key);
	int i;
	char *dst;
	size_t cnt = snprintf(buf, len, "%f", kv->float_[0]);
	for (i = 1, dst = buf + cnt, len = len - cnt; len > 0; len -= cnt, i ++, dst += cnt)
		cnt = snprintf(dst, len, ",%f", kv->float_[i]);
	return buf;
}

static int from_str(ods_key_t key, const char *str)
{
	ods_key_value_t kv = ods_key_value(key);
	int i, off;
	size_t cnt = sscanf(str, " %f%n", &kv->float_[0], &off);
	for (i = 1; cnt == 1; i ++)
		cnt = sscanf(&str[off], " , %f%n", &kv->float_[i], &off);
	kv->len = sizeof(float) * i;
	return off;
}

static size_t size(void)
{
	return -1;
}

static size_t str_size(ods_key_t key)
{
	ods_key_value_t kv = ods_key_value(key);
	char buf[2];
	int i, rem = kv->len;
	size_t cnt = snprintf(buf, 0, "%f", kv->float_[0]);
	for (rem -= sizeof(float), i = 1; rem > 0; rem -= sizeof(float))
		cnt += snprintf(buf, 0, ",%f", kv->float_[i]);
	return cnt;
}

static struct ods_idx_comparator key_comparator = {
	get_type,
	get_doc,
	to_str,
	from_str,
	size,
	str_size,
	comparator
};

struct ods_idx_comparator *get(void)
{
	return &key_comparator;
}

