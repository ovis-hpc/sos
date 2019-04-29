#ifndef _SOS_YAML_H
#define _SOS_YAML_H

/*
 * Copyright (c) 2014-2019 Open Grid Computing, Inc. All rights reserved.
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
enum keyword_e {
	FALSE_KW = 0,
	TRUE_KW = 1,
	ATTRIBUTE_KW = 100,
	INDEXED_KW,
	SCHEMA_KW,
	NAME_KW,
	TYPE_KW,
	COUNT_KW,
	TIME_FUNC_KW,
	RANDOM_FUNC_KW,
};

struct keyword {
	char *str;
	enum keyword_e id;
};

/* This table must be sorted */
struct keyword keyword_table[] = {
	{ "ATTRIBUTE", ATTRIBUTE_KW },
	{ "BYTE_ARRAY", SOS_TYPE_BYTE_ARRAY },
	{ "CHAR_ARRAY", SOS_TYPE_CHAR_ARRAY },
	{ "DOUBLE", SOS_TYPE_DOUBLE },
	{ "DOUBLE_ARRAY", SOS_TYPE_DOUBLE_ARRAY },
	{ "FALSE", FALSE_KW },
	{ "FLOAT", SOS_TYPE_FLOAT },
	{ "FLOAT_ARRAY", SOS_TYPE_FLOAT_ARRAY },
	{ "INDEXED", INDEXED_KW },
	{ "INT16", SOS_TYPE_INT16 },
	{ "INT16_ARRAY", SOS_TYPE_INT16_ARRAY },
	{ "INT32", SOS_TYPE_INT32 },
	{ "INT32_ARRAY", SOS_TYPE_INT32_ARRAY },
	{ "INT64", SOS_TYPE_INT64 },
	{ "INT64_ARRAY", SOS_TYPE_INT64_ARRAY },
	{ "LONG_DOUBLE", SOS_TYPE_LONG_DOUBLE },
	{ "LONG_DOUBLE_ARRAY", SOS_TYPE_LONG_DOUBLE_ARRAY },
	{ "NAME", NAME_KW },
	{ "OBJ", SOS_TYPE_OBJ },
	{ "OBJ_ARRAY", SOS_TYPE_OBJ_ARRAY },
	{ "RANDOM()", RANDOM_FUNC_KW },
	{ "SCHEMA", SCHEMA_KW },
	{ "STRUCT", SOS_TYPE_STRUCT },
	{ "TIME()", TIME_FUNC_KW },
	{ "TIMESTAMP", SOS_TYPE_TIMESTAMP },
	{ "TRUE", TRUE_KW },
	{ "TYPE", TYPE_KW },
	{ "UINT16", SOS_TYPE_UINT16 },
	{ "UINT16_ARRAY", SOS_TYPE_UINT16_ARRAY },
	{ "UINT32", SOS_TYPE_UINT32 },
	{ "UINT32_ARRAY", SOS_TYPE_UINT32_ARRAY },
	{ "UINT64", SOS_TYPE_UINT64 },
	{ "UINT64_ARRAY", SOS_TYPE_UINT64_ARRAY },
};

int compare_keywords(const void *a, const void *b)
{
	struct keyword *kw_a = (struct keyword *)a;
	struct keyword *kw_b = (struct keyword *)b;
	char *str = strdup(kw_a->str);
	str = strtok(str, ",");
	int rc = strcasecmp(str, kw_b->str);
	free(str);
	return rc;
}

#endif
