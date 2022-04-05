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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdlib.h>
#include <strings.h>
#include <errno.h>
#include <ods/ods.h>
#include "ods_priv.h"
#include "ods_log.h"

#define ODS_OPT_VAL_SIZE 64
struct ods_opt {
	const char *name;
	int (*setter)(ods_t ods, struct ods_opt *opt, const char *name, const char *value);
	const char *(*getter)(ods_t ods, struct ods_opt *opt, const char *name);
	char value[ODS_OPT_VAL_SIZE];
};

static int __set_ods_debug(ods_t ods, struct ods_opt *opt, const char *name, const char *value)
{
	int i = strtol(value, NULL, 0);
	if (i)
		__ods_debug = 1;
	else
		__ods_debug = 0;
	return 0;
}

static const char *__get_ods_debug(ods_t ods, struct ods_opt *opt, const char *name)
{
	snprintf(opt->value, sizeof(opt->value), "%d", __ods_debug);
	return opt->value;
}

static int __set_gc_timeout_ms(ods_t ods, struct ods_opt *opt, const char *name, const char *value)
{
	int secs = strtol(value, NULL, 0);
	if (!secs)
		secs = ODS_DEF_GC_TIMEOUT;
	if (secs > 0) {
		__ods_gc_timeout = secs;
		return 0;
	}
	return EINVAL;
}

static const char *__get_gc_timeout_ms(ods_t ods, struct ods_opt *opt, const char *name)
{
	snprintf(opt->value, sizeof(opt->value), "%jd", __ods_gc_timeout);
	return opt->value;
}

struct ods_opt ods_opts[] = {
	{ "gc_timeout_ms", __set_gc_timeout_ms, __get_gc_timeout_ms },
	{ "ods_debug", __set_ods_debug, __get_ods_debug },
};

int compare_opts(const void *a, const void *b)
{
	struct ods_opt const *opt_b = b;
	return strcasecmp(a, opt_b->name);
}

int ods_opt_set(ods_t ods, const char *name, const char *value)
{
	struct ods_opt *opt;
	opt = bsearch(name, ods_opts,
		      sizeof(ods_opts) / sizeof(ods_opts[0]), sizeof(*opt),
		      compare_opts);
	if (!opt)
		return ENOENT;
	return opt->setter(ods, opt, name, value);
}
