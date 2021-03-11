/*
 * Copyright (c) 2008-2015 Open Grid Computing, Inc. All rights reserved.
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
#ifndef _ODS_RBT_T
#define _ODS_RBT_T

#include <stddef.h>
#include <ods/ods_rbt.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Red/Black tree implementation. See
 * http://www.nist.gov/dads/HTML/redblack.html for a definition of
 * this algorithm.
 */

#define ODS_RBN_RED	0
#define ODS_RBN_BLACK	1

/* Red/Black Node */
struct ods_rbn {
	struct ods_rbn    *left;
	struct ods_rbn    *right;
	struct ods_rbn    *parent;
	int               color;
	void              *key;
};

/* Sets key on n.  */
void ods_rbn_init(struct ods_rbn *n, void *key);

/* Comparator callback provided for insert and search operations */
typedef int64_t (*ods_rbn_comparator_t)(void *tree_key, const void *key, void *arg);

/* Processor for each node during traversal. */
typedef int (*ods_rbn_node_fn)(struct ods_rbn *, void *, int);

struct ods_rbt {
	struct ods_rbn *root;
	ods_rbn_comparator_t comparator;
	void *comparator_arg;
	long card;
};

int64_t ods_rbn_cmp(struct ods_rbt *t, struct ods_rbn *a, struct ods_rbn *b);
typedef int (*ods_rbn_printer_t)(struct ods_rbn *rbn, void *printer_arg, int level);
void ods_rbt_print(struct ods_rbt *t, ods_rbn_printer_t printer, void *printer_arg);
void ods_rbt_init(struct ods_rbt *t, ods_rbn_comparator_t cmp, void *cmp_arg);
#define ODS_RBT_INITIALIZER(_c_) { .comparator = _c_ }
void ods_rbt_verify(struct ods_rbt *t);
int ods_rbt_empty(struct ods_rbt *t);
struct ods_rbn *ods_rbt_least_gt_or_eq(struct ods_rbn *n);
struct ods_rbn *ods_rbt_greatest_lt_or_eq(struct ods_rbn *n);
struct ods_rbn *ods_rbt_find_lub(struct ods_rbt *ods_rbt, const void *key);
struct ods_rbn *ods_rbt_find_glb(struct ods_rbt *ods_rbt, const void *key);
struct ods_rbn *ods_rbt_find(struct ods_rbt *t, const void *k);
struct ods_rbn *ods_rbt_min(struct ods_rbt *t);
struct ods_rbn *ods_rbt_max(struct ods_rbt *t);
struct ods_rbn *ods_rbn_succ(struct ods_rbn *n);
struct ods_rbn *ods_rbn_pred(struct ods_rbn *n);
void ods_rbt_ins(struct ods_rbt *t, struct ods_rbn *n);
void ods_rbt_del(struct ods_rbt *t, struct ods_rbn *n);
int ods_rbt_traverse(struct ods_rbt *t, ods_rbn_node_fn f, void *fn_data);
int ods_rbt_is_leaf(struct ods_rbn *n);
#ifndef offsetof
/* C standard since c89 */
#define offsetof(type,member) ((size_t) &((type *)0)->member)
#endif
/* from linux kernel */
#ifndef container_of
#define container_of(ptr, type, member) ({ \
	const __typeof__(((type *)0)->member ) *__mptr = (ptr); \
	(type *)((char *)__mptr - offsetof(type,member));})
#endif
#define ODS_RBT_FOREACH(rbn, ods_rbt) \
	for ((rbn) = ods_rbt_min((ods_rbt)); (rbn); (rbn) = rbn_succ((rbn)))

#ifdef __cplusplus
}
#endif

#endif
