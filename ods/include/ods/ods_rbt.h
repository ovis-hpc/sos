/*
 * Copyright (c) 2021 Open Grid Computing, Inc. All rights reserved.
 *
 * See the file COPYING at the top of this source tree for the terms
 * of the Copyright.
 */
#ifndef _ODS_RBT_T
#define _ODS_RBT_T

#include <stddef.h>
#include <inttypes.h>
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
	struct ods_rbn	*left;
	struct ods_rbn  *right;
	struct ods_rbn  *parent;
	int             color;
	void		*key;
	uint64_t	key64;
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
	uint64_t card;
};

int64_t ods_rbn_cmp(struct ods_rbt *t, struct ods_rbn *a, struct ods_rbn *b);
typedef int (*ods_rbn_printer_t)(struct ods_rbn *rbn, void *printer_arg, int level);
void ods_rbt_print(struct ods_rbt *t, ods_rbn_printer_t printer, void *printer_arg);
void ods_rbt_init(struct ods_rbt *t, ods_rbn_comparator_t cmp, void *cmp_arg);
uint64_t ods_rbt_card(struct ods_rbt *t);
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
int ods_rbt_ins_unique(struct ods_rbt *t, struct ods_rbn *x);
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
	for ((rbn) = ods_rbt_min((ods_rbt)); (rbn); (rbn) = ods_rbn_succ((rbn)))

#ifdef __cplusplus
}
#endif

#endif
