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
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <string.h>
#include <assert.h>
#include <ods/ods.h>
#include "bpt.h"

static int search_leaf_reverse(bpt_t t, ods_obj_t leaf, ods_key_t key);
static int search_leaf(bpt_t t, ods_obj_t leaf, ods_key_t key);
static ods_obj_t leaf_left(bpt_t t, uint64_t node_ref);
static ods_obj_t leaf_right(bpt_t t, uint64_t node_ref);
static int node_neigh(bpt_t t, ods_obj_t node, ods_obj_t *left, ods_obj_t *right);

static void print_node(ods_idx_t idx, int ent, ods_obj_t n, int indent, FILE *fp)
{
	bpt_t t = idx->priv;
	int i;

	if (!n) {
		fprintf(fp, "<nil>\n");
		return;
	}

	/* Print this node */
	if (NODE(n)->is_leaf && NODE(n)->parent)
		indent += 4;
	fprintf(fp, "%p - %*s%s[%d] | %p : ", (void *)(unsigned long)NODE(n)->parent,
		indent, "",
		(NODE(n)->is_leaf?"LEAF":"NODE"),
		ent, n->as.ptr);
	for (i = 0; i < NODE(n)->count; i++) {
		ods_key_t key = ods_ref_as_obj(t->ods, NODE(n)->entries[i].key);
		fprintf(fp, "%s:%p, ",
		       (key ? ods_key_to_str(idx, key) : "-"),
		       (void *)(unsigned long)NODE(n)->entries[i].ref);
		ods_obj_put(key);
	}
	fprintf(fp, "\n");
	fflush(stdout);
	if (NODE(n)->is_leaf)
		return;
	/* Now print all it's children */
	for (i = 0; i < NODE(n)->count; i++) {
		ods_obj_t node = ods_ref_as_obj(t->ods, NODE(n)->entries[i].ref);
		print_node(idx, i, node, indent + 2, fp);
		ods_obj_put(node);
	}
}

static void print_idx(ods_idx_t idx, FILE *fp)
{
	bpt_t t = idx->priv;
	ods_obj_t node = ods_ref_as_obj(t->ods, t->root_ref);
	print_node(idx, 0, node, 0, fp);
	ods_obj_put(node);
}

static int bpt_open(ods_idx_t idx)
{
	size_t udata_sz;
	ods_obj_t udata = ods_get_user_data(idx->ods);
	bpt_t t = malloc(sizeof *t);
	if (!t) {
		ods_obj_put(udata);
		return ENOMEM;
	}
	t->order = UDATA(udata)->order;
	t->root_ref = UDATA(udata)->root;
	t->ods = idx->ods;
	t->comparator = idx->idx_class->cmp->compare_fn;
	idx->priv = t;
	ods_obj_put(udata);
	return 0;
}

static int bpt_init(ods_t ods, va_list argp)
{
	ods_obj_t udata = ods_get_user_data(ods);
	int order = va_arg(argp, int);
	if (order <= 0) {
		/*
		 * Each entry is 16B + 8B for the parent + 8B for the count.
		 * If each node is a page, 4096 / 16B = 256
		 */
		order = 251;
	}
	UDATA(udata)->order = order;
	UDATA(udata)->root = 0;
	ods_obj_put(udata);
	return 0;
}

static void bpt_close(ods_idx_t idx)
{
	ods_obj_t udata = ods_get_user_data(idx->ods);
	bpt_t t = idx->priv;
	UDATA(udata)->root = t->root_ref;
	ods_obj_put(udata);
}

ods_ref_t leaf_find_ref(bpt_t t, ods_key_t key)
{
	ods_ref_t ref = t->root_ref;
	ods_obj_t n;
	int i;

	if (!t->root_ref)
		return 0;

	n = ods_ref_as_obj(t->ods, t->root_ref);
	while (!NODE(n)->is_leaf) {
		int rc;
		for (i = 1; i < NODE(n)->count; i++) {
			ods_key_t entry_key =
				ods_ref_as_obj(t->ods, NODE(n)->entries[i].key);
			rc = t->comparator(key, entry_key);
			ods_obj_put(entry_key);
			if (rc >= 0)
				continue;
			else
				break;
		}
		ref = NODE(n)->entries[i-1].ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	ods_obj_put(n);
	return ref;
}

static int bpt_find(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	int i;
	bpt_t t = idx->priv;
	ods_ref_t leaf_ref = leaf_find_ref(t, key);
	ods_obj_t leaf = ods_ref_as_obj(t->ods, leaf_ref);
	if (!leaf)
		return ENOENT;
	for (i = 0; i < NODE(leaf)->count; i++) {
		int rc;
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, NODE(leaf)->entries[i].key);
		ods_obj_put(leaf);
		leaf = ods_ref_as_obj(t->ods, leaf_ref);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (!rc) {
			*ref = NODE(leaf)->entries[i].ref;
			ods_obj_put(leaf);
			return 0;
		}
	}
	ods_obj_put(leaf);
	return ENOENT;
}

static int bpt_update(ods_idx_t idx, ods_key_t key, ods_ref_t ref)
{
	int i;
	bpt_t t = idx->priv;
	ods_ref_t leaf_ref = leaf_find_ref(t, key);
	ods_obj_t leaf = ods_ref_as_obj(t->ods, leaf_ref);
	if (!leaf)
		return ENOENT;
	for (i = 0; i < NODE(leaf)->count; i++) {
		int rc;
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, NODE(leaf)->entries[i].key);
		leaf = ods_ref_as_obj(t->ods, leaf_ref);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (!rc) {
			NODE(leaf)->entries[i].ref = ref;
			ods_obj_put(leaf);
			return 0;
		}
	}
	ods_obj_put(leaf);
	return ENOENT;
}

static void find_first_dup(bpt_t t, ods_key_t key, ods_ref_t *leaf_ref, int *ent)
{
	int j;
	ods_obj_t left;
	while (NULL != (left = leaf_left(t, *leaf_ref))) {
		/* Search the left sibling to see if there is a match */
		j = search_leaf_reverse(t, left, key);
		if (j < 0) {
			ods_obj_put(left);
			break;
		}
		*leaf_ref = ods_obj_ref(left);
		ods_obj_put(left);
		*ent = j;
		if (j)
			break;
	}
}

static void find_last_dup(bpt_t t, ods_key_t key, ods_ref_t *leaf_ref, int *ent)
{
	int j;
	ods_obj_t right = ods_ref_as_obj(t->ods, *leaf_ref);
	/* Exhaust the current node */
	for (j = *ent; j < NODE(right)->count; j++) {
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, NODE(right)->entries[j].key);
		int rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (rc)
			break;
		*ent = j;
	}
	ods_obj_put(right);
	if (*ent < NODE(right)->count - 1)
		return;
	while (NULL != (right = leaf_right(t, *leaf_ref))) {
		/* Search the right sibling to see if there is a match */
		j = search_leaf(t, right, key);
		if (j < 0)
			break;
		*leaf_ref = ods_obj_ref(right);
		*ent = j;
		if (j < NODE(right)->count - 1)
			break;
		ods_obj_put(right);
	}
	if (right)
		ods_obj_put(right);
}

static int __find_lub(ods_idx_t idx, ods_key_t key,
			ods_ref_t *node_ref, int *ent)
{
	int i;
	bpt_t t = idx->priv;
	ods_ref_t leaf_ref = leaf_find_ref(t, key);
	ods_obj_t leaf = ods_ref_as_obj(t->ods, leaf_ref);
	if (!leaf)
		return 0;
	for (i = 0; i < NODE(leaf)->count; i++) {
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, NODE(leaf)->entries[i].key);
		int rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (rc > 0) {
			continue;
		} else if (rc == 0 || i < NODE(leaf)->count) {
			*ent = i;
			*node_ref = leaf_ref;
			if (rc == 0)
				find_last_dup(t, key, node_ref, ent);
			ods_obj_put(leaf);
			return 1;
		} else {
			break;
		}
	}
	/* LUB is in our right sibling */
	leaf_ref = NODE(leaf)->entries[t->order - 1].ref;
	ods_obj_put(leaf);
	leaf = ods_ref_as_obj(t->ods, leaf_ref);
	if (!leaf)
		return 0;
	*node_ref = leaf_ref;
	*ent = 0;
	return 1;
}

static int bpt_find_lub(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	int ent;
	ods_ref_t leaf_ref;
	ods_obj_t leaf;
	if (__find_lub(idx, key, &leaf_ref, &ent)) {
		leaf = ods_ref_as_obj(idx->ods, leaf_ref);
		*ref = NODE(leaf)->entries[ent].ref;
		ods_obj_put(leaf);
		return 0;
	}
	return ENOENT;
}

static int __find_glb(ods_idx_t idx, ods_key_t key, ods_ref_t *node_ref, int *ent)
{
	int i, rc;
	ods_key_t entry_key;
	bpt_t t = idx->priv;
	ods_ref_t leaf_ref = leaf_find_ref(t, key);
	ods_obj_t leaf = ods_ref_as_obj(t->ods, leaf_ref);
	if (!leaf)
		return 0;
	*node_ref = leaf_ref;
	for (i = NODE(leaf)->count - 1; i >= 0; i--) {
		entry_key = ods_ref_as_obj(t->ods, NODE(leaf)->entries[i].key);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (rc < 0) {
			continue;
		} else if (rc > 0) {
			*ent = i;
		} else {
			*ent = i;
			find_first_dup(t, key, node_ref, ent);
		}
		goto found;
	}
	ods_obj_put(leaf);
	return 0;
 found:
	ods_obj_put(leaf);
	return 1;
}

static int bpt_find_glb(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	int ent;
	ods_ref_t leaf_ref;
	if (__find_glb(idx, key, &leaf_ref, &ent)) {
		ods_obj_t leaf = ods_ref_as_obj(idx->ods, leaf_ref);
		*ref = NODE(leaf)->entries[ent].ref;
		ods_obj_put(leaf);
		return 0;
	}
	return ENOENT;
}


static ods_key_t key_new(ods_idx_t idx, ods_key_t key)
{
	ods_key_t akey = ods_key_alloc(idx, ods_key_len(key));
	if (!akey) {
		ods_extend(idx->ods, ods_size(idx->ods) * 2);
		akey = ods_key_alloc(idx, ods_key_len(key));
	}
	if (akey)
		ods_key_copy(akey, key);
	return akey;
}

static ods_obj_t node_new(ods_idx_t idx, bpt_t t)
{
	size_t sz;
	ods_obj_t obj;
	bpt_node_t n;

	sz = sizeof(struct bpt_node) + (t->order * sizeof(struct bpn_entry));
	obj = ods_obj_alloc(idx->ods, sz);
	if (!obj) {
		ods_extend(idx->ods, ods_size(idx->ods) * 2);
		obj = ods_obj_alloc(idx->ods, sz);
		if (!obj)
			return NULL;
	}
	n = obj->as.ptr;
	memset(n, 0, sz);
	return obj;
}

static struct bpn_entry ENTRY_INITIALIZER = { 0, 0 };

int leaf_insert(bpt_t t, ods_obj_t leaf, ods_key_t key, ods_ref_t obj_ref)
{
	int i, j;

	assert(NODE(leaf)->is_leaf);

	/* Insert the object */
	for (i = 0; i < NODE(leaf)->count; i++) {
		int rc;
		ods_key_t ek = ods_ref_as_obj(t->ods, NODE(leaf)->entries[i].key);
		rc = t->comparator(key, ek);
		ods_obj_put(ek);
		if (rc < 0)
			break;
	}
	/* Move up all the entries to make space */
	for (j = NODE(leaf)->count; j > i; j--)
		NODE(leaf)->entries[j] = NODE(leaf)->entries[j-1];

	/* Put in the new entry and update the count */
	NODE(leaf)->entries[i].key = ods_obj_ref(key);
	NODE(leaf)->entries[i].ref = obj_ref;
	NODE(leaf)->count++;
	return i;
}

static int find_idx(bpt_t t, ods_obj_t leaf, ods_key_t key)
{
	int i;
	for (i = 0; i < NODE(leaf)->count; i++) {
		int rc;
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, NODE(leaf)->entries[i].key);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (rc < 0)
			break;
	}
	return i;
}

static int split_midpoint(int order)
{
	if (order & 1)
		return (order >> 1) + 1;
	return order >> 1;
}

static ods_obj_t leaf_split_insert(ods_idx_t idx, bpt_t t, ods_obj_t left,
				    ods_obj_t new_key, ods_ref_t obj_ref)
{
	ods_obj_t right;
	int i, j;
	int ins_idx;
	int ins_left_n_right;
	int midpoint = split_midpoint(t->order);

	right = node_new(idx, t);
	if (!right)
		return NULL;
	assert(NODE(left)->is_leaf);
	NODE(right)->is_leaf = 1;
	NODE(right)->parent = NODE(left)->parent;

	ins_idx = find_idx(t, left, new_key);
	ins_left_n_right = ins_idx < midpoint;
	if (ins_left_n_right) {
		/*
		 * New entry goes in the left node. This means that
		 * the boundary marking which entries moves from left
		 * to right needs to be shifted left one because the
		 * insertion will eventually shift these entries right.
		 */
		for (i = midpoint - 1, j = 0; i < t->order - 1; i++, j++) {
			NODE(right)->entries[j] = NODE(left)->entries[i];
			NODE(left)->count--;
			NODE(right)->count++;
		}
		/*
		 * Move the objects between the insertion point and
		 * the end one slot to the right.
		 */
		for (i = midpoint - 1; i > ins_idx; i--)
			NODE(left)->entries[i] = NODE(left)->entries[i-1];

		/*
		 * Put the new item in the entry list
		 */
		NODE(left)->entries[ins_idx].ref = obj_ref;
		NODE(left)->entries[ins_idx].key = ods_obj_ref(new_key);
		NODE(left)->count++;
	} else {
		/*
		 * New entry goes in the right node. This means that
		 * as we move the entries from left to right, we need
		 * to leave space for the item that will be added.
		 */
		ins_idx = ins_idx - midpoint;
		for (i = midpoint, j = 0; i < t->order - 1; i++, j++) {
			/*
			 * If this is where the new entry will
			 * go, skip a slot
			 */
			if (ins_idx == j)
				j ++;
			NODE(right)->entries[j] = NODE(left)->entries[i];
			NODE(left)->entries[i] = ENTRY_INITIALIZER;
			NODE(left)->count--;
			NODE(right)->count++;
		}
		/*
		 * Put the new item in the entry list
		 */
		NODE(right)->entries[ins_idx].ref = obj_ref;
		NODE(right)->entries[ins_idx].key = ods_obj_ref(new_key);
		NODE(right)->count++;
	}
	/* Link left --> right */
	NODE(right)->entries[t->order-1].ref =
		NODE(left)->entries[t->order-1].ref; /* NODE(right)->next = NODE(left)->next */
	NODE(left)->entries[t->order-1].ref =
		ods_obj_ref(right); /* NODE(left)->next = right */
	return right;
}

static int verify_node(ods_idx_t idx, ods_obj_t n)
{
	bpt_t t = idx->priv;
	int i;
	int midpoint = split_midpoint(t->order);

	if (!n)
		return 1;

	/* Make sure each entry is lexically >= previous */
	for (i = 0; i < NODE(n)->count-1; i++) {
		ods_key_t e1 = ods_ref_as_obj(t->ods, NODE(n)->entries[i].key);
		ods_key_t e2 = ods_ref_as_obj(t->ods, NODE(n)->entries[i+1].key);
		if (!(t->comparator(e2, e1) >= 0)) {
			print_idx(idx, stdout);
			assert(0);
		}
		ods_obj_put(e1);
		ods_obj_put(e2);
	}
	if (t->root_ref != ODS_PTR_TO_REF(n)) {
		if (NODE(n)->is_leaf)
			midpoint--;
		/* Make sure it has at least midpoint entries */
		if (!NODE(n)->count >= midpoint) {
			print_idx(idx, stdout);
			assert(0);
		}
	}

	if (NODE(n)->is_leaf)
		return 1;

	/* Make certain all n's keys refer to the min. of each child entry */
	for (i = 0; i < NODE(n)->count; i++) {
		ods_key_t parent_key = ods_ref_as_obj(t->ods, NODE(n)->entries[i].key);
		ods_obj_t child = ods_ref_as_obj(t->ods, NODE(n)->entries[i].ref);
		ods_key_t child_key = ods_ref_as_obj(t->ods, NODE(child)->entries[0].key);
		if (!(t->comparator(parent_key, child_key) == 0)) {
			print_idx(idx, stdout);
			assert(0);
		}
		ods_obj_put(parent_key);
		ods_obj_put(child_key);
		ods_obj_put(child);
	}
	/* Now verify each entry */
	for (i = 0; i < NODE(n)->count; i++) {
		ods_obj_t child = ods_ref_as_obj(t->ods, NODE(n)->entries[i].ref);
		verify_node(idx, child);
		ods_obj_put(child);
	}
	return 1;
}

static int debug = 0;
static int verify_tree(ods_idx_t idx)
{
	bpt_t t;
	ods_obj_t root;
	if (!debug)
		return 1;

	t = idx->priv;
	root = ods_ref_as_obj(t->ods, t->root_ref);
	verify_node(idx, root);
	ods_obj_put(root);
	return 1;
}

static void node_insert(bpt_t t, ods_obj_t node, ods_obj_t left,
			ods_ref_t key_ref, ods_obj_t right)
{
	int i, j;
	ods_ref_t left_ref = ods_obj_ref(left);

	assert(!NODE(node)->is_leaf);
	/* Find left's index */
	for (i = 0; i < NODE(node)->count; i++) {
		if (left_ref == NODE(node)->entries[i].ref)
			break;
	}
	assert(i < NODE(node)->count);

	/*
	 * Make room for right after left's current key/ref and the
	 * end of the node
	 */
	for (j = NODE(node)->count; j > i+1; j--)
		NODE(node)->entries[j] = NODE(node)->entries[j-1];

	/* Put in the new entry and update the count */
	NODE(node)->entries[i+1].key = key_ref;
	NODE(node)->entries[i+1].ref = ods_obj_ref(right);
	NODE(node)->count++;
	NODE(left)->parent = NODE(right)->parent = ods_obj_ref(node);
}

static ods_obj_t node_split_insert(ods_idx_t idx, bpt_t t,
				   ods_obj_t left_node,
				   ods_ref_t right_key_ref,
				   ods_obj_t right_node)
{
	ods_obj_t left_parent, right_parent;
	int i, j;
	int ins_idx, ins_left_n_right;
	int count;
	int midpoint = split_midpoint(t->order);

	/* Take our own reference on these parameters */
	left_node = ods_obj_get(left_node);
	right_node = ods_obj_get(right_node);

 split_and_insert:
	/* Right node and parent */
	right_parent = node_new(idx, t);
	if (!right_parent)
		goto err_0;

	/* Left node and parent */
	left_parent = ods_ref_as_obj(t->ods, NODE(left_node)->parent);

	/*
	 * Find the index of the left_node in the parent
	 */
	for (i = 0; i < NODE(left_parent)->count; i++) {
		if (ods_obj_ref(left_node) == NODE(left_parent)->entries[i].ref)
			break;
	}
	/* Right is the succesor of left, so insert it at it's position + 1 */
	ins_idx = i + 1;
	assert(i < NODE(left_parent)->count);

	ins_left_n_right = ins_idx < midpoint;
	if (ins_left_n_right) {
		/*
		 * New entry goes in the left parent. This means that
		 * the boundary marking which entries shift to the
		 * right needs to be moved down one because the
		 * insertion will eventually shift these entries up.
		 */
		count = NODE(left_parent)->count - midpoint + 1;
		for (i = midpoint - 1, j = 0; j < count; i++, j++) {
			ods_obj_t n =
				ods_ref_as_obj(t->ods,
					       NODE(left_parent)->entries[i].ref);
			NODE(n)->parent = ods_obj_ref(right_parent);
			ods_obj_put(n);
			NODE(right_parent)->entries[j] = NODE(left_parent)->entries[i];
			NODE(left_parent)->entries[i] = ENTRY_INITIALIZER;
		}
		NODE(right_parent)->count += count;
		NODE(left_parent)->count -= (count - 1); /* account for the insert below */
		/*
		 * Move the objects between the insertion point and
		 * the end one slot to the right.
		 */
		for (i = midpoint - 1; i > ins_idx; i--)
			NODE(left_parent)->entries[i] = NODE(left_parent)->entries[i-1];

		/*
		 * Put the new item in the entry list. Right is the
		 * successor of left, therefore it's insertion index
		 * cannot be zero.
		 */
		assert(ins_idx);
		NODE(left_parent)->entries[ins_idx].ref = ods_obj_ref(right_node);
		NODE(left_parent)->entries[ins_idx].key = right_key_ref;
		NODE(right_node)->parent = ods_obj_ref(left_parent);
	} else {
		/*
		 * New entry goes in the right node. This means that
		 * as we move the entries from left to right, we need
		 * to leave space for the item that will be added.
		 */
		count = NODE(left_parent)->count;
		ins_idx = ins_idx - midpoint;
		for (i = midpoint, j = 0; i < count; i++, j++) {
			ods_obj_t n =
				ods_ref_as_obj(t->ods,
					       NODE(left_parent)->entries[i].ref);
			/*
			 * If this is where the new entry will
			 * go, skip a slot
			 */
			if (ins_idx == j)
				j ++;
			NODE(n)->parent = ods_obj_ref(right_parent);
			ods_obj_put(n);
			NODE(right_parent)->entries[j] = NODE(left_parent)->entries[i];
			NODE(left_parent)->entries[i] = ENTRY_INITIALIZER;
			NODE(right_parent)->count++;
			NODE(left_parent)->count--;
		}
		/*
		 * Put the new item in the entry list
		 */
		NODE(right_parent)->entries[ins_idx].ref = ods_obj_ref(right_node);
		NODE(right_parent)->entries[ins_idx].key = right_key_ref;
		NODE(right_parent)->count++;
		NODE(right_node)->parent = ods_obj_ref(right_parent);
	}
	assert(NODE(right_parent)->count > 1);
	assert(NODE(left_parent)->count > 1);

	/*
	 * Now insert our new right parent at the next level
	 * up the tree.
	 */
	ods_obj_t next_parent =
		ods_ref_as_obj(t->ods, NODE(left_parent)->parent);
	if (!next_parent) {
		/* Split root */
		next_parent = node_new(idx, t);
		NODE(next_parent)->count = 2;
		NODE(next_parent)->entries[0].ref = ods_obj_ref(left_parent);
		NODE(next_parent)->entries[0].key = NODE(left_parent)->entries[0].key;
		NODE(next_parent)->entries[1].ref = ods_obj_ref(right_parent);
		NODE(next_parent)->entries[1].key = NODE(right_parent)->entries[0].key;
		NODE(left_parent)->parent = ods_obj_ref(next_parent);
		NODE(right_parent)->parent = ods_obj_ref(next_parent);
		t->root_ref = ods_obj_ref(next_parent);
		goto out;
	}
	/* If there is room, insert into the parent */
	if (NODE(next_parent)->count < t->order) {
		node_insert(t, next_parent, left_parent,
			    NODE(right_parent)->entries[0].key, right_parent);
		goto out;
	}
	ods_obj_put(next_parent);
	ods_obj_put(left_node);
	ods_obj_put(right_node);
	/* Go up to the next level and split and insert */
	left_node = left_parent;
	right_node = right_parent;
	right_key_ref = NODE(right_parent)->entries[0].key;
	goto split_and_insert;
 out:
	ods_obj_put(right_parent);
	ods_obj_put(left_parent);
	ods_obj_put(left_node);
	ods_obj_put(right_node);
	return next_parent;
 err_0:
	return NULL;
}

static int bpt_insert(ods_idx_t idx, ods_key_t new_key, ods_ref_t obj_ref)
{
	bpt_t t = idx->priv;
	ods_obj_t parent;
	ods_ref_t leaf_ref;
	ods_obj_t udata = ods_get_user_data(idx->ods);

	if (!t->root_ref) {
		ods_obj_t node = node_new(idx, t);
		if (!node)
			goto err_1;
		t->root_ref = ods_obj_ref(node);
		UDATA(udata)->root = t->root_ref;
		NODE(node)->is_leaf = 1;
		leaf_ref = t->root_ref;
		ods_obj_put(node);
	} else
		leaf_ref = leaf_find_ref(t, new_key);

	/* Check if this is a memory key and allocate an ODS key if so */
	if (!ods_obj_ref(new_key)) {
		ods_key_t key = key_new(idx, new_key);
		if (!key)
			goto err_1;
		new_key = key;
	} else
		ods_obj_get(new_key);

	/* Is there room in the leaf? */
	ods_obj_t leaf = ods_ref_as_obj(t->ods, leaf_ref);
	if (NODE(leaf)->count < t->order - 1) {
		if (!leaf_insert(t, leaf, new_key, obj_ref) && NODE(leaf)->parent) {
			ods_obj_t parent =
				ods_ref_as_obj(t->ods, NODE(leaf)->parent);
			/* Maintain this to simplify other logic */
			if (NODE(parent)->entries[0].ref == leaf_ref)
				NODE(parent)->entries[0].key = NODE(leaf)->entries[0].key;
			ods_obj_put(parent);
		}
		verify_tree(idx);
		ods_obj_put(leaf);
		ods_obj_put(udata);
		ods_obj_put(new_key);
		return 0;
	}
	ods_obj_t new_leaf =
		leaf_split_insert(idx, t, leaf, new_key, obj_ref);
	if (!new_leaf)
		goto err_2;

	parent = ods_ref_as_obj(t->ods, NODE(leaf)->parent);
	if (!parent) {
		parent = node_new(idx, t);
		if (!parent)
			goto err_2;

		NODE(parent)->entries[0].key = NODE(leaf)->entries[0].key;
		NODE(parent)->entries[0].ref = ods_obj_ref(leaf);

		NODE(parent)->entries[1].key = NODE(new_leaf)->entries[0].key;
		NODE(parent)->entries[1].ref = ods_obj_ref(new_leaf);
		NODE(parent)->count = 2;

		NODE(leaf)->parent = ods_obj_ref(parent);
		NODE(new_leaf)->parent = ods_obj_ref(parent);
		t->root_ref = ods_obj_ref(parent);
		goto out;
	}
	if (NODE(parent)->count < t->order) {
		node_insert(t, parent, leaf, NODE(new_leaf)->entries[0].key, new_leaf);
		goto out;
	}
	ods_obj_put(parent);
	parent = node_split_insert(idx, t, leaf,
				   NODE(new_leaf)->entries[0].key, new_leaf);
	if (!parent)
		goto err_3;
 out:
	UDATA(udata)->root = t->root_ref;
	verify_tree(idx);
	ods_obj_put(leaf);
	ods_obj_put(new_leaf);
	ods_obj_put(udata);
	ods_obj_put(parent);
	ods_obj_put(new_key);
	return 0;

 err_3:
	/* TODO: Unsplit the leaf and put the tree back in order. */
	assert(0);
	ods_obj_delete(new_leaf);
	ods_obj_put(new_leaf);
 err_2:
	ods_obj_delete(new_key);
	ods_obj_put(new_key);
 err_1:
	ods_obj_put(udata);
	return ENOMEM;
}

ods_ref_t bpt_min_ref(bpt_t t)
{
	ods_obj_t n;
	ods_ref_t ref = 0;

	if (!t->root_ref)
		return 0;

	/* Walk to the left most leaf and return the 0-th entry  */
	ref = t->root_ref;
	n = ods_ref_as_obj(t->ods, t->root_ref);
	while (!NODE(n)->is_leaf) {
		ref = NODE(n)->entries[0].ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	ods_obj_put(n);
	return ref;
}

static int entry_find(bpt_t t, ods_obj_t node, ods_key_t key, int *idx, ods_ref_t *ref)
{
	int i;
	for (i = 0; i < NODE(node)->count; i++) {
		int rc;
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, NODE(node)->entries[i].key);
		if (!entry_key) {
			/* The last key in an interior node is NUL */
			assert(!NODE(node)->is_leaf);
			break;
		}
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (!rc)
			break;
	}
	if (i < NODE(node)->count) {
		*ref = NODE(node)->entries[i].ref;
		*idx = i;
	} else
		return ENOENT;
	return 0;
}

static ods_obj_t leaf_right_most(bpt_t t, ods_obj_t node)
{
	ods_ref_t next_ref;
	while (!NODE(node)->is_leaf) {
		next_ref = NODE(node)->entries[NODE(node)->count-1].ref;
		ods_obj_put(node);
		node = ods_ref_as_obj(t->ods, next_ref);
	}
	return node;
}

/**
 * Find left sibling of the given \c node in the tree \c t.
 *
 * \param t The tree.
 * \param node The reference leaf node.
 *
 * \returns NULL if the left node is not found.
 * \returns A pointer to the left node, if it is found.
 */
static ods_obj_t leaf_left(bpt_t t, uint64_t node_ref)
{
	int idx;
	ods_obj_t node;
	uint64_t parent_ref;
	ods_obj_t parent;
	ods_obj_t left;

	node = ods_ref_as_obj(t->ods, node_ref);
	assert(NODE(node)->is_leaf);
loop:
	if (t->root_ref == node_ref) {
		ods_obj_put(node);
		return NULL;
	}
	assert(NODE(node)->parent);
	parent_ref = NODE(node)->parent;
	parent = ods_ref_as_obj(t->ods, NODE(node)->parent);
	ods_obj_put(node);

	for (idx = 0; idx < NODE(parent)->count; idx++)
		if (NODE(parent)->entries[idx].ref == node_ref)
			break;
	assert(idx < NODE(parent)->count);

	if (!idx) {
		node_ref = parent_ref;
		node = parent;
		goto loop;
	}

	node = ods_ref_as_obj(t->ods, NODE(parent)->entries[idx-1].ref);
	left = leaf_right_most(t, node);
	ods_obj_put(parent);
	return left;
}

/**
 * Find left sibling of the given \c node in the tree \c t.
 *
 * \param t The tree.
 * \param node The reference leaf node.
 *
 * \returns NULL if the left node is not found.
 * \returns A pointer to the left node, if it is found.
 */
static ods_obj_t leaf_right(bpt_t t, uint64_t node_ref)
{
	ods_obj_t node;
	uint64_t right_ref;
	node = ods_ref_as_obj(t->ods, node_ref);
	assert(NODE(node)->is_leaf);
	right_ref = NODE(node)->entries[t->order - 1].ref;
	ods_obj_put(node);
	return ods_ref_as_obj(t->ods, right_ref);
}

static int node_neigh(bpt_t t, ods_obj_t node, ods_obj_t *left, ods_obj_t *right)
{
	int idx;
	ods_ref_t node_ref;
	ods_obj_t parent;

	*left = *right = NULL;
	node_ref = ods_obj_ref(node);
	if (t->root_ref == node_ref)
		return 0;

	assert(NODE(node)->parent);
	parent = ods_ref_as_obj(t->ods, NODE(node)->parent);
	for (idx = 0; idx < NODE(parent)->count; idx++)
		if (NODE(parent)->entries[idx].ref == node_ref)
			break;
	assert(idx < NODE(parent)->count);

	if (idx)
		*left = ods_ref_as_obj(t->ods, NODE(parent)->entries[idx-1].ref);

	if (idx < NODE(parent)->count-1)
		*right = ods_ref_as_obj(t->ods, NODE(parent)->entries[idx+1].ref);

	ods_obj_put(parent);
	return idx;
}

static int space(bpt_t t, ods_obj_t n)
{
	if (!n)
		return 0;

	if (NODE(n)->is_leaf)
		return t->order - NODE(n)->count - 1;

	return t->order - NODE(n)->count;
}

static int combine_right(bpt_t t, ods_obj_t right, int idx, ods_obj_t node)
{
	int i, j;
	int count = NODE(node)->count - idx;
	ods_ref_t right_ref;
	ods_obj_t entry;

	if (!right || !count)
		return idx;

	/* Make room to the left */
	for (i = NODE(right)->count + count - 1; i > idx; i--)
		NODE(right)->entries[i] = NODE(right)->entries[i-count];

	right_ref = ods_obj_ref(right);
	for (i = 0, j = idx; j < NODE(node)->count; i++, j++) {
		/* Update the entry's parent */
		if (!NODE(node)->is_leaf) {
			entry = ods_ref_as_obj(t->ods, NODE(node)->entries[j].ref);
			NODE(entry)->parent = right_ref;
			ods_obj_put(entry);
		}
		/* Move the entry to the right sibling */
		NODE(right)->entries[i] = NODE(node)->entries[j];
		NODE(right)->count ++;
		idx++;
	}
	return idx;
}

static int combine_left(bpt_t t, ods_obj_t left, int idx, ods_obj_t node)
{
	int i, j;
	int count = NODE(node)->count - idx;
	ods_obj_t entry;
	ods_ref_t left_ref;

	if (!left)
		return idx;

	left_ref = ods_obj_ref(left);
	for (i = NODE(left)->count, j = idx; j < count; i++, j++) {
		/* Update the entry's parent */
		if (!NODE(node)->is_leaf) {
			entry = ods_ref_as_obj(t->ods, NODE(node)->entries[j].ref);
			NODE(entry)->parent = left_ref;
			ods_obj_put(entry);
		}
		/* Move the entry to the left sibling */
		NODE(left)->entries[i] = NODE(node)->entries[j];
		NODE(left)->count++;
		idx++;
	}
	return idx;
}

/*
 * Drops reference on parent and node
 */
static ods_ref_t fixup_parents(bpt_t t, ods_obj_t parent, ods_obj_t node)
{
	int i;
	ods_ref_t parent_ref = ods_obj_ref(parent);
	ods_ref_t node_ref;
	ods_obj_put(parent);
	while (parent_ref) {
		parent = ods_ref_as_obj(t->ods, parent_ref);
		for (i = 0; i < NODE(parent)->count; i++)
			if (NODE(parent)->entries[i].ref == ods_obj_ref(node))
				break;
		assert(i < NODE(parent)->count);
		NODE(parent)->entries[i].key = NODE(node)->entries[0].key;
		ods_obj_put(node);
		node = parent;
		parent_ref = NODE(parent)->parent;
	}
	ods_obj_put(node);
	return t->root_ref;
}

static void merge_from_left(bpt_t t, ods_obj_t left, ods_obj_t node, int midpoint)
{
	int count;
	int i, j;
	ods_ref_t node_ref = ods_obj_ref(node);

	assert(NODE(left)->count > midpoint);
	count = NODE(left)->count - midpoint;

	/* Make room in node */
	for (i = NODE(node)->count + count - 1, j = 0; j < NODE(node)->count; j++, i--)
		NODE(node)->entries[i] = NODE(node)->entries[i-count];

	for (i = 0, j = NODE(left)->count - count; i < count; i++, j++) {
		if (!NODE(node)->is_leaf) {
			ods_obj_t entry =
				ods_ref_as_obj(t->ods, NODE(left)->entries[j].ref);
			NODE(entry)->parent = node_ref;
			ods_obj_put(entry);
		}
		NODE(node)->entries[i] = NODE(left)->entries[j];
		NODE(left)->entries[j] = ENTRY_INITIALIZER;
		NODE(left)->count--;
		NODE(node)->count++;
	}
}

static void merge_from_right(bpt_t t, ods_obj_t right, ods_obj_t node, int midpoint)
{
	int count;
	int i, j;
	ods_obj_t parent;
	ods_ref_t node_ref = ods_obj_ref(node);

	assert(NODE(right)->count > midpoint);
	count = NODE(right)->count - midpoint;
	for (i = NODE(node)->count, j = 0; j < count; i++, j++) {
		if (!NODE(node)->is_leaf) {
			ods_obj_t entry =
				ods_ref_as_obj(t->ods, NODE(right)->entries[j].ref);
			NODE(entry)->parent = node_ref;
			ods_obj_put(entry);
		}
		NODE(node)->entries[i] = NODE(right)->entries[j];
		NODE(right)->count--;
		NODE(node)->count++;
	}
	/* Move right's entries down */
	for (i = 0; i < NODE(right)->count; i++, j++)
		NODE(right)->entries[i] = NODE(right)->entries[j];
	/* Clean up the end of right */
	for (j = NODE(right)->count; j < NODE(right)->count + count; j++)
		NODE(right)->entries[j] = ENTRY_INITIALIZER;

	/* Fixup right's parents. */
	parent = ods_ref_as_obj(t->ods, NODE(right)->parent);
	fixup_parents(t, parent, right);
}

static ods_ref_t entry_delete(bpt_t t, ods_obj_t node, ods_key_t key, int idx)
{
	int i, midpoint;
	ods_obj_t left, right;
	ods_obj_t parent;
	int node_idx;
	int count;

	assert(NODE(node)->is_leaf);
 next_level:
	parent = ods_ref_as_obj(t->ods, NODE(node)->parent);
	/* Remove the key and object from the node */
	for (i = idx; i < NODE(node)->count; i++)
		NODE(node)->entries[i] = NODE(node)->entries[i+1];
	NODE(node)->entries[NODE(node)->count-1] = ENTRY_INITIALIZER;
	NODE(node)->count--;

	if (ods_obj_ref(node) == t->root_ref) {
		/* This is the root of the tree. */
		switch (NODE(node)->count) {
		case 0:
			/* The root is now empty */
			t->root_ref = 0;
			ods_obj_delete(node);
			ods_obj_put(node);
			return 0;
		case 1:
			if (!NODE(node)->is_leaf) {
				/*
				 * The root is not a leaf, but it only
				 * has a single entry left, so promote
				 * it's only child as the new root.
				 */
				ods_ref_t root_ref;
				root_ref = NODE(node)->entries[0].ref;
				parent = ods_ref_as_obj(t->ods, root_ref);
				NODE(parent)->parent = 0;
				ods_obj_put(parent);
				ods_obj_delete(node);
				ods_obj_put(node);
				return root_ref;
			}
		default:
			ods_obj_put(node);
			return t->root_ref;
		}
	}

	midpoint = split_midpoint(t->order);
	if (NODE(node)->is_leaf)
		midpoint--;

	if (NODE(node)->count >= midpoint)
		return fixup_parents(t, parent, node);

	node_idx = node_neigh(t, node, &left, &right);
	count = space(t, left) + space(t, right);
	if (count < NODE(node)->count) {
		/*
		 * There's not enough room in the left and right
		 * siblings to hold node's remainder, so we need to
		 * collect a few entries from the left and right
		 * siblings to put this node over the min.
		 */
		if (left) {
			if (NODE(left)->count > midpoint) {
				merge_from_left(t, left, node, midpoint);
			}
			ods_obj_put(left);
		}

		if (right) {
			if (NODE(right)->count > midpoint
			    && NODE(node)->count < midpoint) {
				/*
				 * Drops the reference on right
				 * because it calls fixup_parents()
				 */
				merge_from_right(t, right, node, midpoint);
			} else
				ods_obj_put(right);
		}

		assert(NODE(node)->count >= midpoint);
		return fixup_parents(t, parent, node);
	}
	/* Node is going away, link our left and right siblings */
	if (NODE(node)->is_leaf && left)
		NODE(left)->entries[t->order-1].ref = ods_obj_ref(right);

	/* Combine as many as possible to the left */
	idx = combine_left(t, left, 0, node);
	if (left) {
		NODE(parent)->entries[node_idx-1].key = NODE(left)->entries[0].key;
		NODE(parent)->entries[node_idx-1].ref = ods_obj_ref(left);
	}
	/* Move the remainder to the right */
	if (idx < NODE(node)->count) {
		combine_right(t, right, idx, node);
		if (right) {
			NODE(parent)->entries[node_idx+1].key = NODE(right)->entries[0].key;
			NODE(parent)->entries[node_idx+1].ref = ods_obj_ref(right);
		}
	}
	/* Remove the node(idx) from the parent. */
	idx = node_idx;
	ods_obj_put(node);
	ods_obj_put(left);
	ods_obj_put(right);
	node = parent;
	goto next_level;
}

static int bpt_delete(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	bpt_t t = idx->priv;
	int ent, rc;
	ods_ref_t leaf_ref, obj_ref;
	ods_obj_t leaf;
	ods_obj_t udata;

	udata =	ods_get_user_data(idx->ods);
	leaf_ref = leaf_find_ref(t, key);
	leaf = ods_ref_as_obj(t->ods, leaf_ref);
	if (!leaf)
		goto noent;

	rc = entry_find(t, leaf, key, &ent, &obj_ref);
	if (rc)
		goto noent;

	/* Ignore the caller's key once we've found the entry */
	key = ods_ref_as_obj(t->ods, NODE(leaf)->entries[ent].key);

	t->root_ref = entry_delete(t, leaf, key, ent);
	UDATA(udata)->root = t->root_ref;
	ods_obj_delete(key);
	ods_obj_put(key);
	*ref = obj_ref;
	ods_obj_put(udata);
	return 0;
 noent:
	ods_obj_put(leaf);
	ods_obj_put(udata);
	return ENOENT;
}

ods_ref_t bpt_max_ref(bpt_t t)
{
	ods_obj_t n;
	ods_ref_t ref = 0;
	if (!t->root_ref)
		return 0;

	/* Walk to the left most leaf and return the 0-th entry  */
	ref = t->root_ref;
	n = ods_ref_as_obj(t->ods, ref);
	while (!NODE(n)->is_leaf) {
		ref = NODE(n)->entries[NODE(n)->count-1].ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	ods_obj_put(n);
	return ref;
}

#define POS_PAD 0x45544552 /* 'ITER' */
struct bpt_pos_s {
	uint32_t pad;
	uint32_t ent;
	ods_ref_t node_ref;
};

static int bpt_iter_set(ods_iter_t oi, const ods_pos_t pos_)
{
	struct bpt_pos_s *pos = (struct bpt_pos_s *)pos_;
	bpt_iter_t i = (bpt_iter_t)oi;
	bpt_t t = i->idx->priv;

	if (pos->pad != POS_PAD)
		return EINVAL;

	i->ent = pos->ent;
	i->node_ref = pos->node_ref;
	return 0;
}

static int bpt_iter_pos(ods_iter_t oi, ods_pos_t pos_)
{
	bpt_iter_t i = (bpt_iter_t)oi;
	bpt_t t = i->idx->priv;
	struct bpt_pos_s *pos = (struct bpt_pos_s *)pos_;

	if (!i->node_ref)
		return ENOENT;

	pos->pad = POS_PAD;
	pos->ent = i->ent;
	pos->node_ref = i->node_ref;
	return 0;
}

static ods_iter_t bpt_iter_new(ods_idx_t idx)
{
	bpt_iter_t iter = calloc(1, sizeof *iter);
	iter->idx = idx;
	return (struct ods_iter *)iter;
}

static void bpt_iter_delete(ods_iter_t i)
{
	free(i);
}

static int bpt_iter_begin(ods_iter_t oi)
{
	ods_obj_t node;
	bpt_iter_t i = (bpt_iter_t)oi;
	bpt_t t = i->idx->priv;
	i->ent = 0;
	i->node_ref = bpt_min_ref(t);
	if (!i->node_ref)
		return ENOENT;
	node = ods_ref_as_obj(t->ods, i->node_ref);
	assert(NODE(node)->is_leaf);
	ods_obj_put(node);
	return 0;
}

static int bpt_iter_end(ods_iter_t oi)
{
	ods_obj_t node;
	bpt_iter_t i = (bpt_iter_t)oi;
	bpt_t t = i->idx->priv;
	i->ent = 0;
	i->node_ref = bpt_max_ref(t);
	if (!i->node_ref)
		return ENOENT;
	node = ods_ref_as_obj(t->ods, i->node_ref);
	i->ent = NODE(node)->count - 1;
	assert(NODE(node)->is_leaf);
	ods_obj_put(node);
	return 0;
}

static ods_key_t bpt_iter_key(ods_iter_t oi)
{
	bpt_iter_t i = (bpt_iter_t)oi;
	ods_obj_t node;
	ods_key_t k = NULL;
	if (i->node_ref) {
		node = ods_ref_as_obj(i->idx->ods, i->node_ref);
		k = ods_ref_as_obj(i->idx->ods, NODE(node)->entries[i->ent].key);
		ods_obj_put(node);
	}
	return k;
}

static ods_ref_t bpt_iter_ref(ods_iter_t oi)
{
	bpt_iter_t i = (bpt_iter_t)oi;
	ods_ref_t ref = 0;
	ods_obj_t node;
	if (i->node_ref) {
		node = ods_ref_as_obj(i->idx->ods, i->node_ref);
		ref = NODE(node)->entries[i->ent].ref;
		ods_obj_put(node);
	}
	return ref;
}

static int search_leaf(bpt_t t, ods_obj_t leaf, ods_key_t key)
{
	int i, rc;
	for (i = 0; i < NODE(leaf)->count; i++) {
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, NODE(leaf)->entries[i].key);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (!rc)
			return i;
		else if (rc < 0)
			return -1;
	}
	return -1;
}

static int search_leaf_reverse(bpt_t t, ods_obj_t leaf, ods_key_t key)
{
	int i, rc, last_match = -1;
	for (i = NODE(leaf)->count-1; i >= 0; i--) {
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, NODE(leaf)->entries[i].key);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (0 == rc)
			last_match = i;
		else
			break;
	}
	return last_match;
}

static int bpt_iter_find(ods_iter_t oi, ods_key_t key)
{
	bpt_iter_t iter = (bpt_iter_t)oi;
	bpt_t t = iter->idx->priv;
	ods_ref_t leaf_ref = leaf_find_ref(t, key);
	ods_obj_t leaf = ods_ref_as_obj(t->ods, leaf_ref);
	int i;

	if (!leaf)
		return ENOENT;

	assert(NODE(leaf)->is_leaf);
	i = search_leaf(t, leaf, key);
	ods_obj_put(leaf);
	if (i < 0)
		return ENOENT;

	/* If the key is not the first element in the leaf, then a duplicate key
	 * cannot be a predecessor.
	 */
	if (i)
		goto found;

	find_first_dup(t, key, &leaf_ref, &i);
 found:
	iter->node_ref = leaf_ref;
	iter->ent = i;
	return 0;
}

static int bpt_iter_find_lub(ods_iter_t oi, ods_key_t key)
{
	bpt_iter_t iter = (bpt_iter_t)oi;
	ods_ref_t leaf_ref;
	int ent;

	if (__find_lub(iter->idx, key, &leaf_ref, &ent)) {
		iter->node_ref = leaf_ref;
		iter->ent = ent;
		return 0;
	}
	return ENOENT;
}

static int bpt_iter_find_glb(ods_iter_t oi, ods_key_t key)
{
	bpt_iter_t iter = (bpt_iter_t)oi;
	ods_ref_t leaf_ref;
	int ent;

	if (__find_glb(iter->idx, key, &leaf_ref, &ent)) {
		iter->node_ref = leaf_ref;
		iter->ent = ent;
		return 0;
	}
	return ENOENT;
}

static int bpt_iter_next(ods_iter_t oi)
{
	ods_obj_t node;
	bpt_iter_t i = (bpt_iter_t)oi;
	bpt_t t = i->idx->priv;
	if (!i->node_ref)
		goto not_found;
	node = ods_ref_as_obj(i->idx->ods, i->node_ref);
	if (i->ent < NODE(node)->count - 1) {
		i->ent++;
	} else {
		i->node_ref = NODE(node)->entries[t->order - 1].ref;
		ods_obj_put(node);
		node = ods_ref_as_obj(i->idx->ods, i->node_ref);
		if (!node)
			goto not_found;
		i->ent = 0;
	}
	ods_obj_put(node);
	return 0;
 not_found:
	return ENOENT;
}

static int bpt_iter_prev(ods_iter_t oi)
{
	bpt_iter_t i = (bpt_iter_t)oi;
	bpt_t t = i->idx->priv;
	if (!i->node_ref)
		goto not_found;
	if (i->ent) {
		i->ent--;
	} else {
		ods_obj_t left = leaf_left(t, i->node_ref);
		if (!left)
			goto not_found;
		i->node_ref = ods_obj_ref(left);
		i->ent = NODE(left)->count - 1;
		ods_obj_put(left);
	}
	return 0;
 not_found:
	return ENOENT;
}

static const char *bpt_get_type(void)
{
	return "BPTREE";
}

static void bpt_commit(ods_idx_t idx)
{
	ods_commit(idx->ods, ODS_COMMIT_SYNC);
}

static struct ods_idx_provider bpt_provider = {
	.get_type = bpt_get_type,
	.init = bpt_init,
	.open = bpt_open,
	.close = bpt_close,
	.commit = bpt_commit,
	.insert = bpt_insert,
	.update = bpt_update,
	.delete = bpt_delete,
	.find = bpt_find,
	.find_lub = bpt_find_lub,
	.find_glb = bpt_find_glb,
	.iter_new = bpt_iter_new,
	.iter_delete = bpt_iter_delete,
	.iter_find = bpt_iter_find,
	.iter_find_lub = bpt_iter_find_lub,
	.iter_find_glb = bpt_iter_find_glb,
	.iter_begin = bpt_iter_begin,
	.iter_end = bpt_iter_end,
	.iter_next = bpt_iter_next,
	.iter_prev = bpt_iter_prev,
	.iter_set = bpt_iter_set,
	.iter_pos = bpt_iter_pos,
	.iter_key = bpt_iter_key,
	.iter_ref = bpt_iter_ref,
	.print_idx = print_idx
};

struct ods_idx_provider *get(void)
{
	return &bpt_provider;
}

