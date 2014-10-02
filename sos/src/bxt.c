/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
 *
 * Confidential and Proprietary
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
#include <sos/ods.h>
#include "bxt.h"

static ods_obj_t leaf_left(bxt_t t, uint64_t node_ref);
static ods_obj_t leaf_right(bxt_t t, uint64_t node_ref);
static int node_neigh(bxt_t t, ods_obj_t node, ods_obj_t *left, ods_obj_t *right);

void dump_node(bxt_t t, ods_idx_t idx, ods_obj_t n, int ent, int indent, FILE *fp)
{
	int i;
	fprintf(fp, "%p - %*s%s[%d] | %p : ", (void *)(unsigned long)NODE(n)->parent,
	       indent, "", "NODE", ent, n->as.ptr);
	for (i = 0; i < NODE(n)->count; i++) {
		ods_key_t key = ods_ref_as_obj(t->ods, N_ENT(n,i).key_ref);
		fprintf(fp, "%s:%p, ",
		       (key ? ods_key_to_str(idx, key) : "-"),
		       (void *)(unsigned long)N_ENT(n, i).node_ref);
		ods_obj_put(key);
	}
	fprintf(fp, "\n");
}

void dump_leaf(bxt_t t, ods_idx_t idx, ods_obj_t n, int ent, int indent, FILE *fp)
{
	int i;
	fprintf(fp, "%p - %*s%s[%d] | %p :\n", (void *)(unsigned long)NODE(n)->parent,
	       indent, "", "LEAF", ent, n->as.ptr);
	for (i = 0; i < NODE(n)->count; i++) {
		ods_obj_t rec;
		ods_key_t key;
		ods_ref_t tail = L_ENT(n,i).tail_ref;
		ods_ref_t head = L_ENT(n,i).head_ref;
		fprintf(fp, "\t\tENTRY#%2d : head %p tail %p\n", i,
			(void *)(unsigned long)head,
			(void *)(unsigned long)tail);
		do {
			rec = ods_ref_as_obj(t->ods, head);
			fprintf(fp,
				"\t\t\trec_ref %p key_ref %p obj_ref %p\n",
				(void *)(unsigned long)head,
				(void *)(unsigned long)REC(rec)->key_ref,
				(void *)(unsigned long)REC(rec)->obj_ref);
			if (head == tail)
				break;
			head = REC(rec)->next_ref;
			ods_obj_put(rec);
		} while (head);
		ods_obj_put(rec);
	}
	fprintf(fp, "\n");
}


static void print_node(ods_idx_t idx, int ent, ods_obj_t n, int indent, FILE *fp)
{
	bxt_t t = idx->priv;
	int i;

	if (!n) {
		fprintf(fp, "<nil>\n");
		return;
	}

	/* Print this node */
	if (NODE(n)->is_leaf && NODE(n)->parent)
		indent += 4;
	if (NODE(n)->is_leaf)
		dump_leaf(t, idx, n, ent, indent, fp);
	else
		dump_node(t, idx, n, ent, indent, fp);
	fflush(stdout);
	if (NODE(n)->is_leaf)
		return;
	/* Now print all it's children */
	for (i = 0; i < NODE(n)->count; i++) {
		ods_obj_t node = ods_ref_as_obj(t->ods, N_ENT(n,i).node_ref);
		print_node(idx, i, node, indent + 2, fp);
		ods_obj_put(node);
	}
}

static void print_idx(ods_idx_t idx, FILE *fp)
{
	bxt_t t = idx->priv;
	ods_obj_t node = ods_ref_as_obj(t->ods, t->root_ref);
	print_node(idx, 0, node, 0, fp);
	ods_obj_put(node);
}

static int bxt_open(ods_idx_t idx)
{
	size_t udata_sz;
	ods_obj_t udata = ods_get_user_data(idx->ods);
	bxt_t t = malloc(sizeof *t);
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

static int bxt_init(ods_t ods, va_list argp)
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

static void bxt_close(ods_idx_t idx)
{
	ods_obj_t udata = ods_get_user_data(idx->ods);
	bxt_t t = idx->priv;
	UDATA(udata)->root = t->root_ref;
	ods_obj_put(udata);
}

ods_obj_t leaf_find(bxt_t t, ods_key_t key)
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
			ods_obj_t entry_key =
				ods_ref_as_obj(t->ods, N_ENT(n,i).key_ref);
			rc = t->comparator(key, entry_key);
			ods_obj_put(entry_key);
			if (rc >= 0)
				continue;
			else
				break;
		}
		ref = N_ENT(n,i-1).node_ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	return n;
}

static ods_obj_t rec_find(bxt_t t, ods_key_t key)
{
	int i;
	ods_obj_t rec;
	ods_obj_t leaf = leaf_find(t, key);
	if (!leaf)
		return NULL;
	for (i = 0; i < NODE(leaf)->count; i++) {
		int rc;
		ods_obj_t rec =
			ods_ref_as_obj(t->ods, L_ENT(leaf,i).head_ref);
		ods_key_t entry_key = ods_ref_as_obj(t->ods, REC(rec)->key_ref);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (!rc)
			goto found;
		ods_obj_put(rec);
		if (rc > 0)
			break;
	}
	rec = NULL;
 found:
	ods_obj_put(leaf);
	return rec;
}

static int bxt_find(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	int i;
	bxt_t t = idx->priv;
	ods_obj_t rec = rec_find(t, key);
	if (!rec)
		return ENOENT;
	*ref = REC(rec)->obj_ref;
	ods_obj_put(rec);
	return 0;
}

static int bxt_update(ods_idx_t idx, ods_key_t key, ods_ref_t ref)
{
	int i;
	bxt_t t = idx->priv;
	ods_obj_t rec = rec_find(t, key);
	if (!rec)
		return ENOENT;
	REC(rec)->obj_ref = ref;
	ods_obj_put(rec);
	return 0;
}

static ods_obj_t __find_lub(ods_idx_t idx, ods_key_t key)
{
	int i;
	bxt_t t = idx->priv;
	ods_obj_t leaf = leaf_find(t, key);
	ods_obj_t rec = NULL;
	if (!leaf)
		return 0;
	for (i = 0; i < NODE(leaf)->count; i++) {
		rec = ods_ref_as_obj(t->ods, L_ENT(leaf,i).head_ref);
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, REC(rec)->key_ref);
		int rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (rc <= 0)
			goto found;
		ods_obj_put(rec);
	}
	ods_obj_put(leaf);
	return NULL;
 found:
	ods_obj_put(leaf);
	return rec;
}

static int bxt_find_lub(ods_idx_t idx, ods_key_t key, ods_ref_t *obj_ref)
{
	ods_obj_t rec = __find_lub(idx, key);
	if (!rec)
		return ENOENT;
	*obj_ref = REC(rec)->obj_ref;
	ods_obj_put(rec);
	return 0;
}

static ods_obj_t __find_glb(ods_idx_t idx, ods_key_t key)
{
	int i;
	bxt_t t = idx->priv;
	ods_obj_t leaf = leaf_find(t, key);
	ods_obj_t rec = NULL;

	if (!leaf)
		goto out;

	for (i = NODE(leaf)->count - 1; i >= 0; i--) {
		rec = ods_ref_as_obj(t->ods, L_ENT(leaf,i).tail_ref);
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, REC(rec)->key_ref);
		int rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (rc < 0) {
			ods_obj_put(rec);
			continue;
		}
		goto out;
	}
	ods_obj_put(leaf);
	return NULL;
 out:
	ods_obj_put(leaf);
	return rec;
}

static int bxt_find_glb(ods_idx_t idx, ods_key_t key, ods_ref_t *obj_ref)
{
	ods_obj_t rec = __find_glb(idx, key);
	if (!rec)
		return ENOENT;
	*obj_ref = REC(rec)->obj_ref;
	ods_obj_put(rec);
	return 0;
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

static ods_obj_t node_new(ods_idx_t idx, bxt_t t)
{
	size_t sz;
	ods_obj_t obj;
	bxt_node_t n;

	sz = sizeof(struct bxt_node) + (t->order * sizeof(struct bxn_entry));
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

static ods_obj_t rec_new(ods_idx_t idx, ods_key_t key, ods_ref_t obj_ref, int is_dup)
{
	ods_obj_t obj;
	bxn_record_t rec;

	obj = ods_obj_alloc(idx->ods, sizeof(struct bxn_record));
	if (!obj) {
		ods_extend(idx->ods, ods_size(idx->ods) * 2);
		obj = ods_obj_alloc(idx->ods, sizeof(struct bxn_record));
		if (!obj)
			return NULL;
	}
	rec = obj->as.ptr;
	memset(rec, 0, sizeof(struct bxn_record));
	if (is_dup == 0) {
		/* Allocate space for the key */
		ods_key_t akey = key_new(idx, key);
		if (!akey)
			goto err_1;
		REC(obj)->key_ref = ods_obj_ref(akey);
		ods_obj_put(akey);
	}
	REC(obj)->obj_ref = obj_ref;
	return obj;
 err_1:
	ods_obj_delete(obj);
	ods_obj_put(obj);
	return NULL;
}

static struct bxn_entry ENTRY_INITIALIZER = { 0, 0 };

static int find_ref_idx(ods_obj_t node, ods_ref_t ref)
{
	int i;
	assert(!NODE(node)->is_leaf);
	for (i = 0; i < NODE(node)->count; i++) {
		if (ref == N_ENT(node,i).node_ref)
			break;
	}
	return i;
}

static int find_key_idx(bxt_t t, ods_obj_t leaf, ods_key_t key, int *found)
{
	int rc = ENOENT;
	int i;
	assert(NODE(leaf)->is_leaf);
	for (i = 0; i < NODE(leaf)->count; i++) {
		ods_obj_t rec =
			ods_ref_as_obj(t->ods, L_ENT(leaf,i).head_ref);
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, REC(rec)->key_ref);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		ods_obj_put(rec);
		if (rc <= 0)
			break;
	}
	*found = (rc == 0);
	return i;
}

int leaf_insert(bxt_t t, ods_obj_t leaf, ods_obj_t new_rec, int ent, int dup)
{
	ods_obj_t entry, next_rec;
	int j;
	if (dup) {
		assert(REC(new_rec)->key_ref == 0);
		/*
		 * Simple case. The entry must exist and leaf does not
		 * need to be expanded or have it's count updated.
		 * Simply chain this new record to the tail of the record
		 * list.
		 */
		entry = ods_ref_as_obj(t->ods, L_ENT(leaf,ent).tail_ref);
		next_rec = ods_ref_as_obj(t->ods, REC(entry)->next_ref);
		REC(new_rec)->key_ref = REC(entry)->key_ref;
		REC(new_rec)->next_ref = REC(entry)->next_ref;
		if (next_rec)
			REC(next_rec)->prev_ref = ods_obj_ref(new_rec);
		REC(new_rec)->prev_ref = ods_obj_ref(entry);
		REC(entry)->next_ref = ods_obj_ref(new_rec);
		L_ENT(leaf,ent).tail_ref = ods_obj_ref(new_rec);
		ods_obj_put(entry);
		ods_obj_put(next_rec);
		return ent;
	}
	entry = ods_ref_as_obj(t->ods, L_ENT(leaf,ent).head_ref);
	if (!entry && ent) {
		/*
		 * This record is being added to the end of the leaf
		 * and the leaf is !empty.
		 */
		entry = ods_ref_as_obj(t->ods, L_ENT(leaf, ent-1).tail_ref);
		assert(entry);
	}
	if (entry) {
		/* Chain this new record to the record list */
		next_rec = ods_ref_as_obj(t->ods, REC(entry)->next_ref);
		REC(new_rec)->next_ref = REC(entry)->next_ref;
		if (next_rec)
			REC(next_rec)->prev_ref = ods_obj_ref(new_rec);
		REC(new_rec)->prev_ref = ods_obj_ref(entry);
		REC(entry)->next_ref = ods_obj_ref(new_rec);
		ods_obj_put(next_rec);
	}
	/* If necessary, move up trailing entries to make space. */
	for (j = NODE(leaf)->count; j > ent; j--)
		NODE(leaf)->entries[j] = NODE(leaf)->entries[j-1];
	NODE(leaf)->count++;
	L_ENT(leaf,ent).head_ref = ods_obj_ref(new_rec);
	L_ENT(leaf,ent).tail_ref = ods_obj_ref(new_rec);
	ods_obj_put(entry);
	return ent;
}

static int split_midpoint(int order)
{
	if (order & 1)
		return (order >> 1) + 1;
	return order >> 1;
}

static ods_obj_t leaf_split_insert(ods_idx_t idx, bxt_t t, ods_obj_t left,
				   ods_obj_t new_key, ods_obj_t new_rec,
				   int ins_idx)
{
	ods_obj_t right;
	int i, j;
	int ins_left_n_right;
	int midpoint = split_midpoint(t->order);

	right = node_new(idx, t);
	if (!right)
		return NULL;
	assert(NODE(left)->is_leaf);
	NODE(right)->is_leaf = 1;
	NODE(right)->parent = NODE(left)->parent;

	/* Insert the new record in the list */
	if (ins_idx < NODE(left)->count) {
		ods_obj_t next_rec = ods_ref_as_obj(t->ods, L_ENT(left,ins_idx).head_ref);
		assert(next_rec);
		REC(new_rec)->next_ref = ods_obj_ref(next_rec);
		REC(new_rec)->prev_ref = REC(next_rec)->prev_ref;
		ods_obj_t prev_rec = ods_ref_as_obj(t->ods, REC(next_rec)->prev_ref);
		if (prev_rec)
			REC(prev_rec)->next_ref = ods_obj_ref(new_rec);
		REC(next_rec)->prev_ref = ods_obj_ref(new_rec);
		ods_obj_put(next_rec);
		ods_obj_put(prev_rec);
	} else {
		ods_obj_t prev_rec = ods_ref_as_obj(t->ods, L_ENT(left,ins_idx-1).tail_ref);
		assert(prev_rec);
		REC(new_rec)->prev_ref = ods_obj_ref(prev_rec);
		REC(new_rec)->next_ref = REC(prev_rec)->next_ref;
		ods_obj_t next_rec = ods_ref_as_obj(t->ods, REC(prev_rec)->next_ref);
		if (next_rec)
			REC(next_rec)->prev_ref = ods_obj_ref(new_rec);
		REC(prev_rec)->next_ref = ods_obj_ref(new_rec);
		ods_obj_put(next_rec);
		ods_obj_put(prev_rec);
	}
	/* Now put the new record in the appropriate leaf */
	ins_left_n_right = ins_idx < midpoint;
	if (ins_left_n_right) {
		/* Move entries to the right node to make room for the new record */
		for (i = midpoint - 1, j = 0; i < t->order; i++, j++) {
			NODE(right)->entries[j] = NODE(left)->entries[i];
			NODE(left)->count--;
			NODE(right)->count++;
		}
		/*
		 * Move left's entries between the insertion point and
		 * the end one slot to the right.
		 */
		for (i = midpoint - 1; i > ins_idx; i--)
			NODE(left)->entries[i] = NODE(left)->entries[i-1];

		/*
		 * Put the new record in the leaf.
		 */
		L_ENT(left, ins_idx).head_ref = ods_obj_ref(new_rec);
		L_ENT(left, ins_idx).tail_ref = ods_obj_ref(new_rec);
		NODE(left)->count++;
	} else {
		/*
		 * New entry goes in the right node. This means that
		 * as we move the entries from left to right, we need
		 * to leave space for the item that will be added.
		 */
		ins_idx = ins_idx - midpoint;
		for (i = midpoint, j = 0; i < t->order; i++, j++) {
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
		L_ENT(right, ins_idx).head_ref = ods_obj_ref(new_rec);
		L_ENT(right, ins_idx).tail_ref = ods_obj_ref(new_rec);
		NODE(right)->count++;
	}
	return right;
}

static void node_insert(bxt_t t, ods_obj_t node, ods_obj_t left,
			ods_ref_t key_ref, ods_obj_t right)
{
	int i, j;
	ods_ref_t left_ref = ods_obj_ref(left);

	assert(!NODE(node)->is_leaf);
	/* Find left's index */
	i = find_ref_idx(node, left_ref);
	assert(i < NODE(node)->count);

	/*
	 * Make room for right after left's current key/ref and the
	 * end of the node
	 */
	for (j = NODE(node)->count; j > i+1; j--)
		NODE(node)->entries[j] = NODE(node)->entries[j-1];

	/* Put in the new entry and update the count */
	N_ENT(node,i+1).key_ref = key_ref;
	N_ENT(node,i+1).node_ref = ods_obj_ref(right);
	NODE(node)->count++;
	NODE(left)->parent = NODE(right)->parent = ods_obj_ref(node);
}

static ods_obj_t node_split_insert(ods_idx_t idx, bxt_t t,
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
		if (ods_obj_ref(left_node) == N_ENT(left_parent,i).node_ref)
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
				ods_ref_as_obj(t->ods, N_ENT(left_parent,i).node_ref);
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
		N_ENT(left_parent,ins_idx).node_ref = ods_obj_ref(right_node);
		N_ENT(left_parent,ins_idx).key_ref = right_key_ref;
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
				ods_ref_as_obj(t->ods, N_ENT(left_parent,i).node_ref);
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
		N_ENT(right_parent,ins_idx).node_ref = ods_obj_ref(right_node);
		N_ENT(right_parent,ins_idx).key_ref = right_key_ref;
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
		N_ENT(next_parent,0).node_ref = ods_obj_ref(left_parent);
		N_ENT(next_parent,0).key_ref = N_ENT(left_parent,0).key_ref;
		N_ENT(next_parent,1).node_ref = ods_obj_ref(right_parent);
		N_ENT(next_parent,1).key_ref = N_ENT(right_parent,0).key_ref;
		NODE(left_parent)->parent = ods_obj_ref(next_parent);
		NODE(right_parent)->parent = ods_obj_ref(next_parent);
		t->root_ref = ods_obj_ref(next_parent);
		goto out;
	}
	/* If there is room, insert into the parent */
	if (NODE(next_parent)->count < t->order) {
		node_insert(t, next_parent, left_parent,
			    N_ENT(right_parent,0).key_ref, right_parent);
		goto out;
	}
	ods_obj_put(next_parent);
	ods_obj_put(left_node);
	ods_obj_put(right_node);
	/* Go up to the next level and split and insert */
	left_node = left_parent;
	right_node = right_parent;
	right_key_ref = N_ENT(right_parent,0).key_ref;
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

static int bxt_insert(ods_idx_t idx, ods_key_t new_key, ods_ref_t obj_ref)
{
	bxt_t t = idx->priv;
	ods_obj_t parent;
	ods_obj_t leaf;
	ods_obj_t udata = ods_get_user_data(idx->ods);
	ods_obj_t new_rec;

	if (!t->root_ref) {
		leaf = node_new(idx, t);
		if (!leaf)
			goto err_1;
		t->root_ref = ods_obj_ref(leaf);
		UDATA(udata)->root = t->root_ref;
		NODE(leaf)->is_leaf = 1;
	} else
		leaf = leaf_find(t, new_key);

	/*
	 * If this is a duplicate key, then the insertion logic reduces to
	 * a list insert in the leaf
	 */
	int is_dup;
	int ent = find_key_idx(t, leaf, new_key, &is_dup);

	/* Allocate a record object */
	new_rec = rec_new(idx, new_key, obj_ref, is_dup);
	if (!new_rec)
		goto err_1;

	if ((NODE(leaf)->count < t->order) || is_dup) {
		if (!leaf_insert(t, leaf, new_rec, ent, is_dup)
		    && NODE(leaf)->parent) {
			ods_obj_t parent =
				ods_ref_as_obj(t->ods, NODE(leaf)->parent);
			/* Maintain this to simplify other logic */
			if (N_ENT(parent,0).node_ref == ods_obj_ref(leaf))
				N_ENT(parent,0).key_ref = N_ENT(leaf,0).key_ref;
			ods_obj_put(parent);
		}
		ods_obj_put(leaf);
		ods_obj_put(udata);
		ods_obj_put(new_rec);
		return 0;
	}
	ods_obj_t new_leaf =
		leaf_split_insert(idx, t, leaf, new_key, new_rec, ent);
	if (!new_leaf)
		goto err_1;

	parent = ods_ref_as_obj(t->ods, NODE(leaf)->parent);
	ods_obj_t leaf_rec = ods_ref_as_obj(t->ods, L_ENT(leaf,0).head_ref);
	ods_obj_t new_leaf_rec = ods_ref_as_obj(t->ods, L_ENT(new_leaf,0).head_ref);
	ods_ref_t leaf_key_ref = REC(leaf_rec)->key_ref;
	ods_ref_t new_leaf_key_ref = REC(new_leaf_rec)->key_ref;
	ods_obj_put(leaf_rec);
	ods_obj_put(new_leaf_rec);

	if (!parent) {
		parent = node_new(idx, t);
		if (!parent)
			goto err_2;

		N_ENT(parent,0).key_ref = leaf_key_ref;
		N_ENT(parent,0).node_ref = ods_obj_ref(leaf);

		N_ENT(parent,1).key_ref = new_leaf_key_ref;
		N_ENT(parent,1).node_ref = ods_obj_ref(new_leaf);

		NODE(parent)->count = 2;

		NODE(leaf)->parent = ods_obj_ref(parent);
		NODE(new_leaf)->parent = ods_obj_ref(parent);
		t->root_ref = ods_obj_ref(parent);
		goto out;
	}
	if (NODE(parent)->count < t->order) {
		node_insert(t, parent, leaf, new_leaf_key_ref, new_leaf);
		goto out;
	}
	ods_obj_put(parent);
	parent = node_split_insert(idx, t, leaf, new_leaf_key_ref, new_leaf);
	if (!parent)
		goto err_3;
 out:
	UDATA(udata)->root = t->root_ref;
	ods_obj_put(leaf);
	ods_obj_put(new_leaf);
	ods_obj_put(udata);
	ods_obj_put(parent);
	ods_obj_put(new_rec);
	return 0;

 err_4:
	/* TODO: Unsplit the leaf and put the tree back in order. */
 err_3:
	ods_obj_delete(new_leaf);
	ods_obj_delete(new_rec);
	ods_obj_put(new_rec);
 err_2:
	ods_obj_put(new_leaf);
 err_1:
	ods_obj_put(udata);
	return ENOMEM;
}

ods_obj_t bxt_min_rec(bxt_t t)
{
	ods_obj_t n;
	ods_obj_t rec;

	if (!t->root_ref)
		return 0;

	/* Walk to the left most leaf and return the 0-th entry  */
	n = ods_ref_as_obj(t->ods, t->root_ref);
	while (!NODE(n)->is_leaf) {
		ods_ref_t ref = N_ENT(n,0).node_ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	rec = ods_ref_as_obj(t->ods, L_ENT(n,0).head_ref);
	ods_obj_put(n);
	return rec;
}

ods_obj_t bxt_max_rec(bxt_t t)
{
	ods_obj_t n;
	ods_obj_t rec;

	if (!t->root_ref)
		return 0;

	/* Walk to the left most leaf and return the 0-th entry  */
	n = ods_ref_as_obj(t->ods, t->root_ref);
	while (!NODE(n)->is_leaf) {
		ods_ref_t ref = N_ENT(n,NODE(n)->count-1).node_ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	rec = ods_ref_as_obj(t->ods, L_ENT(n,NODE(n)->count-1).tail_ref);
	ods_obj_put(n);
	return rec;
}

static ods_obj_t entry_find(bxt_t t, ods_obj_t node,
			    ods_key_t key, int *idx, ods_ref_t *ref)
{
	int i;
	ods_obj_t rec = NULL;
	ods_key_t entry_key;
	assert(NODE(node)->is_leaf);
	for (i = 0; i < NODE(node)->count; i++) {
		int rc;
		rec = ods_ref_as_obj(t->ods, L_ENT(node,i).head_ref);
		entry_key = ods_ref_as_obj(t->ods, REC(rec)->key_ref);
		assert(entry_key);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (!rc)
			break;
		ods_obj_put(rec);
	}
	if (i < NODE(node)->count)
		return rec;
	return NULL;
}

static ods_obj_t leaf_right_most(bxt_t t, ods_obj_t node)
{
	while (!NODE(node)->is_leaf) {
		node = ods_ref_as_obj(t->ods,
				      N_ENT(node,NODE(node)->count-1).node_ref);
		if (!NODE(node)->is_leaf)
			ods_obj_put(node);
	}
	return node;
}

static int node_neigh(bxt_t t, ods_obj_t node, ods_obj_t *left, ods_obj_t *right)
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
	idx = find_ref_idx(parent, node_ref);
	assert(idx < NODE(parent)->count);

	if (idx)
		*left = ods_ref_as_obj(t->ods, N_ENT(parent,idx-1).node_ref);

	if (idx < NODE(parent)->count-1)
		*right = ods_ref_as_obj(t->ods, N_ENT(parent,idx+1).node_ref);

	ods_obj_put(parent);
	return idx;
}

static int space(bxt_t t, ods_obj_t n)
{
	if (!n)
		return 0;

	return t->order - NODE(n)->count;
}

static int combine_right(bxt_t t, ods_obj_t right, int idx, ods_obj_t node)
{
	int i, j;
	int count = NODE(node)->count - idx;
	ods_ref_t right_ref;
	ods_obj_t entry;

	if (!right || !count)
		return idx;

	/* Make room to the left */
	for (i = NODE(right)->count + count - 1; i > idx + count - 1; i--)
		NODE(right)->entries[i] = NODE(right)->entries[i-count];

	right_ref = ods_obj_ref(right);
	for (i = 0, j = idx; j < NODE(node)->count; i++, j++) {
		/* Update the entry's parent */
		if (!NODE(node)->is_leaf) {
			entry = ods_ref_as_obj(t->ods, N_ENT(node,j).node_ref);
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

static int combine_left(bxt_t t, ods_obj_t left, ods_obj_t node)
{
	int i, j;
	int count = NODE(node)->count;
	ods_obj_t entry;
	ods_ref_t left_ref;

	if (!left)
		return 0;

	left_ref = ods_obj_ref(left);
	for (i = NODE(left)->count, j = 0; j < count; i++, j++) {
		/* Update the entry's parent */
		if (!NODE(node)->is_leaf) {
			entry = ods_ref_as_obj(t->ods, N_ENT(node,j).node_ref);
			NODE(entry)->parent = left_ref;
			ods_obj_put(entry);
		}
		/* Move the entry to the left sibling */
		NODE(left)->entries[i] = NODE(node)->entries[j];
		NODE(left)->count++;
	}
	return j;
}

/*
 * Drops reference on parent and node
 */
static ods_ref_t fixup_parents(bxt_t t, ods_obj_t parent, ods_obj_t node)
{
	int i;
	ods_ref_t parent_ref;
	ods_ref_t node_ref;
	parent_ref = ods_obj_ref(parent);
	assert(!NODE(parent)->is_leaf);
	ods_obj_put(parent);
	while (parent_ref) {
		parent = ods_ref_as_obj(t->ods, parent_ref);
		i = find_ref_idx(parent, ods_obj_ref(node));
		assert(i < NODE(parent)->count);
		if (!NODE(node)->is_leaf)
			N_ENT(parent,i).key_ref = N_ENT(node,0).key_ref;
		else {
			ods_obj_t rec = ods_ref_as_obj(t->ods, L_ENT(node,0).head_ref);
			N_ENT(parent,i).key_ref = REC(rec)->key_ref;
			ods_obj_put(rec);
		}
		ods_obj_put(node);
		node = parent;
		parent_ref = NODE(parent)->parent;
	}
	ods_obj_put(node);
	return t->root_ref;
}

static void merge_from_left(bxt_t t, ods_obj_t left, ods_obj_t node, int midpoint)
{
	int count;
	int i, j;
	ods_ref_t node_ref = ods_obj_ref(node);

	assert(NODE(left)->count > midpoint);
	count = NODE(left)->count - midpoint;

	/* Make room in node */
	for (i = NODE(node)->count + count - 1, j = 0; j < NODE(node)->count; j++, i--)
		NODE(node)->entries[i] = NODE(node)->entries[i-count];

	/* Move count entries the left to node */
	for (i = 0, j = NODE(left)->count - count; i < count; i++, j++) {
		if (!NODE(left)->is_leaf) {
			ods_obj_t entry =
				ods_ref_as_obj(t->ods, N_ENT(left,j).node_ref);
			NODE(entry)->parent = node_ref;
			ods_obj_put(entry);
		}
		NODE(node)->entries[i] = NODE(left)->entries[j];
		NODE(left)->entries[j] = ENTRY_INITIALIZER;
		NODE(left)->count--;
		NODE(node)->count++;
	}
}

static void merge_from_right(bxt_t t, ods_obj_t right, ods_obj_t node, int midpoint)
{
	int count;
	int i, j;
	ods_obj_t parent;
	ods_ref_t node_ref = ods_obj_ref(node);

	assert(NODE(right)->count > midpoint);
	count = NODE(right)->count - midpoint;
	for (i = NODE(node)->count, j = 0; j < count; i++, j++) {
		if (!NODE(right)->is_leaf) {
			ods_obj_t entry =
				ods_ref_as_obj(t->ods, N_ENT(right,j).node_ref);
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

static ods_ref_t entry_delete(bxt_t t, ods_obj_t node, ods_obj_t rec, int ent)
{
	int i, midpoint;
	ods_obj_t left, right;
	ods_obj_t parent;
	int node_idx;
	int count;

	assert(NODE(node)->is_leaf);
 next_level:
	parent = ods_ref_as_obj(t->ods, NODE(node)->parent);
	/* Remove the record and object from the node */
	for (i = ent; i < NODE(node)->count - 1; i++)
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
				root_ref = N_ENT(node,0).node_ref;
				ods_obj_put(parent);
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
	if (NODE(node)->count >= midpoint) {
		/* Unless the 0-the element is modified, no fixup is required. */
		if (ent == 0)
			return fixup_parents(t, parent, node);
		ods_obj_put(parent);
		ods_obj_put(node);
		return t->root_ref;
	}
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
	/*
	 * Combine as many as possible to the left
	 * NB: Combining left will never modify the 0-the element,
	 * thefore the parent does not require updating
	 */
	ent = combine_left(t, left, node);

	/* Move the remainder to the right */
	if (ent < NODE(node)->count) {
		/*
		 * Combining right will always modify the 0-th
		 * element, therefore a parent fixup is required.
		 */
		combine_right(t, right, ent, node);
		if (NODE(right)->is_leaf) {
			ods_obj_t rec = ods_ref_as_obj(t->ods, L_ENT(right,0).head_ref);
			N_ENT(parent,node_idx+1).key_ref = REC(rec)->key_ref;
			N_ENT(parent,node_idx+1).node_ref = ods_obj_ref(right);
			ods_obj_put(rec);
		} else {
			N_ENT(parent,node_idx+1).key_ref = N_ENT(right,0).key_ref;
			N_ENT(parent,node_idx+1).node_ref = ods_obj_ref(right);
		}
	}
	/* Remove the node(idx) from the parent. */
	ent = node_idx;
	ods_obj_delete(node);
	ods_obj_put(node);
	ods_obj_put(left);
	ods_obj_put(right);
	node = parent;
	goto next_level;
}

static void delete_head(bxt_t t, ods_obj_t leaf, ods_obj_t rec, int ent)
{
	ods_obj_t next_rec = ods_ref_as_obj(t->ods, REC(rec)->next_ref);
	ods_obj_t prev_rec = ods_ref_as_obj(t->ods, REC(rec)->prev_ref);
	if (prev_rec)
		REC(prev_rec)->next_ref = REC(rec)->next_ref;
	if (next_rec)
		REC(next_rec)->prev_ref = REC(rec)->prev_ref;
	L_ENT(leaf,ent).head_ref = REC(rec)->next_ref;
	ods_obj_put(next_rec);
	ods_obj_put(prev_rec);
	ods_obj_delete(rec);
	ods_obj_put(rec);
}

static int bxt_delete(ods_idx_t idx, ods_key_t key, ods_ref_t *ref)
{
	bxt_t t = idx->priv;
	ods_obj_t udata = ods_get_user_data(idx->ods);
	int ent;
	ods_obj_t leaf, rec;
	int found;

	leaf = leaf_find(t, key);
	if (!leaf)
		goto noent;
	ent = find_key_idx(t, leaf, key, &found);
	if (!found)
		goto noent;
	rec = ods_ref_as_obj(t->ods, L_ENT(leaf, ent).head_ref);
	assert(rec);
	*ref = REC(rec)->obj_ref;

	/*
	 * Trivial case is that this is a dup key. In this case,
	 * delete the first entry on the list and return
	 */
	if (L_ENT(leaf, ent).head_ref != L_ENT(leaf, ent).tail_ref) {
		delete_head(t, leaf, rec, ent);
		ods_obj_put(leaf);
		ods_obj_put(udata);
		return 0;
	}
	t->root_ref = entry_delete(t, leaf, rec, ent);
	ods_obj_delete(key);
	UDATA(udata)->root = t->root_ref;
	ods_obj_put(rec);
	ods_obj_put(udata);
	return 0;
 noent:
	ods_obj_put(leaf);
	ods_obj_put(udata);
	return ENOENT;
}

static ods_iter_t bxt_iter_new(ods_idx_t idx)
{
	bxt_iter_t iter = calloc(1, sizeof *iter);
	iter->idx = idx;
	return (struct ods_iter *)iter;
}

static void bxt_iter_delete(ods_iter_t i)
{
	free(i);
}

static int bxt_iter_begin(ods_iter_t oi)
{
	ods_obj_t node;
	bxt_iter_t i = (bxt_iter_t)oi;
	bxt_t t = i->idx->priv;
	if (i->rec)
		ods_obj_put(i->rec);
	i->rec = bxt_min_rec(t);
	if (i->rec)
		return 0;
	return ENOENT;
}

static int bxt_iter_end(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	bxt_t t = i->idx->priv;
	if (i->rec)
		ods_obj_put(i->rec);
	i->rec = bxt_max_rec(t);
	if (i->rec)
		return 0;
	return ENOENT;
}

static ods_key_t bxt_iter_key(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	bxt_t t = i->idx->priv;
	if (!i->rec)
		return NULL;
	return ods_ref_as_obj(t->ods, REC(i->rec)->key_ref);
}

static ods_ref_t bxt_iter_ref(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	if (!i->rec)
		return 0;
	return REC(i->rec)->obj_ref;
}

static int bxt_iter_find(ods_iter_t oi, ods_key_t key)
{
	bxt_iter_t iter = (bxt_iter_t)oi;
	bxt_t t = iter->idx->priv;
	ods_obj_t leaf = leaf_find(t, key);
	int found;
	int i;

	if (!leaf)
		return ENOENT;

	assert(NODE(leaf)->is_leaf);
	i = find_key_idx(t, leaf, key, &found);
	if (!found) {
		ods_obj_put(leaf);
		return ENOENT;
	}
	iter->rec = ods_ref_as_obj(t->ods, L_ENT(leaf,i).head_ref);
	return 0;
}

static int bxt_iter_find_lub(ods_iter_t oi, ods_key_t key)
{
	bxt_iter_t iter = (bxt_iter_t)oi;
	iter->rec = __find_lub(iter->idx, key);
	if (iter->rec)
		return 0;
	return ENOENT;
}

static int bxt_iter_find_glb(ods_iter_t oi, ods_key_t key)
{
	bxt_iter_t iter = (bxt_iter_t)oi;
	iter->rec = __find_glb(iter->idx, key);
	if (iter->rec)
		return 0;
	return ENOENT;
}

static int bxt_iter_next(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	bxt_t t = i->idx->priv;
	ods_obj_t next_rec;

	if (!i->rec)
		goto not_found;

	next_rec = ods_ref_as_obj(t->ods, REC(i->rec)->next_ref);
	ods_obj_put(i->rec);
	i->rec = next_rec;
	if (!i->rec)
		goto not_found;
	return 0;
 not_found:
	return ENOENT;
}

static int bxt_iter_prev(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	bxt_t t = i->idx->priv;
	ods_obj_t prev_rec;

	if (!i->rec)
		goto not_found;

	prev_rec = ods_ref_as_obj(t->ods, REC(i->rec)->prev_ref);
	ods_obj_put(i->rec);
	i->rec = prev_rec;
	if (!i->rec)
		goto not_found;
	return 0;
 not_found:
	return ENOENT;
}

static const char *bxt_get_type(void)
{
	return "BXTREE";
}

static void bxt_commit(ods_idx_t idx)
{
	ods_commit(idx->ods, ODS_COMMIT_SYNC);
}

static struct ods_idx_provider bxt_provider = {
	.get_type = bxt_get_type,
	.init = bxt_init,
	.open = bxt_open,
	.close = bxt_close,
	.commit = bxt_commit,
	.insert = bxt_insert,
	.update = bxt_update,
	.delete = bxt_delete,
	.find = bxt_find,
	.find_lub = bxt_find_lub,
	.find_glb = bxt_find_glb,
	.iter_new = bxt_iter_new,
	.iter_delete = bxt_iter_delete,
	.iter_find = bxt_iter_find,
	.iter_find_lub = bxt_iter_find_lub,
	.iter_find_glb = bxt_iter_find_glb,
	.iter_begin = bxt_iter_begin,
	.iter_end = bxt_iter_end,
	.iter_next = bxt_iter_next,
	.iter_prev = bxt_iter_prev,
	.iter_key = bxt_iter_key,
	.iter_ref = bxt_iter_ref,
	.print_idx = print_idx
};

struct ods_idx_provider *get(void)
{
	return &bxt_provider;
}

