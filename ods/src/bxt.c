/*
 * Copyright (c) 2014-2015 Open Grid Computing, Inc. All rights reserved.
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
#include <ods/ods.h>
#include "bxt.h"

/* #define BXT_DEBUG */
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

static struct bxt_obj_el *alloc_el(bxt_t t);
ods_atomic_t el_count;
static void free_el(bxt_t t, struct bxt_obj_el *el);
static int node_neigh(bxt_t t, ods_obj_t node, ods_obj_t *left, ods_obj_t *right);

void dump_node(bxt_t t, ods_idx_t idx, ods_obj_t n, int ent, int indent, FILE *fp)
{
	int i;
	if (fp) fprintf(fp, "%p - %*s%s[%d] | %p : ", (void *)(unsigned long)NODE(n)->parent,
	       indent, "", "NODE", ent, n->as.ptr);
	for (i = 0; i < NODE(n)->count; i++) {
		char *keystr = malloc(ods_idx_key_str_size(idx));
		ods_key_t key = ods_ref_as_obj(t->ods, N_ENT(n,i).key_ref);
		if (key) {
			ods_key_to_str(idx, key, keystr);
		} else {
			keystr[0] = '-';
			keystr[1] = '\0';
		}
		if (fp) fprintf(fp, "%s:%p, ",
				keystr,
		       (void *)(unsigned long)N_ENT(n, i).node_ref);
		free(keystr);
		ods_obj_put(key);
	}
	if (fp) fprintf(fp, "\n");
}

void dump_leaf(bxt_t t, ods_idx_t idx, ods_obj_t n, int ent, int indent, FILE *fp)
{
	int i;
	char *keystr = malloc(ods_idx_key_str_size(idx));
	if (fp) fprintf(fp, "%p - %*s%s[%d] | %p :\n", (void *)(unsigned long)NODE(n)->parent,
	       indent-2, "", "LEAF", ent, n->as.ptr);
	for (i = 0; i < NODE(n)->count; i++) {
		ods_obj_t rec;
		ods_key_t key;
		ods_ref_t tail = L_ENT(n,i).tail_ref;
		ods_ref_t head = L_ENT(n,i).head_ref;
		if (fp) fprintf(fp, "%*sENTRY#%2d : head %p tail %p\n", indent+11, "", i,
			(void *)(unsigned long)head,
			(void *)(unsigned long)tail);
		do {
			rec = ods_ref_as_obj(t->ods, head);
			key = ods_ref_as_obj(t->ods, REC(rec)->key_ref);
			if (key) {
				ods_key_to_str(idx, key, keystr);
			} else {
				keystr[0] = '-';
				keystr[1] = '\0';
			}
			if (fp) fprintf(fp,
				"%*srec_ref %p key %s ref %p user %p prev_ref %p next_ref %p\n",
				indent+16, "",
				(void *)(unsigned long)head,
				keystr,
				(void *)*(unsigned long *)&REC(rec)->value.bytes[0],
				(void *)*(unsigned long *)&REC(rec)->value.bytes[8],
				(void *)(unsigned long)REC(rec)->prev_ref,
				(void *)(unsigned long)REC(rec)->next_ref);
			if (head == tail)
				break;
			head = REC(rec)->next_ref;
			ods_obj_put(rec);
		} while (head);
		ods_obj_put(rec);
	}
	free(keystr);
	if (fp) fprintf(fp, "\n");
}


static void print_node(ods_idx_t idx, int ent, ods_obj_t n, int indent, FILE *fp)
{
	bxt_t t = idx->priv;
	int i;

	if (!n) {
		if (fp) fprintf(fp, "<nil>\n");
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
	ods_obj_t node = ods_ref_as_obj(t->ods, t->udata->root_ref);
	print_node(idx, 0, node, 0, fp);
	ods_obj_put(node);
}

static void print_info(ods_idx_t idx, FILE *fp)
{
	bxt_t t = idx->priv;
	fprintf(fp, "%*s : %d\n", 12, "Order", t->udata->order);
	fprintf(fp, "%*s : %lx\n", 12, "Root Ref", t->udata->root_ref);
	fprintf(fp, "%*s : %d\n", 12, "Client Count", t->udata->client_count);
	fprintf(fp, "%*s : %d\n", 12, "Lock", t->udata->lock);
	fprintf(fp, "%*s : %d\n", 12, "Depth", t->udata->depth);
	fprintf(fp, "%*s : %d\n", 12, "Cardinality", t->udata->card);
	fprintf(fp, "%*s : %d\n", 12, "Duplicates", t->udata->dups);
	fflush(fp);
}

#ifdef BXT_DEBUG
static void verify_node(bxt_t t, ods_obj_t node)
{
	int i, j;
	ods_obj_t rec;
	ods_ref_t rec_ref;
	if (!node)
	    return;
	assert(NODE(node)->count <= t->udata->order);
	for (i = 0; i < NODE(node)->count; i++) {
		if (NODE(node)->is_leaf) {
			for (j = i + 1; j < NODE(node)->count; j++) {
				assert(L_ENT(node,i).head_ref != L_ENT(node,j).head_ref);
				assert(L_ENT(node,i).tail_ref != L_ENT(node,j).tail_ref);
			}
			assert(L_ENT(node,i).head_ref && L_ENT(node,i).tail_ref);
			rec_ref = L_ENT(node, i).head_ref;
			rec = ods_ref_as_obj(t->ods, rec_ref);
			assert(REC(rec)->next_ref != rec_ref);
			assert(REC(rec)->prev_ref != rec_ref);
			ods_obj_put(rec);
		} else {
			assert(N_ENT(node,i).node_ref);
			if (i)
				assert(N_ENT(node,i).node_ref != N_ENT(node,i-1).node_ref);
		}
	}
}
#endif

static int bxt_open(ods_idx_t idx)
{
	ods_obj_t udata = ods_get_user_data(idx->ods);
	bxt_t t = calloc(1, sizeof *t);
	if (!t) {
		ods_obj_put(udata);
		return ENOMEM;
	}
	t->udata_obj = udata;
	t->udata = UDATA(udata);
	ods_spin_init(&t->lock, &t->udata->lock);
	t->ods = idx->ods;
	t->comparator = idx->idx_class->cmp->compare_fn;
	idx->priv = t;
	ods_atomic_inc(&t->udata->client_count);
	return 0;
}

static int bxt_init(ods_t ods, const char *argp)
{
	char order_arg[ODS_IDX_ARGS_LEN];
	ods_obj_t udata = ods_get_user_data(ods);
	char *name;
	int order = 0;

	if (argp) {
		strcpy(order_arg, argp);
		name = strtok(order_arg, "=");
		if (strcasecmp(name, "ORDER"))
			order = strtoul(strtok(NULL, "="), NULL, 0);
	}
	if (order <= 0)
		order = 5;

	UDATA(udata)->order = order;
	UDATA(udata)->root_ref = 0;
	UDATA(udata)->client_count = 0;
	UDATA(udata)->lock = 0;
	UDATA(udata)->depth = 0;
	UDATA(udata)->card = 0;
	ods_obj_put(udata);
	return 0;
}

static void bxt_close_(bxt_t t)
{
	struct bxt_obj_el *el;
	ods_obj_t node;

	ods_atomic_dec(&t->udata->client_count);
	ods_obj_put(t->udata_obj);

	/* Clean up any cached node allocations */
	while (!LIST_EMPTY(&t->node_q)) {
		el = LIST_FIRST(&t->node_q);
		LIST_REMOVE(el, entry);
		node = el->obj;
		ods_obj_delete(node);
		ods_obj_put(node);
		free_el(t, el);
	}

	while (!LIST_EMPTY(&t->el_q)) {
		el = LIST_FIRST(&t->el_q);
		LIST_REMOVE(el, entry);
		free(el);
	}

	free(t);
}

static void bxt_close(ods_idx_t idx)
{
	bxt_t t = idx->priv;
	assert(t);
	idx->priv = NULL;
	bxt_close_(t);
}

ods_obj_t leaf_find(bxt_t t, ods_key_t key)
{
	ods_ref_t ref = t->udata->root_ref;
	ods_obj_t n;
	int i;

	if (!t->udata->root_ref)
		return 0;

	n = ods_ref_as_obj(t->ods, t->udata->root_ref);
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

static ods_obj_t rec_find(bxt_t t, ods_key_t key, int first)
{
	int i;
	int rc;
	ods_obj_t rec;
	ods_key_t entry_key;
	ods_obj_t leaf = leaf_find(t, key);
	if (!leaf)
		return NULL;
	for (i = 0; i < NODE(leaf)->count; i++) {
		ods_ref_t ref;
		if (first)
			ref = L_ENT(leaf,i).head_ref;
		else
			ref = L_ENT(leaf,i).tail_ref;
		rec = ods_ref_as_obj(t->ods, ref);
		entry_key = ods_ref_as_obj(t->ods, REC(rec)->key_ref);
		rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (!rc)
			goto found;
		ods_obj_put(rec);
		if (rc < 0)
			break;
	}
	rec = NULL;
 found:
	ods_obj_put(leaf);
	return rec;
}

static int bxt_find(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	bxt_t t = idx->priv;
	ods_obj_t rec = rec_find(t, key, 1);
	if (!rec)
		return ENOENT;
	*data = REC(rec)->value;
	ods_obj_put(rec);
	return 0;
}

static int bxt_update(ods_idx_t idx, ods_key_t key, ods_idx_data_t data)
{
	bxt_t t = idx->priv;
	ods_obj_t rec = rec_find(t, key, 1);
	if (!rec)
		return ENOENT;
	REC(rec)->value = data;
	ods_obj_put(rec);
	return 0;
}

static ods_obj_t __find_lub(ods_idx_t idx, ods_key_t key,
			    ods_iter_flags_t flags)
{
	int i;
	ods_ref_t head_ref = 0;
	ods_ref_t tail_ref = 0;
	bxt_t t = idx->priv;
	ods_obj_t leaf = leaf_find(t, key);
	ods_obj_t rec = NULL;
	if (!leaf)
		return 0;
	for (i = 0; i < NODE(leaf)->count; i++) {
		tail_ref = L_ENT(leaf,i).tail_ref;
		head_ref = L_ENT(leaf,i).head_ref;
		if (rec)
			ods_obj_put(rec);
		rec = ods_ref_as_obj(t->ods, head_ref);
		ods_key_t entry_key =
			ods_ref_as_obj(t->ods, REC(rec)->key_ref);
		int rc = t->comparator(key, entry_key);
		ods_obj_put(entry_key);
		if (rc <= 0)
			goto found;
	}
	/* Our LUB is the first record in the right sibling */
	ods_obj_put(leaf);
	if (tail_ref != head_ref) {
		ods_obj_put(rec);
		rec = ods_ref_as_obj(t->ods, tail_ref);
		assert(rec);
	}
	ods_ref_t next_ref = REC(rec)->next_ref;
	ods_obj_put(rec);
	rec = ods_ref_as_obj(t->ods, next_ref);
	return rec;
 found:
	if (flags & ODS_ITER_F_UNIQUE) {
		ods_obj_put(rec);
		return leaf;
	} else {
		ods_obj_put(leaf);
		return rec;
	}
}

static int bxt_find_lub(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	ods_obj_t rec = __find_lub(idx, key, 0);
	if (!rec)
		return ENOENT;
	*data = REC(rec)->value;
	ods_obj_put(rec);
	return 0;
}

static ods_obj_t __find_glb(ods_idx_t idx, ods_key_t key,
			    ods_iter_flags_t flags)
{
	int i;
	bxt_t t = idx->priv;
	ods_obj_t leaf = leaf_find(t, key);
	ods_obj_t rec = NULL;

	if (!leaf)
		goto out;

	for (i = NODE(leaf)->count - 1; i >= 0; i--) {
		rec = ods_ref_as_obj(t->ods, L_ENT(leaf,i).head_ref);
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
	if (flags & ODS_ITER_F_UNIQUE) {
		ods_obj_put(rec);
		return leaf;
	} else {
		ods_obj_put(leaf);
		return rec;
	}
}

static int bxt_find_glb(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	ods_obj_t rec = __find_glb(idx, key, 0);
	if (!rec)
		return ENOENT;
	*data = REC(rec)->value;
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

struct bxt_context {
	ods_obj_t key;
	ods_obj_t rec;
	LIST_HEAD(bxt_node_list, bxt_obj_el) nodes;
};

static struct bxt_obj_el *alloc_el(bxt_t t)
{
	struct bxt_obj_el *el;
	if (!LIST_EMPTY(&t->el_q)) {
		el = LIST_FIRST(&t->el_q);
		LIST_REMOVE(el, entry);
	} else {
		ods_atomic_inc(&el_count);
		el = malloc(sizeof *el);
	}

	return el;
}

static void free_el(bxt_t t, struct bxt_obj_el *el)
{
	LIST_INSERT_HEAD(&t->el_q, el, entry);
}

static struct bxt_obj_el *node_alloc(bxt_t t)
{
	struct bxt_obj_el *el = alloc_el(t);
	size_t sz = sizeof(struct bxt_node) +
		(t->udata->order * sizeof(struct bxn_entry));
	el->obj = ods_obj_alloc(t->ods, sz);
	if (!el->obj) {
		ods_extend(t->ods, ods_size(t->ods) * 2);
		el->obj = ods_obj_alloc(t->ods, sz);
		if (!el->obj) {
			free_el(t, el);
			goto err;
		}
	}
	NODE(el->obj)->parent = 0;
	NODE(el->obj)->count = 0;
	NODE(el->obj)->is_leaf = 0;
	return el;
 err:
	return NULL;
}

/*
 * Allocate a new node from a pool of B+Tree Depth nodes. If the
 * \c cache parameter is specified, the allocator will ensure that
 * there are sufficient nodes pre-allocated to handle an insertion at
 * the current tree depth.
 */
static ods_obj_t node_new(ods_idx_t idx, bxt_t t, bxt_udata_t udata, int cache)
{
	struct bxt_obj_el *el;
	ods_obj_t node = NULL;

	if (cache == 0 && t->node_q_depth)
		goto alloc_from_cache;

	while (!udata || (t->node_q_depth < udata->depth + 2)) {
		el = node_alloc(t);
		if (!el)
			goto out;
		LIST_INSERT_HEAD(&t->node_q, el, entry);
		ods_atomic_inc(&t->node_q_depth);
		if (!cache)
			break;
	}

 alloc_from_cache:
	if (!LIST_EMPTY(&t->node_q)) {
		el = LIST_FIRST(&t->node_q);
		LIST_REMOVE(el, entry);
		ods_atomic_dec(&t->node_q_depth);
		node = el->obj;
		assert(0 == NODE(node)->is_leaf);
		free_el(t, el);
	}
 out:
	return node;
}

static ods_obj_t rec_new(ods_idx_t idx, ods_key_t key, ods_idx_data_t data, int is_dup)
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
	REC(obj)->value = data;
	return obj;
 err_1:
	ods_obj_delete(obj);
	ods_obj_put(obj);
	return NULL;
}

static struct bxn_entry ENTRY_INITIALIZER = {
	.u.leaf = { 0, 0 }
};

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
	ods_obj_t entry, next_rec, prev_rec;
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
		ods_obj_put(next_rec);
		goto out;
	}
	if (ent < NODE(leaf)->count) {
		entry = ods_ref_as_obj(t->ods, L_ENT(leaf,ent).head_ref);
		assert(entry);
	} else {
		entry = NULL;
	}
	if (!entry && ent) {
		/*
		 * This record is being added to the end of the leaf
		 * and the leaf is !empty.
		 */
		prev_rec = ods_ref_as_obj(t->ods, L_ENT(leaf, ent-1).tail_ref);
		assert(prev_rec);

		REC(new_rec)->prev_ref = ods_obj_ref(prev_rec);
		REC(new_rec)->next_ref = REC(prev_rec)->next_ref;
		if (REC(prev_rec)->next_ref) {
			next_rec = ods_ref_as_obj(t->ods, REC(prev_rec)->next_ref);
			REC(next_rec)->prev_ref = ods_obj_ref(new_rec);
			ods_obj_put(next_rec);
		}
		REC(prev_rec)->next_ref = ods_obj_ref(new_rec);
		ods_obj_put(prev_rec);
	} else if (entry) {
		/*
		 * We are being inserted in front of this entry
		 */
		REC(new_rec)->next_ref = ods_obj_ref(entry);
		REC(new_rec)->prev_ref = REC(entry)->prev_ref;
		if (REC(entry)->prev_ref) {
			prev_rec = ods_ref_as_obj(t->ods, REC(entry)->prev_ref);
			REC(prev_rec)->next_ref = ods_obj_ref(new_rec);
			ods_obj_put(prev_rec);
		}
		REC(entry)->prev_ref = ods_obj_ref(new_rec);
	}
	/* If necessary, move up trailing entries to make space. */
	for (j = NODE(leaf)->count; j > ent; j--)
		NODE(leaf)->entries[j] = NODE(leaf)->entries[j-1];
	NODE(leaf)->count++;
	L_ENT(leaf,ent).head_ref = ods_obj_ref(new_rec);
	L_ENT(leaf,ent).tail_ref = ods_obj_ref(new_rec);
 out:
#ifdef BXT_DEBUG
	{
		ods_key_t next = ods_ref_as_obj(t->ods, REC(new_rec)->next_ref);
		ods_obj_t prev = ods_ref_as_obj(t->ods, REC(new_rec)->prev_ref);
		ods_key_t rec_key = ods_ref_as_obj(t->ods, REC(new_rec)->key_ref);
		ods_key_t prev_key = NULL;
		ods_key_t next_key = NULL;
		if (prev)
			prev_key = ods_ref_as_obj(t->ods, REC(prev)->key_ref);
		if (next)
			next_key = ods_ref_as_obj(t->ods, REC(next)->key_ref);
		if (prev_key)
			assert(t->comparator(prev_key, rec_key) <= 0);
		if (next_key)
			assert(t->comparator(next_key, rec_key) >= 0);
	}
#endif
	ods_obj_put(entry);
	return ent;
}

static int split_midpoint(int order)
{
	if (order & 1)
		return (order >> 1) + 1;
	return order >> 1;
}

static void leaf_split_insert(ods_idx_t idx, bxt_t t, ods_obj_t left,
			      ods_obj_t new_key, ods_obj_t new_rec,
			      int ins_idx, ods_obj_t right)
{
	int i, j;
	int ins_left_n_right;
	int midpoint = split_midpoint(t->udata->order);

	assert(NODE(left)->is_leaf);
	NODE(right)->is_leaf = 1;
	NODE(right)->parent = NODE(left)->parent;

#ifdef BXT_DEBUG
	verify_node(t, left);
	verify_node(t, right);
#endif
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
		for (i = midpoint - 1, j = 0; i < t->udata->order; i++, j++) {
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
		for (i = midpoint, j = 0; i < t->udata->order; i++, j++) {
			/*
			 * If this is where the new entry will
			 * go, skip a slot
			 */
			if (ins_idx == j)
				j ++;
			NODE(right)->entries[j] = NODE(left)->entries[i];
			NODE(left)->entries[i] = ENTRY_INITIALIZER;
#ifdef BXT_DEBUG
			assert(NODE(left)->count <= t->udata->order);
#endif
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
#ifdef BXT_DEBUG
	verify_node(t, left);
	verify_node(t, right);
#endif
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
	int midpoint = split_midpoint(t->udata->order);

#ifdef BXT_DEBUG
	verify_node(t, left_node);
	verify_node(t, right_node);
#endif
	/* Take our own reference on these parameters */
	left_node = ods_obj_get(left_node);
	right_node = ods_obj_get(right_node);

 split_and_insert:
	/* Right node and parent */
	right_parent = node_new(idx, t, NULL, 0);
	assert(right_parent);

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
		next_parent = node_new(idx, t, NULL, 0);
		assert(next_parent);
		NODE(next_parent)->count = 2;
		N_ENT(next_parent,0).node_ref = ods_obj_ref(left_parent);
		N_ENT(next_parent,0).key_ref = N_ENT(left_parent,0).key_ref;
		N_ENT(next_parent,1).node_ref = ods_obj_ref(right_parent);
		N_ENT(next_parent,1).key_ref = N_ENT(right_parent,0).key_ref;
		NODE(left_parent)->parent = ods_obj_ref(next_parent);
		NODE(right_parent)->parent = ods_obj_ref(next_parent);
		t->udata->root_ref = ods_obj_ref(next_parent);
		ods_atomic_inc(&t->udata->depth);
		goto out;
	}
	/* If there is room, insert into the parent */
	if (NODE(next_parent)->count < t->udata->order) {
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
}

static int bxt_insert(ods_idx_t idx, ods_key_t new_key, ods_idx_data_t data)
{
	bxt_t t = idx->priv;
	ods_obj_t parent;
	ods_obj_t leaf;
	ods_obj_t new_rec;
	int is_dup, ent;
	if (ods_spin_lock(&t->lock, -1))
		return EBUSY;

	if (!t->udata->root_ref) {
		leaf = node_new(idx, t, NULL, 0);
		if (!leaf)
			goto err_1;
		NODE(leaf)->is_leaf = 1;
		t->udata->root_ref = ods_obj_ref(leaf);
		ent = 0;
		is_dup = 0;
	} else {
		leaf = leaf_find(t, new_key);
		ent = find_key_idx(t, leaf, new_key, &is_dup);
	}
#ifdef BXT_DEBUG
	verify_node(t, leaf);
#endif

	/* Allocate a record object */
	new_rec = rec_new(idx, new_key, data, is_dup);
	if (!new_rec)
		goto err_1;

	/*
	 * If this is a dup or the new record will fit in the leaf,
	 * then this is a simple insert with no new nodes required.
	 */
	if ((NODE(leaf)->count < t->udata->order) || is_dup) {
		if (is_dup)
			ods_atomic_inc(&t->udata->dups);
		if (!leaf_insert(t, leaf, new_rec, ent, is_dup)
		    && NODE(leaf)->parent) {
			ods_obj_t parent =
				ods_ref_as_obj(t->ods, NODE(leaf)->parent);
			/* Maintain this to simplify other logic */
			if (N_ENT(parent,0).node_ref == ods_obj_ref(leaf)) {
				ods_obj_t rec0 = ods_ref_as_obj(t->ods, L_ENT(leaf,0).head_ref);
				N_ENT(parent,0).key_ref = REC(rec0)->key_ref;
				ods_obj_put(rec0);
			}
			ods_obj_put(parent);
		}
#ifdef BXT_DEBUG
		verify_node(t, leaf);
#endif
		ods_atomic_inc(&t->udata->card);
		ods_obj_put(leaf);
		ods_obj_put(new_rec);
		ods_spin_unlock(&t->lock);
		return 0;
	}

	/*
	 * The new record overflows the leaf. The leaf will need to be split
	 * and the new leaf inserted into the tree.
	 */
	ods_obj_t new_leaf = node_new(idx, t, t->udata, 1);
	if (!new_leaf)
		goto err_1;
	leaf_split_insert(idx, t, leaf, new_key, new_rec, ent, new_leaf);

	ods_obj_t leaf_rec = ods_ref_as_obj(t->ods, L_ENT(leaf,0).head_ref);
	ods_obj_t new_leaf_rec = ods_ref_as_obj(t->ods, L_ENT(new_leaf,0).head_ref);
	ods_ref_t leaf_key_ref = REC(leaf_rec)->key_ref;
	ods_ref_t new_leaf_key_ref = REC(new_leaf_rec)->key_ref;
	ods_obj_put(leaf_rec);
	ods_obj_put(new_leaf_rec);

	parent = ods_ref_as_obj(t->ods, NODE(leaf)->parent);
	if (!parent) {
		parent = node_new(idx, t, NULL, 0);
		assert(parent);
		N_ENT(parent,0).key_ref = leaf_key_ref;
		N_ENT(parent,0).node_ref = ods_obj_ref(leaf);

		N_ENT(parent,1).key_ref = new_leaf_key_ref;
		N_ENT(parent,1).node_ref = ods_obj_ref(new_leaf);

		NODE(parent)->count = 2;

		NODE(leaf)->parent = ods_obj_ref(parent);
		NODE(new_leaf)->parent = ods_obj_ref(parent);
		ods_atomic_inc(&t->udata->depth);
		t->udata->root_ref = ods_obj_ref(parent);
		goto out;
	}
	if (NODE(parent)->count < t->udata->order) {
		node_insert(t, parent, leaf, new_leaf_key_ref, new_leaf);
		goto out;
	}
	ods_obj_put(parent);
	parent = node_split_insert(idx, t, leaf, new_leaf_key_ref, new_leaf);
 out:
#ifdef BXT_DEBUG
	verify_node(t, leaf);
	verify_node(t, new_leaf);
	verify_node(t, parent);
#endif
	ods_atomic_inc(&t->udata->card);
	ods_obj_put(leaf);
	ods_obj_put(new_leaf);
	ods_obj_put(parent);
	ods_obj_put(new_rec);
	ods_spin_unlock(&t->lock);
	return 0;

 err_1:
	ods_spin_unlock(&t->lock);
	return ENOMEM;
}

static ods_obj_t min_in_subtree(bxt_t t, ods_ref_t root)
{
	ods_obj_t n;

	/* Walk to the left most leaf and return the 0-th entry  */
	n = ods_ref_as_obj(t->ods, root);
	while (!NODE(n)->is_leaf) {
		ods_ref_t ref = N_ENT(n,0).node_ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	return n;
}

static ods_obj_t bxt_min_node(bxt_t t)
{
	if (!t->udata->root_ref)
		return 0;

	return min_in_subtree(t, t->udata->root_ref);
}

static ods_obj_t max_in_subtree(bxt_t t, ods_ref_t root)
{
	ods_obj_t n;

	/* Walk to the right most leaf and return the (count-1)-th entry  */
	n = ods_ref_as_obj(t->ods, root);
	while (!NODE(n)->is_leaf) {
		ods_ref_t ref = N_ENT(n,NODE(n)->count-1).node_ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	return n;
}

static ods_obj_t bxt_max_node(bxt_t t)
{
	ods_obj_t n;

	if (!t->udata->root_ref)
		return 0;

	/* Walk to the left most leaf and return the 0-th entry  */
	n = ods_ref_as_obj(t->ods, t->udata->root_ref);
	while (!NODE(n)->is_leaf) {
		ods_ref_t ref = N_ENT(n,NODE(n)->count-1).node_ref;
		ods_obj_put(n);
		n = ods_ref_as_obj(t->ods, ref);
	}
	return n;
}

static ods_obj_t left_sibling(bxt_t t, ods_obj_t node)
{
	int idx = 0;
	ods_ref_t node_ref;
	ods_obj_t pparent, parent, left;

	/*
	 * Root has no left sibling
	 */
	node_ref = ods_obj_ref(node);
	if (t->udata->root_ref == node_ref)
		return NULL;

	/*
	 * Walk up until we reach either the root, or a subtree that
	 * contains a left node/subtree.
	 */
	parent = ods_ref_as_obj(t->ods, NODE(node)->parent);
	while (parent) {
		idx = find_ref_idx(parent, node_ref);
		assert(idx < NODE(parent)->count);
		if (idx > 0)
			break;
		else if (t->udata->root_ref == ods_obj_ref(parent))
			goto not_found;
		pparent = ods_ref_as_obj(t->ods, NODE(parent)->parent);
		node_ref = ods_obj_ref(parent);
		ods_obj_put(parent);
		parent = pparent;
	}
	left = max_in_subtree(t,  N_ENT(parent,idx-1).node_ref);
	ods_obj_put(parent);
	return left;
 not_found:
	ods_obj_put(parent);
	return NULL;
}

static ods_obj_t right_sibling(bxt_t t, ods_obj_t node)
{
	int idx = 0;
	ods_ref_t node_ref;
	ods_obj_t pparent, parent, right;

	/*
	 * Root has no right sibling
	 */
	node_ref = ods_obj_ref(node);
	if (t->udata->root_ref == node_ref)
		return NULL;

	/*
	 * Walk up until we reach either the root, or a subtree that
	 * contains a right node/subtree.
	 */
	parent = ods_ref_as_obj(t->ods, NODE(node)->parent);
	while (parent) {
		idx = find_ref_idx(parent, node_ref);
		assert(idx < NODE(parent)->count);
		if (idx < NODE(parent)->count - 1)
			break;
		else if (t->udata->root_ref == ods_obj_ref(parent))
			goto not_found;
		pparent = ods_ref_as_obj(t->ods, NODE(parent)->parent);
		node_ref = ods_obj_ref(parent);
		ods_obj_put(parent);
		parent = pparent;
	}
	right = min_in_subtree(t,  N_ENT(parent,idx+1).node_ref);
	ods_obj_put(parent);
	return right;
 not_found:
	ods_obj_put(parent);
	return NULL;
}

static int node_neigh(bxt_t t, ods_obj_t node, ods_obj_t *left, ods_obj_t *right)
{
	int idx;
	ods_ref_t node_ref;
	ods_obj_t parent;

	*left = *right = NULL;
	node_ref = ods_obj_ref(node);
	if (t->udata->root_ref == node_ref)
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

	return t->udata->order - NODE(n)->count;
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
	for (i = NODE(right)->count + count - 1; i - count >= 0; i--)
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
#ifdef BXT_DEBUG
	verify_node(t, right);
	verify_node(t, node);
#endif
	return idx;
}

static int combine_left(bxt_t t, ods_obj_t left, ods_obj_t node)
{
	int i, j;
	int count;
	ods_obj_t entry;
	ods_ref_t left_ref;

	if (!left)
		return 0;
	count = NODE(node)->count;
	if (NODE(left)->count + count > t->udata->order)
		count = t->udata->order - NODE(left)->count;
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
#ifdef BXT_DEBUG
	verify_node(t, left);
	verify_node(t, node);
#endif
	return j;
}

/*
 * Drops reference on parent and node
 */
static ods_ref_t fixup_parents(bxt_t t, ods_obj_t parent, ods_obj_t node)
{
	int i;
	ods_ref_t parent_ref;

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
	return t->udata->root_ref;
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
#ifdef BXT_DEBUG
	verify_node(t, node);
	verify_node(t, left);
#endif
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

#ifdef BXT_DEBUG
	verify_node(t, node);
	verify_node(t, right);
#endif
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
	/* Fix up next and prev pointers in record list */
	ods_obj_t next_rec = ods_ref_as_obj(t->ods, REC(rec)->next_ref);
	ods_obj_t prev_rec = ods_ref_as_obj(t->ods, REC(rec)->prev_ref);
	if (prev_rec)
		REC(prev_rec)->next_ref = REC(rec)->next_ref;
	if (next_rec)
		REC(next_rec)->prev_ref = REC(rec)->prev_ref;
	ods_obj_put(next_rec);
	ods_obj_put(prev_rec);

 next_level:
	parent = ods_ref_as_obj(t->ods, NODE(node)->parent);
#ifdef BXT_DEBUG
	verify_node(t, node);
	verify_node(t, parent);
#endif
	/* Remove the record and object from the node */
	for (i = ent; i < NODE(node)->count - 1; i++)
		NODE(node)->entries[i] = NODE(node)->entries[i+1];
	NODE(node)->entries[NODE(node)->count-1] = ENTRY_INITIALIZER;
	NODE(node)->count--;

	if (ods_obj_ref(node) == t->udata->root_ref) {
		/* This is the root of the tree. */
		switch (NODE(node)->count) {
		case 0:
			/* The root is now empty */
			t->udata->root_ref = 0;
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
			return t->udata->root_ref;
		}
	}

	midpoint = split_midpoint(t->udata->order);
	if (NODE(node)->count >= midpoint) {
		/* Unless the 0-the element is modified, no fixup is required. */
		if (ent == 0)
			return fixup_parents(t, parent, node);
		ods_obj_put(parent);
		ods_obj_put(node);
		return t->udata->root_ref;
	}
	node_idx = node_neigh(t, node, &left, &right);
	count = space(t, left) + space(t, right);
#ifdef BXT_DEBUG
	verify_node(t, left);
	verify_node(t, right);
	verify_node(t, node);
#endif
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
#ifdef BXT_DEBUG
			verify_node(t, left);
			verify_node(t, node);
#endif
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
#ifdef BXT_DEBUG
		assert(NODE(right)->count + (NODE(node)->count - ent) <= t->udata->order);
#endif
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

static void delete_head(bxt_t t, ods_obj_t leaf, int ent)
{
	ods_obj_t rec = ods_ref_as_obj(t->ods, L_ENT(leaf,ent).head_ref);
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

static void delete_dup_rec(bxt_t t, ods_obj_t leaf, ods_obj_t rec, int ent)
{
	ods_ref_t rec_ref = ods_obj_ref(rec);
	ods_obj_t next_rec = ods_ref_as_obj(t->ods, REC(rec)->next_ref);
	ods_obj_t prev_rec = ods_ref_as_obj(t->ods, REC(rec)->prev_ref);

	assert(L_ENT(leaf,ent).head_ref != L_ENT(leaf,ent).tail_ref);
	if (prev_rec)
		REC(prev_rec)->next_ref = REC(rec)->next_ref;
	if (next_rec)
		REC(next_rec)->prev_ref = REC(rec)->prev_ref;

	if (rec_ref == L_ENT(leaf,ent).head_ref) {
		assert(REC(rec)->next_ref);
		L_ENT(leaf,ent).head_ref = REC(rec)->next_ref;
	} else if (rec_ref == L_ENT(leaf,ent).tail_ref) {
		assert(REC(rec)->prev_ref);
		L_ENT(leaf,ent).tail_ref = REC(rec)->prev_ref;
	}

	ods_obj_put(next_rec);
	ods_obj_put(prev_rec);
	ods_obj_delete(rec);
	ods_obj_put(rec);
}

static ods_obj_t find_matching_rec(bxt_t t, ods_obj_t leaf, int ent,
				   ods_idx_data_t *data)
{
	ods_obj_t rec;
	ods_ref_t rec_ref;

	/* Find the duplicate that matches the input index data */
	rec_ref = L_ENT(leaf, ent).head_ref;
	do {
		rec = ods_ref_as_obj(t->ods, rec_ref);
		assert(rec);
		if (ods_idx_data_equal(&REC(rec)->value, data))
			return rec;
		if (rec_ref == L_ENT(leaf, ent).tail_ref)
			/* Didn't find a match */
			break;
		rec_ref = REC(rec)->next_ref;
		ods_obj_put(rec);
	} while (rec_ref);
	return NULL;
}

static int bxt_delete(ods_idx_t idx, ods_key_t key, ods_idx_data_t *data)
{
	bxt_t t = idx->priv;
	int ent;
	ods_obj_t leaf, rec;
	int found;

	if (ods_spin_lock(&t->lock, -1))
		return EBUSY;

	leaf = leaf_find(t, key);
	if (!leaf)
		goto noent;
	ent = find_key_idx(t, leaf, key, &found);
	if (!found)
		goto noent;
	ods_atomic_dec(&t->udata->card);
	/*
	 * Trivial case is that this is a dup key. In this case,
	 * delete the first entry on the list and return
	 */
	if (L_ENT(leaf, ent).head_ref != L_ENT(leaf, ent).tail_ref) {
		int rc = 0;
		if (ods_idx_data_null(data)) {
			delete_head(t, leaf, ent);
		} else {
			/* find the matching dup entry and delete it */
			rec = find_matching_rec(t, leaf, ent, data);
			if (rec) {
				delete_dup_rec(t, leaf, rec, ent);
				ods_obj_put(rec);
			} else {
				rc = ENOENT;
			}
		}
		if (!rc)
			ods_atomic_dec(&t->udata->dups);
		ods_obj_put(leaf);
		ods_spin_unlock(&t->lock);
		return rc;
	}
	rec = ods_ref_as_obj(t->ods, L_ENT(leaf, ent).head_ref);
	assert(rec);
	*data = REC(rec)->value;
	t->udata->root_ref = entry_delete(t, leaf, rec, ent);
	ods_obj_delete(key);
	ods_obj_put(rec);
	ods_spin_unlock(&t->lock);
#ifdef BXT_DEBUG
	print_idx(idx, NULL);
#endif
	return 0;
 noent:
	ods_obj_put(leaf);
	ods_spin_unlock(&t->lock);
	return ENOENT;
}

static ods_iter_t bxt_iter_new(ods_idx_t idx)
{
	bxt_iter_t iter = calloc(1, sizeof *iter);
	iter->iter.idx = idx;
	return (struct ods_iter *)iter;
}

static void bxt_iter_delete(ods_iter_t i)
{
	bxt_iter_t bxi = (bxt_iter_t)i;
	if (bxi->rec)
		ods_obj_put(bxi->rec);
	free(i);
}

static int _iter_begin_unique(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	if (i->rec)
		ods_obj_put(i->rec);
	i->ent = 0;
	i->rec = bxt_min_node(t);
	return i->rec ? 0 : ENOENT;
}

static int _iter_begin(bxt_iter_t i)
{
	ods_obj_t node;
	bxt_t t = i->iter.idx->priv;
	if (i->rec)
		ods_obj_put(i->rec);
	node = bxt_min_node(t);
	if (node && (0 == (i->iter.flags & ODS_ITER_F_UNIQUE))) {
		i->rec = ods_ref_as_obj(t->ods, L_ENT(node, 0).head_ref);
		ods_obj_put(node);
	} else
		  i->rec = node;
#ifdef BXT_DEBUG
	if (i->rec)
		assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
#endif
	return i->rec ? 0 : ENOENT;
}

static int bxt_iter_begin(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	if (0 == (i->iter.flags & ODS_ITER_F_UNIQUE))
		return _iter_begin(i);
	return _iter_begin_unique(i);
}

static int _iter_end_unique(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	if (i->rec)
		ods_obj_put(i->rec);
	i->rec = bxt_max_node(t);
	if (i->rec) {
#ifdef BXT_DEBUG
		assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
#endif
		i->ent = NODE(i->rec)->count-1;
	}
	return i->rec ? 0 : ENOENT;
}

static int _iter_end(bxt_iter_t i)
{
	ods_obj_t node;
	bxt_t t = i->iter.idx->priv;
	if (i->rec)
		ods_obj_put(i->rec);
	node = bxt_max_node(t);
	if (node) {
		i->ent = NODE(node)->count-1;
		i->rec = ods_ref_as_obj(t->ods, L_ENT(node, i->ent).tail_ref);
#ifdef BXT_DEBUG
		assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
#endif
		ods_obj_put(node);
	} else
		i->rec = NULL;
	return i->rec ? 0 : ENOENT;
}

static int bxt_iter_end(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	if (0 == (i->iter.flags & ODS_ITER_F_UNIQUE))
		return _iter_end(i);
	return _iter_end_unique(i);
}

static ods_key_t _iter_key_unique(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	ods_obj_t rec, key;
	if (!i->rec)
		return NULL;
	rec = ods_ref_as_obj(t->ods, L_ENT(i->rec, i->ent).head_ref);
	key = ods_ref_as_obj(t->ods, REC(rec)->key_ref);
	ods_obj_put(rec);
	return key;
}

static ods_key_t _iter_key(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	if (!i->rec)
		return NULL;
	return ods_ref_as_obj(t->ods, REC(i->rec)->key_ref);
}

static ods_key_t bxt_iter_key(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	if (0 == (i->iter.flags & ODS_ITER_F_UNIQUE))
		return _iter_key(i);
	return _iter_key_unique(i);
}

static ods_idx_data_t NULL_DATA = {
	.bytes = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

static ods_idx_data_t _iter_data_unique(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	ods_idx_data_t data;
	ods_obj_t rec;
	if (!i->rec)
		return NULL_DATA;
	rec = ods_ref_as_obj(t->ods, L_ENT(i->rec, i->ent).head_ref);
	data = REC(rec)->value;
	ods_obj_put(rec);
	return data;
}

static ods_idx_data_t _iter_data(bxt_iter_t i)
{
	if (!i->rec)
		return NULL_DATA;
	return REC(i->rec)->value;
}

static ods_idx_data_t bxt_iter_data(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	if (0 == (i->iter.flags & ODS_ITER_F_UNIQUE))
		return _iter_data(i);
	return _iter_data_unique(i);
}

static int _iter_find_dup(ods_iter_t oi, ods_key_t key, int first)
{
	bxt_iter_t iter = (bxt_iter_t)oi;
	bxt_t t = iter->iter.idx->priv;
	ods_obj_t leaf = leaf_find(t, key);
	ods_ref_t ref;
	int found;
	int i;

	if (iter->rec) {
		ods_obj_put(iter->rec);
		iter->rec = NULL;
	}
	if (!leaf)
		return ENOENT;

	if (oi->flags & ODS_ITER_F_UNIQUE)
		return EINVAL;

	assert(NODE(leaf)->is_leaf);
	i = find_key_idx(t, leaf, key, &found);
	if (!found) {
		ods_obj_put(leaf);
		return ENOENT;
	}
	if (first)
		ref = L_ENT(leaf,i).head_ref;
	else
		ref = L_ENT(leaf,i).tail_ref;
	iter->rec = ods_ref_as_obj(t->ods, ref);
	ods_obj_put(leaf);
	iter->ent = i;
	return 0;
}

static int bxt_iter_find_first(ods_iter_t oi, ods_key_t key)
{
	return _iter_find_dup(oi, key, 1);
}

static int bxt_iter_find_last(ods_iter_t oi, ods_key_t key)
{
	return _iter_find_dup(oi, key, 0);
}

static int bxt_iter_find(ods_iter_t oi, ods_key_t key)
{
	bxt_iter_t iter = (bxt_iter_t)oi;
	bxt_t t = iter->iter.idx->priv;
	ods_obj_t leaf = leaf_find(t, key);
	int found;
	int i;

	if (iter->rec) {
		ods_obj_put(iter->rec);
		iter->rec = NULL;
	}
	if (!leaf)
		return ENOENT;

	assert(NODE(leaf)->is_leaf);
	i = find_key_idx(t, leaf, key, &found);
	if (!found) {
		ods_obj_put(leaf);
		return ENOENT;
	}
	if (0 == (oi->flags & ODS_ITER_F_UNIQUE)) {
		iter->rec = ods_ref_as_obj(t->ods, L_ENT(leaf,i).head_ref);
		ods_obj_put(leaf);
		iter->ent = i;
	} else {
		iter->rec = leaf;
		iter->ent = i;
	}
	return 0;
}

static int bxt_iter_find_lub(ods_iter_t oi, ods_key_t key)
{
	bxt_iter_t iter = (bxt_iter_t)oi;
	if (iter->rec) {
		ods_obj_put(iter->rec);
		iter->rec = NULL;
	}
	iter->rec = __find_lub(iter->iter.idx, key, iter->iter.flags);
	return iter->rec ? 0 : ENOENT;
}

static int bxt_iter_find_glb(ods_iter_t oi, ods_key_t key)
{
	bxt_iter_t iter = (bxt_iter_t)oi;
	if (iter->rec) {
		ods_obj_put(iter->rec);
		iter->rec = NULL;
	}
	iter->rec = __find_glb(iter->iter.idx, key, iter->iter.flags);
	return iter->rec ? 0 : ENOENT;
}

static int _iter_next_unique(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	ods_obj_t right;

	if (!i->rec)
		return ENOENT;

	if (i->ent < NODE(i->rec)->count - 1)
		i->ent++;
	else {
		right = right_sibling(t, i->rec);
		ods_obj_put(i->rec);
		i->rec = right;
		i->ent = 0;
#ifdef BXT_DEBUG
		if (i->rec)
			assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
#endif
	}
	return i->rec ? 0 : ENOENT;
}

static int _iter_next(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	ods_obj_t next_rec;

	if (i->rec) {
		ods_ref_t next_ref = REC(i->rec)->next_ref;
#ifdef BXT_DEBUG
		ods_ref_t rec_ref = ods_obj_ref(i->rec);
		ods_ref_t prev_ref = REC(i->rec)->prev_ref;
		assert(next_ref != rec_ref);
		assert(prev_ref != rec_ref);
		if (next_ref)
			assert(prev_ref != next_ref);
#endif
		next_rec = ods_ref_as_obj(t->ods, next_ref);
		ods_obj_put(i->rec);
		i->rec = next_rec;
#ifdef BXT_DEBUG 
		if (i->rec)
			assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
#endif
	}
	return i->rec ? 0 : ENOENT;
}

static int bxt_iter_next(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	if (0 == (i->iter.flags & ODS_ITER_F_UNIQUE))
		return _iter_next(i);
	return _iter_next_unique(i);
}

static int _iter_prev_unique(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	ods_obj_t left;

	if (!i->rec)
		return ENOENT;

	if (i->ent > 0)
		i->ent--;
	else {
		left = left_sibling(t, i->rec);
		ods_obj_put(i->rec);
		i->rec = left;
		if (left)
			i->ent = NODE(left)->count - 1;
	}
	return i->rec ? 0 : ENOENT;
}

static int _iter_prev(bxt_iter_t i)
{
	bxt_t t = i->iter.idx->priv;
	ods_obj_t prev_rec;

	if (i->rec) {
		prev_rec = ods_ref_as_obj(t->ods, REC(i->rec)->prev_ref);
		ods_obj_put(i->rec);
		i->rec = prev_rec;
#ifdef BXT_DEBUG
		if (i->rec)
			assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
#endif
	}
	return i->rec ? 0 : ENOENT;
}

static int bxt_iter_prev(ods_iter_t oi)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	if (0 == (i->iter.flags & ODS_ITER_F_UNIQUE))
		return _iter_prev(i);
	return _iter_prev_unique(i);
}

#define POS_PAD 0x54584220 /* 'BXT ' */

struct bxt_pos_s {
	uint32_t pad;
	uint32_t ent;
	ods_ref_t rec_ref;
};

static int bxt_iter_set(ods_iter_t oi, const ods_pos_t pos_)
{
	struct bxt_pos_s *pos = (struct bxt_pos_s *)pos_;
	bxt_iter_t i = (bxt_iter_t)oi;
	bxt_t t = i->iter.idx->priv;
	ods_obj_t rec;

	if (pos->pad != POS_PAD)
		return EINVAL;

	rec = ods_ref_as_obj(t->ods, pos->rec_ref);
	if (!rec)
		return EINVAL;

	if (i->rec)
		ods_obj_put(i->rec);

	i->rec = rec;
	i->ent = pos->ent;
#ifdef BXT_DEBUG
	if (i->rec)
		assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
#endif
	return 0;
}

static int bxt_iter_pos(ods_iter_t oi, ods_pos_t pos_)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	struct bxt_pos_s *pos = (struct bxt_pos_s *)pos_;

	if (!i->rec)
		return ENOENT;
#ifdef BXT_DEBUG
	assert(REC(i->rec)->next_ref != 0xffffffffffffffff);
#endif
	pos->pad = POS_PAD;
	pos->ent = i->ent;
	pos->rec_ref = ods_obj_ref(i->rec);
	return 0;
}

static int bxt_iter_pos_delete(ods_iter_t oi, ods_pos_t pos_)
{
	bxt_iter_t i = (bxt_iter_t)oi;
	struct bxt_pos_s *pos = (struct bxt_pos_s *)pos_;
	bxt_t t = i->iter.idx->priv;
	int ent;
	ods_obj_t leaf, rec;
	int found;
	ods_key_t key;
	ods_ref_t prev_ref, next_ref;

	if (ods_spin_lock(&t->lock, -1))
		return EBUSY;

	uint32_t status = ods_ref_status(t->ods, pos->rec_ref);
	assert(0 == (status & ODS_REF_STATUS_FREE));
	rec = ods_ref_as_obj(t->ods, pos->rec_ref);
	if (!rec)
		goto norec;
#ifdef BXT_DEBUG
	assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
#endif
	key = ods_ref_as_obj(t->ods, REC(rec)->key_ref);
	leaf = leaf_find(t, key);
	if (!leaf)
		goto noent;

	ent = find_key_idx(t, leaf, key, &found);
	if (!found)
		goto noent;

	prev_ref = REC(rec)->prev_ref;
	next_ref = REC(rec)->next_ref;
#ifdef BXT_DEBUG
	assert(prev_ref != pos->rec_ref);
	assert(next_ref != pos->rec_ref);
#endif
	ods_atomic_dec(&t->udata->card);

	/* If the rec has a prev, move the iterator to it, if not move to the next */
	if (prev_ref)
		i->rec = ods_ref_as_obj(t->ods, prev_ref);
	else
		i->rec = ods_ref_as_obj(t->ods, next_ref);
#ifdef BXT_DEBUG
	if (i->rec) {
		assert(i->rec->ref != 0xFFFFFFFFFFFFFFFF);
		assert(REC(i->rec)->next_ref != 0xFFFFFFFFFFFFFFFF);
	}
#endif
	/*
	 * Trivial case is that this is a dup key. In this case,
	 * delete the pos entry on the REC list and return
	 */
	if (L_ENT(leaf, ent).head_ref != L_ENT(leaf, ent).tail_ref) {
		ods_atomic_dec(&t->udata->dups);
		delete_dup_rec(t, leaf, rec, ent);
		ods_obj_put(leaf);
		ods_spin_unlock(&t->lock);
		return 0;
	}

	t->udata->root_ref = entry_delete(t, leaf, rec, ent);
	ods_obj_delete(key);
	ods_obj_put(rec);
	ods_spin_unlock(&t->lock);
	return 0;
 noent:
	ods_obj_put(key);
	ods_obj_put(leaf);
 norec:
	ods_spin_unlock(&t->lock);
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

int bxt_stat(ods_idx_t idx, ods_idx_stat_t idx_sb)
{
	struct stat sb;
	bxt_t t = idx->priv;
	idx_sb->cardinality = t->udata->card;
	idx_sb->duplicates = t->udata->dups;
	ods_stat(idx->ods, &sb);
	idx_sb->size = sb.st_size;
	return 0;
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
	.stat = bxt_stat,
	.iter_new = bxt_iter_new,
	.iter_delete = bxt_iter_delete,
	.iter_find = bxt_iter_find,
	.iter_find_lub = bxt_iter_find_lub,
	.iter_find_glb = bxt_iter_find_glb,
	.iter_find_first = bxt_iter_find_first,
	.iter_find_last = bxt_iter_find_last,
	.iter_begin = bxt_iter_begin,
	.iter_end = bxt_iter_end,
	.iter_next = bxt_iter_next,
	.iter_prev = bxt_iter_prev,
	.iter_set = bxt_iter_set,
	.iter_pos = bxt_iter_pos,
	.iter_pos_delete = bxt_iter_pos_delete,
	.iter_key = bxt_iter_key,
	.iter_data = bxt_iter_data,
	.print_idx = print_idx,
	.print_info = print_info
};

struct ods_idx_provider *get(void)
{
	return &bxt_provider;
}

static void __attribute__ ((constructor)) bxt_lib_init(void)
{
}

static void __attribute__ ((destructor)) bxt_lib_term(void)
{
}
