/*
 * Copyright (c) 2021 Open Grid Computing, Inc. All rights reserved.
 *
 * See the file COPYING at the top of this source tree for the terms
 * of the Copyright.
 */
#include <inttypes.h>
#include <stdlib.h>
#include <ods/ods_rbt.h>
#include <assert.h>
#include <stdio.h>

/* A LEAF (NULL) is considered BLACK */
static int is_red(struct ods_rbn *x)
{
	if (!x)
		return 0;
	return x->color == ODS_RBN_RED;
}

static void rotate_left(struct ods_rbt *t, struct ods_rbn *x)
{
	struct ods_rbn *y = x->right;
	struct ods_rbn *parent = x->parent;

	/* Link y's left to x's right and update parent if not a leaf */
	x->right = y->left;
	if (y->left)
		y->left->parent = x;

	/* Attach y to x's parent if x is not the root.
	 * If x == t->root, then x->parent == NULL */
	y->parent = x->parent;
	if (t->root != x) {
		assert(x->parent);
		if (parent->left == x)
			parent->left = y;
		else
			parent->right = y;
	} else {
		assert(x->parent == NULL);
		t->root = y;
		y->parent = NULL;
	}

	/* Attach x as y's new left */
	y->left = x;
	x->parent = y;
}

static void rotate_right(struct ods_rbt *t, struct ods_rbn *x)
{
	struct ods_rbn *y = x->left;
	struct ods_rbn *parent = x->parent;

	/* Link y's right to x's left and update parent */
	x->left = y->right;
	if (y->right)
		y->right->parent = x;

	/* Attach y to x's parent */
	if (t->root != x) {
		y->parent = parent;
		if (parent->right == x)
			parent->right = y;
		else
			parent->left = y;
	} else {
		t->root = y;
		y->parent = NULL;
	}

	/* Attach x as y's new left */
	y->right = x;
	x->parent = y;
}

/**
 * \brief Initialize an ODS_RBT.
 *
 * \param t	Pointer to the ODS_RBT.
 * \param cmp	Pointer to the function that compares nodes in the
 *		ODS_RBT.
 * \param cmp_arg Additional argument to pass to the comparator function
 */
void ods_rbt_init(struct ods_rbt *t, ods_rbn_comparator_t cmp, void *cmp_arg)
{
	t->root = NULL;
	t->comparator = cmp;
	t->comparator_arg = cmp_arg;
	t->card = 0;
}

/**
 * @brief Return the number of elements in the tree
 *
 * @param t The tree handle
 * @return uint64_t The number of elements in the tree
 */
uint64_t ods_rbt_card(struct ods_rbt *t)
{
	return t->card;
}

/**
 * \brief Returns TRUE if the tree is empty.
 *
 * \param t	Pointer to the ods_rbt.
 * \retval 0	The tree is not empty
 * \retval 1	The tree is empty
 */
int ods_rbt_empty(struct ods_rbt *t)
{
	return (t->root == NULL);
}

/**
 * \brief Initialize an ODS_RBN node.
 *
 * Initialize an ODS_RBN node. This is a convenience function to avoid
 * having the application know about the internals of the ODS_RBN while
 * still allowing the ODS_RBN to be embedded in the applications object
 * and avoiding a second allocation in ods_rbn_ins.
 *
 * \param n The ODS_RBN to initialize
 * \param key Pointer to the key
 */
void ods_rbn_init(struct ods_rbn *n, void *key)
{
	n->key = key;
}

/**
 * @brief Compare the keys of two RBN
 *
 * Use the tree's configured comparator to compare two
 * RBN from the tree \c t.
 *
 * @param t The tree handle
 * @param a The RBN pointer
 * @param b An RBN pointer
 * @return <0 if a->key < b->key
 * @return =0 if a->key == b->key
 * @return >0 if a->key > b->key
 */
int64_t ods_rbn_cmp(struct ods_rbt *t, struct ods_rbn *a, struct ods_rbn *b)
{
	return t->comparator(a->key, b->key, t->comparator_arg);
}

static int __ods_rbt_ins(struct ods_rbt *t, struct ods_rbn *x, int allow_dup)
{
	struct ods_rbn *parent = NULL;
	struct ods_rbn *n;
	int64_t c = 0;

	/* Initialize new node */
	x->left = x->right = NULL;

	/* Trivial root insertion */
	if (!t->root) {
		x->color = ODS_RBN_BLACK;
		t->root = x;
		x->parent = NULL;
		t->card += 1;
		return 1;
	}

	/* Always insert a RED node */
	x->color = ODS_RBN_RED;
	for (n = t->root; n; ) {
		parent = n;
		c = t->comparator(n->key, x->key, t->comparator_arg);
		if (c == 0 && 0 == allow_dup)
			return 0;
		if (c > 0)
			n = n->left;
		else
			n = n->right;
	}
	/* Replace leaf with new node */
	assert(parent);
	x->parent = parent;
	if (c > 0)
		parent->left = x;
	else
		parent->right = x;

	/*
	 * While x is not the root and x's parent is red. Note that if x's
	 * parent is RED, then x's parent is also not the root
	 */
	while (x != t->root && is_red(x->parent)) {
		struct ods_rbn *uncle;
		if (x->parent == x->parent->parent->left) {
			uncle = x->parent->parent->right;
			if (is_red(uncle)) {
				x->parent->color = ODS_RBN_BLACK;
				uncle->color = ODS_RBN_BLACK;
				x->parent->parent->color = ODS_RBN_RED;
				x = x->parent->parent;
			} else {
				if (x == x->parent->right) {
					x = x->parent;
					rotate_left(t, x);
				}
				x->parent->color = ODS_RBN_BLACK;
				x->parent->parent->color = ODS_RBN_RED;
				rotate_right(t, x->parent->parent);
			}
		} else {
			uncle = x->parent->parent->left;
			if (is_red(uncle)) {
				x->parent->color = ODS_RBN_BLACK;
				uncle->color = ODS_RBN_BLACK;
				x->parent->parent->color = ODS_RBN_RED;
				x = x->parent->parent;
			} else {
				if (x == x->parent->left) {
					x = x->parent;
					rotate_right(t, x);
				}
				x->parent->color = ODS_RBN_BLACK;
				if (x->parent->parent) {
					x->parent->parent->color = ODS_RBN_RED;
					rotate_left(t, x->parent->parent);
				}
			}
		}
	}
	t->root->color = ODS_RBN_BLACK;
	t->card += 1;
	return 1;
}

/**
 * \brief Insert a new node into the ODS_RBT.
 *
 * Insert a new node into a ODS_RBT. The node is allocated by the user and
 * must be freed by the user when removed from the tree. The 'key'
 * field of the provided node must have already been set up by the
 * caller.
 *
 * \param t	Pointer to the ODS_RBT.
 * \param x	Pointer to the node to insert.
 */
void ods_rbt_ins(struct ods_rbt *t, struct ods_rbn *x)
{
    (void)__ods_rbt_ins(t, x, 1);
}
/**
 * @brief Insert a unique node into the RBT.
 *
 * If the key already exists, the input rbn is not modified and FALSE
 * (0) is returned.
 *
 * @param t     Pointer to the RBT.
 * @param x     Pointer to the node to insert.
 * @return TRUE(1) if the node is inserted
 * @return FALSE(0) if the key already exists
 */
int ods_rbt_ins_unique(struct ods_rbt *t, struct ods_rbn *x)
{
	return __ods_rbt_ins(t, x, 0);
}

static inline struct ods_rbn *find_pred(struct ods_rbn *n)
{
	struct ods_rbn *x;
	for (x = n->left; x->right; x = x->right);
	return x;
}

static inline struct ods_rbn *find_succ(struct ods_rbn *n)
{
	struct ods_rbn *x;
	for (x = n->right; x->left; x = x->left);
	return x;
}

/**
 * \brief Delete a node from the ODS_RBT.
 *
 * \param t	Pointer to the ODS_RBT.
 * \param z	Pointer to the node to delete.
 */
/**
 * \brief Return the largest sibling less than or equal
 *
 * \param n	Pointer to the node
 * \return !0	Pointer to the lesser sibling
 * \return NULL	The node specified is the min
 */
struct ods_rbn *ods_rbt_greatest_lt_or_eq(struct ods_rbn *n)
{
	if (n->left)
		return n->left;
	if (n->parent && n->parent->left == n)
		return n->parent;
	return NULL;
}

/**
 * \brief Find the largest node less than or equal to a key
 *
 * \param t	Pointer to the tree
 * \param key   Pointer to the key value
 * \return !0	Pointer to the lesser sibling
 * \return NULL	The node specified is the min
 */
struct ods_rbn *ods_rbt_find_glb(struct ods_rbt *t, const void *key)
{
	struct ods_rbn *x;
	struct ods_rbn *glb = NULL;

	for (x = t->root; x; ) {
		int64_t c;
		c = t->comparator(x->key, key, t->comparator_arg);
		if (!c)
			return x;
		if (c > 0) {
			x = x->left;
		} else {
			glb = x;
			x = x->right;
		}
	}
	return glb;
}

/**
 * \brief Return the smallest sibling greater than or equal
 *
 * \param n	Pointer to the node
 * \retval !0	Pointer to the greater sibling
 * \retval NULL	The node specified is the max
 */
struct ods_rbn *ods_rbt_least_gt_or_eq(struct ods_rbn *n)
{
	if (n->right)
		return n->right;
	if (n->parent && n->parent->right == n)
		return n->parent;
	return NULL;
}

/**
 * \brief Find the smallest node greater than or equal to a key
 *
 * \param t	Pointer to the tree
 * \param key	Pointer to the key
 * \retval !0	Pointer to the greater sibling
 * \retval NULL	The node specified is the max
 */
struct ods_rbn *ods_rbt_find_lub(struct ods_rbt *t, const void *key)
{
	struct ods_rbn *x;
	struct ods_rbn *lub = NULL;

	for (x = t->root; x; ) {
		int64_t c;
		c = t->comparator(x->key, key, t->comparator_arg);
		if (!c)
			return x;
		if (c < 0) {
			x = x->right;
		} else {
			lub = x;
			x = x->left;
		}
	}
	return lub;
}

/**
 * \brief Find a node in the ODS_RBT that matches a key
 *
 * \param t	Pointer to the ODS_RBT.
 * \param key	Pointer to the key.
 * \retval !NULL Pointer to the node with the matching key.
 * \retval NULL  No node in the tree matches the key.
 */
struct ods_rbn *ods_rbt_find(struct ods_rbt *t, const void *key)
{
	struct ods_rbn *x;

	for (x = t->root; x; ) {
		int64_t c;
		c = t->comparator(x->key, key, t->comparator_arg);
		if (!c)
			return x;

		if (c > 0)
			x = x->left;
		else
			x = x->right;
	}
	return NULL;
}

struct ods_rbn *__ods_rbn_min(struct ods_rbn *n)
{
	for (; n && n->left; n = n->left);
	return n;
}

struct ods_rbn *__ods_rbn_max(struct ods_rbn *n)
{
	for (; n && n->right; n = n->right);
	return n;
}

/**
 * \brief Return the smallest (i.e leftmost) node in the ODS_RBT.
 *
 * \param t	Pointer to the ODS_RBT.
 * \return	Pointer to the node or NULL if the tree is empty.
 */
struct ods_rbn *ods_rbt_min(struct ods_rbt *t)
{
	return __ods_rbn_min(t->root);
}

/**
 * \brief Return the largest (i.e. rightmost) node in the ODS_RBT.
 *
 * \param t	Pointer to the ODS_RBT.
 * \return	Pointer to the node or NULL if the tree is empty.
 */
struct ods_rbn *ods_rbt_max(struct ods_rbt *t)
{
	return __ods_rbn_max(t->root);
}

static int ods_rbt_traverse_subtree(struct ods_rbn *n, ods_rbn_node_fn f,
				 void *fn_data, int level)
{
	int rc;
	if (n) {
		rc = ods_rbt_traverse_subtree(n->left, f, fn_data, level+1);
		if (rc)
			goto err;
		rc = f(n, fn_data, level);
		if (rc)
			goto err;
		rc = ods_rbt_traverse_subtree(n->right, f, fn_data, level+1);
		if (rc)
			goto err;
	}
	return 0;
 err:
	return rc;
}

/**
 * \brief Traverse an ODS_RBT
 *
 * Perform a recursive traversal of an ODS_RBT from left to right. For
 * each non-leaf node, a callback function is invoked with a pointer
 * to the node.
 *
 * \param t	A pointer to the ODS_RBT.
 * \param f	A pointer to the function to call as each ODS_RBT node is
 *		visited.
 * \param p	Pointer to provide as an argument to the callback
 *		function along with the ODS_RBT node pointer.
 */
int ods_rbt_traverse(struct ods_rbt *t, ods_rbn_node_fn f, void *p)
{
	if (t->root)
		return ods_rbt_traverse_subtree(t->root, f, p, 0);
	return 0;
}

/**
 * \brief Return the successor node
 *
 * Given a node in the tree, return it's successor.
 *
 * \param n	Pointer to the current node
 */
struct ods_rbn *ods_rbn_succ(struct ods_rbn *n)
{
	if (n->right)
		return __ods_rbn_min(n->right);

	if (n->parent) {
		while (n->parent && n == n->parent->right)
			n = n->parent;
		return n->parent;
	}

	return NULL;
}

/**
 * \brief Return the predecessor node
 *
 * Given a node in the tree, return it's predecessor.
 *
 * \param n	Pointer to the current node
 */
struct ods_rbn *ods_rbn_pred(struct ods_rbn *n)
{
	if (n->left)
		return __ods_rbn_max(n->left);

	if (n->parent) {
		while (n->parent && n == n->parent->left)
			n = n->parent;
		return n->parent;
	}

	return NULL;
}

static struct ods_rbn *sibling(struct ods_rbn *n)
{
	assert (n != NULL);
	assert (n->parent != NULL); /* Root node has no sibling */
	if (n == n->parent->left)
		return n->parent->right;
	else
		return n->parent->left;
}

void replace_node(struct ods_rbt *t, struct ods_rbn *oldn, struct ods_rbn *newn)
{
	if (oldn->parent == NULL) {
		t->root = newn;
		newn->parent = NULL;
	} else {
		if (oldn == oldn->parent->left)
			oldn->parent->left = newn;
		else
			oldn->parent->right = newn;
	}
	if (newn != NULL) {
		newn->parent = oldn->parent;
	}
}

static int node_color(struct ods_rbn *n) {
	return n == NULL ? ODS_RBN_BLACK : n->color;
}

/*
 * 1. Each node is either red or black:
 */
static void verify_property_1(struct ods_rbn *n) {
	assert(node_color(n) == ODS_RBN_RED || node_color(n) == ODS_RBN_BLACK);
	if (n == NULL) return;
	verify_property_1(n->left);
	verify_property_1(n->right);
}

/*
 * 2. The root node is black.
 */
static void verify_property_2(struct ods_rbn *root) {
	assert(node_color(root) == ODS_RBN_BLACK);
}

/*
* 3. All NULL leaves are black and contain no data.
* This property is assured by always treating NULL as black.
*/

/*
 * 4. Every red node has two children, and both are black (or
 * equivalently, the parent of every red node is black).
 */
static void verify_property_4(struct ods_rbn * n)
{
	if (node_color(n) == ODS_RBN_RED) {
		assert (node_color(n->left)   == ODS_RBN_BLACK);
		assert (node_color(n->right)  == ODS_RBN_BLACK);
		assert (node_color(n->parent) == ODS_RBN_BLACK);
	}
	if (n == NULL) return;
	verify_property_4(n->left);
	verify_property_4(n->right);
}

/*
 * 5. All paths from any given node to its leaf nodes contain the same
 * number of black nodes. Traverse the tree counting black nodes until
 * a leaf is reached; save this count. Compare this saved count to all
 * other paths to a leaf.
 */
static void verify_property_5_helper(struct ods_rbn * n, int black_count, int* path_black_count)
{
	if (node_color(n) == ODS_RBN_BLACK) {
		black_count++;
	}
	if (n == NULL) {
		if (*path_black_count == -1) {
			*path_black_count = black_count;
		} else {
			assert (black_count == *path_black_count);
		}
		return;
	}
	verify_property_5_helper(n->left,  black_count, path_black_count);
	verify_property_5_helper(n->right, black_count, path_black_count);
}

static void verify_property_5(struct ods_rbn *n)
{
	int black_count_path = -1;
	verify_property_5_helper(n, 0, &black_count_path);
}

static void verify_tree(struct ods_rbt *t) {
	verify_property_1(t->root);
	verify_property_2(t->root);
	/* Property 3 is implicit */
	verify_property_4(t->root);
	verify_property_5(t->root);
}

/**
 * \brief verify that the tree is correct
 */
void ods_rbt_verify(struct ods_rbt *t)
{
	verify_tree(t);
}

static int __ods_rbn_print(struct ods_rbn *ods_rbn, void *fn_data, int level)
{
	printf("%p %*c%-2d: %ld\n", ods_rbn, 80 - (level * 6), (ods_rbn->color?'B':'R'),
	       level, *((int64_t *)ods_rbn->key));
	return 0;
}

void ods_rbt_print(struct ods_rbt *t, ods_rbn_printer_t rbn_printer, void *arg)
{
	if (!rbn_printer)
		rbn_printer = __ods_rbn_print;
	ods_rbt_traverse(t, rbn_printer, arg);
}

static void delete_case1(struct ods_rbt *t, struct ods_rbn *n);
static void delete_case2(struct ods_rbt * t, struct ods_rbn *n);
static void delete_case3(struct ods_rbt *t, struct ods_rbn *n);
static void delete_case4(struct ods_rbt *t, struct ods_rbn *n);
static void delete_case5(struct ods_rbt *t, struct ods_rbn *n);
static void delete_case6(struct ods_rbt *t, struct ods_rbn *n);

void ods_rbt_del(struct ods_rbt *t, struct ods_rbn * n)
{
	struct ods_rbn *pred;
	struct ods_rbn *child;
	assert(n);
	assert(t->card);
	t->card -= 1;
	if (n->left != NULL && n->right != NULL) {
		/*
		 * Swap n with it's predecessor's location in the
		 * tree, and then delete 'n'. The simplifies the
		 * delete to the case where n->right is NULL.
		 */
		pred = find_pred(n);
		struct ods_rbn P = *pred;

		assert(pred->right == NULL);
		assert(pred->parent != NULL);

		/* Move pred to n's location */
		if (pred->parent != n) {
			/* n's predecessor is max of left subtree */
			/*
			 *            R           R
			 *             \           \
			 *              N           Pr
			 *             / \         / \
			 *            L   R  ==>  L   R
			 *           / \         / \
			 *          X   Pr      X   N
			 *             /           /
			 *            Y           Y
			 */
			pred->left = n->left;
			pred->right = n->right;
			pred->parent = n->parent;
			pred->color = n->color;
			n->color = P.color;
			n->left->parent = n->right->parent = pred;
			if (n->parent) {
				if (n->parent->left == n)
					n->parent->left = pred;
				else
					n->parent->right = pred;
			} else {
				t->root = pred;
			}

			if (node_color(n) == ODS_RBN_RED) {
				/*
				 * Removing a red node does not
				 * require rebalancing
				 */
				P.parent->right = P.left;
				return;
			}
			/* Move n to pred's location */
			n->left = P.left;
			n->right = NULL;
			n->parent = P.parent;
			if (P.left)
				P.left->parent = n;
			assert(P.parent->right == pred);
			P.parent->right = n;
		} else {
			/* n's predecessor is its left child */
			/*
			 *            R           R
			 *             \           \
			 *              N           Pr
			 *             / \         / \
			 *            Pr  R  ==>  N   R
			 *           / \         /
			 *          X   -       X
			 */
			/* pred becomes 'n' */
			pred->color = n->color;
			n->color = P.color;
			pred->parent = n->parent;
			pred->right = n->right;
			n->right->parent = pred;

			/* Attach P to parent */
			if (n->parent) {
				if (n == n->parent->left)
					n->parent->left = pred;
				else
					n->parent->right = pred;
				pred->parent = n->parent;
			} else {
				t->root = pred;
				pred->parent = NULL;
			}

			if (node_color(n) == ODS_RBN_RED) {
				/* n->left == pred. pred->left still
				 * poits to it's old left, which is
				 * correct when we're deleting 'n'.
				 * Deleting RED, no rebalance */
				return;
			}

			/* Attach 'n' to pred->left */
			n->left = P.left;
			if (P.left)
				P.left->parent = n;
			pred->left = n;
			n->parent = pred;
			n->right = NULL;
			n->parent = pred;
		}
	}

	assert(n->left == NULL || n->right == NULL);
	child = n->right == NULL ? n->left : n->right;
	struct ods_rbn *delete_me = n;

	/* If 'n' is RED, we don't need to rebalance, just remove it. */
	if (node_color(n) == ODS_RBN_RED) {
		replace_node(t, n, child);
		return;
	}

	n->color = node_color(child);

	/*
	 * If n is the root, just remove it since it will change the
	 * number of black nodes on every path to a leaf and not
	 * require rebalancing the tree.
	 */
	if (n->parent == NULL) {
		t->root = child;
		if (child) {
			child->color = ODS_RBN_BLACK;
			child->parent = NULL;
		}
		return;
	}

#ifndef __unroll_tail_recursion__
	delete_case2(t, n);
#else
 case_1:
	if (n->parent == NULL)
		return;
	/* Case 2: */
	struct ods_rbn *s = sibling(n);
	struct ods_rbn *p = n->parent;
	if (node_color(s) == ODS_RBN_RED) {
		p->color = ODS_RBN_RED;
		s->color = ODS_RBN_BLACK;
		if (n == p->left)
			rotate_left(t, p);
		else
			rotate_right(t, p);
		s = sibling(n);
		p = n->parent;
	}
	/* Case 3: */
	if (node_color(p) == ODS_RBN_BLACK &&
	    node_color(s) == ODS_RBN_BLACK &&
	    node_color(s->left) == ODS_RBN_BLACK &&
	    node_color(s->right) == ODS_RBN_BLACK) {
		s->color = ODS_RBN_RED;
		n = p;
		goto case_1;
	}
	/* Case 4: */
	if (node_color(p) == ODS_RBN_RED &&
	    node_color(s) == ODS_RBN_BLACK &&
	    node_color(s->left) == ODS_RBN_BLACK &&
	    node_color(s->right) == ODS_RBN_BLACK) {
		s->color = ODS_RBN_RED;
		p->color = ODS_RBN_BLACK;
		goto delete;
	}
	/* Case 5 */
	if (n == p->left &&
	    node_color(s) == ODS_RBN_BLACK &&
	    node_color(s->left) == ODS_RBN_RED &&
	    node_color(s->right) == ODS_RBN_BLACK) {
		s->color = ODS_RBN_RED;
		s->left->color = ODS_RBN_BLACK;
		rotate_right(t, s);
	} else if (n == p->right &&
		 node_color(s) == ODS_RBN_BLACK &&
		 node_color(s->right) == ODS_RBN_RED &&
		 node_color(s->left) == ODS_RBN_BLACK) {
		s->color = ODS_RBN_RED;
		s->right->color = ODS_RBN_BLACK;
		rotate_left(t, s);
	}
	/* Case 6 */
	s->color = p->color;
	p->color = ODS_RBN_BLACK;
	if (n == p->left) {
		assert (node_color(s->right) == ODS_RBN_RED);
		s->right->color = ODS_RBN_BLACK;
		rotate_left(t, p);
	} else {
		assert (node_color(s->left) == ODS_RBN_RED);
		s->left->color = ODS_RBN_BLACK;
		rotate_right(t, p);
	}
delete:
#endif
	replace_node(t, delete_me, child);
	if (n->parent == NULL && child != NULL) // root should be black
		child->color = ODS_RBN_BLACK;
}

static void delete_case1(struct ods_rbt * t, struct ods_rbn *n)
{
	if (n->parent == NULL)
		return;
	delete_case2(t, n);
}

/*
 * Case 2: N's sibling is RED
 *
 * In this case we exchange the colors of the parent and sibling, then
 * rotate about the parent so that the sibling becomes the parent of
 * its former parent. This does not restore the tree properties, but
 * reduces the problem to one of the remaining cases.
 */
static void delete_case2(struct ods_rbt * t, struct ods_rbn *n)
{
	struct ods_rbn *s = sibling(n);
	struct ods_rbn *p = n->parent;
	if (node_color(s) == ODS_RBN_RED) {
		p->color = ODS_RBN_RED;
		s->color = ODS_RBN_BLACK;
		if (n == p->left)
			rotate_left(t, p);
		else
			rotate_right(t, p);
	}
	delete_case3(t, n);
}

/*
 * Case 3: N's parent, sibling, and sibling's children are black.
 *
 * Paint the sibling red. Now all paths passing through N's parent
 * have one less black node than before the deletion, so we must
 * recursively run this procedure from case 1 on N's parent.
 */
static void delete_case3(struct ods_rbt *t, struct ods_rbn *n)
{
	struct ods_rbn *s = sibling(n);
	struct ods_rbn *p = n->parent;
	assert(s);
	if (node_color(p) == ODS_RBN_BLACK &&
	    node_color(s) == ODS_RBN_BLACK &&
	    node_color(s->left) == ODS_RBN_BLACK &&
	    node_color(s->right) == ODS_RBN_BLACK) {
		s->color = ODS_RBN_RED;
		delete_case1(t, p);
	} else {
		delete_case4(t, n);
	}
}

/*
 * Case 4: N's sibling and sibling's children are black, but its parent is red
 *
 * Exchange the colors of the sibling and parent; this restores the
 * tree properties.
 */
void delete_case4(struct ods_rbt *t, struct ods_rbn *n)
{
	struct ods_rbn *s = sibling(n);
	struct ods_rbn *p = n->parent;
	assert(s);
	if (node_color(p) == ODS_RBN_RED &&
	    node_color(s) == ODS_RBN_BLACK &&
	    node_color(s->left) == ODS_RBN_BLACK &&
	    node_color(s->right) == ODS_RBN_BLACK) {
		s->color = ODS_RBN_RED;
		p->color = ODS_RBN_BLACK;
	} else {
		delete_case5(t, n);
	}
}

/*
 * Case 5: There are two cases handled here which are mirror images of
 * one another:
 *
 * N's sibling S is black, S's left child is red, S's right child is
 * black, and N is the left child of its parent. We exchange the
 * colors of S and its left sibling and rotate right at S.  N's
 * sibling S is black, S's right child is red, S's left child is
 * black, and N is the right child of its parent. We exchange the
 * colors of S and its right sibling and rotate left at S.
 *
 * Both of these function to reduce us to the situation described in case 6.
 */
static void delete_case5(struct ods_rbt * t, struct ods_rbn *n)
{
	struct ods_rbn *s = sibling(n);
	struct ods_rbn *p = n->parent;
	assert(s);
	if (n == p->left &&
	    node_color(s) == ODS_RBN_BLACK &&
	    node_color(s->left) == ODS_RBN_RED &&
	    node_color(s->right) == ODS_RBN_BLACK) {
		s->color = ODS_RBN_RED;
		s->left->color = ODS_RBN_BLACK;
		rotate_right(t, s);
	} else if (n == p->right &&
		 node_color(s) == ODS_RBN_BLACK &&
		 node_color(s->right) == ODS_RBN_RED &&
		 node_color(s->left) == ODS_RBN_BLACK) {
		s->color = ODS_RBN_RED;
		s->right->color = ODS_RBN_BLACK;
		rotate_left(t, s);
	}
	delete_case6(t, n);
}

/*
 * Case 6: There are two cases handled here which are mirror images of
 * one another:
 *
 * N's sibling S is black, S's right child is red, and N is the left
 * child of its parent. We exchange the colors of N's parent and
 * sibling, make S's right child black, then rotate left at N's
 * parent.  N's sibling S is black, S's left child is red, and N is
 * the right child of its parent. We exchange the colors of N's
 * parent and sibling, make S's left child black, then rotate right
 * at N's parent.
 *
 * This accomplishes three things at once:
 *
 * - We add a black node to all paths through N, either by adding a
 * -   black S to those paths or by recoloring N's parent black.  We
 * -   remove a black node from all paths through S's red child,
 * -   either by removing P from those paths or by recoloring S.  We
 * -   recolor S's red child black, adding a black node back to all
 * -   paths through S's red child.
 *
 * S's left child has become a child of N's parent during the rotation and so is unaffected.
 */
static void delete_case6(struct ods_rbt *t, struct ods_rbn *n)
{
	struct ods_rbn *s = sibling(n);
	struct ods_rbn *p = n->parent;
	assert(s);
	s->color = p->color;
	p->color = ODS_RBN_BLACK;
	if (n == p->left) {
		assert (node_color(s->right) == ODS_RBN_RED);
		s->right->color = ODS_RBN_BLACK;
		rotate_left(t, p);
	} else {
		assert (node_color(s->left) == ODS_RBN_RED);
		s->left->color = ODS_RBN_BLACK;
		rotate_right(t, p);
	}
}

#ifdef ODS_RBT_TEST
#include <inttypes.h>
#include <time.h>
#include "ovis-test/test.h"
struct test_key {
	struct ods_rbn n;
	int64_t key;
	int ord;
};

int test_comparator(void *a, const void *b, void *arg)
{
	int64_t ai = *(int64_t *)a;
	int64_t bi = *(int64_t *)b;
	if (ai < bi)
		return -1;
	if (ai > bi)
		return 1;
	return 0;
}

int ods_rbn_print(struct ods_rbn *ods_rbn, void *fn_data, int level)
{
	struct test_key *k = (struct test_key *)ods_rbn;
	printf("%p %*c%-2d: %12d (%d)\n", ods_rbn, 80 - (level * 6), (ods_rbn->color?'B':'R'),
	       level, k->key, k->ord);
	return 0;
}

int main(int argc, char *argv[])
{
	struct ods_rbt ods_rbt;
	struct ods_rbt ods_rbtB;
	int key_count, iter_count;
	int max = -1;
	int min = 0x7FFFFFFF;
	struct test_key key;
	int x;
	int64_t kv;
	time_t t = time(NULL);
	struct test_key** keys;

	if (argc != 3) {
		printf("usage: ./ods_rbt {key-count} {iter-count}\n");
		exit(1);
	}
	key_count = atoi(argv[1]);
	iter_count = atoi(argv[2]);
	keys = calloc(key_count, sizeof(struct test_key*));

	ods_rbt_init(&ods_rbtB, test_comparator, NULL);

	/*
	 * Test Duplicates
	 */
	for (x = 0; x < key_count; x++) {
		struct test_key *k = calloc(1, sizeof *k);
		k->ord = x;
		k->key = 1000;
		ods_rbn_init(&k->n, &k->key);
		ods_rbt_ins(&ods_rbtB, &k->n);
		keys[x] = k;
	}
	struct ods_rbn *n;
	kv = 1000;
	for (n = ods_rbt_min(&ods_rbtB); n; n = ods_rbn_succ(n)) {
		struct test_key *k = container_of(n, struct test_key, n);
		TEST_ASSERT(k->key == kv, "k->key(%ld) == %ld\n", k->key, kv);
	}
	kv = 1000;
	for (n = ods_rbt_min(&ods_rbtB); n; n = ods_rbn_succ(n)) {
		struct test_key *k = container_of(n, struct test_key, n);
		TEST_ASSERT(k->key == kv, "k->key(%ld) == %ld\n", k->key, kv);
	}

	ods_rbt_verify(&ods_rbtB);
	for (x = 0; x < key_count; x++) {
		struct test_key *k = keys[x];
		ods_rbt_del(&ods_rbtB, &k->n);
	}
	TEST_ASSERT(ods_rbtB.card == 0, "Tree is empty\n");

	/*
	 * Test LUB/GLB
	 */
	int64_t test_keys[] = { 1, 3, 5, 7, 9 };
	int i;
	struct test_key *k;
	for (x = 0; x < 100; ) {
		for (i = 0; i < sizeof(test_keys) / sizeof(test_keys[0]); i++) {
			k = calloc(1, sizeof *k);
			k->ord = x++;
			ods_rbn_init(&k->n, &k->key);
			k->key = test_keys[i];
			ods_rbt_ins(&ods_rbtB, &k->n);
		}
	}

	kv = 0;
	n = ods_rbt_find_glb(&ods_rbtB, &kv);
	TEST_ASSERT(n == NULL, "glb(0) == NULL\n");
	for (i = 0; i < sizeof(test_keys) / sizeof(test_keys[0]); i++) {
		n = ods_rbt_find_glb(&ods_rbtB, &test_keys[i]);
		k = container_of(n, struct test_key, n);
		TEST_ASSERT(k->key == test_keys[i], "glb(%ld) == %ld\n", k->key, test_keys[i]);

		kv = test_keys[i] + 1;
		n = ods_rbt_find_glb(&ods_rbtB, &kv);
		k = container_of(n, struct test_key, n);
		TEST_ASSERT(k->key == test_keys[i], "glb(%ld) == %ld\n", k->key, test_keys[i]);
	}
	kv = 10;
	n = ods_rbt_find_lub(&ods_rbtB, &kv);
	TEST_ASSERT(n == NULL, "lub(10) == NULL\n");

	/* Empty the tree */
	for (n = ods_rbt_min(&ods_rbtB); n; n = ods_rbt_min(&ods_rbtB)) {
		k = container_of(n, struct test_key, n);
		ods_rbt_del(&ods_rbtB, n);
		free(k);
	}
	for (i = 0; i < 100; i++) {
		k = calloc(1, sizeof(*k));
		k->ord = x++;
		k->key = i;
		ods_rbn_init(&k->n, &k->key);
		ods_rbt_ins(&ods_rbtB, &k->n);
	}
	for (x = 0; x < 100; x += 2) {
		kv = x;
		struct ods_rbn *ods_rbn = ods_rbt_find(&ods_rbtB, &kv);
		TEST_ASSERT((ods_rbn != NULL), "%ld found.\n", kv);
	}
	srandom(t);
	ods_rbt_init(&ods_rbt, test_comparator, NULL);
	key_count = atoi(argv[1]);
	while (key_count--) {
		struct test_key *k = calloc(1, sizeof *k);
		struct ods_rbn *ods_rbn;
		ods_rbn_init(&k->n, &k->key);
		k->key = (int)random();
		ods_rbn = ods_rbt_find(&ods_rbt, &k->key);
		if (ods_rbn) {
			printf("FAIL -- DUPLICATE %d.\n", k->key);
			continue;
		}
		ods_rbt_ins(&ods_rbt, &k->n);
		if (k->key > max)
			max = k->key;
		else if (k->key < min)
			min = k->key;
	}
	// ods_rbt_traverse(&ods_rbt, ods_rbn_print, NULL);
	struct ods_rbn *min_ods_rbn = ods_rbt_min(&ods_rbt);
	struct ods_rbn *max_ods_rbn = ods_rbt_max(&ods_rbt);
	TEST_ASSERT((min_ods_rbn && ((struct test_key *)min_ods_rbn)->key == min),
		    "The min (%d) is in the tree.\n", min);
	TEST_ASSERT((max_ods_rbn && ((struct test_key *)max_ods_rbn)->key == max),
		    "The max (%d) is in the tree.\n", max);
	TEST_ASSERT((min < max),
		    "The min (%d) is less than the max (%d).\n",
		    min, max);
	if (min_ods_rbn)
		ods_rbt_del(&ods_rbt, min_ods_rbn);
	TEST_ASSERT((ods_rbt_find(&ods_rbt, &min) == NULL),
		    "Delete %d and make certain it's not found.\n",
		    min);
	if (max_ods_rbn)
		ods_rbt_del(&ods_rbt, max_ods_rbn);
	TEST_ASSERT((ods_rbt_find(&ods_rbt, &max) == NULL),
		    "Delete %d and make certain it's not found.\n", max);
	int test_iter = 0;
	while (test_iter < iter_count) {
		ods_rbt_init(&ods_rbt, test_comparator, NULL);
		/* Generate batch of random keys */
		srandom(time(NULL));
		key_count = atoi(argv[1]);
		for (x = 0; x < key_count; x++) {
			keys[x]->key = (int)random();
			keys[x]->ord = x;
			ods_rbn_init(&keys[x]->n, &keys[x]->key);
			ods_rbt_ins(&ods_rbt, &keys[x]->n);
		}
		verify_tree(&ods_rbt);
		printf("Created %d keys.\n", key_count);

		/* Test that all the inserted keys are present */
		for (x = 0; x < key_count; x++) {
			struct ods_rbn *ods_rbn = ods_rbt_find(&ods_rbt, &keys[x]->key);
			if (!ods_rbn) {
				TEST_ASSERT(ods_rbn != NULL,
					    "Key[%d] ==  %ld is not present in the tree\n",
					    x, keys[x]->key);
			}
		}
		/* Now delete them all */
		for (x = 0; x < key_count; x++) {
			int y;
			struct ods_rbn *ods_rbn;
			/* Ensure that the remaining keys are still present */
			for (y = x; y < key_count; y++) {
				struct ods_rbn *ods_rbn = ods_rbt_find(&ods_rbt, &keys[y]->key);
				if (!ods_rbn) {
					TEST_ASSERT(ods_rbn != NULL,
						    "Key[%d,%d] ==  %ld is not present "
						    "in the tree\n",
						    x, y, keys[y]->key);
				}
			}
			ods_rbn = ods_rbt_find(&ods_rbt, &keys[x]->key);
			if (ods_rbn) {
				ods_rbt_del(&ods_rbt, ods_rbn);
			} else {
				TEST_ASSERT(ods_rbn != NULL,
					    "Key[%d] ==  %ld is not present in the tree\n",
					    x, keys[x]->key);
			}
			verify_tree(&ods_rbt);
			/* Ensure that the remaining keys are still present */
			for (y = x+1; y < key_count; y++) {
				struct ods_rbn *ods_rbn = ods_rbt_find(&ods_rbt, &keys[y]->key);
				if (!ods_rbn) {
					TEST_ASSERT(ods_rbn != NULL,
						    "Key[%d,%d] ==  %ld is not present in the tree\n",
						    x, y, keys[y]->key);
				}
			}
		}
		ods_rbt_traverse(&ods_rbt, ods_rbn_print, NULL);
		test_iter += 1;
		printf("test iteration %d\n", test_iter);
	}
	return 0;
}
#endif
