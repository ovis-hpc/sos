#ifndef _AST_H_
#define _AST_H_
#include <regex.h>
#include <sos/sos.h>
#include <ods/ods_rbt.h>
#include <inttypes.h>

enum ast_token_e {
	ASTT_ERR = -1,	    /* Invalid token */
	ASTT_LPAREN = 1,    /* '(' */
	ASTT_RPAREN,	    /* ')' */
	ASTT_ASTERISK,	    /* '*' */
	ASTT_DQSTRING,	    /* "[^"]*" */
	ASTT_SQSTRING,	    /* '[^']*' */
	ASTT_INTEGER,	    /* [:digit:]+ */
	ASTT_FLOAT,	    /* e.g. -1234.345e-10 */
	ASTT_LT,	    /* '<' */
	ASTT_LE,	    /* '<=' */
	ASTT_EQ,	    /* '==' */
	ASTT_GE,	    /* '>=' */
	ASTT_GT,	    /* '>' */
	ASTT_NE,	    /* '!=' */
	ASTT_COMMA,         /* ',' */
	ASTT_OR,	    /* 'or' */
	ASTT_AND,	    /* 'and' */
	ASTT_NOT,	    /* 'not' */
	ASTT_EOF,	    /* End of expression */
	/* KEYWORD */
	ASTT_SELECT,        /* 'select' */
	ASTT_KEYWORD = ASTT_SELECT,
	ASTT_FROM,          /* 'from' */
	ASTT_ORDER_BY,      /* 'order_by' */
	ASTT_WHERE,         /* 'where' */
	ASTT_LIMIT,	    /* 'limit' */
	ASTT_RESAMPLE,	    /* 'resample' */
	ASTT_GROUP_BY,	    /* 'group_by' */
	ASTT_NAME,          /* An alphanumeric name that doesn't match
			     * any of the above */
};

enum ast_parse_e {
	ASTP_OK,
	ASTP_ERROR,
	ASTP_UNBALANCED_PAREN,
	ASTP_BAD_SCHEMA_NAME,
	ASTP_BAD_ATTR_NAME,
	ASTP_BAD_ATTR_TYPE,
	ASTP_BAD_CONST_TYPE,
	ASTP_BAD_OP_NAME,
	ASTP_SYNTAX,
	ASTP_ENOMEM,
};

typedef struct ast_attr_entry_s *ast_attr_entry_t;
struct ast_term_attr {
	sos_attr_t attr;
	struct ast_attr_limits *limits;
	ast_attr_entry_t entry;
};

struct ast_term {
	enum ast_value_kind_e {
		ASTV_CONST,
		ASTV_ATTR,
		ASTV_BINOP,
	} kind;
	struct sos_value_s value_;
	sos_value_t value;
	union {
		struct ast_term_attr *attr;
		struct ast_term_binop *binop;
	};
};

struct ast_term_binop {
	struct ast_term *lhs;
	enum ast_token_e op;
	struct ast_term *rhs;
	LIST_ENTRY(ast_term_binop) entry;
};

struct ast;

struct ast_operator_s {
	const char *key;
	void (*op)(struct ast *ast, struct ast_attr_entry_s *ae, size_t count,
		   sos_obj_t res_obj, sos_obj_t src_obj);
};

typedef struct ast_schema_entry_s *ast_schema_entry_t;
struct ast_attr_entry_s {
	const char *name;
	sos_attr_t src_attr;    /* attribute in source object */
	sos_attr_t res_attr;    /* attribute in query result object */
	sos_attr_t join_attr;   /* Best join-attr this attribute appears */
	int  join_attr_idx;	/* Position the attribute appears in the join */
	unsigned min_join_idx;
	int  rank;
	struct ast_operator_s *op;
	struct ast_term *binop;	/* The expression that this attr appears in */
	ast_schema_entry_t schema;
	struct ast_term_attr *value_attr;
	TAILQ_ENTRY(ast_attr_entry_s) link;
	LIST_ENTRY(ast_attr_entry_s) join_link;
};

struct ast_schema_entry_s {
	const char *name;
	sos_schema_t schema;
	uint64_t schema_id;
	TAILQ_ENTRY(ast_schema_entry_s) link;
	LIST_HEAD(join_list_head, ast_attr_entry_s) join_list;
};

typedef struct ast_attr_limits {
	const char *name;
	sos_attr_t attr;
	sos_value_t min_v;
	sos_value_t max_v;
	int join_idx;
	sos_attr_t join_attr;
	struct ods_rbn rbn;
} *ast_attr_limits_t;

enum ast_succ_e {
	 AST_KEY_NONE = 0,	/* No possible additional matches for this key */
	 AST_KEY_MORE = 1,
	 AST_KEY_MIN = 2, /* Key is at its min value */
	 AST_KEY_MAX = 3, /* Key is at its max value */
};

#define AST_MAX_JOIN_KEYS 16
struct ast {
	regex_t dqstring_re;
	regex_t sqstring_re;
	regex_t float_re;
	regex_t int_re;

	/* The SOS container */
	sos_t sos;
	sos_iter_t sos_iter;
 	sos_schema_t sos_iter_schema;
	char *iter_attr_name;
	struct ast_attr_entry_s *iter_attr_e;
	sos_schema_t result_schema;
	int result_count;	/* Number of objects returned so far */
	int result_limit;	/* Maximum objects returned */
	double bin_width;	/* Resample bin width */
	struct ods_rbt bin_tree;/* Tree of bin objects */
	struct ods_rbt group_tree;/* Tree of group objects */
	ods_idx_t bin_cmp_arg;

	/* Key used to position iterator */
	sos_key_t key;

	/* Number of attributes in the iterator key */
	int key_count;

	/* Attribute key limits in precedence order */
	struct ast_attr_limits *key_limits[AST_MAX_JOIN_KEYS];

	/* !0 if there may be successor matching recods in the
	 * index. Indexed by the key order */
	enum ast_succ_e succ_key[AST_MAX_JOIN_KEYS];

	uint64_t query_id;
	struct ast_term *where;

	TAILQ_HEAD(ast_schema_list, ast_schema_entry_s) schema_list;
	TAILQ_HEAD(select_attr_list, ast_attr_entry_s) select_list;
	TAILQ_HEAD(where_attr_list, ast_attr_entry_s) where_list;
	TAILQ_HEAD(index_attr_list, ast_attr_entry_s) index_list;
	TAILQ_HEAD(group_by_list, ast_attr_entry_s) group_list;
	sos_comp_key_spec_t group_comp_key_spec;
	struct ods_idx_comparator *group_cmp;
	LIST_HEAD(binop_list, ast_term_binop) binop_list;

	struct ods_rbt attr_tree;

	/* location where error occurred */
	int pos;

	/* 0 on successful parse, !0 otherwise */
	int result;

	/* Error message when result != 0 */
	char error_msg[256];
};

enum ast_eval_e {
	 AST_EVAL_MATCH = 1,
	 AST_EVAL_NOMATCH = 2,
	 AST_EVAL_EMPTY = 3
};

extern struct ast *ast_create(sos_t sos, uint64_t query_id);
extern void ast_destroy(struct ast *);
extern int ast_parse(struct ast *ast, char *expr);
extern enum ast_eval_e ast_eval(struct ast *ast, sos_obj_t obj);
extern enum ast_eval_e ast_eval_limits(struct ast *ast, sos_obj_t obj);
extern struct ast_term *ast_find_term(struct ast_term *term, const char *name);
extern int ast_start_key(struct ast *ast, sos_key_t start_key);
extern int ast_resample_obj_add(struct ast *ast, sos_obj_t obj);
extern sos_obj_t ast_resample_obj_next(struct ast *ast);
extern int ast_group_obj_add(struct ast *ast, sos_obj_t obj);
sos_obj_t ast_group_obj_next(struct ast *ast);
#endif
