#ifndef _AST_H_
#define _AST_H_
#include <regex.h>
#include <sos/sos.h>
#include <inttypes.h>

enum ast_token_e {
	ASTT_LPAREN = 1,	/* '(' */
	ASTT_RPAREN,        /* ')' */
	ASTT_ASTERISK,
	ASTT_DQSTRING,		/* "[^"]*" */
	ASTT_SQSTRING,		/* '[^']*' */
	ASTT_INTEGER,		/* [:digit:]+ */
	ASTT_FLOAT,			/* e.g. -1234.345e-10 */
	ASTT_LT,			/* '<' */
	ASTT_LE,			/* '<=' */
	ASTT_EQ,			/* '==' */
	ASTT_GE,			/* '>=' */
	ASTT_GT,			/* '>' */
	ASTT_NE,			/* '!=' */
	ASTT_COMMA,         /* ',' */
	ASTT_OR,			/* 'or' */
	ASTT_AND,			/* 'and' */
	ASTT_NOT,			/* '!' */
	ASTT_EOF,			/* End of expression */
	/* KEYWORD */
	ASTT_SELECT,        /* 'select' */
	ASTT_FROM,          /* 'from' */
	ASTT_ORDER_BY,      /* 'order_by' */
	ASTT_WHERE,         /* 'where' */
	ASTT_NAME,          /* An alphanumeric name that doesn't match
			     * any of the above */
	ASTT_ERR = 255		/* Invalid token */
};

enum ast_parse_e {
	ASTP_OK,
	ASTP_ERROR,
	ASTP_UNBALANCED_PAREN,
	ASTP_BAD_SCHEMA_NAME,
	ASTP_BAD_ATTR_NAME,
	ASTP_BAD_ATTR_TYPE,
	ASTP_SYNTAX,
};

struct ast_value_attr {
	sos_schema_t schema;
	sos_attr_t attr;
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
		struct ast_value_attr *attr;
		struct ast_term_binop *binop;
	};
};

struct ast_term_binop {
	struct ast_term *lhs;
	enum ast_token_e op;
	struct ast_term *rhs;
};

typedef struct ast_schema_entry_s *ast_schema_entry_t;
typedef struct ast_attr_entry_s {
	const char *name;
	sos_attr_t sos_attr;    /* attribute in source object */
	sos_attr_t res_attr;    /* attribute in query result object */
	sos_attr_t join_attr;   /* Best join-attr this attribute appears */
	uint32_t join_attr_idx;      /* Position the attribute appears in the join */
	uint32_t min_join_idx;
	uint32_t rank;
	ast_schema_entry_t schema;
	struct ast_value_attr *value_attr;
	TAILQ_ENTRY(ast_attr_entry_s) link;
	LIST_ENTRY(ast_attr_entry_s) join_link;
} *ast_attr_entry_t;

struct ast_schema_entry_s {
	const char *name;
	sos_schema_t schema;
	uint64_t schema_id;
	TAILQ_ENTRY(ast_name_entry_s) link;
	LIST_HEAD(join_list_head, ast_attr_entry_s) join_list;
};

struct ast {
	regex_t dqstring_re;
	regex_t sqstring_re;
	regex_t float_re;
	regex_t int_re;

	/* The SOS container */
	sos_t sos;
	sos_iter_t sos_iter;
	sos_schema_t sos_iter_schema;
	struct ast_attr_entry_s *iter_attr_e;
	sos_schema_t result_schema;

	uint64_t query_id;
	struct ast_term *where;
	struct ast_term *order_by;
	TAILQ_HEAD(ast_schema_list, ast_schema_entry_s) schema_list;
	TAILQ_HEAD(select_attr_list, ast_attr_entry_s) select_list;
	TAILQ_HEAD(where_attr_list, ast_attr_entry_s) where_list;

	/* location where error occurred */
	int pos;

	/* 0 on successful parse, !0 otherwise */
	int result;

	/* Error message when result != 0 */
	char error_msg[256];
};

extern struct ast *ast_create(sos_t sos, uint64_t query_id);
extern int ast_parse(struct ast *ast, char *expr);
extern int ast_eval(struct ast *ast, sos_obj_t obj);

#endif