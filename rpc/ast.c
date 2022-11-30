/* -*- c-basic-offset: 8 -*- */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <regex.h>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <sos/sos.h>
#include "sos_priv.h"
#include "ast.h"

static struct ast_term *ast_parse_binop(struct ast *ast, const char *expr, int *ppos);
static void ast_term_destroy(struct ast *ast, struct ast_term *term);

static int is_comparator(enum ast_token_e token)
{
	if ((token >= ASTT_LT && token <= ASTT_NE)
		|| token == ASTT_AND
		|| token == ASTT_OR)
		return 1;
	return 0;
}

struct ast_key_word_s {
	const char *key;
	enum ast_token_e token;
};

static int key_word_comparator(const void *a_, const void *b_)
{
	const char *a = a_;
	struct ast_key_word_s const *b = b_;
	return strcasecmp(a, b->key);
}

static struct ast_key_word_s key_words[] = {
	{ "and", ASTT_AND },
	{ "from", ASTT_FROM },
	{ "not", ASTT_NOT },
	{ "or", ASTT_OR },
	{ "order_by", ASTT_ORDER_BY },
	{ "select", ASTT_SELECT },
	{ "where", ASTT_WHERE },
};

static void ast_enomem(struct ast *ast, int pos)
{
	ast->result = ASTP_ENOMEM;
	ast->pos = pos;
	snprintf(ast->error_msg, sizeof(ast->error_msg), "Insufficient memory");
}

enum ast_token_e ast_lex(struct ast *ast, const char *expr, int *ppos, char **token_val)
{
	static char token_str[256];
	const char *s = &expr[*ppos];
	int rc;

	if (*ppos >= strlen(expr))
		return ASTT_EOF;
	while (isspace(*s)) {
		s++;
		*ppos +=1 ;
	}
	if (*ppos >= strlen(expr))
		return ASTT_EOF;

	token_str[0] = '\0';
	*token_val = token_str;

	/* DQuoted string */
	if (*s == '\"') {
		regmatch_t match[2];
		rc = regexec(&ast->dqstring_re, s, 2, match, REG_EXTENDED);
		if (rc == 0) {
			strncpy(token_str,
				&s[1],					/* skip starting quote */
				match[0].rm_eo - 1);
			token_str[match[0].rm_eo - 1] = '\0';
			*ppos += match[0].rm_eo + 1;	/* skip closing quote */;
			return ASTT_DQSTRING;
		}
		strncpy(token_str, s, sizeof(token_str));
		return ASTT_ERR;
	}
	/* SQuoted string */
	if (*s == '\'') {
		regmatch_t match[2];
		rc = regexec(&ast->sqstring_re, s, 2, match, REG_EXTENDED);
		if (rc == 0) {
			strncpy(token_str,
				&s[1],					/* skip starting quote */
				match[0].rm_eo - 1);
			token_str[match[0].rm_eo-1] = '\0';
			*ppos += match[0].rm_eo + 1;	/* skip closing quote */;
			return ASTT_SQSTRING;
		}
		strncpy(token_str, s, sizeof(token_str));
		return ASTT_ERR;
	}
	if (*s == '-' || *s == '+' || isdigit(*s)) {
		char number[255];
		regmatch_t match[2];
		int i;
		for (i = 0; i < sizeof(number) - 1
			     && (isdigit(s[i])
				 || s[i] == '+' || s[i] == '-'
				 || s[i] == '.'
				 || s[i] == 'e' || s[i] == 'E'); i++)
			{
				number[i] = s[i];
			}
		number[i] = '\0';

		/* Float? */
		rc = regexec(&ast->float_re, number, 2, match, REG_EXTENDED);
		if (rc == 0) {
			strcpy(token_str, number);
			*ppos += strlen(number);
			return ASTT_FLOAT;
		}
		/* Integer? */
		rc = regexec(&ast->int_re, number, 2, match, REG_EXTENDED);
		if (rc == 0) {
			strcpy(token_str, number);
			*ppos += strlen(number);
			return ASTT_INTEGER;
		}
		strncpy(token_str, s, sizeof(token_str));
		return ASTT_ERR;
	}
	if (s[0] == '*') {
		*ppos += 1;
		token_str[0] = s[0];
		token_str[1] = '\0';
		return ASTT_ASTERISK;
	}
	if (s[0] == ',') {
		*ppos += 1;
		token_str[0] = s[0];
		token_str[1] = '\0';
		return ASTT_COMMA;
	}
	if (s[0] == '(') {
		*ppos += 1;
		token_str[0] = s[0];
		token_str[1] = '\0';
		return ASTT_LPAREN;
	}
	if (s[0] == ')') {
		*ppos += 1;
		token_str[0] = s[0];
		token_str[1] = '\0';
		return ASTT_RPAREN;
	}
	if (s[0] == '=' && s[1] == '=') {
		*ppos += 2;
		strncpy(token_str, s, 2);
		token_str[2] = '\0';
		return ASTT_EQ;
	}
	if (s[0] == '<') {
		if (s[1] == '=') {
			*ppos +=2 ;
			strncpy(token_str, s, 2);
			token_str[2] = '\0';
			return ASTT_LE;
		}
		token_str[0] = s[0];
		token_str[1] = '\0';
		*ppos += 1;
		return ASTT_LT;
	}
	if (s[0] == '>') {
		if (s[1] == '=') {
			*ppos += 2;
			strncpy(token_str, s, 2);
			token_str[2] = '\0';
			return ASTT_GE;
		}
		*ppos += 1;
		token_str[0] = s[0];
		token_str[1] = '\0';
		return ASTT_GT;
	}
	if (s[0] == '!' && s[1] == '=') {
		*ppos += 2;
		strncpy(token_str, s, 2);
		token_str[2] = '\0';
		return ASTT_NE;
	}
	/* Look for a keyword */
	if (isalpha(*s)) {
		char keyword[255];
		struct ast_key_word_s *kw;
		rc = 0;
		while ((isalnum(*s)
			|| *s == '_' || *s == '$' || *s == '#' || *s == '.')
		       && rc < sizeof(keyword) - 1) {
			keyword[rc++] = *s;
			s++;
		}
		keyword[rc] = '\0';

		/* Keywords comparator */
		kw = bsearch(keyword,
			     key_words, sizeof(key_words) / sizeof(key_words[0]),
			     sizeof (struct ast_key_word_s),
			     key_word_comparator);
		strcpy(token_str, keyword);
		*ppos += strlen(keyword);
		if (kw)
			return kw->token;
		else
			return ASTT_NAME;
	}
	for (rc = 0; !isspace(*s) && rc < sizeof(token_str) - 1; rc++)
		token_str[rc] = *s++;
	token_str[rc] = '\0';
	return ASTT_ERR;
}

/*
 * Attribute names may encode a schema name as follows:
 *      <schema_name> '[' <attr_name> ']'
 * If the <attr_name> alone is specified, the default schema is used.
 */
static enum ast_parse_e parse_attr(struct ast *ast, const char *name, struct ast_term_attr *value_attr)
{
	ast_attr_entry_t ae = calloc(1, sizeof *ae);
	if (!ae) {
		ast_enomem(ast, 0);
		return ast->result;
	}
	ae->value_attr = value_attr;
	ae->name = strdup(name);
	if (!ae->name) {
		ast_enomem(ast, 0);
		free(ae);
		return ast->result;
	}
	ae->rank = 0;
	ae->join_attr_idx = -1;
	value_attr->entry = ae;
	TAILQ_INSERT_TAIL(&ast->where_list, ae, link);
	return 0;
}

/*
 * A term is one of:
 * - string,
 * - integer,
 * - float,
 * - attribute
 */
static struct ast_term *ast_parse_term(struct ast *ast, const char *expr, int *ppos)
{
	char *token_str;
	enum ast_token_e token;
	struct ast_term *term;
	enum ast_parse_e err;
	int rparen;
	struct ast_term *newb;

	token = ast_lex(ast, expr, ppos, &token_str);
	switch (token) {
	case ASTT_LPAREN:
		rparen = 0;
		term = ast_parse_binop(ast, expr, ppos);
		if (!term)
			return NULL;
		while (!rparen) {
			token = ast_lex(ast, expr, ppos, &token_str);
			switch (token) {
			case ASTT_RPAREN:
				rparen = 1;
				break;
			case ASTT_OR:
			case ASTT_AND:
			case ASTT_NOT:
				newb = calloc(1, sizeof(*newb));
				newb->kind = ASTV_BINOP;
				newb->binop = calloc(1, sizeof(*newb->binop));
				LIST_INSERT_HEAD(&ast->binop_list, newb->binop, entry);
				newb->binop->lhs = term;
				newb->binop->op = token;
				newb->binop->rhs = ast_parse_binop(ast, expr, ppos);
				if (ast->result)
					break;
				term = newb;
				break;
			default:
				ast->result = ASTP_UNBALANCED_PAREN;
				ast->pos = *ppos;
				snprintf(ast->error_msg, sizeof(ast->error_msg),
					 "Expected ')' but got '%s'", token_str);
				ast_term_destroy(ast, term);
				term = NULL;
				rparen = 1;
				break;
			}
		}
		break;
	case ASTT_NAME:	/* Attribute */
		term = calloc(1, sizeof(*term));
		if (!term) {
			ast_enomem(ast, *ppos);
			break;
		}
		term->kind = ASTV_ATTR;
		term->attr = calloc(1, sizeof(*term->attr));
		if (!term->attr) {
			ast_enomem(ast, *ppos);
			free(term);
			term = NULL;
			break;
		}
		err = parse_attr(ast, token_str, term->attr);
		if (err) {
			free(term->attr);
			free(term);
			term = NULL;
			break;
		}
		break;
	case ASTT_DQSTRING:
	case ASTT_SQSTRING:	/* String value */
		term = calloc(1, sizeof(*term));
		if (!term) {
			ast_enomem(ast, *ppos);
			break;
		}
		term->kind = ASTV_CONST;
		term->value = sos_value_init_const(&term->value_,
						   SOS_TYPE_CHAR_ARRAY,
						   token_str,
						   strlen(token_str));
		break;
	case ASTT_INTEGER:
		term = calloc(1, sizeof(*term));
		if (!term) {
			ast_enomem(ast, *ppos);
			break;
		}
		term->kind = ASTV_CONST;
		term->value = sos_value_init_const(&term->value_, SOS_TYPE_INT64,
						   strtol(token_str, NULL, 0));
		break;
	case ASTT_FLOAT:
		term = calloc(1, sizeof(*term));
		if (!term) {
			ast_enomem(ast, *ppos);
			break;
		}
		term->kind = ASTV_CONST;
		term->value = sos_value_init_const(
						   &term->value_,
						   SOS_TYPE_DOUBLE,
						   strtod(token_str, NULL));
		break;
	default:
		ast->result = ASTP_ERROR;
		ast->pos = *ppos;
		term = NULL;
	}
	return term;
}

/*
 * Update the attributes value min and max. If the value_term is not a
 * CONST, ignore the update
 */
static int update_attr_limits(struct ast *ast, struct ast_term *attr_term,
			      struct ast_term *value_term, enum ast_token_e op)
{
	struct ast_attr_limits *limits;
	const char *attr_name;
	sos_type_t attr_type;
	sos_type_t value_type;
	struct ods_rbn *rbn;

	/* Ignore values that are not constants */
	if (value_term->kind != ASTV_CONST)
		return 0;

	attr_name = sos_attr_name(attr_term->attr->attr);
	attr_type = sos_attr_type(attr_term->attr->attr);
	value_type = sos_value_type(value_term->value);

	/* Ensure that the attribute type is compatible with the constant value */
	switch (attr_type) {
	case SOS_TYPE_INT16:
	case SOS_TYPE_UINT16:
	case SOS_TYPE_INT32:
	case SOS_TYPE_UINT32:
	case SOS_TYPE_INT64:
	case SOS_TYPE_UINT64:
		if (value_type < SOS_TYPE_INT16 || value_type > SOS_TYPE_UINT64)
			goto type_err;
		break;
	case SOS_TYPE_FLOAT:
	case SOS_TYPE_DOUBLE:
		switch (value_type) {
		case SOS_TYPE_FLOAT:
		case SOS_TYPE_DOUBLE:
		case SOS_TYPE_INT64:
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT32:
		case SOS_TYPE_UINT32:
			break;
		default:
			goto type_err;
		}
		break;
	case SOS_TYPE_TIMESTAMP:
		switch (value_type) {
		case SOS_TYPE_DOUBLE:
		case SOS_TYPE_FLOAT:
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT64:
		case SOS_TYPE_UINT32:
		case SOS_TYPE_INT32:
		case SOS_TYPE_TIMESTAMP:
		case SOS_TYPE_STRING:
			break;
		default:
			goto type_err;
		}
		break;
	case SOS_TYPE_LONG_DOUBLE:
	default:
		if (attr_type != sos_value_type(value_term->value))
			goto type_err;
	}

	rbn = ods_rbt_find(&ast->attr_tree, attr_name);
	if (!rbn) {
		/* Allocate a new attribute */
		limits = calloc(1, sizeof(*limits));
		limits->name = strdup(attr_name);
		limits->attr = attr_term->attr->attr;
		if (attr_type != SOS_TYPE_BYTE_ARRAY &&
		    /* TODO: Array support? */
		    attr_type != SOS_TYPE_STRING) {
			limits->min_v = NULL;
			limits->max_v = NULL;
		} else {
			/* Variable length strings don't have a bounded min/max */
			limits->min_v = value_term->value;
			limits->max_v = value_term->value;
		}
		limits->join_idx = -1;
		limits->join_attr = NULL;
		ods_rbn_init(&limits->rbn, (void *)limits->name);
		ods_rbt_ins(&ast->attr_tree, &limits->rbn);
	} else {
		limits = container_of(rbn, struct ast_attr_limits, rbn);
	}
	attr_term->attr->limits = limits;
	/* Cast the value_term based on the type of the attribute */
	switch (sos_attr_type(attr_term->attr->attr)) {
	case SOS_TYPE_INT16:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_UINT64:
			value_term->value->type = SOS_TYPE_INT16;
			value_term->value->data->prim.int16_ =
				(int16_t)value_term->value->data->prim.uint64_;
			break;
		case SOS_TYPE_INT64:
			value_term->value->type = SOS_TYPE_INT16;
			value_term->value->data->prim.int16_ =
				(int16_t)value_term->value->data->prim.int64_;
			break;
		case SOS_TYPE_UINT32:
			value_term->value->type = SOS_TYPE_INT16;
			value_term->value->data->prim.int16_ =
				(int16_t)value_term->value->data->prim.uint32_;
			break;
		case SOS_TYPE_INT32:
			value_term->value->type = SOS_TYPE_INT16;
			value_term->value->data->prim.int16_ =
				(int16_t)value_term->value->data->prim.int32_;
			break;
		case SOS_TYPE_FLOAT:
			value_term->value->type = SOS_TYPE_INT16;
			value_term->value->data->prim.int16_ =
				(int16_t)value_term->value->data->prim.float_;
			break;
		case SOS_TYPE_DOUBLE:
			value_term->value->type = SOS_TYPE_INT16;
			value_term->value->data->prim.int16_ =
				(int16_t)value_term->value->data->prim.double_;
			break;
		default:
			break;
		}
		break;
	case SOS_TYPE_UINT16:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_UINT64:
			value_term->value->type = SOS_TYPE_UINT16;
			value_term->value->data->prim.uint16_ =
				(uint16_t)value_term->value->data->prim.uint64_;
			break;
		case SOS_TYPE_INT64:
			value_term->value->type = SOS_TYPE_UINT16;
			value_term->value->data->prim.uint16_ =
				(uint16_t)value_term->value->data->prim.int64_;
			break;
		case SOS_TYPE_UINT32:
			value_term->value->type = SOS_TYPE_UINT16;
			value_term->value->data->prim.uint16_ =
				(uint16_t)value_term->value->data->prim.uint32_;
			break;
		case SOS_TYPE_INT32:
			value_term->value->type = SOS_TYPE_UINT16;
			value_term->value->data->prim.uint16_ =
				(uint16_t)value_term->value->data->prim.int32_;
			break;
		case SOS_TYPE_FLOAT:
			value_term->value->type = SOS_TYPE_UINT16;
			value_term->value->data->prim.uint16_ =
				(uint16_t)value_term->value->data->prim.float_;
			break;
		case SOS_TYPE_DOUBLE:
			value_term->value->type = SOS_TYPE_UINT16;
			value_term->value->data->prim.uint16_ =
				(uint16_t)value_term->value->data->prim.double_;
			break;
		default:
			break;
		}
		break;
	case SOS_TYPE_INT32:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_UINT64:
			value_term->value->type = SOS_TYPE_INT32;
			value_term->value->data->prim.int32_ =
				(int32_t)value_term->value->data->prim.uint64_;
			break;
		case SOS_TYPE_INT64:
			value_term->value->type = SOS_TYPE_INT32;
			value_term->value->data->prim.int32_ =
				(int32_t)value_term->value->data->prim.int64_;
			break;
		case SOS_TYPE_UINT32:
			value_term->value->type = SOS_TYPE_INT32;
			value_term->value->data->prim.int32_ =
				(int32_t)value_term->value->data->prim.uint32_;
			break;
		case SOS_TYPE_UINT16:
			value_term->value->type = SOS_TYPE_INT32;
			value_term->value->data->prim.int32_ =
				(int32_t)value_term->value->data->prim.uint16_;
			break;
		case SOS_TYPE_INT16:
			value_term->value->type = SOS_TYPE_INT32;
			value_term->value->data->prim.int32_ =
				(int32_t)value_term->value->data->prim.int16_;
			break;
		case SOS_TYPE_FLOAT:
			value_term->value->type = SOS_TYPE_INT32;
			value_term->value->data->prim.int32_ =
				(int32_t)value_term->value->data->prim.float_;
			break;
		case SOS_TYPE_DOUBLE:
			value_term->value->type = SOS_TYPE_INT32;
			value_term->value->data->prim.int32_ =
				(int32_t)value_term->value->data->prim.double_;
			break;
		default:
			break;
		}
		break;
	case SOS_TYPE_UINT32:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_UINT64:
			value_term->value->type = SOS_TYPE_UINT32;
			value_term->value->data->prim.uint32_ =
				(uint32_t)value_term->value->data->prim.uint64_;
			break;
		case SOS_TYPE_INT64:
			value_term->value->type = SOS_TYPE_UINT32;
			value_term->value->data->prim.uint32_ =
				(uint32_t)value_term->value->data->prim.int64_;
			break;
		case SOS_TYPE_INT32:
			value_term->value->type = SOS_TYPE_INT32;
			value_term->value->data->prim.uint32_ =
				(uint32_t)value_term->value->data->prim.int32_;
			break;
		case SOS_TYPE_UINT16:
			value_term->value->type = SOS_TYPE_UINT32;
			value_term->value->data->prim.uint32_ =
				(uint32_t)value_term->value->data->prim.uint16_;
			break;
		case SOS_TYPE_INT16:
			value_term->value->type = SOS_TYPE_UINT32;
			value_term->value->data->prim.uint32_ =
				(uint32_t)value_term->value->data->prim.int16_;
			break;
		case SOS_TYPE_FLOAT:
			value_term->value->type = SOS_TYPE_UINT32;
			value_term->value->data->prim.uint32_ =
				(uint32_t)value_term->value->data->prim.float_;
			break;
		case SOS_TYPE_DOUBLE:
			value_term->value->type = SOS_TYPE_UINT32;
			value_term->value->data->prim.uint32_ =
				(uint32_t)value_term->value->data->prim.double_;
			break;
		default:
			break;
		}
		break;
	case SOS_TYPE_INT64:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_UINT64:
			value_term->value->type = SOS_TYPE_INT64;
			value_term->value->data->prim.int64_ =
				(int64_t)value_term->value->data->prim.uint64_;
			break;
		case SOS_TYPE_UINT32:
			value_term->value->type = SOS_TYPE_INT64;
			value_term->value->data->prim.int64_ =
				(int64_t)value_term->value->data->prim.uint32_;
			break;
		case SOS_TYPE_INT32:
			value_term->value->type = SOS_TYPE_INT64;
			value_term->value->data->prim.int32_ =
				(int64_t)value_term->value->data->prim.int32_;
			break;
		case SOS_TYPE_UINT16:
			value_term->value->type = SOS_TYPE_INT64;
			value_term->value->data->prim.int64_ =
				(int64_t)value_term->value->data->prim.uint16_;
			break;
		case SOS_TYPE_INT16:
			value_term->value->type = SOS_TYPE_INT64;
			value_term->value->data->prim.int64_ =
				(int64_t)value_term->value->data->prim.int16_;
			break;
		case SOS_TYPE_FLOAT:
			value_term->value->type = SOS_TYPE_INT64;
			value_term->value->data->prim.int64_ =
				(int64_t)value_term->value->data->prim.float_;
			break;
		case SOS_TYPE_DOUBLE:
			value_term->value->type = SOS_TYPE_INT64;
			value_term->value->data->prim.int64_ =
				(int64_t)value_term->value->data->prim.double_;
			break;
		default:
			break;
		}
		break;
	case SOS_TYPE_UINT64:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_INT64:
			value_term->value->type = SOS_TYPE_UINT64;
			value_term->value->data->prim.uint64_ =
				(uint64_t)value_term->value->data->prim.int64_;
			break;
		case SOS_TYPE_UINT32:
			value_term->value->type = SOS_TYPE_UINT64;
			value_term->value->data->prim.uint64_ =
				(uint64_t)value_term->value->data->prim.uint32_;
			break;
		case SOS_TYPE_INT32:
			value_term->value->type = SOS_TYPE_UINT64;
			value_term->value->data->prim.uint32_ =
				(uint64_t)value_term->value->data->prim.int32_;
			break;
		case SOS_TYPE_UINT16:
			value_term->value->type = SOS_TYPE_UINT64;
			value_term->value->data->prim.uint64_ =
				(uint64_t)value_term->value->data->prim.uint16_;
			break;
		case SOS_TYPE_INT16:
			value_term->value->type = SOS_TYPE_UINT64;
			value_term->value->data->prim.uint64_ =
				(uint64_t)value_term->value->data->prim.int16_;
			break;
		case SOS_TYPE_FLOAT:
			value_term->value->type = SOS_TYPE_UINT64;
			value_term->value->data->prim.uint64_ =
				(uint64_t)value_term->value->data->prim.float_;
			break;
		case SOS_TYPE_DOUBLE:
			value_term->value->type = SOS_TYPE_UINT64;
			value_term->value->data->prim.uint64_ =
				(uint64_t)value_term->value->data->prim.double_;
			break;
		default:
			break;
		}
		break;
	case SOS_TYPE_DOUBLE:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_FLOAT:
			value_term->value->type = SOS_TYPE_DOUBLE;
			value_term->value->data->prim.double_ =
				(double)value_term->value->data->prim.float_;
			break;
		case SOS_TYPE_UINT64:
			value_term->value->type = SOS_TYPE_DOUBLE;
			value_term->value->data->prim.double_ =
				(double)value_term->value->data->prim.uint64_;
			break;
		case SOS_TYPE_INT64:
			value_term->value->type = SOS_TYPE_DOUBLE;
			value_term->value->data->prim.double_ =
				(double)value_term->value->data->prim.int64_;
			break;
		case SOS_TYPE_UINT32:
			value_term->value->type = SOS_TYPE_DOUBLE;
			value_term->value->data->prim.double_ =
				(double)value_term->value->data->prim.uint32_;
			break;
		case SOS_TYPE_INT32:
			value_term->value->type = SOS_TYPE_DOUBLE;
			value_term->value->data->prim.double_ =
				(double)value_term->value->data->prim.int32_;
			break;
		default:
			break;
		}
		break;
	case SOS_TYPE_FLOAT:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_DOUBLE:
			value_term->value->type = SOS_TYPE_FLOAT;
			value_term->value->data->prim.float_ =
				(float)value_term->value->data->prim.double_;
			break;
		case SOS_TYPE_UINT64:
			value_term->value->type = SOS_TYPE_FLOAT;
			value_term->value->data->prim.float_ =
				(float)value_term->value->data->prim.uint64_;
			break;
		case SOS_TYPE_INT64:
			value_term->value->type = SOS_TYPE_FLOAT;
			value_term->value->data->prim.float_ =
				(float)value_term->value->data->prim.int64_;
			break;
		case SOS_TYPE_UINT32:
			value_term->value->type = SOS_TYPE_FLOAT;
			value_term->value->data->prim.float_ =
				(float)value_term->value->data->prim.uint32_;
			break;
		case SOS_TYPE_INT32:
			value_term->value->type = SOS_TYPE_FLOAT;
			value_term->value->data->prim.float_ =
				(float)value_term->value->data->prim.int32_;
			break;
		default:
			break;
		}
		break;
	case SOS_TYPE_TIMESTAMP:
		switch (sos_value_type(value_term->value)) {
		case SOS_TYPE_STRING:
			/* Expecting secs,usecs */
			{
				char *s = strtok(value_term->value->data->array.data.char_, ":.");
				char *u = strtok(NULL, ":.");
				if (!s || !u)
					goto type_err;
				value_term->value->type = SOS_TYPE_TIMESTAMP;
				value_term->value->data->prim.timestamp_.tv.tv_sec = atoi(s);
				value_term->value->data->prim.timestamp_.tv.tv_usec = atoi(u);
			}
			break;
		case SOS_TYPE_UINT64:
		case SOS_TYPE_INT64:
			value_term->value->type = SOS_TYPE_TIMESTAMP;
			value_term->value->data->prim.timestamp_.tv.tv_sec =
				value_term->value->data->prim.uint64_;
			value_term->value->data->prim.timestamp_.tv.tv_usec = 0;
			break;
		case SOS_TYPE_UINT32:
		case SOS_TYPE_INT32:
			value_term->value->type = SOS_TYPE_TIMESTAMP;
			value_term->value->data->prim.timestamp_.tv.tv_sec =
				value_term->value->data->prim.uint32_;
			value_term->value->data->prim.timestamp_.tv.tv_usec = 0;
			break;
		case SOS_TYPE_DOUBLE:
			value_term->value->type = SOS_TYPE_TIMESTAMP;
			value_term->value->data->prim.timestamp_.tv.tv_sec =
				(uint32_t)floor(value_term->value->data->prim.double_);
			value_term->value->data->prim.timestamp_.tv.tv_usec =
				(uint32_t)(value_term->value->data->prim.double_ -
					   floor(value_term->value->data->prim.double_)
					   * 1.e6);
			break;
		case SOS_TYPE_FLOAT:
			value_term->value->type = SOS_TYPE_TIMESTAMP;
			value_term->value->data->prim.timestamp_.tv.tv_sec =
				(uint32_t)floor(value_term->value->data->prim.float_);
			value_term->value->data->prim.timestamp_.tv.tv_usec =
				(uint32_t)(value_term->value->data->prim.float_ -
					   floor(value_term->value->data->prim.float_)
					   * 1.e6);
			break;
		default:
			break;
		}
		break;
	default:
		/*
		 * Coerce the type of the value to the attribute
		 * type. This is necessary because all integer values
		 * in the SQL are signed while the attributes may be
		 * unsigned.
		 */
		value_term->value->type = sos_attr_type(attr_term->attr->attr);
		break;
	}

	assert(limits);
	switch (op) {
	case ASTT_LT:
	case ASTT_LE:
		if (NULL == limits->max_v || sos_value_cmp(value_term->value, limits->max_v) > 0)
			limits->max_v = value_term->value;
		break;
	case ASTT_EQ:
		if (NULL == limits->min_v || sos_value_cmp(value_term->value, limits->min_v) < 0)
			limits->min_v = value_term->value;
		if (NULL == limits->max_v || sos_value_cmp(value_term->value, limits->max_v) > 0)
			limits->max_v = value_term->value;
		break;
	case ASTT_GE:
	case ASTT_GT:
		if (NULL == limits->min_v || sos_value_cmp(value_term->value, limits->min_v) < 0)
			limits->min_v = value_term->value;
		break;
	case ASTT_NE:
	default:
		break;
	}
	return 0;
 type_err:
	snprintf(ast->error_msg, sizeof(ast->error_msg),
		 "The attribute %s with type %s does not match the constant of type %s",
		 attr_name, sos_value_type_name(attr_type),
		 sos_value_type_name(sos_value_type(value_term->value)));
	ast->result = ASTP_BAD_CONST_TYPE;
	return ast->result;
}

/*
 * <term> <op> <term>
 */
static struct ast_term *ast_parse_binop(struct ast *ast, const char *expr, int *ppos)
{
	char *token_str;
	int next_pos;
	struct ast_term *term;
	struct ast_term_binop *binop = calloc(1, sizeof(*binop));
	binop->lhs = ast_parse_term(ast, expr, ppos);
	if (!binop->lhs) {
		free(binop);
		return NULL;
	}
	next_pos = *ppos;
	binop->op = ast_lex(ast, expr, &next_pos, &token_str);
	if (binop->op == ASTT_EOF || binop->op >= ASTT_KEYWORD) {
		/*
		 * If lhs is a binop, it's ok, otherwise, it's a
		 * syntax error
		 */
		if (binop->lhs->kind == ASTV_BINOP) {
			term = binop->lhs;
			free(binop);
			return term;
		} else {
			ast->result = ASTP_SYNTAX;
			ast->pos = next_pos;
			snprintf(ast->error_msg, sizeof(ast->error_msg),
				"Unexpected token '%s'", token_str);
			ast_term_destroy(ast, binop->lhs);
			free(binop);
			return NULL;
		}
	}
	*ppos = next_pos;
	if (!is_comparator(binop->op)) {
		ast->result = ASTP_SYNTAX;
		ast->pos = *ppos;
		snprintf(ast->error_msg, sizeof(ast->error_msg),
			"Expected 'and', 'or', '<', '<=', '==', '>=', or '>' but got '%s'", token_str);
		ast_term_destroy(ast, binop->lhs);
		free(binop);
		return NULL;
	}

	switch (binop->op) {
	case ASTT_AND:
	case ASTT_OR:
	case ASTT_NOT:
		/* Has to be a binop */
		binop->rhs = ast_parse_binop(ast, expr, ppos);
		break;
	default:
		/* Can be any term */
		binop->rhs = ast_parse_term(ast, expr, ppos);
		break;
	}
	if (!binop->rhs) {
		ast_term_destroy(ast, binop->lhs);
		free(binop);
		return NULL;
	}
	term = calloc(1, sizeof(*term));
	term->kind = ASTV_BINOP;
	term->binop = binop;
	LIST_INSERT_HEAD(&ast->binop_list, binop, entry);

	return term;
}

int ast_parse_select_clause(struct ast *ast, const char *expr, int *ppos)
{
	char *token_str;
	enum ast_token_e token;
	int next_pos = *ppos;
	TAILQ_INIT(&ast->select_list);

	for (token = ast_lex(ast, expr, &next_pos, &token_str);
	     token == ASTT_NAME || token == ASTT_ASTERISK;
	     token = ast_lex(ast, expr, &next_pos, &token_str)) {
		struct ast_attr_entry_s *ae = calloc(1, sizeof *ae);
		ae->name = strdup(token_str);
		*ppos = next_pos;	/* consume this token */
		TAILQ_INSERT_TAIL(&ast->select_list, ae, link);
		/* Check for a ',' indicating another name */
		token = ast_lex(ast, expr, &next_pos, &token_str);
		if (token != ASTT_COMMA)
			/* End of schema name list, do not consume token */
			break;
	}
	return ast->result;
}

int ast_parse_from_clause(struct ast *ast, const char *expr, int *ppos)
{
	char *token_str;
	enum ast_token_e token;
	int next_pos = *ppos;
	TAILQ_INIT(&ast->schema_list);

	/*
	 * Quoted strings are supported to allow for schema with special
	 * characters such as '-', '@', etc...
	 */
	for (token = ast_lex(ast, expr, &next_pos, &token_str);
	     token == ASTT_NAME || token == ASTT_DQSTRING || token == ASTT_SQSTRING;
	     token = ast_lex(ast, expr, &next_pos, &token_str))
	{
		struct ast_schema_entry_s *se = calloc(1, sizeof *se);
		se->name = strdup(token_str);
		LIST_INIT(&se->join_list);
		*ppos = next_pos;	/* consume this token */
		TAILQ_INSERT_TAIL(&ast->schema_list, se, link);
		/* Check for a ',' indicating another name */
		token = ast_lex(ast, expr, &next_pos, &token_str);
		if (token != ASTT_COMMA)
			/* end of schema name list, do not consume token */
			break;
	}
	return ast->result;
}

int ast_parse_order_by_clause(struct ast *ast, const char *expr, int *ppos)
{
	char *token_str;
	enum ast_token_e token;
	struct ast_term *term;

	token = ast_lex(ast, expr, ppos, &token_str);
	if (token != ASTT_NAME) {
		ast->result = ASTP_ERROR;
		ast->pos = *ppos;
		snprintf(ast->error_msg, sizeof(ast->error_msg),
			 "Expected attribute name but found '%s'", token_str);
		goto out;
	}
	term = calloc(1, sizeof(*term));
	term->value = calloc(1, sizeof(*term->value));
	term->kind = ASTV_ATTR;
	term->attr = calloc(1, sizeof(*term->attr));
	ast_attr_entry_t ae = calloc(1, sizeof *ae);
	ae->value_attr = term->attr;
	ae->name = strdup(token_str);
	ae->rank = 0;
	ae->join_attr_idx = -1;
	term->attr->entry = ae;
	TAILQ_INSERT_TAIL(&ast->index_list, ae, link);
 out:
	return ast->result;
}

int ast_parse_where_clause(struct ast *ast, const char *expr, int *ppos)
{
	char *token_str;
	enum ast_token_e token;
	struct ast_term *newb;
	int next_pos;

	/* A where clause has at least one binary op */
	ast->where = ast_parse_binop(ast, expr, ppos);
	if (ast->result)
		return ast->result;
	/*
	 * Subsequent binary ops becomes the RHS of another
	 * binary op that then becomes the root of the AST.
	 */
	next_pos = *ppos;
	for (token = ast_lex(ast, expr, &next_pos, &token_str);
	     is_comparator(token);
	     token = ast_lex(ast, expr, &next_pos, &token_str)) {
		*ppos = next_pos;
		newb = calloc(1, sizeof(*newb));
		newb->kind = ASTV_BINOP;
		newb->binop = calloc(1, sizeof(*newb->binop));
		LIST_INSERT_HEAD(&ast->binop_list, newb->binop, entry);
		newb->binop->lhs = ast->where;
		newb->binop->op = token;
		switch (token) {
		case ASTT_OR:
		case ASTT_AND:
		case ASTT_NOT:
			newb->binop->rhs = ast_parse_binop(ast, expr, &next_pos);
			break;
		default:
			newb->binop->rhs = ast_parse_term(ast, expr, &next_pos);
			break;
		}
		if (ast->result)
			break;
		*ppos = next_pos;
		ast->where = newb;
	}
	if ((ast->result == 0) && (token == ASTT_ERR)) {
		/*
		 * If we exited the loop due to a bad token, report the
		 * error, but don't squash an earlier error.
		 */
		ast->result = ASTP_SYNTAX;
		ast->pos = *ppos;
		snprintf(ast->error_msg, sizeof(ast->error_msg),
			"Unexpected token '%s' in where clause.", token_str);
	}
	return ast->result;
}

static struct ast_attr_entry_s
*attr_alloc(sos_attr_t sos_attr, struct ast_schema_entry_s *schema_e)
{
	struct ast_attr_entry_s *ae = calloc(1, sizeof(*ae));
	assert(ae);
	ae->sos_attr = sos_attr;
	ae->min_join_idx = UINT32_MAX;
	ae->name = strdup(sos_attr_name(sos_attr));
	ae->schema = schema_e;
	return ae;
}

static int __resolve_sos_entities(struct ast *ast)
{
	struct ast_attr_entry_s *attr_e;
	struct ast_schema_entry_s *schema_e;
	sos_array_t join_list;
	ast_attr_entry_t join_attr_e;

	/* Resolve all the schema in the 'from' clause. */
	TAILQ_FOREACH(schema_e, &ast->schema_list, link) {
		schema_e->schema = sos_schema_by_name(ast->sos, schema_e->name);
		if (!schema_e->schema) {
			ast->result = ASTP_BAD_SCHEMA_NAME;
			snprintf(ast->error_msg, sizeof(ast->error_msg),
				 "The schema '%s' was not found in the container",
				 schema_e->name);
			return ASTP_BAD_SCHEMA_NAME;
		}
		int attr_id, attr_count = sos_schema_attr_count(schema_e->schema);
		for (attr_id = 0; attr_id < attr_count; attr_id++) {
			sos_attr_t sos_attr = sos_schema_attr_by_id(schema_e->schema, attr_id);
			if (SOS_TYPE_JOIN != sos_attr_type(sos_attr))
				continue;
			struct ast_attr_entry_s *ae = attr_alloc(sos_attr, schema_e);
			LIST_INSERT_HEAD(&schema_e->join_list, ae, join_link);
		}
	}

	/*
	 * Resolve all of the attributes in the 'select' clause.
	 * If '*' is present, use the schema directly from the list,
	 * otherwise, create a query-specific schema for the result.
	 */
	attr_e = TAILQ_FIRST(&ast->select_list);
	if (0 == strcmp(attr_e->name, "*")) {
		/* Replace the '*' with all the attributes from the schema */
		TAILQ_REMOVE(&ast->select_list, attr_e, link);
		free((void *)attr_e->name);
		free(attr_e);
		TAILQ_FOREACH(schema_e, &ast->schema_list, link) {
			sos_attr_t attr;
			for (attr = sos_schema_attr_first(schema_e->schema); attr; attr = sos_schema_attr_next(attr)) {
				if (sos_attr_type(attr) == SOS_TYPE_JOIN)
					continue;
				attr_e = calloc(1, sizeof(*attr_e));
				attr_e->name = strdup(sos_attr_name(attr));
				attr_e->sos_attr = attr;
				attr_e->schema = schema_e;
				TAILQ_INSERT_TAIL(&ast->select_list, attr_e, link);
			}
		}
	}
	/*
	 * Create a query specific result schema and attr_id map between
	 * the source and destination schema. This schema will include all
	 * of the attributes in the select clause plus the attribute elected
	 * for the iterator + all of the attributes in the join if that
	 * attribute is SOS_TYPE_JOIN.
	 */
	char res_schema_name[255];
	snprintf(res_schema_name, sizeof(res_schema_name), "query.%ld", ast->query_id);
	sos_schema_t res_schema = sos_schema_new(res_schema_name);

	TAILQ_FOREACH(attr_e, &ast->select_list, link) {
		TAILQ_FOREACH(schema_e, &ast->schema_list, link) {
			attr_e->sos_attr = sos_schema_attr_by_name(schema_e->schema, attr_e->name);
			attr_e->schema = schema_e;
			if (attr_e->sos_attr)
				break;
		}
		if (!attr_e->sos_attr) {
			ast->result = ASTP_BAD_ATTR_NAME;
			snprintf(ast->error_msg, sizeof(ast->error_msg),
				 "The '%s' attribute was not found in any schema in the 'from' clause.",
				 attr_e->name);
			return ast->result;
		}
		int rc = sos_schema_attr_add(res_schema, attr_e->name, sos_attr_type(attr_e->sos_attr));
		assert(!rc);
		sos_attr_t res_attr = sos_schema_attr_by_name(res_schema, attr_e->name);
		attr_e->res_attr = res_attr;
	}
	ast->result_schema = res_schema;

	/*
	 * Resolve all of the attributes in the 'where' clause
	 */
	TAILQ_FOREACH(attr_e, &ast->where_list, link) {
		TAILQ_FOREACH(schema_e, &ast->schema_list, link) {
			attr_e->sos_attr = sos_schema_attr_by_name(schema_e->schema, attr_e->name);
			assert(attr_e->value_attr);
			attr_e->value_attr->attr = attr_e->sos_attr;
			if (attr_e->sos_attr) {
				attr_e->schema = schema_e;
				break;
			}
		}
		if (!attr_e->sos_attr) {
			ast->result = ASTP_BAD_ATTR_NAME;
			snprintf(ast->error_msg, sizeof(ast->error_msg),
				 "The '%s' attribute was not found in any schema in the 'from' clause.",
				 attr_e->name);
			return ast->result;
		}
		if (SOS_TYPE_JOIN == sos_attr_type(attr_e->sos_attr)) {
			ast->result = ASTP_BAD_ATTR_TYPE;
			snprintf(ast->error_msg, sizeof(ast->error_msg),
				 "The '%s' attribute is SOS_TYPE_JOIN and "
				 "cannot appear in the 'where' clause.",
				 attr_e->name);
			return ast->result;
		}
	}

	/*
	 * Resolve all of the attributes in the 'order_by' clause
	 */
	TAILQ_FOREACH(attr_e, &ast->index_list, link) {
		TAILQ_FOREACH(schema_e, &ast->schema_list, link) {
			attr_e->sos_attr = sos_schema_attr_by_name(schema_e->schema, attr_e->name);
			assert(attr_e->value_attr);
			attr_e->value_attr->attr = attr_e->sos_attr;
			if (attr_e->sos_attr) {
				attr_e->schema = schema_e;
				break;
			}
		}
		if (!attr_e->sos_attr) {
			ast->result = ASTP_BAD_ATTR_NAME;
			snprintf(ast->error_msg, sizeof(ast->error_msg),
				 "The '%s' attribute was not found in any schema in the 'from' clause.",
				 attr_e->name);
			return ast->result;
		}
	}

	struct ast_term_binop *binop;
	LIST_FOREACH(binop, &ast->binop_list, entry) {
		/* Update the attribute min/max values in the attr tree */
		if (binop->lhs->kind == ASTV_ATTR) {
			if (update_attr_limits(ast, binop->lhs, binop->rhs, binop->op))
				return ast->result;
		} else if (binop->rhs->kind == ASTV_ATTR) {
			if (update_attr_limits(ast, binop->rhs, binop->lhs, binop->op))
				return ast->result;
		}
	}
	/*
	 * Run through the attributes in the 'where' clause; compute the
	 * number of times attributes in the where clause appear in the join and
	 * the attribute that is closest to the front of the join.
	 */
	TAILQ_FOREACH(attr_e, &ast->where_list, link) {
		attr_e->min_join_idx = UINT32_MAX;
		LIST_FOREACH(join_attr_e, &attr_e->schema->join_list, join_link) {
			join_list = sos_attr_join_list(join_attr_e->sos_attr);
			int join_idx, join_count = join_list->count;
			for (join_idx = 0; join_idx < join_count; join_idx++) {
				if (join_list->data.uint32_[join_idx]
				    == sos_attr_id(attr_e->sos_attr)) {
					join_attr_e->rank += join_count - join_idx;
					/* If this join is preferred over ones previously found, replace it */
					if (join_idx < attr_e->min_join_idx) {
						attr_e->join_attr = attr_e->sos_attr;
						attr_e->join_attr_idx = join_idx;
						attr_e->min_join_idx = join_idx;
					}
					break;
				}
			}
		}
	}

	struct ast_attr_entry_s *best_attr_e = TAILQ_FIRST(&ast->index_list);
	if (best_attr_e)
		goto create_iterator;

	/*
	 * Find the join index with the highest rank and use it to
	 * create the iterator
	 */
	TAILQ_FOREACH(schema_e, &ast->schema_list, link) {
		struct ast_attr_entry_s *join_e;
		LIST_FOREACH(join_e, &schema_e->join_list, join_link) {
			if (!best_attr_e) {
				best_attr_e = join_e;
				continue;
			}
			if (join_e->rank > best_attr_e->rank)
				best_attr_e = join_e;
		}
	}
	if (best_attr_e)
		goto create_iterator;

	if (!TAILQ_EMPTY(&ast->where_list))
		goto process_where_list;

	TAILQ_FOREACH(schema_e, &ast->schema_list, link) {
		sos_attr_t a;

		/* Search for something named 'timestamp' */
		for (a = sos_schema_attr_first(schema_e->schema); a;
		     a = sos_schema_attr_next(a)) {
			if (!sos_attr_is_indexed(a))
				continue;
			if (0 == strcmp(sos_attr_name(a), "timestamp")) {
				best_attr_e = attr_alloc(a, schema_e);
				goto create_iterator;
			}
		}
		/* Search for something having the type TIMESTAMP */
		for (a = sos_schema_attr_first(schema_e->schema); a;
		     a = sos_schema_attr_next(a)) {
			if (!sos_attr_is_indexed(a))
				continue;
			if (sos_attr_type(a) == SOS_TYPE_TIMESTAMP) {
				best_attr_e = attr_alloc(a, schema_e);
				goto create_iterator;
			}
		}
		/* Pick the 1st indexed attribute in the schema */
		for (a = sos_schema_attr_first(schema_e->schema); a;
		     a = sos_schema_attr_next(a)) {
			if (sos_attr_is_indexed(a))
				continue;
			best_attr_e = attr_alloc(a, schema_e);
			goto create_iterator;
		}
	}

	/* There are no indexed attributees in the schema, it cannot
	 * be queried */
	return ASTP_BAD_SCHEMA_NAME;

 process_where_list:
	/* Use the attribute attribute index that appears the
	 * most in the where clause */
	TAILQ_FOREACH(attr_e, &ast->where_list, link) {
		if (!sos_attr_is_indexed(attr_e->sos_attr))
			continue;
		TAILQ_FOREACH(best_attr_e, &ast->where_list, link) {
			if (!sos_attr_is_indexed(attr_e->sos_attr))
				continue;
			if (0 == strcmp(best_attr_e->name, attr_e->name))
				attr_e->rank += 1;
		}
	}
	best_attr_e = NULL;
	TAILQ_FOREACH(attr_e, &ast->where_list, link) {
		if (!sos_attr_is_indexed(attr_e->sos_attr))
			continue;
		if (best_attr_e == NULL) {
			best_attr_e = attr_e;
			continue;
		}
		if (attr_e->rank > best_attr_e->rank)
			best_attr_e = attr_e;
	}

 create_iterator:
	/* Create the SOS iterator using the best attribute */
	ast->iter_attr_e = best_attr_e;
	ast->sos_iter = sos_attr_iter_new(best_attr_e->sos_attr);
	if (!ast->sos_iter)
		return errno;
	ast->sos_iter_schema = sos_schema_by_name(ast->sos, best_attr_e->schema->name);
	if (SOS_TYPE_JOIN == sos_attr_type(best_attr_e->sos_attr)) {
		/* Ensure all attributes in the join and the join attr are in the result schema */
		sos_array_t ja = sos_attr_join_list(best_attr_e->sos_attr);
		char **join_list = calloc(ja->count, sizeof(char *));
		assert(join_list);
		int i, rc;
		ast->key_count = ja->count;
		for (i = 0; i < ja->count; i++) {
			sos_attr_t jaa = sos_schema_attr_by_id(ast->sos_iter_schema, ja->data.uint32_[i]);
			sos_attr_t resa = sos_schema_attr_by_name(res_schema, sos_attr_name(jaa));
			ast->succ_key[i] = AST_KEY_MORE;
			join_list[i] = (char *)sos_attr_name(jaa);
			TAILQ_FOREACH(attr_e, &ast->where_list, link) {
				if (0 == strcmp(attr_e->name, join_list[i])) {
					attr_e->join_attr_idx = i;
					/* Update the attribute limits */
					struct ast_attr_limits *limits;
					struct ods_rbn *rbn = ods_rbt_find(&ast->attr_tree, attr_e->name);
					assert(rbn);
					limits = container_of(rbn, struct ast_attr_limits, rbn);
					limits->join_idx = i;
					limits->join_attr = jaa;
					ast->key_limits[i] = limits;
				}
			}
			if (!resa) {
				attr_e = calloc(1, sizeof(*attr_e));
				attr_e->name = strdup(sos_attr_name(jaa));
				attr_e->sos_attr = jaa;
				int rc = sos_schema_attr_add(res_schema, sos_attr_name(jaa), sos_attr_type(jaa),
								sos_attr_size(jaa), NULL);
				attr_e->res_attr = sos_schema_attr_by_name(res_schema, sos_attr_name(jaa));
				TAILQ_INSERT_TAIL(&ast->select_list, attr_e, link);
				assert(0 == rc);
			}
		}
		rc = sos_schema_attr_add(res_schema,
					sos_attr_name(best_attr_e->sos_attr),
					SOS_TYPE_JOIN,
					ja->count, join_list
					);
		attr_e = calloc(1, sizeof(*attr_e));
		attr_e->name = strdup(sos_attr_name(best_attr_e->sos_attr));
		attr_e->sos_attr = best_attr_e->sos_attr;
		attr_e->res_attr = sos_schema_attr_by_name(res_schema, sos_attr_name(best_attr_e->sos_attr));
		TAILQ_INSERT_TAIL(&ast->select_list, attr_e, link);
		assert(0 == rc);
		free(join_list);
	} else {
		ast->key_count = 1;
		ast->succ_key[0] = AST_KEY_MORE;
		ast->key_limits[0] = NULL;
		/* Ensure that best_attr is in the result schema */
		sos_attr_t res_a = sos_schema_attr_by_name(res_schema, best_attr_e->name);
		if (!res_a) {
			sos_schema_attr_add(res_schema,
					sos_attr_name(best_attr_e->sos_attr),
					sos_attr_type(best_attr_e->sos_attr),
					sos_attr_size(best_attr_e->sos_attr),
					NULL
					);
			attr_e = calloc(1, sizeof(*attr_e));
			attr_e->name = strdup(sos_attr_name(best_attr_e->sos_attr));
			attr_e->sos_attr = best_attr_e->sos_attr;
			attr_e->res_attr = sos_schema_attr_by_name(res_schema, sos_attr_name(best_attr_e->sos_attr));
			TAILQ_INSERT_TAIL(&ast->select_list, attr_e, link);
		}
	}
	ast->sos_iter_schema = sos_schema_by_name(ast->sos, best_attr_e->schema->name);
	return 0;
}

/*
 * select <select-clause> from <from-clase> where <where-clause> order_by <order-by-clause>
 *
 * The <select-clause> is a comma-separated list of attributes from the schema in the <from-clause>
 * The <from-clause> specifies the schema from which attributes are gathered.
 * The <where-clause> specifies the conditions under which the object data is returned in the result.
 * The <order-by> specifies the name of the attribute (must be indexed) used to order data
 */
int ast_parse(struct ast *ast, char *expr)
{
	char *token_str;
	enum ast_token_e token;
	int rc, pos;

	ast->pos = pos = 0;
	ast->error_msg[0] = '\0';
	ast->result = rc = 0;
	TAILQ_INIT(&ast->schema_list);
	TAILQ_INIT(&ast->select_list);
	TAILQ_INIT(&ast->where_list);

	for (token = ast_lex(ast, expr, &pos, &token_str);
	     ast->result == 0
		     && token != ASTT_EOF && token != ASTT_ERR;
	     token = ast_lex(ast, expr, &pos, &token_str))
	{
		switch (token) {
		case ASTT_SELECT:
			rc = ast_parse_select_clause(ast, expr, &pos);
			break;
		case ASTT_FROM:
			rc = ast_parse_from_clause(ast, expr, &pos);
			break;
		case ASTT_ORDER_BY:
			rc = ast_parse_order_by_clause(ast, expr, &pos);
			break;
		case ASTT_WHERE:
			rc = ast_parse_where_clause(ast, expr, &pos);
			break;
		default:
			rc = EINVAL;
			ast->result = ASTP_ERROR;
			ast->pos = pos;
			snprintf(ast->error_msg, sizeof(ast->error_msg),
				"Expected 'select', 'from', 'where', or 'order_by', but found '%s'",
				token_str);
		}
	}
	if (!rc) {
		/* The parse was valid, resolve any SOS attributes */
		__resolve_sos_entities(ast);
	}
	return ast->result;
}

/*
 * Determine if given the iterator key and the assumption that the
 * keys are in ascending order, if it is possible that there is
 * another match in the index.
 */

static sos_value_t ast_term_visit(struct ast *ast, struct ast_term *term, sos_obj_t obj)
{
	sos_value_t lhs, rhs;
	int true_n_false;
	if (term->kind != ASTV_BINOP) {
		if (term->kind == ASTV_ATTR) {
			term->value =
				sos_value_init(&term->value_,
					       obj, term->attr->attr);
		}
		return term->value;
	}

	lhs = ast_term_visit(ast, term->binop->lhs, obj);
	switch (term->binop->op) {
	case ASTT_OR:
		if (sos_value_true(lhs)) {
			if (lhs->obj)
				sos_value_put(lhs);
			return sos_value_init_const(&term->value_, SOS_TYPE_INT32, (1 == 1));
		}
		rhs = ast_term_visit(ast, term->binop->rhs, obj);
		true_n_false = sos_value_true(rhs);
		if (rhs->obj)
			sos_value_put(rhs);
		return sos_value_init_const(&term->value_, SOS_TYPE_INT32, true_n_false);
	case ASTT_AND:
		if (!sos_value_true(lhs)) {
			if (lhs->obj)
				sos_value_put(lhs);
			return sos_value_init_const(&term->value_, SOS_TYPE_INT32, 0);
		}
		rhs = ast_term_visit(ast, term->binop->rhs, obj);
		true_n_false = sos_value_true(rhs);
		if (rhs->obj)
			sos_value_put(rhs);
		return sos_value_init_const(&term->value_, SOS_TYPE_INT32, true_n_false);
	case ASTT_NOT:
		if (!sos_value_true(lhs)) {
			if (lhs->obj)
				sos_value_put(lhs);
			return sos_value_init_const(&term->value_, SOS_TYPE_INT32, 0);
		}
		rhs = ast_term_visit(ast, term->binop->rhs, obj);
		true_n_false = sos_value_true(rhs);
		if (rhs->obj)
			sos_value_put(rhs);
		return sos_value_init_const(&term->value_, SOS_TYPE_INT32, !true_n_false);
	default:
		rhs = ast_term_visit(ast, term->binop->rhs, obj);
		break;
	}

	/* If the lhs and rhs are 'strings', normalize the '\0' */
	if (SOS_TYPE_STRING == lhs->type) {
		if (lhs->obj) {
			char *s = sos_array_data(lhs, char_);
			size_t slen = sos_array_count(lhs);
			if (slen && s[slen-1] != '\0' && NULL == rhs->obj) {
				size_t tlen = sos_array_count(rhs);
				char *t = sos_array_data(rhs, char_);
				if (tlen && t[tlen-1] == '\0')
					rhs->data->array.count -= 1;
			}
		}
	}
	int rc = sos_value_cmp(lhs, rhs);
	if (lhs->obj)
		sos_value_put(lhs);
	if (rhs->obj)
		sos_value_put(rhs);
	switch (term->binop->op) {
	case ASTT_LT:
		rc = (rc < 0);
		break;
	case ASTT_LE:
		rc = (rc <= 0);
		break;
	case ASTT_EQ:
		rc = (rc == 0);
		break;
	case ASTT_GE:
		rc = (rc >= 0);
		break;
	case ASTT_GT:
		rc = (rc > 0);
		break;
	case ASTT_NE:
		rc = (rc != 0);
		break;
	case ASTT_AND:
		rc = (sos_value_true(lhs) && sos_value_true(rhs));
		break;
	case ASTT_OR:
		rc = (sos_value_true(lhs) || sos_value_true(rhs));
		break;
	case ASTT_NOT:
		rc = (sos_value_true(rhs) && !sos_value_true(rhs));
		break;
	default:
		assert(0 == "Invalid comparator");
	}
	term->value = sos_value_init_const(
					   &term->value_,
					   SOS_TYPE_INT32,
					   rc);
	return term->value;
}

/*
 * Run through the attributes check if the where condition could
 * possibly match
 */
static int limit_check(struct ast *ast, sos_obj_t obj)
{
	int idx;
	struct ast_attr_limits *limits;
	struct sos_value_s v_;
	sos_value_t v;
	sos_attr_t attr;
	for (idx = 0; idx < ast->key_count; idx++) {
		limits = ast->key_limits[idx];
		if (!limits)
			continue;
		assert(limits->join_idx == idx);
		if (limits->max_v) {
			v = sos_value_init(&v_, obj, limits->attr);
			assert(v);
			int rc = sos_value_cmp(v, limits->max_v);
			if (rc >= 0) {
				if (rc == 0) {
					ast->succ_key[idx] = AST_KEY_MAX;
				} else if (idx == 0) {
					ast->succ_key[idx] = AST_KEY_NONE;
				} else if (ast->succ_key[idx-1] != AST_KEY_MORE) {
					ast->succ_key[idx] = AST_KEY_NONE;
				}
			}
			sos_value_put(v);
		} else {
			ast->succ_key[idx] = AST_KEY_MORE;
		}
	}
	for (idx = 0; idx < ast->key_count; idx++) {
		if (ast->succ_key[idx] == AST_KEY_NONE)
			return 1;
	}
	return 0;
}

/*
 * All components of the key except the last one should be
 * between min and max
 */
enum ast_eval_e ast_eval_limits(struct ast *ast, sos_obj_t obj)
{
	int idx, rc;
	struct ast_attr_limits *limits;
	struct sos_value_s v_;
	sos_value_t v;
	sos_attr_t attr;

	for (idx = 0; idx < ast->key_count; idx++) {
		limits = ast->key_limits[idx];
		if (!limits) {
			/* All but the last key must have limits or we can't check the object */
			if (idx < ast->key_count - 1)
				return AST_EVAL_MATCH;
			else
				continue;
		}
		assert(limits->join_idx == idx);
		v = sos_value_init(&v_, obj, limits->attr);
		assert(v);

		if (limits->min_v) {
			rc = sos_value_cmp(v, limits->min_v);
			if (rc < 0 && idx < ast->key_count - 1)
				return AST_EVAL_NOMATCH;
		}
		if (limits->max_v) {
			rc = sos_value_cmp(v, limits->max_v);
			if (rc > 0)
				return AST_EVAL_NOMATCH;
		}
	}
	return AST_EVAL_MATCH;
}

enum ast_eval_e ast_eval(struct ast *ast, sos_obj_t obj)
{
	if (limit_check(ast, obj))
	    return AST_EVAL_EMPTY;

	if (ast->where) {
		if (sos_value_true(ast_term_visit(ast, ast->where, obj)))
			return AST_EVAL_MATCH;
		return AST_EVAL_NOMATCH;
	}
	return AST_EVAL_MATCH;
}

struct ast_term *ast_find_term(struct ast_term *term, const char *attr_name)
{
	struct ast_term *lhs, *rhs;
	int join_idx, join_count;
	sos_array_t join_list;
	if (!term)
		return NULL;

	switch (term->kind) {
	case ASTV_CONST:
		return NULL;
	case ASTV_ATTR:
		if (0 == strcmp(attr_name, sos_attr_name(term->attr->attr)))
			return term;
		return NULL;
	case ASTV_BINOP:
		lhs = ast_find_term(term->binop->lhs, attr_name);
		rhs = ast_find_term(term->binop->rhs, attr_name);
		if (lhs || rhs)
			return term;
		break;
	default:
		assert(0 == "Invalid term in expression");
	}
	return NULL;
}

static int64_t attr_cmp(void *a, const void *b, void *arg)
{
	return strcmp((char *)a,(char *)b);
}

struct ast *ast_create(sos_t sos, uint64_t query_id)
{
	struct ast *ast = calloc(1, sizeof *ast);
	if (!ast)
		return NULL;
	ast->sos = sos;
	ast->query_id = query_id;
	int rc = regcomp(&ast->dqstring_re, "\"([^\"]*)|(\\.{1})\"", REG_EXTENDED);
	assert(0 == rc);
	rc = regcomp(&ast->sqstring_re, "\'([^\']*)|(\\.{1})\'", REG_EXTENDED);
	assert(0 == rc);
	rc =  regcomp(&ast->float_re, "[+-]?(([0-9]+)?(\\.[0-9]*))([eE][0-9]+)?", REG_EXTENDED);
	assert(0 == rc);
	rc =  regcomp(&ast->int_re, "[+-]?[0-9]+", REG_EXTENDED);
	assert(0 == rc);
	TAILQ_INIT(&ast->schema_list);
	TAILQ_INIT(&ast->select_list);
	TAILQ_INIT(&ast->where_list);
	TAILQ_INIT(&ast->index_list);
	LIST_INIT(&ast->binop_list);
	ods_rbt_init(&ast->attr_tree, attr_cmp, NULL);
	return ast;
}

static void ast_term_destroy(struct ast *ast, struct ast_term *term)
{
	sos_value_t lhs, rhs;

	if (!term)
		return;

	switch (term->kind) {
	case ASTV_ATTR:
		free(term->attr);
		free(term);
		break;
	case ASTV_CONST:
		sos_value_put(term->value);
		free(term);
		break;
	case ASTV_BINOP:
		ast_term_destroy(ast, term->binop->lhs);
		ast_term_destroy(ast, term->binop->rhs);
		sos_value_put(term->value);
		free(term->binop);
		free(term);
		break;
	}
}


void ast_destroy(struct ast *ast)
{
	regfree(&ast->dqstring_re);
	regfree(&ast->sqstring_re);
	regfree(&ast->float_re);
	regfree(&ast->int_re);

	while (!ods_rbt_empty(&ast->attr_tree)) {
		struct ods_rbn *rbn = ods_rbt_min(&ast->attr_tree);
		ods_rbt_del(&ast->attr_tree, rbn);
		struct ast_attr_limits *limits = container_of(rbn, struct ast_attr_limits, rbn);
		free((char *)limits->name);
		free(limits);
	}
	if (ast->sos_iter)
		sos_iter_free(ast->sos_iter);
	free(ast->iter_attr_name);
	sos_schema_free(ast->result_schema);
	ast_term_destroy(ast, ast->where);

	while (!TAILQ_EMPTY(&ast->schema_list)) {
		struct ast_schema_entry_s *se = TAILQ_FIRST(&ast->schema_list);
		TAILQ_REMOVE(&ast->schema_list, se, link);
		while (!LIST_EMPTY(&se->join_list)) {
			struct ast_attr_entry_s *ae = LIST_FIRST(&se->join_list);
			LIST_REMOVE(ae, join_link);
			free((void *)ae->name);
			free(ae);
		}
		free((char *)se->name);
		free(se);
	}
	while (!TAILQ_EMPTY(&ast->select_list)) {
		struct ast_attr_entry_s *ae = TAILQ_FIRST(&ast->select_list);
		TAILQ_REMOVE(&ast->select_list, ae, link);
		free((void *)ae->name);
		free(ae);
	}
	while (!TAILQ_EMPTY(&ast->where_list)) {
		struct ast_attr_entry_s *ae = TAILQ_FIRST(&ast->where_list);
		TAILQ_REMOVE(&ast->where_list, ae, link);
		free((void *)ae->name);
		free(ae);
	}
	while (!TAILQ_EMPTY(&ast->index_list)) {
		struct ast_attr_entry_s *ae = TAILQ_FIRST(&ast->index_list);
		TAILQ_REMOVE(&ast->index_list, ae, link);
		free((void *)ae->name);
		free(ae);
	}
	free(ast);
}

typedef int (*ast_binop_find_fn_t)(struct ast_term *binop, const void *arg);

static struct ast_term *
__ast_binop_find_term(struct ast_term *term,
		      ast_binop_find_fn_t cmp_fn, const void *cmp_arg, enum ast_token_e *op)
{
	struct ast_term *lhs, *rhs;

	if (term->kind != ASTV_BINOP)
		return NULL;

	int rc = cmp_fn(term, cmp_arg);
	if (rc)
		*op = term->binop->op;
	if (rc > 0)
		return term->binop->rhs;
	else if (rc < 0)
		return term->binop->lhs;

	lhs = __ast_binop_find_term(term->binop->lhs, cmp_fn, cmp_arg, op);
	if (lhs)
		return lhs;

	rhs = __ast_binop_find_term(term->binop->rhs, cmp_fn, cmp_arg, op);
	if (rhs)
		return rhs;

	return NULL;
}

struct ast_term *ast_binop_find(struct ast *ast, ast_binop_find_fn_t cmp_fn, const void *cmp_arg, enum ast_token_e *op)
{
	struct ast_term *term = ast->where;
	if (!term || term->kind != ASTV_BINOP)
		return NULL;

	return __ast_binop_find_term(term, cmp_fn, cmp_arg, op);
}

/**
 * @brief Compare rhs and lhs based on attribute name
 *
 * Return the constant value for the specified attribute name
 *
 * >0 rhs
 * <0 lhs
 * 0 No match
 */
static int __attr_ge(struct ast_term *term, const void *name)
{
	struct ast_term *lhs;
	struct ast_term *rhs;
	enum ast_token_e op;
	assert(term->kind == ASTV_BINOP);

	lhs = term->binop->lhs;
	rhs = term->binop->rhs;
	op = term->binop->op;

	if (lhs->kind == ASTV_ATTR) {
		/* Check if it matches our name */
		const char *aname =
			sos_attr_name(lhs->attr->attr);
		if (0 == strcmp(aname, name)) {
			/* The rhs must be a constant */
			if  (rhs->kind != ASTV_CONST)
				return 0;
			if (op != ASTT_EQ
			    && op != ASTT_GE
			    && op != ASTT_GT)
				return 0;
			return 1;/* rhs is const value for attr */
		}
		return 0;
	}

	if (rhs->kind == ASTV_ATTR) {
		/* Check if it matches our name */
		const char *aname =
			sos_attr_name(rhs->attr->attr);
		if (0 == strcmp(aname, name)) {
			/* The lhs must be a constant */
			if  (lhs->kind != ASTV_CONST)
				return 0;
			if (op != ASTT_EQ
			    && op != ASTT_LE
			    && op != ASTT_LT)
				return 0;
			return -1; /* lhs is constant value for attr */
		}
		return 0;
	}

	return 0;
}

/**
 * @brief  Find the value for attribute \c name in the were cause
 *
 * Find the value for attribute \c name that appears in the where
 * clause that is compared to a constant value and for which the
 * comparason is '>', '>=', or '=='. Note the following equivalencies:
 *
 * const <= attr <==> attr >= const
 * const == attr <==> attr == const
 * const < attr  <==> attr > const
 *
 */
struct ast_term *ast_attr_value_ge(struct ast *ast, const char *name, enum ast_token_e *op)
{
	return ast_binop_find(ast, __attr_ge, name, op);
}

sos_key_t __sos_key_from_const(sos_key_t key, sos_attr_t attr, struct ast_attr_limits *limits)
{
	sos_value_data_t data = limits->min_v->data;
	switch (sos_attr_type(attr)) {
	case SOS_TYPE_INT16:
		return sos_key_for_attr(key, attr, data->prim.int16_);
	case SOS_TYPE_INT32:
		return sos_key_for_attr(key, attr, data->prim.int32_);
	case SOS_TYPE_INT64:
		return sos_key_for_attr(key, attr, data->prim.int64_);
	case SOS_TYPE_UINT16:
		return sos_key_for_attr(key, attr, data->prim.uint16_);
	case SOS_TYPE_UINT32:
		return sos_key_for_attr(key, attr, data->prim.uint32_);
	case SOS_TYPE_UINT64:
		return sos_key_for_attr(key, attr, data->prim.uint64_);
	case SOS_TYPE_FLOAT:
		return sos_key_for_attr(key, attr, data->prim.float_);
	case SOS_TYPE_DOUBLE:
		return sos_key_for_attr(key, attr, data->prim.double_);
	case SOS_TYPE_LONG_DOUBLE:
		return sos_key_for_attr(key, attr, data->prim.long_double_);
	case SOS_TYPE_TIMESTAMP:
		return sos_key_for_attr(key, attr, data->prim.timestamp_);
	case SOS_TYPE_BYTE_ARRAY:
	case SOS_TYPE_STRING:
		return sos_key_for_attr(key, attr, data->array.data.char_, data->array.count);
	case SOS_TYPE_OBJ:
	case SOS_TYPE_STRUCT:
	case SOS_TYPE_JOIN:
	case SOS_TYPE_INT16_ARRAY:
	case SOS_TYPE_INT32_ARRAY:
	case SOS_TYPE_INT64_ARRAY:
	case SOS_TYPE_UINT16_ARRAY:
	case SOS_TYPE_UINT32_ARRAY:
	case SOS_TYPE_UINT64_ARRAY:
	case SOS_TYPE_FLOAT_ARRAY:
	case SOS_TYPE_DOUBLE_ARRAY:
	case SOS_TYPE_LONG_DOUBLE_ARRAY:
	case SOS_TYPE_OBJ_ARRAY:
		break;
	}
	return NULL;
}

/**
 * Build up a key that the caller can use to search for an index entry
 * that is the 1st possible match given the where clause.
 *
 * The key assumes that the iterator will be traversed from glb to
 * max-match.
 *
 * returns ESRCH if the key has been prepared for sos_iter_find_glb(),
 * otherwise, the caller should assume that the key is of no use
 * and should call sos_iter_begin()
 */
int ast_start_key(struct ast *ast, sos_key_t key)
{
	ast_attr_entry_t attr;
	sos_attr_t iter_attr = sos_iter_attr(ast->sos_iter);
	int rc = 0;
	int search = 0;
	int join_idx;
	ods_comp_key_t comp_key;
	ods_key_comp_t key_comp;
	sos_array_t attr_ids;
	struct ods_rbn *rbn;
	struct ast_attr_limits *limits;

	if (sos_attr_type(iter_attr) != SOS_TYPE_JOIN) {
		rbn = ods_rbt_find(&ast->attr_tree, sos_attr_name(iter_attr));
		if (!rbn)
			return 0;
		limits = container_of(rbn, struct ast_attr_limits, rbn);
		key = __sos_key_from_const(key, iter_attr, limits);
		return ESRCH;
	}
	/*
	 * Build up a component key with values from the where clause
	 */
	attr_ids = sos_attr_join_list(iter_attr);
	comp_key = (ods_comp_key_t)ods_key_value(key);
	key_comp = comp_key->value;
	comp_key->len = 0;

	for (join_idx = 0; join_idx < attr_ids->count; join_idx++) {
		int join_attr_id = attr_ids->data.uint32_[join_idx];
		size_t comp_len;
		enum ast_token_e op;
		sos_attr_t attr =
			sos_schema_attr_by_id(sos_attr_schema(iter_attr),
					      join_attr_id);
		rbn = ods_rbt_find(&ast->attr_tree, sos_attr_name(attr));
		if (rbn) {
			limits = container_of(rbn, struct ast_attr_limits, rbn);
			if (limits->min_v)
				key_comp = __sos_set_key_comp(key_comp, limits->min_v, &comp_len);
			else
				key_comp = __sos_set_key_comp(key_comp, sos_attr_min(attr), &comp_len);
		} else {
			key_comp = __sos_set_key_comp_to_min(key_comp, attr, &comp_len);
		}
		search = ESRCH;
		comp_key->len += comp_len;
	}
	return search;
}
