#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <regex.h>
#include <assert.h>
#include <errno.h>
#include <sos/sos.h>
#include "ast.h"

static int is_comparator(enum ast_token_e token)
{
	if (token >= ASTT_LT && token <= ASTT_NOT)
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
	{ "or", ASTT_OR },
	{ "order_by", ASTT_ORDER_BY },
	{ "select", ASTT_SELECT },
	{ "where", ASTT_WHERE },
};

static struct ast_term *ast_parse_binop(struct ast *ast, const char *expr, int *ppos);
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
	if (s[0] == '=' && s[1] == '=') {
		*ppos += 2;
		return ASTT_EQ;
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
	if (s[0] == '!') {
		if (s[1] == '=') {
			*ppos += 2;
			strncpy(token_str, s, 2);
			token_str[2] = '\0';
			return ASTT_NE;
		}
		token_str[0] = s[0];
		token_str[1] = '\0';
		*ppos += 1;
		return ASTT_NOT;
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
	return ASTT_ERR;
}

/*
 * Attribute names may encode a schema name as follows:
 *      <schema_name> '[' <attr_name> ']'
 * If the <attr_name> alone is specified, the default schema is used.
 */
static enum ast_parse_e parse_attr(struct ast *ast, const char *name, struct ast_value_attr *value_attr)
{
	ast_attr_entry_t ae = calloc(1, sizeof *ae);
	ae->value_attr = value_attr;
	ae->name = strdup(name);
	ae->rank = 0;
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

	token = ast_lex(ast, expr, ppos, &token_str);
	switch (token) {
	case ASTT_LPAREN:
		term = ast_parse_binop(ast, expr, ppos);
		token = ast_lex(ast, expr, ppos, &token_str);
		if (token != ASTT_RPAREN)
			ast->result = ASTP_UNBALANCED_PAREN;
		break;
	case ASTT_NAME:	/* Attribute */
		term = calloc(1, sizeof(*term));
		term->value = calloc(1, sizeof(*term->value));
		term->kind = ASTV_ATTR;
		term->attr = calloc(1, sizeof(*term->attr));
		err = parse_attr(ast, token_str, term->attr);
		if (err) {
			free(term->value->attr);
			ast->result = err;
		}
		break;
	case ASTT_DQSTRING:	/* Attribute */
		term = calloc(1, sizeof(*term));
		term->value = calloc(1, sizeof(*term->value));
		term->kind = ASTV_ATTR;
		term->attr = calloc(1, sizeof(*term->attr));
		err = parse_attr(ast, token_str, term->attr);
		if (err) {
			free(term->value->attr);
			ast->result = err;
		}
		break;
	case ASTT_SQSTRING:	/* String value */
		term = calloc(1, sizeof(*term));
		term->value = calloc(1, sizeof(*term->value));
		term->kind = ASTV_CONST;
		term->value = sos_value_init_const(&term->value_,
						   SOS_TYPE_CHAR_ARRAY,
						   token_str);
		break;
	case ASTT_INTEGER:
		term = calloc(1, sizeof(*term));
		term->value = calloc(1, sizeof(*term->value));
		term->kind = ASTV_CONST;
		term->value = sos_value_init_const(&term->value_, SOS_TYPE_INT64,
						   strtol(token_str, NULL, 0));
		break;
	case ASTT_FLOAT:
		term = calloc(1, sizeof(*term));
		term->value = calloc(1, sizeof(*term->value));
		term->kind = ASTV_CONST;
		term->value = sos_value_init_const(
						   &term->value_,
						   SOS_TYPE_DOUBLE,
						   strtod(token_str, NULL));
		break;
	default:
		ast->result = ASTP_ERROR;
		term = NULL;
	}
	return term;
}

/*
 * <term> <op> <term>
 */
static struct ast_term *ast_parse_binop(struct ast *ast, const char *expr, int *ppos)
{
	char *token_str;

	struct ast_term_binop *binop = calloc(1, sizeof(*binop));
	binop->lhs = ast_parse_term(ast, expr, ppos);
	binop->op = ast_lex(ast, expr, ppos, &token_str);
	if (!is_comparator(binop->op)) {
		ast->result = ASTP_ERROR;
		free(binop);
		return NULL;
	}
	binop->rhs = ast_parse_term(ast, expr, ppos);
	struct ast_term *term = calloc(1, sizeof(*term));
	term->kind = ASTV_BINOP;
	term->binop = binop;
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
	     token = ast_lex(ast, expr, &next_pos, &token_str))
		{
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

	for (token = ast_lex(ast, expr, &next_pos, &token_str);
	     token == ASTT_NAME;
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

int ast_parse_where_clause(struct ast *ast, const char *expr, int *ppos)
{
	char *token_str;
	enum ast_token_e token;
	struct ast_term *binop, *newb;
	int next_pos;

	/* A where clause has at least one binary op */
	ast->where = binop = ast_parse_binop(ast, expr, ppos);
	if (ast->result)
		return ast->result;
	/*
	 * Subsequent binary ops becomes the RHS of another
	 * binary op that then becomes the root of the AST.
	 */
	next_pos = *ppos;
	for (token = ast_lex(ast, expr, &next_pos, &token_str);
	     is_comparator(token);
	     token = ast_lex(ast, expr, &next_pos, &token_str))
		{
			*ppos = next_pos;
			newb = calloc(1, sizeof(*newb));
			newb->kind = ASTV_BINOP;
			newb->binop = calloc(1, sizeof(*binop));
			newb->binop->lhs = ast->where;
			newb->binop->op = token;
			newb->binop->rhs = ast_parse_term(ast, expr, &next_pos);
			if (ast->result)
				break;
			*ppos = next_pos;
			ast->where = binop = newb;
		}
	return ast->result;
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
			struct ast_attr_entry_s *ae = calloc(1, sizeof(*ae));
			ae->sos_attr = sos_attr;
			ae->min_join_idx = UINT32_MAX;
			ae->name = strdup(sos_attr_name(sos_attr));
			ae->schema = schema_e;
			ae->rank = 0;
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
				attr_e = calloc(1, sizeof(*attr_e));
				attr_e->name = strdup(sos_attr_name(attr));
				attr_e->sos_attr = attr;
				attr_e->schema = schema_e;
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
			attr_e->value_attr->schema = schema_e->schema;
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
				 "The '%s' attribute is SOS_TYPE_JOIN and cannot appear in the 'where' clause.",
				 attr_e->name);
			return ast->result;
		}
	}
	/*
	 * Run through the attributes in the 'where' clause; compute the
	 * Nnumber of times attributes in the where clause appear in the join and
	 * the attribute that has the minimum attribute index
	 */
	TAILQ_FOREACH(attr_e, &ast->where_list, link) {
		attr_e->min_join_idx = UINT32_MAX;
		LIST_FOREACH(join_attr_e, &attr_e->schema->join_list, join_link) {
			join_list = sos_attr_join_list(join_attr_e->sos_attr);
			int join_idx, join_count = join_list->count;
			for (join_idx = 0; join_idx < join_count; join_idx++) {
				if (join_list->data.uint32_[join_idx] == sos_attr_id(attr_e->sos_attr)) {
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
	/* Find the join index with the highest rank and use it to create the iterator */
	struct ast_attr_entry_s *join_e;
	struct ast_attr_entry_s *best_attr_e = NULL;
	TAILQ_FOREACH(schema_e, &ast->schema_list, link) {
		LIST_FOREACH(join_e, &schema_e->join_list, join_link) {
			if (!best_attr_e) {
				best_attr_e = join_e;
				continue;
			}
			if (join_e->rank > best_attr_e->rank)
				best_attr_e = join_e;
		}
	}
	if (!best_attr_e) {
		/* Use the attribute attribute index that appears the most in the where clause */
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
	}
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
		for (i = 0; i < ja->count; i++) {
			sos_attr_t jaa = sos_schema_attr_by_id(ast->sos_iter_schema, ja->data.uint32_[i]);
			sos_attr_t resa = sos_schema_attr_by_name(res_schema, sos_attr_name(jaa));
			join_list[i] = (char *)sos_attr_name(jaa);
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
			break;
		case ASTT_WHERE:
			rc = ast_parse_where_clause(ast, expr, &pos);
			break;
		default:
			rc = EINVAL;
			ast->result = ASTP_ERROR;
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

static sos_value_t ast_term_visit(struct ast *ast, struct ast_term *term, sos_obj_t obj)
{
	sos_value_t lhs, rhs;
	if (term->kind != ASTV_BINOP) {
		if (term->kind == ASTV_ATTR)
			term->value =
				sos_value_init(&term->value_,
					       obj, term->attr->attr);
		return term->value;
	}

	lhs = ast_term_visit(ast, term->binop->lhs, obj);
	rhs = ast_term_visit(ast, term->binop->rhs, obj);

	switch (term->binop->op) {
	case ASTT_OR:
		return sos_value_init_const(
					    &term->value_, SOS_TYPE_INT32,
					    sos_value_true(lhs) || sos_value_true(rhs)
					    );
	case ASTT_AND:
		return sos_value_init_const(
					    &term->value_, SOS_TYPE_INT32,
					    sos_value_true(lhs) && sos_value_true(rhs)
					    );
	default:
		break;
	}

	int rc = sos_value_cmp(lhs, rhs);
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
	default:
		assert(0 == "Invalid comparator");
	}
	term->value = sos_value_init_const(
					   &term->value_,
					   SOS_TYPE_INT32,
					   rc);
	return term->value;
}

int ast_eval(struct ast *ast, sos_obj_t obj)
{
	if (ast->where)
		return sos_value_true(ast_term_visit(ast, ast->where, obj));
	return 1;
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
	return ast;
}
