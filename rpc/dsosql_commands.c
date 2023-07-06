/* -*- c-basic-offset : 8 -*- */
#include <pthread.h>
#include <sys/time.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <assert.h>
#include <sos/sos.h>
#include "dsos.h"
#include <ods/ods_atomic.h>
#include <jansson.h>
#include "dsosql.h"
#undef VERSION

int add_filter(sos_schema_t schema, sos_filter_t filt, const char *str);
char *strcasestr(const char *haystack, const char *needle);

/*
 * These are type conversion helpers for various types
 */
int value_from_str(sos_attr_t attr, sos_value_t cond_value, char *value_str, char **endptr)
{
	double ts;

	int rc = sos_value_from_str(cond_value, value_str, endptr);
	if (!rc)
		return 0;

	switch (sos_attr_type(attr)) {
	case SOS_TYPE_TIMESTAMP:
		/* Try to convert from double (which should handle int too) */
		ts = strtod(value_str, endptr);
		if (ts == 0 && *endptr == value_str)
			break;
		cond_value->data->prim.timestamp_.tv.tv_sec = (int)ts;
		uint32_t usecs = (uint32_t)((double)(ts - (int)ts) * 1.0e6);
		cond_value->data->prim.timestamp_.tv.tv_usec = usecs;
		rc = 0;
		break;
	default:
		break;
	}
	return rc;
}

int create(const char *path, int o_mode)
{
	int rc = 0;
	sos_t sos = sos_container_open(path, SOS_PERM_CREAT|SOS_PERM_RW, o_mode);
	if (!sos) {
		rc = errno;
		perror("The container could not be created");
	} else {
		sos_container_close(sos, SOS_COMMIT_ASYNC);
	}
	return rc;
}

int col_widths[] = {
	[SOS_TYPE_INT16] = 6,
	[SOS_TYPE_INT32] = 12,
	[SOS_TYPE_INT64] = 18,
	[SOS_TYPE_UINT16] = 6,
	[SOS_TYPE_UINT32] = 12,
	[SOS_TYPE_UINT64] = 18,
	[SOS_TYPE_FLOAT] = 12,
	[SOS_TYPE_DOUBLE] = 24,
	[SOS_TYPE_LONG_DOUBLE] = 48,
	[SOS_TYPE_TIMESTAMP] = 32,
	[SOS_TYPE_OBJ] = 8,
	[SOS_TYPE_STRUCT] = 32,
	[SOS_TYPE_JOIN] = 32,
	[SOS_TYPE_BYTE_ARRAY] = -1,
	[SOS_TYPE_CHAR_ARRAY] = -1,
	[SOS_TYPE_INT16_ARRAY] = -1,
	[SOS_TYPE_INT32_ARRAY] = -1,
	[SOS_TYPE_INT64_ARRAY] = -1,
	[SOS_TYPE_UINT16_ARRAY] = -1,
	[SOS_TYPE_UINT32_ARRAY] = -1,
	[SOS_TYPE_UINT64_ARRAY] = -1,
	[SOS_TYPE_FLOAT_ARRAY] = -1,
	[SOS_TYPE_DOUBLE_ARRAY] = -1,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = -1,
	[SOS_TYPE_OBJ_ARRAY] = -1,
};

int schema_dir(sos_t sos)
{
	sos_schema_t schema;
	for (schema = sos_schema_first(sos); schema;
	     schema = sos_schema_next(schema))
		sos_schema_print(schema, stdout);
	printf("\n");
	return 0;
}

/*
 * Add a column. The format is:
 * <name>[col_width]
 */
int add_column(sos_schema_t schema, const char *str, struct col_list_s *col_list)
{
	char *width;
	char *s = NULL;
	struct col_s *col = calloc(1, sizeof *col);
	if (!col)
		goto err;
	s = strdup(str);
	if (!s)
		goto err;
	width = strchr(s, '[');
	if (width) {
		*width = '\0';
		width++;
		col->width = strtoul(width, NULL, 0);
	}
	col->name = s;
	if (!col->name)
		goto err;
	TAILQ_INSERT_TAIL(col_list, col, entry);
	sos_attr_t attr = sos_schema_attr_by_name(schema, s);
	if (!attr)
goto err;
	col->id = sos_attr_id(attr);
	return 0;
 err:
	if (col)
		free(col);
	if (s)
		free(s);
	printf("Could not allocate memory for the column.\n");
	return ENOMEM;
}

struct cfg_s {
	char *kv;
	TAILQ_ENTRY(cfg_s) entry;
};
TAILQ_HEAD(cfg_list_s, cfg_s) cfg_list = TAILQ_HEAD_INITIALIZER(cfg_list);

struct clause_s {
	const char *str;
	TAILQ_ENTRY(clause_s) entry;
};
TAILQ_HEAD(clause_list_s, clause_s) clause_list = TAILQ_HEAD_INITIALIZER(clause_list);

enum query_fmt {
	TABLE_FMT,
	CSV_FMT,
	JSON_FMT
} format;

void table_header(FILE *outp, sos_attr_t index_attr, struct col_list_s *col_list)
{
	struct col_s *col;
	if (index_attr) {
		if (sos_attr_type(index_attr) != SOS_TYPE_JOIN) {
			fprintf(outp, "ORDER_BY \"%s\"\n", sos_attr_name(index_attr));
		} else {
			int join_idx;
			sos_array_t join_list = sos_attr_join_list(index_attr);
			sos_schema_t schema = sos_attr_schema(index_attr);
			fprintf(outp, "ORDER_BY ");
			for (join_idx = 0; join_idx < join_list->count; join_idx++) {
				sos_attr_t join_attr = sos_schema_attr_by_id(schema,
									     join_list->data.uint32_[join_idx]);
				fprintf(outp, "\"%s\"", sos_attr_name(join_attr));
				if (join_idx < join_list->count - 1) {
					fprintf(outp, ", ");
				}
			}
			fprintf(outp, "\n");
		}
	}
	/* Print the header labels */
	TAILQ_FOREACH(col, col_list, entry) {
		if (col->width > 0)
			fprintf(outp, "%-*s ", col->width, col->name);
		else
			fprintf(outp, "%-s ", col->name);
	}
	fprintf(outp, "\n");

	/* Print the header separators */
	TAILQ_FOREACH(col, col_list, entry) {
		int i;
		if (col->width > 0)
			for (i = 0; i < col->width; i++)
				fprintf(outp, "-");
		else
			for (i = 0; i < strlen(col->name); i++)
				fprintf(outp, "-");
		fprintf(outp, " ");
	}
	fprintf(outp, "\n");
}

static void csv_header(FILE *outp, sos_attr_t index_attr, struct col_list_s *col_list)
{
	struct col_s *col;
	int first = 1;
	/* Print the header labels */
	fprintf(outp, "# ");
	TAILQ_FOREACH(col, col_list, entry) {
		if (!first)
			fprintf(outp, ",");
		fprintf(outp, "%s", col->name);
		first = 0;
	}
	fprintf(outp, "\n");
}

static void json_header(FILE *outp, sos_attr_t index_attr, struct col_list_s *col_list)
{
	fprintf(outp, "{ \"data\" : [\n");
}

void table_row(FILE *outp, sos_schema_t schema, sos_obj_t obj, struct col_list_s *col_list)
{
	struct col_s *col;
	size_t col_len;
	sos_attr_t attr;
	char *col_str;
	char str[80];
	TAILQ_FOREACH(col, col_list, entry) {
		attr = sos_schema_attr_by_id(schema, col->id);
		if (col->width > 0 && col->width < sizeof(str)) {
			col_len = col->width;
			col_str = str;
		} else {
			if (col->width > 0)
				col_len = col->width;
			else
				col_len = sos_obj_attr_strlen(obj, attr);
			if (col_len < sizeof(str))
				col_str = str;
			else
				col_str = malloc(col_len);
			if (col_len < strlen(col->name))
				col_len = strlen(col->name);
		}
		if (col->width > 0) {
			fprintf(outp, "%*s ", col->width,
				sos_obj_attr_to_str(obj, attr, col_str, col_len));
		} else {
			fprintf(outp, "%s ",
				sos_obj_attr_to_str(obj, attr, col_str, col_len));
		}
		if (col_str != str)
			free(col_str);
	}
	fprintf(outp, "\n");
}

static void csv_row(FILE *outp, sos_schema_t schema, sos_obj_t obj, struct col_list_s *col_list)
{
	struct col_s *col;
	int first = 1;
	sos_attr_t attr;
	size_t col_len;
	char *col_str;
	char str[80];
	TAILQ_FOREACH(col, col_list, entry) {
		attr = sos_schema_attr_by_id(schema, col->id);
		if (!first)
			fprintf(outp, ",");
		col_len = sos_obj_attr_strlen(obj, attr);
		if (col_len < sizeof(str)) {
			col_str = str;
			col_len = sizeof(str);
		} else {
			col_str = malloc(col_len);
		}
		fprintf(outp, "%s", sos_obj_attr_to_str(obj, attr, col_str, col_len));
		if (col_str != str)
			free(col_str);
		first = 0;
	}
	fprintf(outp, "\n");
}

static void json_row(FILE *outp, sos_schema_t schema, sos_obj_t obj, struct col_list_s *col_list)
{
	struct col_s *col;
	static int first_row = 1;
	int first = 1;
	sos_attr_t attr;
	size_t col_len;
	char *col_str;
	static char str[80];
	if (!first_row)
		fprintf(outp, ",\n");
	first_row = 0;
	fprintf(outp, "{");
	TAILQ_FOREACH(col, col_list, entry) {
		attr = sos_schema_attr_by_id(schema, col->id);
		if (!first)
			fprintf(outp, ",");
		col_len = sos_obj_attr_strlen(obj, attr);
		if (col_len < sizeof(str)) {
			col_str = str;
			col_len = sizeof(str);
		} else {
			col_str = malloc(col_len);
		}
		if (sos_attr_is_array(attr) && sos_attr_type(attr) != SOS_TYPE_CHAR_ARRAY) {
			fprintf(outp, "\"%s\" : [%s]", col->name,
				sos_obj_attr_to_str(obj, attr, col_str, col_len));
		} else {
			fprintf(outp, "\"%s\" : \"%s\"", col->name,
				sos_obj_attr_to_str(obj, attr, col_str, col_len));
		}
		if (col_str != str)
			free(col_str);
		first = 0;
	}
	fprintf(outp, "}");
}

void table_footer(FILE *outp, int rec_count, int iter_count, struct col_list_s *col_list)
{
	struct col_s *col;

	/* Print the footer separators */
	TAILQ_FOREACH(col, col_list, entry) {
		int i;
		for (i = 0; i < col->width; i++)
			fprintf(outp, "-");
		fprintf(outp, " ");
	}
	fprintf(outp, "\n");
	fprintf(outp, "Records %d\n", rec_count);
}

static void csv_footer(FILE *outp, int rec_count, int iter_count, struct col_list_s *col_list)
{
	fprintf(outp, "# Records %d/%d.\n", rec_count, iter_count);
}

static void json_footer(FILE *outp, int rec_count, int iter_count, struct col_list_s *col_list)
{
	fprintf(outp, "], \"%s\" : %d, \"%s\" : %d}\n",
		"totalRecords", rec_count, "recordCount", iter_count);
}

ods_atomic_t records;
size_t col_count;
int *col_map;
sos_attr_t *attr_map;

struct type_def_s {
	const char *name;
	sos_type_t type;
};

struct type_def_s types[] = {
	{ "bytes", SOS_TYPE_BYTE_ARRAY },
	{ "double", SOS_TYPE_DOUBLE },
	{ "double_array", SOS_TYPE_DOUBLE_ARRAY },
	{ "float", SOS_TYPE_FLOAT },
	{ "float_array", SOS_TYPE_FLOAT_ARRAY },
	{ "int16", SOS_TYPE_INT16 },
	{ "int16_array", SOS_TYPE_INT16_ARRAY },
	{ "int32", SOS_TYPE_INT32 },
	{ "int32_array", SOS_TYPE_INT32_ARRAY },
	{ "int64", SOS_TYPE_INT64 },
	{ "int64_array", SOS_TYPE_INT64_ARRAY },
	{ "join", SOS_TYPE_JOIN },
	{ "long_double_array", SOS_TYPE_LONG_DOUBLE_ARRAY },
	{ "string", SOS_TYPE_STRING },
	{ "struct", SOS_TYPE_STRUCT },
	{ "timestamp", SOS_TYPE_TIMESTAMP },
	{ "uint16", SOS_TYPE_UINT16 },
	{ "uint16_array", SOS_TYPE_UINT16_ARRAY },
	{ "uint32", SOS_TYPE_UINT32 },
	{ "uint32_array", SOS_TYPE_UINT32_ARRAY },
	{ "uint64", SOS_TYPE_UINT64 },
	{ "uint64_array", SOS_TYPE_UINT64_ARRAY },
};

int type_cmp(const void *a_, const void *b_)
{
	const char *a = a_;
	struct type_def_s *b = (struct type_def_s *)b_;
	return strcmp(a, b->name);
}

sos_type_t lookup_type(const char *name)
{
	struct type_def_s *def =
		bsearch(name, types,
			sizeof(types) / sizeof(types[0]),
			sizeof(types[0]),
			type_cmp);
	return def->type;
}

int dsosql_create_schema(dsos_container_t cont, char *schema_name, char *template)
{
	sos_schema_t schema;
	dsos_schema_t dschema;
	dsos_res_t res;
	json_error_t error;
	json_t *t, *i, *j, *attr;
	int rc = 0;

	t = json_loads(template, 0, &error);
	if (!t) {
		printf("Error parsing the schema template:\n");
		printf("    %s\n", error.text);
		printf("    %s\n", error.source);
		printf("    %*s\n", error.column, "^");
		rc = EINVAL;
		goto err;
	}
	schema = sos_schema_new(schema_name);
	if (!schema) {
		printf("Error %d creating the schema %s\n", errno, schema_name);
		rc = errno;
		goto err;
	}
	/* Each item is a dictionary of 'name', 'type', + optional 'size' and 'join_list' */
	if (!json_is_object(t)) {
		rc = EINVAL;
		printf("Thhe schema template must be a dictionary.\n");
		goto err;
	}
	json_t *uuid = json_object_get(t, "uuid");
	if (!uuid || !json_is_string(uuid)) {
		printf("The \"uuid\" attribute is missing from the template or is not a string.\n");
		rc = EINVAL;
		goto err;
	}
	json_t *attr_list = json_object_get(t, "attrs");
	if (!attr_list || !json_is_array(attr_list)) {
		printf("The \"attrs\" attribute is missing from the template or is not an array.\n");
		rc = EINVAL;
		goto err;
	}
	int attr_id;
	json_array_foreach(attr_list, attr_id, attr) {
		if (!json_is_object(attr)) {
			printf("The attribute entry is not an object.\n");
			rc = EINVAL;
			goto err;
		}

		i = json_object_get(attr, "name");
		if (!i || !json_is_string(i)) {
			printf("The \"name\" attribute is missing from the object.\n");
			rc = EINVAL;
			goto err;
		}
		const char *attr_name = json_string_value(i);

		i = json_object_get(attr, "type");
		if (!i || !json_is_string(i)) {
			printf("The \"type\" attribute is missing from the object.\n");
			rc = EINVAL;
			goto err;
		}
		const char *type_str = json_string_value(i);
		sos_type_t type = lookup_type(type_str);
		int size = 0;
		if (type == SOS_TYPE_STRUCT) {
			j = json_object_get(i, "size");
			if (j) {
				if (!json_is_integer(j)) {
					printf("The template file has an invalid attribute 'size' value, must be an integer\n");
					rc = EINVAL;
					goto err;
				}
				size = json_integer_value(j);
			}
		}

		char **attr_names = NULL;
		if (type == SOS_TYPE_JOIN) {
			json_t *join = json_object_get(i, "join_attrs");
			json_t *join_entry;
			if (!join || !json_is_array(join)) {
				printf("The template file has an invalid or missing join_attrs value, must be a list []\n");
				rc = EINVAL;
				goto err;
			}
			int item_no;
			size = json_array_size(join);
			attr_names = calloc(size, sizeof(char *));
			json_array_foreach(join, item_no, join_entry) {
				if (!json_is_string(join_entry)) {
					printf("The template file attribute join_list members must be quoted strings.\n");
					rc = EINVAL;
					goto err;
				}
				attr_names[item_no] = strdup(json_string_value(join_entry));
			}
		}
		rc = sos_schema_attr_add(schema, attr_name, type, size, attr_names);
		if (json_object_get(i, "index"))
			rc = sos_schema_index_add(schema, attr_name);
	}
	json_decref(t);
	dschema = dsos_schema_create(cont, schema, &res);
	printf("%p\n", dschema);
 err:
	return rc;
}

int dsosql_import_csv(dsos_container_t cont, FILE* fp, char *schema_name, char *col_spec)
{
	char lbuf[4096];
	struct timeval t0, t1, tr;
	int rc, cols;
	dsos_schema_t schema;
	char *inp, *tok;
	ods_atomic_t prev_recs = 0;
	int items_queued = 0;
	void *retval;
	dsos_res_t res;

	/* Get the schema */
	schema = dsos_schema_by_name(cont, schema_name);
	if (!schema) {
		printf("The schema '%s' was not found.\n", schema_name);
		return ENOENT;
	}

	if (!col_spec) {
		col_spec = lbuf;
		col_spec = fgets(lbuf, sizeof(lbuf), fp);
		if (col_spec[0] != '#') {
			printf("CSV file must have header to map columns to schmea.\n");
			return EINVAL;
		}
		col_spec++;
	}
	/* Count the number of columns  in the input spec */
	for (cols = 1, inp = col_spec; *inp != '\0'; inp++)
		if (*inp == ',')
			cols ++;

	/* Allocate a column map to contain the mapping */
	col_count = cols;
	col_map = calloc(cols, sizeof(int));
	attr_map = calloc(cols, sizeof(sos_attr_t));
	for (cols =  0, tok = strtok(col_spec, ","); tok; tok = strtok(NULL, ",")) {
		if (isdigit(*tok)) {
			/* Attribute id */
			col_map[cols] = atoi(tok);
			attr_map[cols] = dsos_schema_attr_by_id(schema, col_map[cols]);
		} else if (isalpha(*tok)) {
			/* Attribute name */
			attr_map[cols] = dsos_schema_attr_by_name(schema, tok);
			if (attr_map[cols])
				col_map[cols] = sos_attr_id(attr_map[cols]);
			else
				col_map[cols] = -1;
		} else {
			/* skip it */
			col_map[cols] = -1;
			attr_map[cols] = NULL;
		}
		cols++;
	}

	/*
	 * Read each line of the input CSV file. Separate the lines
	 * into columns delimited by the ',' character. Assign each
	 * column to the attribute id specified in the col_map[col_no]. The
	 * col_no in the input string are numbered 0..n
	 */
	records = 0;
	gettimeofday(&t0, NULL);
	tr = t0;
	rc = dsos_transaction_begin(cont, NULL);
	if (rc)
		return rc;
	while (1) {
		char *inp;
		char buf[4096];
		do {
			inp = fgets(buf, sizeof(buf), fp);
			if (!inp)
				goto out;
		} while (inp[0] == '#' || inp[0] == '\0');

		sos_obj_t obj = dsos_obj_new(schema);
		cols = 0;
		char *pos = NULL;
		for (tok = strtok_r(buf, ",", &pos); tok;
		     tok = strtok_r(NULL, ",", &pos)) {
			if (pos && pos[-1] == '\n')
				pos[-1] = '\0';
			if (cols >= col_count) {
				printf("Warning: line contains more columns "
				       "%d than are in column map.\n\"%s\"\n",
				       cols, buf);
				break;
			}
			int id = col_map[cols];
			if (id < 0) {
				cols++;
				continue;
			}
			rc = sos_obj_attr_from_str(obj, attr_map[cols], tok, NULL);
			if (rc) {
				printf("Warning: formatting error setting %s = %s.\n",
				       sos_attr_name(attr_map[cols]), tok);
			}
			cols++;
		}
		dsos_obj_create(cont, NULL, schema, obj);
		sos_obj_put(obj);
		if (rc) {
			printf("Error %d adding object to indices.\n", rc);
		}
		ods_atomic_inc(&records);

		if (records && (0 == (records % 10000))) {
			dsos_transaction_end(cont);
			double ts, tsr;
			gettimeofday(&t1, NULL);
			ts = (double)(t1.tv_sec - t0.tv_sec);
			ts += (double)(t1.tv_usec - t0.tv_usec) / 1.0e6;
			tsr = (double)(t1.tv_sec - tr.tv_sec);
			tsr += (double)(t1.tv_usec - tr.tv_usec) / 1.0e6;

			printf("Added %d records in %f seconds: %.0f/%.0f records/second.\n",
			       records, ts, (double)records / ts,
			       (double)(records - prev_recs) / tsr
			       );
			prev_recs = records;
			tr = t1;
			dsos_transaction_begin(cont, NULL);
		}
	}
 out:
	dsos_transaction_end(cont);
	printf("Added %d records.\n", records);
	return 0;
}

struct cond_key_s {
	char *name;
	enum sos_cond_e cond;
};

struct cond_key_s cond_keys[] = {
	{ "eq", SOS_COND_EQ },
	{ "ge", SOS_COND_GE },
	{ "gt", SOS_COND_GT },
	{ "le", SOS_COND_LE },
	{ "lt", SOS_COND_LT },
	{ "ne", SOS_COND_NE },
};

int compare_key(const void *a, const void *b)
{
	const char *str = a;
	struct cond_key_s const *cond = b;
	return strcmp(str, cond->name);
}

int add_filter(sos_schema_t schema, sos_filter_t filt, const char *str)
{
	struct cond_key_s *cond_key;
	sos_attr_t attr;
	sos_value_t cond_value;
	char attr_name[64];
	char cond_str[16];
	char value_str[256];
	int rc;

	/*
	 * See if str contains the special keyword 'unique'
	 */
	if (strcasestr(str, "unique")) {
		rc = sos_filter_flags_set(filt, SOS_ITER_F_UNIQUE);
		if (rc)
			printf("Error %d setting the filter flags.\n", rc);
		return rc;
	}

	rc = sscanf(str, "%64[^:]:%16[^:]:%256[^\t\n]", attr_name, cond_str, value_str);
	if (rc != 3) {
		printf("Error %d parsing the filter clause '%s'.\n", rc, str);
		return EINVAL;
	}

	/*
	 * Get the condition
	 */
	cond_key = bsearch(cond_str,
			   cond_keys, sizeof(cond_keys)/sizeof(cond_keys[0]),
			   sizeof(*cond_key),
			   compare_key);
	if (!cond_key) {
		printf("Invalid comparason, '%s', specified.\n", cond_str);
		return EINVAL;
	}

	/*
	 * Get the attribute
	 */
	attr = sos_schema_attr_by_name(schema, attr_name);
	if (!attr) {
		printf("The '%s' attribute is not a member of the schema.\n", attr_name);
		return EINVAL;
	}

	/*
	 * Create a value and set it
	 */
	cond_value = sos_value_init(sos_value_new(), NULL, attr);
	rc = value_from_str(attr, cond_value, value_str, NULL);
	if (rc) {
		printf("The value '%s' specified for the attribute '%s' is invalid.\n",
		       value_str, attr_name);
		return EINVAL;
	}

	rc = sos_filter_cond_add(filt, attr, cond_key->cond, cond_value);
	if (rc) {
		printf("The value could not be created, error %d.\n", rc);
		return EINVAL;
	}
	return 0;
}

int dsosql_query_select(dsos_container_t cont, const char *select_clause)
{
	dsos_query_t query = dsos_query_create(cont);
	sos_schema_t schema;
	sos_obj_t obj;
	struct col_s *col;
	int rec_count;
	struct col_list_s col_list = TAILQ_HEAD_INITIALIZER(col_list);
	void (*header)(FILE *outp, sos_attr_t index_attr, struct col_list_s *col_list);
	void (*row)(FILE *outp, sos_schema_t schema, sos_obj_t obj, struct col_list_s *col_list);
	void (*footer)(FILE *outp, int rec_count, int iter_count, struct col_list_s *col_list);

	if (!query) {
		printf("%s\n", dsos_last_errmsg());
		return dsos_last_err();
	}
	int rc = dsos_query_select(query, select_clause);
	if (rc) {
		printf("Error %d: \"%s\"\n", rc, dsos_last_errmsg());
		return rc;
	}
	schema = dsos_query_schema(query);
	/* Add all the attributes in the schema to the col_list */
	sos_attr_t attr;
	for (attr = sos_schema_attr_first(schema); attr; attr = sos_schema_attr_next(attr)) {
		if (sos_attr_type(attr) != SOS_TYPE_JOIN) {
			add_column(schema, sos_attr_name(attr), &col_list);
		}
	}

	TAILQ_FOREACH(col, &col_list, entry) {
		attr = sos_schema_attr_by_name(schema, col->name);
		if (!attr) {
			printf("The attribute %s from the view is not "
			       "in the schema.\n", col->name);
			return ENOENT;
		}
		col->id = sos_attr_id(attr);
		if (!col->width)
			col->width = col_widths[sos_attr_type(attr)];
	}

	switch (output_format) {
	case TABLE:
		header = table_header;
		row = table_row;
		footer = table_footer;
		break;
	case CSV:
		header = csv_header;
		row = csv_row;
		footer = csv_footer;
		break;
	case JSON:
		header = json_header;
		row = json_row;
		footer = json_footer;
		break;
	default:
		assert(NULL == "Invalid output format");
	}
	header(stdout, dsos_query_index_attr(query), &col_list);
	rec_count = 0;
	for (obj = dsos_query_next(query); obj; obj = dsos_query_next(query)) {
		row(stdout, schema, obj, &col_list);
		sos_obj_put(obj);
		rec_count += 1;
	}
	footer(stdout, rec_count, 0, &col_list);
	while (!TAILQ_EMPTY(&col_list)) {
		col = TAILQ_FIRST(&col_list);
		TAILQ_REMOVE(&col_list, col, entry);
		free((char *)col->name);
		free(col);
	}
	dsos_query_destroy(query);
	return 0;
}
