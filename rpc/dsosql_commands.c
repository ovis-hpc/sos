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
#include "json.h"
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

struct col_s {
	const char *name;
	int id;
	int width;
	TAILQ_ENTRY(col_s) entry;
};
TAILQ_HEAD(col_list_s, col_s) col_list = TAILQ_HEAD_INITIALIZER(col_list);

/*
 * Add a column. The format is:
 * <name>[col_width]
 */
static int add_column(const char *str)
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
	TAILQ_INSERT_TAIL(&col_list, col, entry);
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

static void table_header(FILE *outp, sos_attr_t index_attr)
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
	TAILQ_FOREACH(col, &col_list, entry) {
		if (col->width > 0)
			fprintf(outp, "%-*s ", col->width, col->name);
		else
			fprintf(outp, "%-s ", col->name);
	}
	fprintf(outp, "\n");

	/* Print the header separators */
	TAILQ_FOREACH(col, &col_list, entry) {
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

static void csv_header(FILE *outp)
{
	struct col_s *col;
	int first = 1;
	/* Print the header labels */
	fprintf(outp, "# ");
	TAILQ_FOREACH(col, &col_list, entry) {
		if (!first)
			fprintf(outp, ",");
		fprintf(outp, "%s", col->name);
		first = 0;
	}
	fprintf(outp, "\n");
}

static void json_header(FILE *outp)
{
	fprintf(outp, "{ \"data\" : [\n");
}

static void table_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
{
	struct col_s *col;
	size_t col_len;
	sos_attr_t attr;
	char *col_str;
	char str[80];
	TAILQ_FOREACH(col, &col_list, entry) {
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

static void csv_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
{
	struct col_s *col;
	int first = 1;
	sos_attr_t attr;
	size_t col_len;
	char *col_str;
	char str[80];
	TAILQ_FOREACH(col, &col_list, entry) {
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

static void json_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
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
	TAILQ_FOREACH(col, &col_list, entry) {
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

static void table_footer(FILE *outp, int rec_count, int iter_count)
{
	struct col_s *col;

	/* Print the footer separators */
	TAILQ_FOREACH(col, &col_list, entry) {
		int i;
		for (i = 0; i < col->width; i++)
			fprintf(outp, "-");
		fprintf(outp, " ");
	}
	fprintf(outp, "\n");
	fprintf(outp, "Records %d\n", rec_count);
}

static void csv_footer(FILE *outp, int rec_count, int iter_count)
{
	fprintf(outp, "# Records %d/%d.\n", rec_count, iter_count);
}

static void json_footer(FILE *outp, int rec_count, int iter_count)
{
	fprintf(outp, "], \"%s\" : %d, \"%s\" : %d}\n",
		"totalRecords", rec_count, "recordCount", iter_count);
}

int dsosql_query(sos_t sos, const char *schema_name, const char *index_name)
{
	sos_schema_t schema;
	sos_attr_t attr;
	sos_iter_t iter;
	size_t attr_count, attr_id;
	int rc;
	struct col_s *col;
	sos_filter_t filt;

	schema = sos_schema_by_name(sos, schema_name);
	if (!schema) {
		printf("The schema '%s' was not found, error %d.\n", schema_name, errno);
		return ENOENT;
	}
	attr = sos_schema_attr_by_name(schema, index_name);
	if (!attr) {
		printf("The attribute '%s' does not exist in '%s', error %d.\n",
		       index_name, schema_name, errno);
		return ENOENT;
	}
	if (!sos_attr_index(attr)) {
		printf("The attribute '%s' is not indexed in '%s', error %d.\n",
		       index_name, schema_name, errno);
		return ENOENT;
	}
	iter = sos_attr_iter_new(attr);
	if (!iter) {
		printf("Error %d creating and iterator for the index '%s'.\n",
		       errno, index_name);
		return errno;
	}
	filt = sos_filter_new(iter);
	if (!filt) {
		printf("Error %d creating a filter for the index '%s'.\n",
		       errno, index_name);
		return errno;
	}

	/* Create the col_list from the schema if the user didn't specify one */
	if (TAILQ_EMPTY(&col_list)) {
		attr_count = sos_schema_attr_count(schema);
		for (attr_id = 0; attr_id < attr_count; attr_id++) {
			attr = sos_schema_attr_by_id(schema, attr_id);
			if (add_column(sos_attr_name(attr)))
				return ENOMEM;
		}
	}
	/* Query the schema for each attribute's id, and compute width */
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

	/* Build the index filter */
	struct clause_s *clause;
	TAILQ_FOREACH(clause, &clause_list, entry) {
		rc = add_filter(schema, filt, clause->str);
		if (rc)
			return rc;
	}

	switch (format) {
	case JSON_FMT:
		json_header(stdout);
		break;
	case CSV_FMT:
		csv_header(stdout);
		break;
	default:
		table_header(stdout, NULL);
		break;
	}

	int rec_count;
	int iter_count;
	sos_obj_t obj;
	void (*printer)(FILE *outp, sos_schema_t schema, sos_obj_t obj);
	switch (format) {
	case JSON_FMT:
		printer = json_row;
		break;
	case CSV_FMT:
		printer = csv_row;
		break;
	default:
		printer = table_row;
		break;
	}

	for (rec_count = 0, iter_count = 0, obj = sos_filter_begin(filt);
	     obj; obj = sos_filter_next(filt), iter_count++) {
		printer(stdout, schema, obj);
		rec_count++;
		sos_obj_put(obj);
	}
	switch (format) {
	case JSON_FMT:
		json_footer(stdout, rec_count, iter_count);
		break;
	case CSV_FMT:
		csv_footer(stdout, rec_count, iter_count);
		break;
	default:
		table_footer(stdout, rec_count, iter_count);
		break;
	}
	sos_filter_free(filt);
	return 0;
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
	json_parser_t parser = json_parser_new(0);
	json_entity_t e, i, attr, value;
	int rc = json_parse_buffer(parser, template, strlen(template), &e);
	if (rc) {
		printf("Error %d parsing the schema template.\n", rc);
		goto err;
	}
	schema = sos_schema_new(schema_name);
	if (!schema) {
		printf("Error creating the schema %s\n", schema_name);
		rc = errno;
		goto err;
	}
	json_entity_t l = json_attr_find(e, "attrs");
	if (!l) {
		printf("The attribute list is missing from the template.\n");
		rc = EINVAL;
		goto err;
	}
	l = json_attr_value(l);
	/* Each item is a dictionary of 'name', 'type', + optional 'size' and 'join_list' */
	for (i = json_item_first(l); i; i = json_item_next(i)) {
		attr = json_attr_find(i, "type");
		json_entity_t value = json_attr_value(attr);
		if (json_entity_type(value) != JSON_STRING_VALUE) {
			printf("The template file has an invalid attribute 'type' value. Must be a quoted string\n");
			rc = EINVAL;
			goto err;
		}
		json_str_t str = json_value_str(value);
		sos_type_t type = lookup_type(str->str);
		if (!type) {
			printf("The template file has an invalid attribute 'type' value.\n");
			rc = EINVAL;
			goto err;
		}
		attr = json_attr_find(i, "name");
		if (json_entity_type(value) != JSON_STRING_VALUE) {
			printf("The template file has an invalid attribute 'name' value. Must be a quoted string\n");
			rc = EINVAL;
			goto err;
		}
		char *name = json_value_str(json_attr_value(attr))->str;
		int size = 0;
		if (type == SOS_TYPE_STRUCT) {
			attr = json_attr_find(i, "size");
			if (attr) {
				if (json_entity_type(value) != JSON_INT_VALUE) {
					printf("The template file has an invalid attribute 'size' value, must be an integer\n");
					rc = EINVAL;
					goto err;
				}
				size = json_value_int(json_attr_value(attr));
			}
		}
		char **attr_names = NULL;
		if (type == SOS_TYPE_JOIN) {
			attr = json_attr_find(i, "join_attrs");
			if (!attr) {
				printf("The template file has an invalid or missing join_attrs value, must be an list []\n");
				rc = EINVAL;
				goto err;
			}
			json_entity_t value = json_attr_value(attr);
			if (json_entity_type(value) != JSON_LIST_VALUE) {
				printf("The template file join_list attribute must be a list, [] of quoted strings.\n");
				rc = EINVAL;
				goto err;
			}
			json_list_t join_list = json_value_list(value);
			size = join_list->item_count;
			json_entity_t join_attr;
			attr_names = calloc(size, sizeof(char *));
			int item_no = 0;
			for (join_attr = json_item_first(&join_list->base);
			     join_attr;
			     join_attr = json_item_next(join_attr)) {
				if (json_entity_type(join_attr) != JSON_STRING_VALUE) {
					printf("The template file attribute join_list members muse be quoted strings.\n");
					rc = EINVAL;
					goto err;
				}
				attr_names[item_no] = json_value_str(join_attr)->str;
				item_no += 1;
			}
		}
		rc = sos_schema_attr_add(schema, name, type, size, attr_names);
		attr = json_attr_find(i, "index");
		if (attr)
			rc = sos_schema_index_add(schema, name);
	}
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
		dsos_obj_create(cont, schema, obj);
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
	int rc = dsos_query_select(query, select_clause);
	if (rc) {
		printf("Error %d: \"%s\"\n", rc, dsos_last_errmsg());
		return rc;
	}
	schema = dsos_query_schema(query);
	/* Add all the attributes in the schema to the col_list */
	sos_attr_t attr;
	while (!TAILQ_EMPTY(&col_list)) {
		col = TAILQ_FIRST(&col_list);
		TAILQ_REMOVE(&col_list, col, entry);
		free(col);
	}
	for (attr = sos_schema_attr_first(schema); attr; attr = sos_schema_attr_next(attr))
		if (sos_attr_type(attr) != SOS_TYPE_JOIN)
			add_column(sos_attr_name(attr));
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

	table_header(stdout, dsos_query_index_attr(query));
	rec_count = 0;
	for (obj = dsos_query_next(query); obj; obj = dsos_query_next(query)) {
		table_row(stdout, schema, obj);
		sos_obj_put(obj);
		rec_count += 1;
	}
	table_footer(stdout, rec_count, 0);
	while (!TAILQ_EMPTY(&col_list)) {
		col = TAILQ_FIRST(&col_list);
		TAILQ_REMOVE(&col_list, col, entry);
		free((char *)col->name);
		free(col);
	}
	dsos_query_destroy(query);
	return 0;
}
