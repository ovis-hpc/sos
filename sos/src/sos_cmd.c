/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
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

/**
 * \page commands Commands
 *
 * The \c sos_cmd command is used to create containers, add schema,
 * import objects and query containers. See \ref partition_overview
 * for more information on partitions.
 */
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
#include <ods/ods_atomic.h>
#ifdef ENABLE_YAML
#include <yaml.h>
#include <sos/sos_yaml.h>
#endif

int add_filter(sos_schema_t schema, sos_filter_t filt, const char *str);
char *strcasestr(const char *haystack, const char *needle);

#ifdef ENABLE_YAML
const char *short_options = "f:I:M:C:K:O:S:X:V:F:T:s:o:icql";
#else
const char *short_options = "f:I:M:C:K:O:S:X:V:F:T:icql";
#endif

struct option long_options[] = {
	{"format",      required_argument,  0,  'f'},
	{"info",	no_argument,	    0,  'i'},
	{"create",	no_argument,	    0,  'c'},
	{"query",	no_argument,        0,  'q'},
	{"dir",         no_argument,        0,  'l'},
#ifdef ENABLE_YAML
	{"schema",      required_argument,  0,  's'},
	{"object",	required_argument,  0,  'o'},
#endif
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"mode",	required_argument,  0,  'O'},
	{"schema_name",	required_argument,  0,  'S'},
	{"index",	required_argument,  0,  'X'},
	{"csv",		required_argument,  0,  'I'},
	{"map",         required_argument,  0,  'M'},
	{"filter",	required_argument,  0,  'F'},
	{"threads",	required_argument,  0,  'T'},
	{"option",      optional_argument,  0,  'K'},
	{"column",      optional_argument,  0,  'V'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
#ifdef ENABLE_YAML
	printf("sos_cmd { -l | -i | -c | -o | -s | -K | -q } -C <container> "
	       "[-O <mode_mask>]\n");
#else
	printf("sos_cmd { -l | -i | -c | -K | -q } -C <container> "
	       "[-O <mode_mask>]\n");
#endif
	printf("    -C <path>      The path to the container. Required for all options.\n");
	printf("\n");
	printf("    -K <key>=<value> Set a container configuration option.\n");
	printf("\n");
	printf("    -l             Print a directory of the schemas.\n");
	printf("\n");
	printf("    -i		   Show debug information for the container.\n");
	printf("\n");
	printf("    -c             Create the container.\n");
	printf("       -O <mode>   The file mode bits for the container files,\n"
	       "                   see the open() system call.\n");
	printf("\n");
#ifdef ENABLE_YAML
	printf("    -s <path>      Add a schema to a container.\n"
	       "                   <path> is the path to the YAML file defining the schema\n");
	printf("\n");
	printf("    -o <path>      Add an object to a container.\n"
	       "                   <path> is the path to the YAML file containing the data\n");
	printf("\n");
#endif
	printf("    -I <csv_file>  Import a CSV file into the container.\n");
	printf("       -S <schema> The schema for objects.\n");
	printf("       -M <map>    String that maps CSV columns to object attributes.\n");
	printf("\n");
	printf("    -q             Query the container.\n");
	printf("       -S <schema> Schema of objects to query.\n");
	printf("       -X <index>  Attribute's index or name to query.\n");
	printf("       [-f <fmt>]  Specifies the format of the output data. Valid formats are:\n");
	printf("                   table  - Tabular format, one row per object. [default]\n");
	printf("                   csv    - Comma separated file with a single header row defining columns\n");
	printf("                   json   - JSON Objects.\n");
	printf("       [-F <rule>] Add a filter rule to the index.\n");
	printf("       [-V <col>]  Add an object attribute (i.e. column) to the output.\n");
	printf("                   Use '<col>[width]' to specify the desired column width\n");
	exit(1);
}

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
		cond_value->data->prim.timestamp_.fine.secs = (int)ts;
		uint32_t usecs = (uint32_t)((double)(ts - (int)ts) * 1.0e6);
		cond_value->data->prim.timestamp_.fine.usecs = usecs;
		rc = 0;
		break;
	default:
		break;
	}
	return rc;
}

int create(const char *path, int o_mode)
{
	int rc = sos_container_new(path, o_mode);
	if (rc) {
		errno = rc;
		perror("The container could not be created");
	}
	return rc;
}

int col_widths[] = {
	[SOS_TYPE_INT16] = 6,
	[SOS_TYPE_INT32] = 10,
	[SOS_TYPE_INT64] = 18,
	[SOS_TYPE_UINT16] = 6,
	[SOS_TYPE_UINT32] = 10,
	[SOS_TYPE_UINT64] = 18,
	[SOS_TYPE_FLOAT] = 12,
	[SOS_TYPE_DOUBLE] = 24,
	[SOS_TYPE_LONG_DOUBLE] = 48,
	[SOS_TYPE_TIMESTAMP] = 32,
	[SOS_TYPE_OBJ] = 8,
	[SOS_TYPE_STRUCT] = 32,
	[SOS_TYPE_BYTE_ARRAY] = 32,
	[SOS_TYPE_CHAR_ARRAY] = 32,
 	[SOS_TYPE_INT16_ARRAY] = 6,
 	[SOS_TYPE_INT32_ARRAY] = 8,
	[SOS_TYPE_INT64_ARRAY] = 8,
	[SOS_TYPE_UINT16_ARRAY] = 6,
	[SOS_TYPE_UINT32_ARRAY] = 8,
	[SOS_TYPE_UINT64_ARRAY] = 8,
	[SOS_TYPE_FLOAT_ARRAY] = 8,
	[SOS_TYPE_DOUBLE_ARRAY] = 8,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = 8,
	[SOS_TYPE_OBJ_ARRAY] = 8,
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
int add_column(const char *str)
{
	char *width;
	char *s;
	struct col_s *col = calloc(1, sizeof *col);
	if (!col)
		goto err;
	s = strdup(str);
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
	printf("Could not allocate memory for the column.\n");
	return ENOMEM;
}

struct cfg_s {
	char *kv;
	TAILQ_ENTRY(cfg_s) entry;
};
TAILQ_HEAD(cfg_list_s, cfg_s) cfg_list = TAILQ_HEAD_INITIALIZER(cfg_list);

/*
 * Add a configuration option. The format is:
 * <key>=<value>
 */
int add_config(const char *str)
{
	struct cfg_s *cfg = calloc(1, sizeof *cfg);
	if (!cfg)
		goto err;
	cfg->kv = strdup(str);
	TAILQ_INSERT_TAIL(&cfg_list, cfg, entry);
	return 0;
 err:
	printf("Could not allocate memory for the configuration option.\n");
	return ENOMEM;
}

struct clause_s {
	const char *str;
	TAILQ_ENTRY(clause_s) entry;
};
TAILQ_HEAD(clause_list_s, clause_s) clause_list = TAILQ_HEAD_INITIALIZER(clause_list);

int add_clause(const char *str)
{
	struct clause_s *clause = malloc(sizeof *clause);
	if (!clause)
		return ENOMEM;
	clause->str = strdup(str);
	TAILQ_INSERT_TAIL(&clause_list, clause, entry);
	return 0;
}

enum query_fmt {
	TABLE_FMT,
	CSV_FMT,
	JSON_FMT
} format;

void table_header(FILE *outp)
{
	struct col_s *col;
	/* Print the header labels */
	TAILQ_FOREACH(col, &col_list, entry)
		fprintf(outp, "%-*s ", col->width, col->name);
	fprintf(outp, "\n");

	/* Print the header separators */
	TAILQ_FOREACH(col, &col_list, entry) {
		int i;
		for (i = 0; i < col->width; i++)
			fprintf(outp, "-");
		fprintf(outp, " ");
	}
	fprintf(outp, "\n");
}

void csv_header(FILE *outp)
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

void json_header(FILE *outp)
{
	fprintf(outp, "{ \"data\" : [\n");
}

void table_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
{
	struct col_s *col;
	sos_attr_t attr;
	static char str[80];
	TAILQ_FOREACH(col, &col_list, entry) {
		attr = sos_schema_attr_by_id(schema, col->id);
		fprintf(outp, "%*s ", col->width,
			sos_obj_attr_to_str(obj, attr, str, 80));
	}
	fprintf(outp, "\n");
}

void csv_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
{
	struct col_s *col;
	int first = 1;
	sos_attr_t attr;
	static char str[80];
	TAILQ_FOREACH(col, &col_list, entry) {
		attr = sos_schema_attr_by_id(schema, col->id);
		if (!first)
			fprintf(outp, ",");
		fprintf(outp, "%s", sos_obj_attr_to_str(obj, attr, str, 80));
		first = 0;
	}
	fprintf(outp, "\n");
}

void json_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
{
	struct col_s *col;
	static int first_row = 1;
	int first = 1;
	sos_attr_t attr;
	static char str[80];
	if (!first_row)
		fprintf(outp, ",\n");
	first_row = 0;
	fprintf(outp, "{");
	TAILQ_FOREACH(col, &col_list, entry) {
		attr = sos_schema_attr_by_id(schema, col->id);
		if (!first)
			fprintf(outp, ",");
		fprintf(outp, "\"%s\" : \"%s\"", col->name, sos_obj_attr_to_str(obj, attr, str, 80));
		first = 0;
	}
	fprintf(outp, "}");
}

void table_footer(FILE *outp, int rec_count, int iter_count)
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
	fprintf(outp, "Records %d/%d.\n", rec_count, iter_count);
}

void csv_footer(FILE *outp, int rec_count, int iter_count)
{
	fprintf(outp, "# Records %d/%d.\n", rec_count, iter_count);
}

void json_footer(FILE *outp, int rec_count, int iter_count)
{
	fprintf(outp, "], \"%s\" : %d, \"%s\" : %d}\n",
		"totalRecords", rec_count, "recordCount", iter_count);
}

int query(sos_t sos, const char *schema_name, const char *index_name)
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
		printf("The schema '%s' was not found.\n", schema_name);
		return ENOENT;
	}
	attr = sos_schema_attr_by_name(schema, index_name);
	if (!attr) {
		printf("The attribute '%s' does not exist in '%s'.\n",
		       index_name, schema_name);
		return ENOENT;
	}
	iter = sos_attr_iter_new(attr);
	if (!iter)
		return ENOMEM;
	filt = sos_filter_new(iter);

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
		table_header(stdout);
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
	sos_iter_free(iter);
	return 0;
}
#ifdef ENABLE_YAML
int add_schema(sos_t sos, FILE *fp)
{
	yaml_parser_t parser;
	yaml_document_t document;

	memset(&parser, 0, sizeof(parser));
	memset(&document, 0, sizeof(document));

	if (!yaml_parser_initialize(&parser))
		return EINVAL;

	/* Set the parser parameters. */
	yaml_parser_set_input_file(&parser, fp);

	enum sos_parser_state {
		START,
		SCHEMA_DEF,
		SCHEMA_NAME_DEF,
		ATTR_DEF,
		ATTR_NAME_DEF,
		ATTR_TYPE_DEF,
		ATTR_INDEXED_DEF,
		STOP
	} state = START;
	struct keyword kw_;
	struct keyword *kw;
	yaml_event_t event;
	sos_schema_t schema = NULL;
	char *schema_name = NULL;
	char *attr_name = NULL;
	char *key_type = NULL;
	char *index_type = NULL;
	char *index_str = NULL;
	int attr_indexed = 0;
	sos_type_t attr_type = SOS_TYPE_FIRST;
	int attr_size = 0, rc = 0;
	do {
		if (!yaml_parser_parse(&parser, &event)) {
			printf("Error Line %zu Column %zu : %s.\n",
			       parser.context_mark.line,
			       parser.context_mark.column,
			       parser.problem);
			return EINVAL;
		}
		switch(event.type) {
		case YAML_NO_EVENT:
		case YAML_STREAM_START_EVENT:
		case YAML_STREAM_END_EVENT:
		case YAML_DOCUMENT_START_EVENT:
		case YAML_DOCUMENT_END_EVENT:
		case YAML_MAPPING_START_EVENT:
			switch (state) {
			case START:
				state = SCHEMA_DEF;
				break;
			default:
				break;
			}
			break;
		case YAML_SEQUENCE_END_EVENT:
			if (sos && schema)
				rc = sos_schema_add(sos, schema);
			state = SCHEMA_DEF;
			break;
		case YAML_SEQUENCE_START_EVENT:
			switch (state) {
			case START:
				state = SCHEMA_DEF;
				break;
			case SCHEMA_DEF:
				state = SCHEMA_DEF;
				break;
			default:
				printf("Unexpected new block in state %d.\n", state);
				return EINVAL;
			}
			break;
		case YAML_MAPPING_END_EVENT:
			if (!schema_name) {
				printf("The 'name' keyword must be used to "
				       "specify the schema name before "
				       "attributes are defined.\n");
				return EINVAL;
			}
			switch (state) {
			case SCHEMA_DEF:
				state = STOP;
				break;
			case SCHEMA_NAME_DEF:
				state = SCHEMA_DEF;
				break;
			case ATTR_DEF:
				rc = sos_schema_attr_add(schema, attr_name,
							 attr_type, attr_size);
				if (rc) {
					printf("Error %d adding attribute '%s'.\n",
					       rc, attr_name);
					return rc;
				}
				if (attr_indexed) {
					rc = sos_schema_index_add(schema, attr_name);
					if (rc) {
						printf("Error %d adding the index "
						       " for attribute '%s'.\n",
						       rc, attr_name);
						return rc;
					}
					if (key_type && key_type[0] != '\0') {
						if (!index_type)
							index_type = "BXTREE";
						rc = sos_schema_index_modify
							(
							 schema, attr_name,
							 index_type, key_type
							 );
						if (rc)
							printf("Warning: The key '%s' or index type '%s' "
							       "were not recognizerd.\n", key_type, index_type);
						free(index_str);
					}
				}
				state = SCHEMA_DEF;
				break;
			default:
				assert(0);
				state = SCHEMA_DEF;
				break;
			}
			break;
		case YAML_ALIAS_EVENT:
			break;
		case YAML_SCALAR_EVENT:
			kw_.str = (char *)event.data.scalar.value;
			kw = bsearch(&kw_, keyword_table,
				     sizeof(keyword_table)/sizeof(keyword_table[0]),
				     sizeof(keyword_table[0]),
				     compare_keywords);
			switch (state) {
			case SCHEMA_DEF:
				if (!kw) {
					printf("Unexpected keyword %s.\n",
					       event.data.scalar.value);
					return EINVAL;
				}
				switch (kw->id) {
				case NAME_KW:
					state = SCHEMA_NAME_DEF;
					break;
				case ATTRIBUTE_KW:
					state = ATTR_NAME_DEF;
					if (attr_name)
						free(attr_name);
					attr_name = NULL;
					attr_size = 0;
					attr_indexed = 0;
					attr_type = -1;
					break;
				case SCHEMA_KW:
					break;
				default:
					printf("The '%s' keyword is not "
					       "expected here and is being ignored.\n",
					       kw->str);
					break;
				}
				break;
			case ATTR_DEF:
				if (!kw) {
					printf("Unexpected keyword %s.\n",
					       event.data.scalar.value);
					break;
				}
				switch (kw->id) {
				case NAME_KW:
					printf("The 'name' keyword is not "
					       "expected in an attribute definition.\n");
					state = ATTR_DEF;
					break;
				case INDEXED_KW:
					state = ATTR_INDEXED_DEF;
					break;
				case TYPE_KW:
					state = ATTR_TYPE_DEF;
					break;
				default:
					printf("The keyword '%s' is not expected here.\n",
					       kw_.str);
					break;
				}
				break;
			case SCHEMA_NAME_DEF:
				if (schema_name)
					free(schema_name);
				schema_name = strdup((char *)event.data.scalar.value);
				schema = sos_schema_new(schema_name);
				if (!schema) {
					printf("The schema '%s' could not be "
					       "created, errno %d.\n",
					       schema_name, errno);
					return errno;
				}
				break;
			case ATTR_NAME_DEF:
				if (isdigit(event.data.scalar.value[0])) {
					printf("The first character of the attribute"
					       "named '%s' cannot be a number.\n",
					       event.data.scalar.value);
					return errno;
				}
				attr_name = strdup((char *)event.data.scalar.value);
				state = ATTR_DEF;
				break;
			case ATTR_INDEXED_DEF:
				if (!kw) {
					printf("The 'indexed' value must be "
					       "'true' or 'false', %s is not "
					       "recognized.\n", kw_.str);
					break;
				}
				attr_indexed = kw->id;
				index_str = strdup((char *)event.data.scalar.value);
				index_str = strtok(index_str, ",");
				key_type = strtok(NULL, ",");
				if (key_type) {
					while (isspace(*key_type)) key_type++;
					index_type = strtok(NULL, ",");
					if (index_type)
						while (isspace(*index_type)) index_type++;
				}
				state = ATTR_DEF;
				break;
			case ATTR_TYPE_DEF:
				if (!kw) {
					printf("Unrecognized 'type' name "
					       "'%s'.\n", kw_.str);
					break;
				}
				attr_type = kw->id;
				if (attr_type == SOS_TYPE_STRUCT) {
					/* decode the size */
					char *size_str, *str;
					str = strdup((char *)event.data.scalar.value);
					size_str = strtok(str, ",");
					if (!size_str)
						return EINVAL;
					size_str = strtok(NULL, ",");
					if (!size_str)
						return EINVAL;
					attr_size = strtoul(size_str, NULL, 0);
					if (errno)
						return EINVAL;
					free(str);
				} else
					attr_size = 0;
				state = ATTR_DEF;
				break;
			default:
				printf("Parser error!\n");
				return EINVAL;
			}
		}
		if(event.type != YAML_STREAM_END_EVENT)
			yaml_event_delete(&event);

	} while(event.type != YAML_STREAM_END_EVENT);
	if (schema_name)
		free(schema_name);
	yaml_event_delete(&event);
	yaml_parser_delete(&parser);

	return rc;
}
#endif

int import_done = 0;

struct obj_entry_s {
	char buf[3 * 1024];
	sos_obj_t obj;
	TAILQ_ENTRY(obj_entry_s) entry;
	LIST_ENTRY(obj_entry_s) free;
};
pthread_mutex_t free_lock = PTHREAD_MUTEX_INITIALIZER;
LIST_HEAD(obj_entry_list, obj_entry_s) free_list =
	LIST_HEAD_INITIALIZER(free_list);

struct obj_q_s {
	sos_t sos;
	int depth;
	pthread_mutex_t lock;
	pthread_cond_t wait;
	TAILQ_HEAD(obj_q_list, obj_entry_s) queue;
};

#define ADD_THREADS 4
#define DRAIN_WAIT (ADD_THREADS * 10 * 1024)
int thread_count = ADD_THREADS;

struct obj_q_s *work_queues;
pthread_mutex_t drain_lock;
pthread_cond_t drain_wait;
ods_atomic_t queued_work;

struct obj_entry_s *alloc_work()
{
	struct obj_entry_s *entry;
	pthread_mutex_lock(&free_lock);
	if (!LIST_EMPTY(&free_list)) {
		entry = LIST_FIRST(&free_list);
		LIST_REMOVE(entry, free);
	} else
		entry = malloc(sizeof *entry);
	pthread_mutex_unlock(&free_lock);
	return entry;
}

void free_work(struct obj_entry_s *entry)
{
	pthread_mutex_lock(&free_lock);
	LIST_INSERT_HEAD(&free_list, entry, free);
	pthread_mutex_unlock(&free_lock);

	ods_atomic_dec(&queued_work);
	if (queued_work < DRAIN_WAIT)
		pthread_cond_signal(&drain_wait);
}

static int next_q;
void queue_work(struct obj_entry_s *work)
{
	ods_atomic_inc(&queued_work);
	pthread_mutex_lock(&work_queues[next_q].lock);
	TAILQ_INSERT_TAIL(&work_queues[next_q].queue, work, entry);
	work_queues[next_q].depth++;
	pthread_cond_signal(&work_queues[next_q].wait);
	pthread_mutex_unlock(&work_queues[next_q].lock);

	next_q++;
	if (next_q >= thread_count)
		next_q = 0;
}

ods_atomic_t records;
size_t col_count;
int *col_map;
sos_attr_t *attr_map;

void *add_proc(void *arg)
{
	struct obj_q_s *queue = arg;
	struct obj_entry_s *work;
	char *tok;
	int rc;
	int cols;

	while (!import_done || !TAILQ_EMPTY(&queue->queue)) {
		pthread_mutex_lock(&queue->lock);
		while (!queue->depth && !import_done) {
			rc = pthread_cond_wait(&queue->wait, &queue->lock);
			if (rc == EINTR)
				continue;
		}
		if (!TAILQ_EMPTY(&queue->queue)) {
			work = TAILQ_FIRST(&queue->queue);
			TAILQ_REMOVE(&queue->queue, work, entry);
			queue->depth--;
		} else
			work = NULL;
		pthread_mutex_unlock(&queue->lock);
		if (import_done && !work)
			break;
		cols = 0;
		char *pos = NULL;
		for (tok = strtok_r(work->buf, ",", &pos); tok;
		     tok = strtok_r(NULL, ",", &pos)) {
			if (pos && pos[-1] == '\n')
				pos[-1] = '\0';
			if (cols >= col_count) {
				printf("Warning: line contains more columns "
				       "than are in column map.\n\"%s\"",
				       work->buf);
				break;
			}
			int id = col_map[cols];
			if (id < 0) {
				cols++;
				continue;
			}
			rc = sos_obj_attr_from_str(work->obj, attr_map[cols], tok, NULL);
			if (rc) {
				printf("Warning: formatting error setting %s = %s.\n",
				       sos_attr_name(attr_map[cols]), tok);
			}
			cols++;
		}
		rc = sos_obj_index(work->obj);
		sos_obj_put(work->obj);
		if (rc) {
			printf("Error %d adding object to indices.\n", rc);
		}
		ods_atomic_inc(&records);
		free_work(work);
	}

	return NULL;
}

int import_csv(sos_t sos, FILE* fp, char *schema_name, char *col_spec)
{
	struct timeval t0, t1, tr;
	int rc, cols;
	sos_schema_t schema;
	char *inp, *tok;
	ods_atomic_t prev_recs = 0;
	int items_queued = 0;
	void *retval;

	/* Get the schema */
	schema = sos_schema_by_name(sos, schema_name);
	if (!schema) {
		printf("The schema '%s' was not found.\n", schema_name);
		return ENOENT;
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
			col_map[cols] = atoi(tok);
			attr_map[cols] = sos_schema_attr_by_id(schema, col_map[cols]);
		} else {
			col_map[cols] = -1;
			attr_map[cols] = NULL;
		}
		cols++;
	}

	pthread_t *add_thread = calloc(thread_count, sizeof *add_thread);
	work_queues = calloc(thread_count, sizeof *work_queues);
	int i;
	for (i = 0; i < thread_count; i++) {
		work_queues[i].sos = sos;
		TAILQ_INIT(&work_queues[i].queue);
		pthread_mutex_init(&work_queues[i].lock, NULL);
		pthread_cond_init(&work_queues[i].wait, NULL);
		rc = pthread_create(&add_thread[i], NULL, add_proc, &work_queues[i]);
	}

	/*
	 * Read each line of the input CSV file. Separate the lines
	 * into columns delimited by the ',' character. Assign each
	 * column to the attribute id specified in the col_map[col_no]. The
	 * col_no in the input string are numbered 0..n
	 */
	pthread_cond_init(&drain_wait, NULL);
	pthread_mutex_init(&drain_lock, NULL);
	records = 0;
	gettimeofday(&t0, NULL);
	tr = t0;
	while (1) {
		char *inp;
		struct obj_entry_s *work;

		pthread_mutex_lock(&drain_lock);
		while (queued_work > DRAIN_WAIT) {
			rc = pthread_cond_wait(&drain_wait, &drain_lock);
			if (rc) {
				perror("pthread_cond_wait: ");
				printf("Wait error %d\n", rc);
			}
		}
		pthread_mutex_unlock(&drain_lock);

		work = alloc_work();
		do {
			inp = fgets(work->buf, sizeof(work->buf), fp);
			if (!inp)
				goto out;
		} while (inp[0] == '#' || inp[0] == '\0');
		work->obj = sos_obj_new(schema);
		if (!work->obj) {
			printf("Memory allocation failure!\n");
			break;
		}
		queue_work(work);
		items_queued++;
		if (records && (0 == (records % 10000))) {
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
		}
	}
 out:
	import_done = 1;
	printf("queued %d items...joining.\n", items_queued);
	for (i = 0; i < thread_count; i++)
		pthread_cond_signal(&work_queues[i].wait);
	for (i = 0; i < thread_count; i++)
		pthread_join(add_thread[i], &retval);
	printf("Added %d records.\n", records);
	return 0;
}

#ifdef ENABLE_YAML
int add_object(sos_t sos, FILE* fp)
{
	yaml_parser_t parser;
	yaml_document_t document;

	memset(&parser, 0, sizeof(parser));
	memset(&document, 0, sizeof(document));

	if (!yaml_parser_initialize(&parser))
		return 0;

	/* Set the parser parameters. */
	yaml_parser_set_input_file(&parser, fp);

	enum parser_state {
		START,
		NEED_SCHEMA,
		SCHEMA_VALUE,
		NEXT_ATTR,
		NEXT_VALUE,
		STOP
	}  state = START;
	struct keyword kw_;
	struct keyword *kw;
	yaml_event_t event;
	char *attr_name = NULL;
	char *attr_value = NULL;
	int rc;
	int obj_count = 0;
	sos_obj_t sos_obj = NULL;
	sos_schema_t schema = NULL;
	srandom(time(NULL));
	do {
		if (!yaml_parser_parse(&parser, &event)) {
			printf("Error Line %zu Column %zu : %s.\n",
			       parser.context_mark.line,
			       parser.context_mark.column,
			       parser.problem);
			return obj_count;
		}
		switch(event.type) {
		case YAML_NO_EVENT:
		case YAML_STREAM_START_EVENT:
		case YAML_STREAM_END_EVENT:
		case YAML_DOCUMENT_END_EVENT:
		case YAML_DOCUMENT_START_EVENT:
		case YAML_MAPPING_START_EVENT:
		case YAML_MAPPING_END_EVENT:
		case YAML_ALIAS_EVENT:
			break;
		case YAML_SEQUENCE_START_EVENT:
			state = NEED_SCHEMA;
			break;
		case YAML_SEQUENCE_END_EVENT:
			if (schema) {
				schema = NULL;
			}
			if (sos_obj) {
				rc = sos_obj_index(sos_obj);
				if (rc) {
					printf("Error %d adding object to it's indices.\n",
					       rc);
				}
			}
			sos_obj_put(sos_obj);
			sos_obj = NULL;
			state = START;
			break;
		case YAML_SCALAR_EVENT:
			kw_.str = (char *)event.data.scalar.value;
			kw = bsearch(&kw_, keyword_table,
				     sizeof(keyword_table)/sizeof(keyword_table[0]),
				     sizeof(keyword_table[0]),
				     compare_keywords);
			switch (state) {
			case NEED_SCHEMA:
				if (!kw || kw->id != SCHEMA_KW) {
					printf("Expected the 'schema' keyword.\n");
					return obj_count;
				}
				state = SCHEMA_VALUE;
				break;
			case SCHEMA_VALUE:
				schema = sos_schema_by_name(sos, kw_.str);
				if (!schema) {
					printf("The schema '%s' was not found.\n",
					       kw_.str);
					return obj_count;
				}
				sos_obj = sos_obj_new(schema);
				if (!sos_obj) {
					printf("Error %d creating the '%s' object.\n",
					       errno,
					       kw_.str);
					return obj_count;
				}
				obj_count++;
				state = NEXT_ATTR;
				break;
			case NEXT_ATTR:
				if (attr_name)
					free(attr_name);
				attr_name = strdup(kw_.str);
				state = NEXT_VALUE;
				break;
			case NEXT_VALUE:
				if (attr_value)
					free(attr_value);
				if (kw) {
					long val;
					switch (kw->id) {
					case TIME_FUNC_KW:
						val = time(NULL);
						attr_value = malloc(32);
						sprintf(attr_value, "%ld", val);
						break;
					case RANDOM_FUNC_KW:
						val = random();
						attr_value = malloc(32);
						sprintf(attr_value, "%ld", val);
						break;
					default:
						attr_value = strdup(kw_.str);
						break;
					}
				} else
					attr_value = strdup(kw_.str);
				rc = sos_obj_attr_by_name_from_str(sos_obj,
								   attr_name, attr_value,
								   NULL);
				if (rc) {
					printf("Error %d setting attribute '%s' to '%s'.\n",
					       rc, attr_name, attr_value);
				}
				state = NEXT_ATTR;
				break;
			default:
				break;
			}
		}
		if(event.type != YAML_STREAM_END_EVENT)
			yaml_event_delete(&event);

	} while(event.type != YAML_STREAM_END_EVENT);
	yaml_event_delete(&event);
	yaml_parser_delete(&parser);
	return 0;
}
#endif

#define INFO		0x001
#define CREATE		0x002
#define SCHEMA  	0x004
#define OBJECT		0x008
#define QUERY		0x010
#define SCHEMA_DIR	0x020
#define CSV		0x080
#define CONFIG  	0x100

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
	char value_str[64];
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

	rc = sscanf(str, "%64[^:]:%16[^:]:%64[^\t\n]", attr_name, cond_str, value_str);
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

int main(int argc, char **argv)
{
	char *path = NULL;
	char *col_map = NULL;
	int o, rc = 0;
	int o_mode = 0664;
	int action = 0;
	sos_t sos;
	char *index_name = NULL;
	char *schema_name = NULL;
	FILE *schema_file = NULL;
	FILE *obj_file = NULL;
	FILE *csv_file = NULL;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'c':
			action |= CREATE;
			break;
		case 'i':
			action |= INFO;
			break;
		case 's':
			action |= SCHEMA;
			schema_file = fopen(optarg, "r");
			if (!schema_file) {
				perror("Error opening the schema file: ");
				exit(9);
			}
			break;
		case 'o':
			action |= OBJECT;
			obj_file = fopen(optarg, "r");
			if (!obj_file) {
				perror("Error opening the schema file: ");
				exit(9);
			}
			break;
		case 'l':
			action |= SCHEMA_DIR;
			break;
		case 'q':
			action |= QUERY;
			break;
		case 'f':
			if (0 == strcasecmp("table", optarg))
				format = TABLE_FMT;
			else if (0 == strcasecmp("csv", optarg))
				format = CSV_FMT;
			else if (0 == strcasecmp("json", optarg))
				format = JSON_FMT;
			else {
				fprintf(stderr, "Ignoring unrecognized output format '%s'\n",
					optarg);
				format = TABLE_FMT;
			}
			break;
		case 'C':
			path = strdup(optarg);
			break;
		case 'K':
			action |= CONFIG;
			if (optarg && add_config(optarg))
				exit(11);
			break;
		case 'O':
			o_mode = strtol(optarg, NULL, 0);
			break;
		case 'V':
			if (add_column(optarg))
				exit(11);
			break;
		case 'S':
			schema_name = strdup(optarg);
			break;
		case 'X':
			index_name = strdup(optarg);
			break;
		case 'M':
			col_map = strdup(optarg);
			break;
		case 'F':
			if (add_clause(optarg))
				exit(11);
			break;
		case 'I':
			action |= CSV;
			csv_file = fopen(optarg, "r");
			if (!csv_file) {
				perror("Error opening CSV file: ");
				exit(9);
			}
			break;
		case 'T':
			thread_count = atoi(optarg);
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (!path)
		usage(argc, argv);

	if (!action) {
		printf("No action was requested.\n");
		usage(argc, argv);
	}
	if (action & CREATE) {
		rc = create(path, o_mode);
		action &= ~CREATE;
	}

	if (!action)
		return 0;

	if (action & CONFIG) {
		struct cfg_s *cfg;
		TAILQ_FOREACH(cfg, &cfg_list, entry) {
			char *option, *value;
			option = strtok(cfg->kv, "=");
			value = strtok(NULL, "=");
			rc = sos_container_config_set(path, option, value);
			if (rc)
				printf("Warning: The '%s' option was ignored.\n",
				       cfg->kv);
		}
		sos_config_print(path, stdout);
		action &= ~CONFIG;
	}
	if (!action)
		return 0;

	sos_perm_t mode;
	if (!(action & (CSV | OBJECT | SCHEMA)))
		mode = SOS_PERM_RO;
	else
		mode = SOS_PERM_RW;
	sos = sos_container_open(path, mode);
	if (!sos) {
		printf("Error %d opening the container %s.\n",
		       errno, path);
		exit(1);
	}
#ifdef ENABLE_YAML
	if (action & OBJECT) {
		rc = add_object(sos, obj_file);
		if (rc) {
			printf("Error %d processing objects file.\n", rc);
			exit(3);
		}
	}

	if (action & SCHEMA) {
		rc = add_schema(sos, schema_file);
		if (rc) {
			printf("Error %d processing schema file.\n", rc);
			exit(2);
		}
	}
#endif

	if (action & INFO)
		sos_container_info(sos, stdout);

	if (action & QUERY) {
		if (!index_name || !schema_name) {
			printf("The -X and -S options must be specified with "
			       "the query flag.\n");
			usage(argc, argv);
		}
		rc = query(sos, schema_name, index_name);
	}

	if (action & SCHEMA_DIR)
		rc = schema_dir(sos);

	if (action & CSV) {
		if (!schema_name || !csv_file || !col_map)
			usage(argc, argv);
		rc = import_csv(sos, csv_file, schema_name, col_map);
	}
	sos_container_close(sos, SOS_COMMIT_SYNC);
	return rc;
}
