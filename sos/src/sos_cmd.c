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

#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <yaml.h>
#include <assert.h>
#include <sos/sos.h>
#include "sos_yaml.h"

const char *short_options = "C:O:Y:y:S:A:icsoq";

struct option long_options[] = {
	{"info",	no_argument,	    0,  'i'},
	{"create",	no_argument,	    0,  'c'},
	{"query",	no_argument,        0,  'q'},
	{"schema",      no_argument,        0,  's'},
	{"object",	no_argument,        0,  'o'},
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"mode",	required_argument,  0,  'O'},
	{"schema_file",	required_argument,  0,  'Y'},
	{"obj_file",	required_argument,  0,  'y'},
	{"schema_name",	required_argument,  0,  'S'},
	{"attr_name",	required_argument,  0,  'A'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos { -i | -c | -s | -q } -C <container> [-o <mode_mask>] [-Y <yaml-file>]\n");
	printf("    -C <path>      The path to the container. Required for all options.\n");
	printf("\n");
	printf("    -i		   Show information for the container.\n");
	printf("\n");
	printf("    -c             Create the container.\n");
	printf("       -O <mode>   The file mode bits for the container files,\n"
	       "                   see the open() system call.\n");
	printf("\n");
	printf("    -s             Add a schema to a container.\n");
	printf("       -Y <path>   Path to a schema description file.\n");
	printf("\n");
	printf("    -o             Add an object to a container.\n");
	printf("       -y <path>   Path to an object description file.\n");
	printf("\n");
	printf("    -q             Query the container.\n");
	printf("       -S <schema> Object type to query.\n");
	printf("       -A <attr>   Attribute's index to query.\n");
	exit(1);
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
	[SOS_TYPE_INT32] = 10,
	[SOS_TYPE_INT64] = 18,
	[SOS_TYPE_UINT32] = 10,
	[SOS_TYPE_UINT64] = 18,
	[SOS_TYPE_FLOAT] = 12,
	[SOS_TYPE_DOUBLE] = 24,
	[SOS_TYPE_LONG_DOUBLE] = 48,
	[SOS_TYPE_OBJ] = 8,
	[SOS_TYPE_BYTE_ARRAY] = 32,
 	[SOS_TYPE_INT32_ARRAY] = 8,
	[SOS_TYPE_INT64_ARRAY] = 8,
	[SOS_TYPE_UINT32_ARRAY] = 8,
	[SOS_TYPE_UINT64_ARRAY] = 8,
	[SOS_TYPE_FLOAT_ARRAY] = 8,
	[SOS_TYPE_DOUBLE_ARRAY] = 8,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = 8,
	[SOS_TYPE_OBJ_ARRAY] = 8,
};

int query(sos_t sos, const char *schema_name, const char *attr_name)
{
	sos_schema_t schema;
	sos_attr_t attr;
	sos_iter_t iter;
	size_t attr_count, attr_id;
	int rc;

	schema = sos_schema_find(sos, schema_name);
	if (!schema) {
		printf("The schema '%s' was not found.\n", schema_name);
		return ENOENT;
	}
	attr = sos_attr_by_name(schema, attr_name);
	if (!attr) {
		printf("The attribute '%s' does not exist in '%s'.\n",
		       attr_name, schema_name);
		return ENOENT;
	}
	iter = sos_iter_new(attr);
	if (!iter)
		return ENOMEM;


	attr_count = sos_schema_attr_count(schema);
	/* Print the header labels */
	for (attr_id = 0; attr_id < attr_count; attr_id++) {
		attr = sos_attr_by_id(schema, attr_id);
		col_widths[attr_id] = sos_value_size(attr, NULL);
		if (sos_attr_type(attr) < SOS_TYPE_OBJ)
			col_widths[attr_id] = (col_widths[attr_id] * 2) + 2;
		printf("%-*s ",
		       col_widths[attr_id],
		       sos_attr_name(attr));
	}
	printf("\n");
	/* Print the header separators */
	for (attr_id = 0; attr_id < attr_count; attr_id++) {
		int i;
		attr = sos_attr_by_id(schema, attr_id);
		for (i = 0; i < col_widths[attr_id]; i++)
			printf("-");
		printf(" ");
	}
	printf("\n");
	for (rc = sos_iter_begin(iter); !rc; rc = sos_iter_next(iter)) {
		char str[80];
		sos_obj_t obj = sos_iter_obj(iter);
		for (attr_id = 0; attr_id < attr_count; attr_id++) {
			attr = sos_attr_by_id(schema, attr_id);
			printf("%-*s ",
			       col_widths[attr_id],
			       sos_value_to_str(obj, attr, str, 80));
		}
		printf("\n");
		sos_obj_put(obj);
	}
}

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
		ATTR_LEN_DEF,
		STOP
	} state = START;
	struct keyword kw_;
	struct keyword *kw;
	yaml_event_t event;
	sos_schema_t schema = NULL;
	char *schema_name = NULL;
	char *attr_name = NULL;
	int attr_indexed;
	sos_type_t attr_type;
	int arry_count;
	int rc = 0;
	do {
		if (!yaml_parser_parse(&parser, &event)) {
			printf("Parser error %d\n", parser.error);
			return EINVAL;
		}
		switch(event.type) {
		case YAML_NO_EVENT:
		case YAML_STREAM_START_EVENT:
		case YAML_STREAM_END_EVENT:
		case YAML_DOCUMENT_START_EVENT:
		case YAML_DOCUMENT_END_EVENT:
		case YAML_SEQUENCE_START_EVENT:
		case YAML_SEQUENCE_END_EVENT:
			break;
		case YAML_MAPPING_START_EVENT:
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
				rc = sos_attr_add(schema, attr_name, attr_type, arry_count);
				if (rc) {
					printf("Error %d adding attribute '%s'.\n",
					       rc, attr_name);
					return rc;
				}
				if (attr_indexed) {
					rc = sos_index_add(schema, attr_name);
					if (rc) {
						printf("Error %d adding the index "
						       " for attribute '%s'.\n",
						       rc, attr_name);
						return rc;
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
			kw_.str = event.data.scalar.value;
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
					arry_count = 0;
					if (attr_name)
						free(attr_name);
					attr_name = NULL;
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
				case LENGTH_KW:
					state = ATTR_LEN_DEF;
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
				schema_name = strdup(event.data.scalar.value);
				schema = sos_schema_new(schema_name);
				if (!schema) {
					printf("The schema '%s' could not be "
					       "created, errno %d.\n",
					       schema_name, errno);
					return errno;
				}
				break;
			case ATTR_NAME_DEF:
				attr_name = strdup(event.data.scalar.value);
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
				state = ATTR_DEF;
				break;
			case ATTR_TYPE_DEF:
				if (!kw) {
					printf("Unrecognized 'type' name "
					       "'%s'.\n", kw_.str);
					break;
				}
				attr_type = kw->id;
				state = ATTR_DEF;
				break;
			case ATTR_LEN_DEF:
				arry_count = strtoul(kw_.str, NULL, 0);
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

	rc = sos_schema_add(sos, schema);
	return rc;
}

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
			printf("Parser error %d\n", parser.error);
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
				sos_schema_put(schema);
				schema = NULL;
			}
			if (sos_obj) {
				rc = sos_obj_index(sos, sos_obj);
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
			kw_.str = event.data.scalar.value;
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
				if (schema)
					sos_schema_put(schema);
				schema = sos_schema_find(sos, kw_.str);
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
				rc = sos_attr_by_name_from_str(schema, sos_obj,
							       attr_name, attr_value);
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

#define INFO	0x01
#define CREATE	0x02
#define SCHEMA  0x04
#define OBJECT	0x08
#define QUERY	0x10

int main(int argc, char **argv)
{
	char *path = NULL;
	int o, rc;
	int o_mode = 0660;
	int action = 0;
	sos_t sos;
	char *attr_name = NULL;
	char *schema_name = NULL;
	FILE *schema_file = stdin;
	FILE *obj_file = stdin;
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
			break;
		case 'o':
			action |= OBJECT;
			break;
		case 'q':
			action |= QUERY;
			break;
		case 'C':
			path = optarg;
			break;
		case 'O':
			o_mode = strtol(optarg, NULL, 0);
			break;
		case 'Y':
			schema_file = fopen(optarg, "r");
			if (!schema_file) {
				perror("Error opening the schema file: ");
				exit(9);
			}
			break;
		case 'y':
			obj_file = fopen(optarg, "r");
			if (!obj_file) {
				perror("Error opening the schema file: ");
				exit(9);
			}
			break;
		case 'S':
			schema_name = strdup(optarg);
			break;
		case 'A':
			attr_name = strdup(optarg);
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (!path)
		usage(argc, argv);

	if (action & CREATE) {
		create(path, o_mode);
		action &= ~CREATE;
	}

	if (action) {
		rc = sos_container_open(path, O_RDWR, &sos);
		if (rc) {
			printf("Error %d opening the container %s.\n",
			       rc, path);
			exit(1);
		}
	}

	if (action & SCHEMA) {
		rc = add_schema(sos, schema_file);
		if (rc) {
			printf("Error %d processing schema file.\n", rc);
			exit(2);
		}
	}

	if (action & OBJECT) {
		rc = add_object(sos, obj_file);
		if (rc) {
			printf("Error %d processing objects file.\n", rc);
			exit(3);
		}
	}

	if (action & INFO)
		sos_container_info(sos, stdout);

	if(action & QUERY) {
		if (!attr_name || !schema_name) {
			printf("The -A and -S options must be specified with "
			       "the query flag.\n");
			usage(argc, argv);
		}
		rc = query(sos, schema_name, attr_name);
	}

	return 0;
}
