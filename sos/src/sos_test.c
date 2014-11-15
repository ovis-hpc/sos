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

#include <sys/queue.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>
#include <stdint.h>
#include <inttypes.h>
#include <getopt.h>
#include <assert.h>
#include <yaml.h>

#include <sos/sos.h>

const char *short_options = "Cc:S:A:b:s:l:rRi?";

struct option long_options[] = {
	{"create",      no_argument,  0,  'C'},
	{"container",   required_argument,  0,  'c'},
	{"schema",	required_argument,  0,  'S'},
	{"attr",	required_argument,  0,  'A'},
	{"remove",      no_argument,        0,  'r'},
	{"rotate",      no_argument,        0,  'R'},
	{"keep-index",  no_argument,        0,  'i'},
	{"limit",       required_argument,  0,  'l'},
	{"backups",     required_argument,  0,  'l'},
	{"help",        no_argument,        0,  '?'},
	{0,             0,                  0,  0}
};

enum keyword_e {
	FALSE_KW = 0,
	TRUE_KW = 1,
	ATTRIBUTE_KW = 100,
	INDEXED_KW,
	SCHEMA_KW,
	NAME_KW,
	TYPE_KW,
	COUNT_KW,
};

struct keyword {
	char *str;
	enum keyword_e id;
};

struct keyword keyword_table[] = {
	{ "ATTRIBUTE", ATTRIBUTE_KW },
	{ "DOUBLE", SOS_TYPE_DOUBLE },
	{ "FALSE", FALSE_KW },
	{ "FLOAT", SOS_TYPE_FLOAT },
	{ "INDEXED", INDEXED_KW },
	{ "INT32", SOS_TYPE_INT32 },
	{ "INT64", SOS_TYPE_INT64 },
	{ "LONG_DOUBLE", SOS_TYPE_LONG_DOUBLE },
	{ "NAME", NAME_KW },
	{ "OBJ", SOS_TYPE_OBJ },
	{ "SCHEMA", SCHEMA_KW },
	{ "TRUE", TRUE_KW },
	{ "TYPE", TYPE_KW },
	{ "UINT32", SOS_TYPE_UINT32 },
	{ "UINT64", SOS_TYPE_UINT64 },
	{ "BYTE_ARRAY", SOS_TYPE_BYTE_ARRAY },
	{ "INT32_ARRAY", SOS_TYPE_INT32_ARRAY },
	{ "INT64_ARRAY", SOS_TYPE_INT64_ARRAY },
	{ "UINT32_ARRAY", SOS_TYPE_UINT32_ARRAY },
	{ "UINT64_ARRAY", SOS_TYPE_UINT64_ARRAY },
	{ "FLOAT_ARRAY", SOS_TYPE_FLOAT_ARRAY },
	{ "DOUBLE_ARRAY", SOS_TYPE_DOUBLE_ARRAY },
	{ "LONG_DOUBLE_ARRAY", SOS_TYPE_LONG_DOUBLE_ARRAY },
	{ "OBJ_ARRAY", SOS_TYPE_OBJ_ARRAY },
};

int compare_keywords(const void *a, const void *b)
{
	struct keyword *kw_a = (struct keyword *)a;
	struct keyword *kw_b = (struct keyword *)b;
	return strcasecmp(kw_a->str, kw_b->str);
}

sos_schema_t parse_schema(int argc, char *argv[])
{
	yaml_parser_t parser;
	yaml_emitter_t emitter;
	yaml_document_t document;

	memset(&parser, 0, sizeof(parser));
	memset(&document, 0, sizeof(document));

	/* Initialize the parser and emitter objects. */
	if (!yaml_parser_initialize(&parser))
		return NULL;

	/* Set the parser parameters. */
	yaml_parser_set_input_file(&parser, stdin);

	enum parser_state {
		START,
		SCHEMA_DEF,
		SCHEMA_NAME_DEF,
		ATTR_DEF,
		ATTR_NAME_DEF,
		ATTR_TYPE_DEF,
		ATTR_INDEXED_DEF,
		ATTR_COUNT_DEF,
		STOP
	} state = START;
	struct keyword kw_;
	struct keyword *kw;
	yaml_event_t event;
	sos_schema_t schema;
	char *attr_name;
	char *schema_name;
	int attr_indexed;
	sos_type_t attr_type;
	int arry_count = 0;
	int rc;
	do {
		if (!yaml_parser_parse(&parser, &event)) {
			printf("Parser error %d\n", parser.error);
			exit(EXIT_FAILURE);
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
				state = ATTR_DEF;
				break;
			default:
				assert(0);
			}
			break;
		case YAML_MAPPING_END_EVENT:
			if (!schema_name) {
				printf("The 'name' keyword must be used to "
				       "specify the schema name before "
				       "attributes are defined.\n");
				exit(1);
			}
			switch (state) {
			case SCHEMA_DEF:
				state = STOP;
				break;
			case ATTR_DEF:
				rc = sos_attr_add(schema, attr_name, attr_type, arry_count);
				if (rc) {
					printf("Error %d adding attribute '%s'.\n",
					       rc, attr_name);
					exit(2);
				}
				break;
			default:
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
					state = SCHEMA_NAME_DEF;
					break;
				case ATTRIBUTE_KW:
					state = ATTR_NAME_DEF;
					arry_count = 0;
					attr_name = NULL;
					attr_indexed = 0;
					attr_type = -1;
					break;
				case INDEXED_KW:
					state = ATTR_INDEXED_DEF;
					break;
				case TYPE_KW:
					state = ATTR_TYPE_DEF;
					break;
				}
				break;
			case SCHEMA_NAME_DEF:
				schema_name = event.data.scalar.value;
				state = ATTR_DEF;
				schema = sos_schema_new(schema_name);
				if (!schema) {
					printf("Error allocating schema for %s\n",
					       schema_name);
					exit(2);
				}
				break;
			case ATTR_NAME_DEF:
				attr_name = event.data.scalar.value;
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
			case ATTR_COUNT_DEF:
				arry_count = strtoul(kw_.str, NULL, 0);
				state = ATTR_DEF;
				break;
			default:
				printf("Parser error!\n");
				exit(1);
			}
		}
		if(event.type != YAML_STREAM_END_EVENT)
			yaml_event_delete(&event);

	} while(event.type != YAML_STREAM_END_EVENT);
	yaml_event_delete(&event);
	yaml_parser_delete(&parser);
	yaml_emitter_delete(&emitter);
	return schema;
}

void usage()
{
	printf("\n\
Usage: sos_test [OPTIONS]\n\
\n\
OPTIONS: \n\
    -c,--container <path>\n\
	The container name.\n\
\n\
    -S,--schema <schema_name>\n\
	The schema name to add to the object store.\n\
\n\
    -R,--rotate \n\
	Enable store rotation\n\
\n\
    -i,--keep-index \n\
	SOS rotation keep indices\n\
\n\
    -r,--remove \n\
	Enable old data removal (disabled if -R is given)\n\
\n\
    -l,--limit <TIME_LIMIT>\n\
	Time limitation in seconds for store rotation or data removal\n\
	(default: 60 (sec))\n\
\n\
    -b,--backups <BACKUPS_LIMIT>\n\
	The number of backups for store rotation (default: unlimited)\n\
"
	);
	_exit(-1);
}

int main(int argc, char **argv)
{
	const char *path = NULL;
	sos_schema_t schema;
	int o;

	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'c':
			path = optarg;
			break;
		case 'r':
			remove = 1;
			break;
		case 'R':
			rotate = 1;
			break;
		case 'i':
			keep_index = 1;
			break;
		case 'l':
			limit = atoi(optarg);
			break;
		case 'b':
			backups = atoi(optarg);
			break;
		case 'S':
			schema = parse_schema(argc, argv);
			break;
		case 'A':
			break;
		case '?':
		default:
			usage();
		}
	}
	if (!path)
		usage();

	if (create) {
		rc = sos_container_new(path, 0660);
		if (rc) {
			printf("The container could not be created.\n");
			usage();
		}
	}
	sos_t c;
	rc = sos_container_open(path, O_RDWR, &c);
	if (rc) {
		printf("Error %d opening the container.\n", rc);
		usage();
	}
	optind = 0;
	sos_schema_t schema = NULL;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'S':
			schema = sos_schema_new(optarg);
			if (!schema) {
				printf("Error creating schema.\n");
				usage();
			}
			break;
		case 'A':
			rc = sos_attr_add(schema, optarg, SOS_TYPE_UINT64, 1);
			if (rc) {
				printf("Error %d adding %s attribute to schema.\n",
				       rc, optarg);
				usage();
			}
			break;
		case 'C':
		case 'c':
		case 'r':
		case 'R':
		case 'i':
		case 'l':
		case 'b':
			break;
		case '?':
		default:
			usage();
		}
	}
	if (schema) {
		rc = sos_schema_add(c, schema);
		if (rc) {
			printf("Error %d adding schema to container.\n", rc);
			usage();
		}
	}
#if 0
	assert(sos);

	while ((s = fgets(buf, sizeof(buf), stdin)) != NULL) {
		n = sscanf(buf, "%"PRIu32".%"PRIu32" %"PRIu64" %"PRIu64,
				&sec, &usec, &metric_id, &value);

		if (!sec0)
			sec0 = sec;

		if (n != 4)
			break;

		obj = sos_obj_new(sos);
		assert(obj);

		sos_obj_attr_set_uint32(sos, 0, obj, sec);
		sos_obj_attr_set_uint32(sos, 1, obj, usec);
		sos_obj_attr_set_uint64(sos, 2, obj, metric_id);
		sos_obj_attr_set_uint64(sos, 3, obj, value);

		/* Add it to the indexes */
		rc = sos_obj_add(sos, obj);
		assert(rc == 0);
		if (remove)
			remove_data(sos, sec, limit);
		if (rotate && (sec - sec0) >= limit) {
			sos_t new_sos;
			if (keep_index)
				new_sos = sos_rotate_i(sos, backups);
			else
				new_sos = sos_rotate(sos, backups);
			assert(new_sos);
			sos = new_sos;
			sec0 = sec;
			p = sos_post_rotation(sos, "SOS_TEST_POSTROTATE");
		}
	}

	sos_close(sos, ODS_COMMIT_SYNC);
#endif
	return 0;
}
