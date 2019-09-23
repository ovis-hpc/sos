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
#include "config.h"
#undef VERSION

int add_filter(sos_schema_t schema, sos_filter_t filt, const char *str);
char *strcasestr(const char *haystack, const char *needle);

const char *short_options = "f:I:M:m:C:K:O:S:X:V:F:T:tidcqlLRv";

struct option long_options[] = {
	{"format",      required_argument,  0,  'f'},
	{"locks",	no_argument,	    0,  'L'},
	{"cleanup",	no_argument,	    0,  'R'},
	{"info",	no_argument,	    0,  'i'},
	{"debug",	no_argument,	    0,  'd'},
	{"move",	no_argument,	    0,  'm'},
	{"create",	no_argument,	    0,  'c'},
	{"query",	no_argument,        0,  'q'},
	{"dir",         no_argument,        0,  'l'},
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"mode",	required_argument,  0,  'O'},
	{"schema_name",	required_argument,  0,  'S'},
	{"index",	required_argument,  0,  'X'},
	{"csv",		required_argument,  0,  'I'},
	{"map",         required_argument,  0,  'M'},
	{"filter",	required_argument,  0,  'F'},
	{"test",	no_argument,        0,  't'},
	{"threads",	required_argument,  0,  'T'},
	{"option",      optional_argument,  0,  'K'},
	{"column",      optional_argument,  0,  'V'},
	{"version",     optional_argument,  0,  'v'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_cmd { -l | -i | -c | -K | -q } -C <path> -m <new_path>"
	       "[-O <mode_mask>]\n");
	printf("    -C <path>      The path to the container. Required for all options.\n");
	printf("    -v             Print the container version information and exit.\n");
	printf("    -m <new_path>  Use to modify the path saved internally after the container is copied.\n");
	printf("\n");
	printf("    -K <key>=<value> Set a container configuration option.\n");
	printf("\n");
	printf("    -l             Print a directory of the schemas.\n");
	printf("\n");
	printf("    -i		   Show config information for the container.\n");
	printf("    -d		   Show debug data for the container.\n");
	printf("\n");
	printf("    -c             Create the container.\n");
	printf("       -O <mode>   The file mode bits for the container files,\n"
	       "                   see the open() system call.\n");
	printf("\n");
	printf("    -I <csv_file>  Import a CSV file into the container.\n");
	printf("       -S <schema> The schema for objects.\n");
	printf("       -M <map>    String that maps CSV columns to object attributes.\n");
	printf("\n");
	printf("    -t             Test indices in the database for consistency.\n");
	printf("    -L             Show database lock information.\n");
	printf("    -R             Clean up locks held by dead processes.\n");
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
	printf("                   If not specified, all attributes in the object are output.\n");
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
	int rc = sos_container_new(path, o_mode);
	if (rc) {
		errno = rc;
		perror("The container could not be created");
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
int add_column(const char *str)
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

void csv_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
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

void json_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
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
	return 0;
}

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
				       "%d than are in column map.\n\"%s\"\n",
				       cols, work->buf);
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

#define INFO		0x0001
#define CREATE		0x0002
#define QUERY		0x0010
#define SCHEMA_DIR	0x0020
#define CSV		0x0080
#define CONFIG  	0x0100
#define LOCKS		0x0200
#define CLEANUP		0x0400
#define MOVE		0x0800
#define DEBUG		0x1000
#define VERSION		0x2000
#define TEST		0x4000

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

int main(int argc, char **argv)
{
	char *path = NULL;
	char *new_path = NULL;
	char *col_map = NULL;
	int o, rc = 0;
	int o_mode = 0664;
	int action = 0;
	sos_t sos;
	char *index_name = NULL;
	char *schema_name = NULL;
	FILE *csv_file = NULL;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'c':
			action |= CREATE;
			break;
		case 'i':
			action |= INFO;
			break;
		case 'd':
			action |= DEBUG;
			break;
		case 'L':
			action |= LOCKS;
			break;
		case 'R':
			action |= CLEANUP;
			break;
		case 'l':
			action |= SCHEMA_DIR;
			break;
		case 'q':
			action |= QUERY;
			break;
		case 'v':
			action |= VERSION;
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
		case 'm':
			new_path = strdup(optarg);
			action |= MOVE;
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
				fprintf(stderr, "Bad -I %s\n", optarg);
				perror("Error opening CSV file: ");
				exit(9);
			}
			break;
		case 't':
			action |= TEST;
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

	if (action & MOVE) {
		rc = sos_container_move(path, new_path);
		if (rc) {
			printf("Error %d updating the container's internal path data.\n", rc);
			exit(1);
		}
		action &= ~MOVE;
	}

	if (!action)
		return 0;

	if (action & VERSION) {
		printf("Library Version   : %d.%d.%d\n", ODS_VER_MAJOR, ODS_VER_MINOR, ODS_VER_FIX);
		printf("Git Commit ID     : %s\n", ODS_COMMIT_ID);
	}
	if (action & CLEANUP)
		sos_container_lock_cleanup(path);

	if (action & LOCKS)
		return sos_container_lock_info(path, stdout);

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
	if (!(action & CSV))
		mode = SOS_PERM_RO;
	else
		mode = SOS_PERM_RW;
	sos = sos_container_open(path, mode);
	if (!sos) {
		printf("Error %d opening the container %s.\n",
		       errno, path);
		exit(1);
	}
	if (action & VERSION) {
		struct sos_version_s vers = sos_container_version(sos);
		printf("Container Version : %d.%d.%d\n", vers.major, vers.minor, vers.fix);
		printf("Git Commit ID     : %s\n", vers.git_commit_id);
	}

	if (action & INFO)
		sos_config_print(path, stdout);

	if (action & DEBUG)
		sos_container_info(sos, stdout);

	if (action & TEST) {
		rc = sos_container_verify(sos);
		action &= TEST;
		return rc;
	}

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
	free(path);
	return rc;
}
