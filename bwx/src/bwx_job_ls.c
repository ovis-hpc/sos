/* -*- c-basic-offset: 8 -*-
 * Copyright (c) 2015 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2015 Sandia Corporation. All rights reserved.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S. Government.
 * Export of this program may require a license from the United States
 * Government.
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
 *      Neither the name of Sandia nor the names of any contributors may
 *      be used to endorse or promote products derived from this software
 *      without specific prior written permission.
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
#define _GNU_SOURCE
#include <inttypes.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <assert.h>
#include <sos/sos.h>
#include <ods/ods_atomic.h>
#include <ods/rbt.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <bwx/bwx.h>


#pragma pack(4)
struct Kvs {
	uint32_t secondary;
	uint32_t primary;
};

struct Job {
	uint32_t id;
	struct sos_timestamp_s start;
	struct sos_timestamp_s end;
};

struct Sample {
	struct sos_timestamp_s time;
	struct Kvs job_time;
	uint32_t comp_id;
};
#pragma pack()

const char *short_options = "C:D:X:ja:b:tu:V:J:cdv";

struct option long_options[] = {
	{"container",   required_argument,  0,  'C'},
	{"uid",         required_argument,  0,  'u'},
	{"after",       required_argument,  0,  'a'},
	{"before",      required_argument,  0,  'b'},
	{"by_component",no_argument,        0,  'c'},
	{"component",   required_argument,  0,  'D'},
	{"by_job",      no_argument,        0,  'j'},
	{"job",         required_argument,  0,  'J'},
	{"diff",        no_argument,        0,  'd'},
	{"verbose",     no_argument,        0,  'v'},
	{"view",        required_argument,  0,  'V'},
	{"bin-width",   required_argument,  0,  'w'},
	{0,             0,                  0,    0}
};
static int verbose = 0;
static char *job_str = NULL;
static char *comp_str = NULL;
static char *uid_str = NULL;
static char *start_str = NULL;
static struct sos_timestamp_s start_time = { 0, 0 };
static char *end_str = NULL;
static struct sos_timestamp_s end_time = { 0xffffffff, 0xffffffff };

struct col_s {
	const char *name;
	int id;
	int width;
	TAILQ_ENTRY(col_s) entry;
};
TAILQ_HEAD(col_list_s, col_s) col_list = TAILQ_HEAD_INITIALIZER(col_list);

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

/*
 * Add a column. The format is:
 * <name>[~col_width]
 */
int add_column(const char *str)
{
	char *width;
	char *s;
	struct col_s *col = calloc(1, sizeof *col);
	if (!col)
		goto err;
	s = strdup(str);
	width = strchr(s, '~');
	if (width) {
		*width = '\0';
		width++;
		col->width = atoi(width);
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

void show_by_time(sos_t sos)
{
	sos_schema_t job_schema = sos_schema_by_name(sos, "Job");
	sos_attr_t StartTime_attr = sos_schema_attr_by_name(job_schema, "StartTime");
	sos_iter_t job_iter = sos_attr_iter_new(StartTime_attr);
	if (!job_iter) {
		perror("sos_attr_iter_new: ");
		exit(1);
	}
	sos_obj_t job_obj;
	char str[80];
	int rc;
	SOS_KEY(start_key);
	sos_key_set(start_key, &start_time, sizeof(start_time));
	for (rc = sos_iter_sup(job_iter, start_key);
	     !rc; rc = sos_iter_next(job_iter)) {
		job_obj = sos_iter_obj(job_iter);
		job_t job = sos_obj_ptr(job_obj);
		if (job->StartTime.secs >= end_time.secs) {
			sos_obj_put(job_obj);
			break;
		}
		sos_key_t key = sos_iter_key(job_iter);
		printf("%-12s ", sos_obj_attr_by_name_to_str(job_obj, "Id", str, sizeof(str)));
		printf("%22s ", sos_obj_attr_by_name_to_str(job_obj, "StartTime", str, sizeof(str)));
		printf("%22s ", sos_obj_attr_by_name_to_str(job_obj, "EndTime", str, sizeof(str)));
		printf("%7s ", sos_obj_attr_by_name_to_str(job_obj, "JobName", str, sizeof(str)));
		printf("%s\n", sos_obj_attr_by_name_to_str(job_obj, "UserName", str, sizeof(str)));
		sos_obj_put(job_obj);
		sos_key_put(key);
	}
	sos_iter_free(job_iter);
}

const char *attr_as_str(sos_obj_t obj, const char *attr_name)
{
	static char str[1024];
	return sos_obj_attr_by_name_to_str(obj, attr_name, str, sizeof(str));
}

void show_by_comp(sos_t sos)
{
	sos_index_t comptime_idx = sos_index_open(sos, "CompTime");
	sos_iter_t job_iter = sos_index_iter_new(comptime_idx);
	if (!job_iter) {
		perror("sos_index_iter_new: ");
		exit(1);
	}
	sos_obj_t job_obj;
	uint32_t comp_id, last_comp_id;
	last_comp_id = -1;
	int rc;
	printf("%-12s %-12s %-22s %-22s %-12s %-s\n",
	       "Comp Id", "Job Id", "Start Time", "End Time", "Username", "Job Name");
	printf("------------ ------------ ---------------------- ---------------------- ------------ --------\n");
	if (comp_str) {
		SOS_KEY(comp_key);
		comp_id = strtoul(comp_str, NULL, 0);
		struct Kvs kv = {
			.primary = comp_id,
			.secondary = 0
		};
		sos_key_set(comp_key, &kv, sizeof(kv));
		rc = sos_iter_sup(job_iter, comp_key);
	} else {
		rc = sos_iter_begin(job_iter);
	}
	for (; !rc; rc = sos_iter_next(job_iter)) {
		job_obj = sos_iter_obj(job_iter);
		sos_key_t key = sos_iter_key(job_iter);
		struct Kvs *kv = (struct Kvs *)sos_key_value(key);
		if (comp_str) {
			if (comp_id != kv->primary) {
				sos_obj_put(job_obj);
				sos_key_put(key);
				break;
			}
		} else
			comp_id = kv->primary;
		if (comp_id != last_comp_id) {
			printf("%12d ", comp_id);
			last_comp_id = comp_id;
		} else {
			printf("             ");
		}
		printf("%12s ", attr_as_str(job_obj, "Id"));
		printf("%22s ", attr_as_str(job_obj, "StartTime"));
		printf("%22s ", attr_as_str(job_obj, "EndTime"));
		printf("%-12s ", attr_as_str(job_obj, "UserName"));
		printf("%s\n", attr_as_str(job_obj, "JobName"));
		sos_obj_put(job_obj);
		sos_key_put(key);
	}
	sos_index_close(comptime_idx, SOS_COMMIT_ASYNC);
}

int col_widths[] = {
	[SOS_TYPE_INT32] = 10,
	[SOS_TYPE_INT64] = 18,
	[SOS_TYPE_UINT32] = 10,
	[SOS_TYPE_UINT64] = 18,
	[SOS_TYPE_FLOAT] = 12,
	[SOS_TYPE_DOUBLE] = 24,
	[SOS_TYPE_LONG_DOUBLE] = 48,
	[SOS_TYPE_TIMESTAMP] = 32,
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
void show_samples(sos_t sos, uint32_t job_id)
{
	sos_attr_t attr;
	size_t attr_count, attr_id;
	struct col_s *col;
	sos_schema_t sample_schema;
	sos_attr_t jobtime_attr;
	sos_iter_t sample_iter;
	struct Kvs kvs;
	int rc;
	SOS_KEY(sample_key);

	sample_schema = sos_schema_by_name(sos, "Sample");
	if (!sample_schema) {
		perror("sos_schema_by_name: ");
		exit(2);
	}
	jobtime_attr = sos_schema_attr_by_name(sample_schema, "JobTime");
	if (!jobtime_attr) {
		perror("sos_schema_attr_by_name: ");
		exit(2);
	}
	sample_iter = sos_attr_iter_new(jobtime_attr);
	if (!sample_iter) {
		perror("sos_attr_iter_new :");
		exit(2);
	}

	/* Create the col_list from the schema if the user didn't specify one */
	if (TAILQ_EMPTY(&col_list)) {
		attr_count = sos_schema_attr_count(sample_schema);
		for (attr_id = 0; attr_id < attr_count; attr_id++) {
			attr = sos_schema_attr_by_id(sample_schema, attr_id);
			if (add_column(sos_attr_name(attr)))
				return;
		}
	}
	/* Query the schema for each attribute's id, and compute width */
	TAILQ_FOREACH(col, &col_list, entry) {
		attr = sos_schema_attr_by_name(sample_schema, col->name);
		if (!attr) {
			printf("The attribute %s from the view is not "
			       "in the schema.\n", col->name);
			return;
		}
		col->id = sos_attr_id(attr);
		if (!col->width)
			col->width = col_widths[sos_attr_type(attr)];
	}

	kvs.primary = job_id;
	kvs.secondary = 0;
	sos_key_set(sample_key, &kvs, sizeof(kvs));
	for (rc = sos_iter_sup(sample_iter, sample_key);
	     rc == 0; rc = sos_iter_next(sample_iter)) {
		sos_obj_t obj = sos_iter_obj(sample_iter);
		struct Sample *sample = sos_obj_ptr(obj);
		if (sample->job_time.primary != job_id)
			break;
		table_row(stdout, sample_schema, obj);
		sos_obj_put(obj);
	}

}

void show_sample_by_job(sos_t sos)
{
	sos_schema_t job_schema;
	sos_attr_t id_attr;
	sos_iter_t job_iter;
	sos_obj_t job_obj;
	struct Job *job;
	uint32_t job_id = 0;
	int rc;

	job_schema = sos_schema_by_name(sos, "Job");
	if (!job_schema) {
		perror("sos_schema_by_name: ");
		exit(2);
	}
	id_attr = sos_schema_attr_by_name(job_schema, "Id");
	if (!id_attr) {
		perror("sos_schema_attr_by_name: ");
		exit(2);
	}
	job_iter = sos_attr_iter_new(id_attr);
	if (!job_iter) {
		perror("sos_attr_iter_new: ");
		exit(1);
	}

	printf("%-12s %-22s %-22s %-12s %-s\n",
	       "Job Id", "Start Time", "End Time", "Username", "Job Name");
	printf("------------ ---------------------- "
	       "---------------------- ------------ --------\n");
	if (!job_str)
		rc = sos_iter_begin(job_iter);
	else {
		job_id = strtoul(job_str, NULL, 0);
		SOS_KEY(job_key);
		sos_key_set(job_key, &job_id, sizeof(job_id));
		rc = sos_iter_sup(job_iter, job_key);
	}
	for (; !rc; rc = sos_iter_next(job_iter)) {
		job_obj = sos_iter_obj(job_iter);
		job = sos_obj_ptr(job_obj);
		if (job_id && job_id != job->id) {
			sos_obj_put(job_obj);
			break;
		}
		printf("%12d ", job->id);
		printf("%22s ", attr_as_str(job_obj, "StartTime"));
		printf("%22s ", attr_as_str(job_obj, "EndTime"));
		printf("%-12s ", attr_as_str(job_obj, "UserName"));
		printf("%s\n", attr_as_str(job_obj, "JobName"));
		if (verbose)
			show_samples(sos, job->id);
		sos_obj_put(job_obj);
	}
	sos_iter_free(job_iter);
}

void show_comp_by_job(sos_t sos)
{
	sos_index_t jobcomp_idx = sos_index_open(sos, "JobComp");
	sos_iter_t job_iter = sos_index_iter_new(jobcomp_idx);
	if (!job_iter) {
		perror("sos_index_iter_new: ");
		exit(1);
	}
	sos_obj_t job_obj;
	uint32_t job_id, last_job_id;
	last_job_id = -1;
	int rc;
	printf("%-12s %-22s %-22s %-12s %-s\n",
	       "Job Id", "Start Time", "End Time", "Username", "Job Name");
	printf("------------ ---------------------- ---------------------- ------------ --------\n");
	int comp_count = 1;
	for (rc = sos_iter_begin(job_iter); !rc; rc = sos_iter_next(job_iter)) {
		job_obj = sos_iter_obj(job_iter);
		sos_key_t key = sos_iter_key(job_iter);
		struct Kvs *kv = (struct Kvs *)sos_key_value(key);
		job_id = kv->primary;
		if (job_id != last_job_id) {
			printf("\n%12d ", job_id);
			last_job_id = job_id;
			printf("%22s ", attr_as_str(job_obj, "StartTime"));
			printf("%22s ", attr_as_str(job_obj, "EndTime"));
			printf("%-12s ", attr_as_str(job_obj, "UserName"));
			printf("%s\n            %8d ",
			       attr_as_str(job_obj, "JobName"), kv->secondary);
			comp_count = 1;
		} else {
			if (!(comp_count % 8))
				printf("\n            ");
			printf("%8d ", kv->secondary);
			comp_count++;
		}
		sos_obj_put(job_obj);
		sos_key_put(key);
	}
	sos_index_close(jobcomp_idx, SOS_COMMIT_ASYNC);
}

void usage(int argc, char *argv[])
{
	printf("%s: Query Jobs and Job Sample data in a Container:\n"
	       "    -C <container>  The Container name\n"
	       "    -j <job_id>     An optional Job Id\n"
	       "    -a <start-time> Show jobs starting on or after start-time\n"
	       "    -b <start-time> Show jobs starting before start-time\n"
	       "    -u <user-id>    Show only jobs for this user-id\n"
	       "    -c              Order results by component\n"
	       "    -t <bin-width>  Order results by time; bin-width specifies the\n"
	       "                    width of the time window to use for grouping.\n"
	       "                    The default is 1 second.\n"
	       "    -d              Treat the data as a cumulative value, process\n"
	       "                    differences\n"
	       "    -v              Dump all sample data for the Job\n"
	       "    -V              Add this metric to the set of processed data\n",
	       argv[0]);
	exit(1);
}


#define TIME_FMT "%Y/%m/%d %H:%M:%S"

void parse_time_data(const char *str, struct sos_timestamp_s *ts)
{
	char *s;
	struct tm tm;
	memset(&tm, 0, sizeof(tm));
	s = strptime(str, TIME_FMT, &tm);
	if (!s) {
		printf("Invalid Time format '%s', correct format is %s\n",
		       str, TIME_FMT);
		exit(1);
	}
	ts->secs = mktime(&tm);
	ts->usecs = 0;
}

int main(int argc, char *argv[])
{
	sos_t sos;
	char *path = NULL;
	int o;
	int by_time = 0;
	int by_job = 0;
	int comp_by_job = 0;
	int by_comp = 0;

	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 't':
			by_time = 1;
			verbose = 1;
			break;
		case 'c':
			by_comp = 1;
			verbose = 1;
			break;
		case 'D':
			comp_str = strdup(optarg);
			break;
		case 'X':
			comp_by_job = 1;
			break;
		case 'j':
			by_job = 1;
			break;
		case 'J':
			job_str = strdup(optarg);
			break;
		case 'a':
			start_str = strdup(optarg);
			parse_time_data(start_str, &start_time);
			break;
		case 'b':
			end_str = strdup(optarg);
			parse_time_data(end_str, &end_time);
			break;
		case 'C':
			path = strdup(optarg);
			break;
		case 'u':
			uid_str = strdup(optarg);
			break;
		case 'v':
			verbose = 1;
			break;
		case 'V':
			if (add_column(optarg))
				exit(11);
			break;
		default:
			usage(argc, argv);
		}
	}
	if (!path)
		usage(argc, argv);

	sos = sos_container_open(path, SOS_PERM_RW);
	if (!sos) {
		perror("could not open container:");
		return errno;
	}
	if (by_job)
		show_sample_by_job(sos);
	if (comp_by_job)
		show_comp_by_job(sos);
	if (by_time)
		show_by_time(sos);
	if (by_comp)
		show_by_comp(sos);
	return 0;
}
