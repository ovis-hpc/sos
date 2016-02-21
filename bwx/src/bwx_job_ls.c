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


const char *short_options = "C:X:ja:b:tV:J:cv";

struct option long_options[] = {
	{"container",   required_argument,  0,  'C'},
	{"after",       required_argument,  0,  'a'},
	{"before",      required_argument,  0,  'b'},
	{"component",   required_argument,  0,  'D'},
	{"by_job",      no_argument,        0,  'j'},
	{"job",         required_argument,  0,  'J'},
	{"verbose",     no_argument,        0,  'v'},
	{"view",        required_argument,  0,  'V'},
	{0,             0,                  0,    0}
};
static int verbose = 0;
static char *job_str = NULL;
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

const char *fmt_timestamp(sos_obj_t obj, sos_attr_t attr)
{
	static char str[80];
	sos_value_t v = sos_value(obj, attr);
	time_t t = v->data->prim.timestamp_.fine.secs;
	sos_value_put(v);
	struct tm *tm = localtime(&t);
	strftime(str, sizeof(str), "%F %R", tm);
	return str;
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
	[SOS_TYPE_TIMESTAMP] = 16,
	[SOS_TYPE_OBJ] = 8,
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

void table_header(FILE *outp)
{
	struct col_s *col;
	/* Print the header labels */
	fprintf(outp, "    ");
	TAILQ_FOREACH(col, &col_list, entry)
		fprintf(outp, "%-*s ", col->width, col->name);
	fprintf(outp, "\n");

	/* Print the header separators */
	fprintf(outp, "    ");
	TAILQ_FOREACH(col, &col_list, entry) {
		int i;
		for (i = 0; i < col->width; i++)
			fprintf(outp, "-");
		fprintf(outp, " ");
	}
	fprintf(outp, "\n");
}

void table_row(FILE *outp, sos_schema_t schema, sos_obj_t obj)
{
	struct col_s *col;
	sos_attr_t attr;
	static char str[80];
	fprintf(outp, "    ");
	TAILQ_FOREACH(col, &col_list, entry) {
		attr = sos_schema_attr_by_id(schema, col->id);
		if (sos_attr_type(attr) != SOS_TYPE_TIMESTAMP)
			fprintf(outp, "%*s ", col->width,
				sos_obj_attr_to_str(obj, attr, str, 80));
		else
			fprintf(outp, "%-*s ", col->width, fmt_timestamp(obj, attr));
	}
	fprintf(outp, "\n");
}

void show_samples(sos_t sos, uint32_t job_id)
{
	sos_attr_t attr;
	size_t attr_count, attr_id;
	struct col_s *col;
	sos_schema_t sample_schema;
	sos_attr_t jobtime_attr;
	sos_iter_t sample_iter;
	struct job_time_key_s job_time_key;
	int rc;
	SOS_KEY(sample_key);

	sample_schema = sos_schema_by_name(sos, "Sample");
	if (!sample_schema) {
		perror("sos_schema_by_name: ");
		exit(2);
	}
	jobtime_attr = sos_schema_attr_by_name(sample_schema, "job_time");
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
	job_time_key.job_id = job_id;
	job_time_key.secs = 0;
	sos_key_set(sample_key, &job_time_key, sizeof(job_time_key));
	table_header(stdout);
	for (rc = sos_iter_sup(sample_iter, sample_key);
	     rc == 0; rc = sos_iter_next(sample_iter)) {
		sos_obj_t obj = sos_iter_obj(sample_iter);
		job_sample_t sample = (job_sample_t)sos_obj_ptr(obj);
		if (sample->job_time.job_id != job_id)
			break;
		table_row(stdout, sample_schema, obj);
		sos_obj_put(obj);
	}
}

void show_sample_by_job(sos_t sos)
{
	sos_schema_t job_schema;
	sos_attr_t StartTime_attr, EndTime_attr, UserName_attr, JobName_attr;
	sos_attr_t id_attr;
	sos_iter_t job_iter;
	sos_obj_t job_obj;
	job_t job;
	uint32_t job_id = 0;
	int rc;
	static char str[80];

	job_schema = sos_schema_by_name(sos, "Job");
	if (!job_schema) {
		perror("sos_schema_by_name: ");
		exit(2);
	}
	StartTime_attr = sos_schema_attr_by_name(job_schema, "StartTime");
	EndTime_attr = sos_schema_attr_by_name(job_schema, "EndTime");
	UserName_attr = sos_schema_attr_by_name(job_schema, "UserName");
	JobName_attr = sos_schema_attr_by_name(job_schema, "JobName");
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

	printf("%-12s %-16s %-16s %-16s %s\n",
	       "Job Id", "Start Time", "End Time", "Username", "Job Name");
	printf("------------ ---------------- "
	       "---------------- ---------------- --------------\n");
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
		job = (job_t)sos_obj_ptr(job_obj);
		if (job_id && job_id != job->job_id) {
			sos_obj_put(job_obj);
			break;
		}
		printf("%12d ", job->job_id);
		printf("%-16s ", fmt_timestamp(job_obj, StartTime_attr));
		printf("%-16s ", fmt_timestamp(job_obj, EndTime_attr));
		printf("%-16s ", sos_obj_attr_to_str(job_obj, UserName_attr,
						     str, sizeof(str)));
		printf("%s\n", sos_obj_attr_to_str(job_obj, JobName_attr,
						   str, sizeof(str)));
		if (verbose)
			show_samples(sos, job->job_id);
		sos_obj_put(job_obj);
	}
	sos_iter_free(job_iter);
}

void show_by_time(sos_t sos)
{
	sos_schema_t job_schema = sos_schema_by_name(sos, "Job");
	sos_attr_t Id_attr = sos_schema_attr_by_name(job_schema, "Id");
	sos_attr_t StartTime_attr = sos_schema_attr_by_name(job_schema, "StartTime");
	sos_attr_t EndTime_attr = sos_schema_attr_by_name(job_schema, "EndTime");
	sos_attr_t UserName_attr = sos_schema_attr_by_name(job_schema, "UserName");
	sos_attr_t JobName_attr = sos_schema_attr_by_name(job_schema, "JobName");
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

	printf("%-12s %-16s %-16s %-16s %s\n",
	       "Job Id", "Start Time", "End Time", "User Name", "Job Name");
	printf("------------ ---------------- ---------------- ---------------- ----------------\n");
	for (rc = sos_iter_sup(job_iter, start_key);
	     !rc; rc = sos_iter_next(job_iter)) {
		job_obj = sos_iter_obj(job_iter);
		job_t job = sos_obj_ptr(job_obj);
		if (job->StartTime.secs >= end_time.secs) {
			sos_obj_put(job_obj);
			break;
		}
		sos_key_t key = sos_iter_key(job_iter);
		printf("%-12s ", sos_obj_attr_to_str(job_obj, Id_attr,
							     str, sizeof(str)));
		printf("%-16s ", fmt_timestamp(job_obj, StartTime_attr));
		printf("%-16s ", fmt_timestamp(job_obj, EndTime_attr));
		printf("%-16s ", sos_obj_attr_to_str(job_obj, UserName_attr,
						     str, sizeof(str)));
		printf("%s\n", sos_obj_attr_to_str(job_obj, JobName_attr,
						   str, sizeof(str)));
		if (verbose)
			show_samples(sos, job->job_id);
		sos_obj_put(job_obj);
		sos_key_put(key);
	}
	sos_iter_free(job_iter);
}

void usage(int argc, char *argv[])
{
	printf("%s: Query Jobs and Job Sample data in a Container:\n"
	       "    -C <container>  The Container name\n"
	       "    -j              order the output by Job Id\n"
	       "    -J <job_id>     Show results for only this Job Id\n"
	       "    -a <start-time> Show jobs starting on or after start-time\n"
	       "    -b <start-time> Show jobs starting before start-time\n"
	       "    -t              Order results by time [default].\n"
	       "    -v              Dump all sample data for the Job\n"
	       "    -V              Add this metric to the sample output\n",
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
	int by_time = 1;
	int by_job = 0;

	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 't':
			by_job = 0;
			by_time = 1;
			break;
		case 'j':
			by_job = 1;
			by_time = 0;
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
	if (by_time)
		show_by_time(sos);
	return 0;
}
