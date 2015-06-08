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
#include <inttypes.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <yaml.h>
#include <assert.h>
#include <sos/sos.h>
#include <ods/ods_atomic.h>
#include <ods/rbt.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <sos/job.h>
#include "sos_yaml.h"

const char *short_options = "C:j:a:b:t:u:V:cdv";

struct option long_options[] = {
	{"container",   required_argument,  0,  'C'},
	{"job",         required_argument,  0,  'j'},
	{"uid",         required_argument,  0,  'u'},
	{"after",       required_argument,  0,  'a'},
	{"before",      required_argument,  0,  'b'},
	{"by_component",no_argument,        0,  'c'},
	{"by_time",     no_argument,        0,  't'},
	{"diff",        no_argument,        0,  'd'},
	{"verbose",     no_argument,        0,  'v'},
	{"view",        required_argument,  0,  'V'},
	{"bin-width",   required_argument,  0,  'w'},
	{0,             0,                  0,    0}
};

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

void print_by_component(sos_t sos,
			sos_schema_t job_schema, sos_schema_t comp_schema, sos_schema_t sample_schema,
		   sos_obj_t job_obj,
		  sos_attr_t comp_head_attr, sos_attr_t sample_head_attr,
		  sos_attr_t next_comp_attr, sos_attr_t next_sample_attr)
{
	char str[80];

	printf("%-12s ", sos_obj_attr_by_name_to_str(job_obj, "JobId", str, sizeof(str)));
	printf("%22s ", sos_obj_attr_by_name_to_str(job_obj, "StartTime", str, sizeof(str)));
	printf("%22s ", sos_obj_attr_by_name_to_str(job_obj, "EndTime", str, sizeof(str)));
	printf("%7s ", sos_obj_attr_by_name_to_str(job_obj, "UID", str, sizeof(str)));
	printf("%s\n", sos_obj_attr_by_name_to_str(job_obj, "Name", str, sizeof(str)));

	sos_value_t comp_head = sos_value(job_obj, comp_head_attr);
	sos_obj_t comp_obj = sos_obj_from_value(sos, comp_head);
	while (comp_obj) {
		sos_value_t sample_head = sos_value(comp_obj, sample_head_attr);
		sos_obj_t sample_obj = sos_obj_from_value(sos, sample_head);
		printf("    %-12s\n", sos_obj_attr_by_name_to_str(comp_obj, "CompId",
								  str, sizeof(str)));
		while (sample_obj) {
			sos_value_t next = sos_value(sample_obj, next_sample_attr);
			printf("             %22s ",
			       sos_obj_attr_by_name_to_str(sample_obj,
							   "Time",
							   str,
							   sizeof(str)));
			printf("%12s ",
			       sos_obj_attr_by_name_to_str(sample_obj,
							   "CompId",
							   str,
							   sizeof(str)));
			printf("\n");
			sample_obj = sos_obj_from_value(sos, next);
			sos_value_put(next);
		}
		sos_value_t next_comp = sos_value(comp_obj, next_comp_attr);
		sos_obj_put(comp_obj);
		comp_obj = sos_obj_from_value(sos, next_comp);
		sos_value_put(next_comp);
	}
	sos_value_put(comp_head);
	sos_obj_put(job_obj);
}

struct component {
	uint64_t comp_id;
	sos_obj_t comp_obj;
	sos_obj_t sample_obj;
	sos_value_t sample_head;
};

struct metric {
	sos_attr_t attr;
	double i;
	double comp_i;
	double time_i;
	double max_xi;
	double min_xi;
	double comp_mean_xi;
	double time_mean_xi;
	double sum_xi;
	double sum_xi_sq;
};

void print_job(sos_obj_t job_obj, FILE *outp, job_metric_vector_t mvec)
{
	char str[256];
	printf("%8s ", sos_obj_attr_by_name_to_str(job_obj, "JobId", str, sizeof(str)));
	printf("%8s ", sos_obj_attr_by_name_to_str(job_obj, "JobSize", str, sizeof(str)));
	printf("%8s ", sos_obj_attr_by_name_to_str(job_obj, "UserName", str, sizeof(str)));
	printf("%-18s ", sos_obj_attr_by_name_to_str(job_obj, "StartTime", str, sizeof(str)));
	printf("%s ", sos_obj_attr_by_name_to_str(job_obj, "JobName", str, sizeof(str)));
	printf("\n");
	if (mvec) {
		int i;
		for (i = 0; i < mvec->count; i++) {
			printf("%-12s ", sos_attr_name(mvec->vec[i].attr));
			printf("%-12s ", "Min");
			printf("%-12s ", "Max");
			printf("%-12s ", "STD");
		}
		printf("\n");
		for (i = 0; i < mvec->count; i++)
			printf("------------ ------------ ------------ ------------ ");
		printf("\n");
	}
}


double xi_val(job_metric_t m, int diff)
{
	if (!diff)
		return m->comp_mean_xi;
	return m->diff_xi;
}

void print_mvec(job_metric_vector_t v, job_metric_flags_t flags, FILE *outp)
{
	job_metric_t m;
	char str[80];
	int i;

	for (i = 0; i < v->count; i++) {
		job_metric_t m = &v->vec[i];
		double xi = xi_val(m, flags);
		if (i == 0) {
			double jitter =  xi - floor(xi);
			time_t t = (time_t)floor(xi);
			struct tm *tm = localtime(&t);
			strftime(str, sizeof(str), "%H:%M:%S", tm);
			printf("%8s.%03d ", str, (int)(jitter * 1.0e3));
			printf("%12.6G ", m->min_xi - floor(m->min_xi));
			printf("%12.6G ", m->max_xi - floor(m->max_xi));
		} else {
			printf("%12.4G ", xi);
			printf("%12.4G ", m->min_xi);
			printf("%12.4G ", m->max_xi);
		}
		double std = m->comp_sum_xi / (m->comp_i - 1.0);
		std = sqrt(std);
		printf("%12.4G ", std);
	}
	printf("\n");
}

void print_stats(job_metric_vector_t m, FILE *outp)
{
	char str[80];
	int i;

	for (i = 0; i < m->count; i++)
		printf("------------ ------------ ------------ ------------ ");
	printf("\n");
	for (i = 0; i < m->count; i++) {
		double std;
		double xi = m->vec[i].time_mean_xi;
		printf("%12.4G ", m->vec[i].time_mean_xi);
		printf("%12.4G ", m->vec[i].min_xi);
		printf("%12.4G ", m->vec[i].max_xi);
		if (i == 0) xi = xi - floor(xi);
		std = m->vec[i].time_sum_xi_sq
			- (m->vec[i].time_sum_xi * m->vec[i].time_sum_xi) / (m->vec[i].time_i);
		std /= m->vec[i].time_i - 1.0;
		printf("%12.4G ", sqrt(std));
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	int verbose = 0;
	sos_t sos;
	job_metric_vector_t mvec = NULL;
	job_metric_flags_t flags = JOB_METRIC_VAL;
	char *start_str = NULL;
	char *end_str = NULL;
	char *job_str = NULL;
	char *path = NULL;
	char *uid_str = NULL;
	int rc, o;
	int by_time = 1;
	const char **attrs = NULL;
	size_t buf_count = 0;
	size_t attr_count = 0;
	double bin_width;
	SOS_KEY(job_key);

	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 't':
			by_time = 1;
			verbose = 1;
			bin_width = strtod(optarg, NULL);
			break;
		case 'c':
			by_time = 0;
			verbose = 1;
			break;
		case 'j':
			job_str = strdup(optarg);
			break;
		case 'a':
			start_str = strdup(optarg);
			break;
		case 'b':
			end_str = strdup(optarg);
			break;
		case 'C':
			path = strdup(optarg);
			break;
		case 'u':
			uid_str = strdup(optarg);
			break;
		case 'd':
			flags = JOB_METRIC_CUM;
			break;
		case 'v':
			verbose = 1;
			break;
		case 'V':
			if (attr_count + 1 > buf_count) {
				buf_count += 10;
				attrs = realloc(attrs, buf_count * sizeof(char *));
			}
			attrs[attr_count] = strdup(optarg);
			attr_count += 1;
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
	job_iter_t job_iter = job_iter_new(sos, JOB_ORDER_BY_TIME);
	if (!job_iter) {
		printf("Could not create the Job iterator.\n");
		exit(1);
	}
	if (verbose) {
		job_iter_set_bin_width(job_iter, bin_width);
		if (0 == attr_count) {
			printf("At least one metric name must be "
			       "specified in verbose mode.\n");
			usage(argc, argv);
		}
		mvec = job_iter_mvec_new(job_iter, attr_count, attrs);
		if (!mvec) {
			printf("Invalid attribute name specified.\n");
			usage(argc, argv);
		}
	}
	sos_obj_t job_obj;
	if (job_str) {
		long job_id = strtol(job_str, NULL, 0);
		job_obj = job_iter_find_job_by_id(job_iter, job_id);
	} else
		job_obj = job_iter_begin_job(job_iter);

	for (; job_obj; job_obj = job_iter_next_job(job_iter)) {
		print_job(job_obj, stdout, mvec);
		if (verbose) {
			for (rc = job_iter_begin_sample(job_iter, flags, mvec);
			     !rc; rc = job_iter_next_sample(job_iter)) {
				print_mvec(mvec, flags, stdout);
			}
			print_stats(mvec, stdout);
		}
		sos_obj_put(job_obj);
		if (job_str)
			break;
	}
	job_iter_free(job_iter);
	return 0;
}
