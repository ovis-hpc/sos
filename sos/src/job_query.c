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
#include "sos_yaml.h"

const char *short_options = "C:j:s:e:u:";

struct option long_options[] = {
	{"container",   required_argument,  0,  'C'},
	{"job",         required_argument,  0,  'j'},
	{"uid",         required_argument,  0,  'u'},
	{"start",       required_argument,  0,  's'},
	{"end",         required_argument,  0,  's'},
	{0,             0,                  0,    0}
};

void usage(int argc, char *argv[])
{
	printf("%s: -j <job_id> -s <start_time> "
	       "-e <end_time> -u <uid>\n", argv[0]);
	exit(1);
}

int main(int argc, char *argv[])
{
	sos_t sos;
	char *schema_name = "Sample";
	char *index_name = "JobId";
	sos_schema_t sample_schema;
	sos_schema_t comp_schema;
	sos_attr_t comp_attr;
	sos_iter_t time_iter;
	sos_attr_t time_attr;
	sos_filter_t time_filt;
	sos_value_t start_value;
	sos_value_t end_value;
	sos_value_t job_value;
	sos_value_t comp_value;
	sos_obj_t job_obj, sample_obj, comp_obj;
	char *start_str = NULL;
	char *end_str = NULL;
	char *job_str = NULL;
	char *path = NULL;
	char *name_str = NULL;
	char *uid_str = NULL;
	FILE *comp_file = stdin;
	int rc, o;
	uint64_t comp_id;
	SOS_KEY(job_key);

	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'j':
			job_str = strdup(optarg);
			break;
		case 's':
			start_str = strdup(optarg);
			break;
		case 'e':
			end_str = strdup(optarg);
			break;
		case 'C':
			path = strdup(optarg);
			break;
		case 'u':
			uid_str = strdup(optarg);
			break;
		case 'n':
			name_str = strdup(optarg);
			break;
		case 'c':
			comp_file = fopen(optarg, "r");
			if (!comp_file) {
				printf("Could not open the specified component file.\n");
				usage(argc, argv);
			}
			break;
		default:
			usage(argc, argv);
		}
	}
	if (!path)
		usage(argc, argv);

	rc = sos_container_open(path, SOS_PERM_RW, &sos);
	if (rc) {
		perror("could not open container:");
		return rc;
	}
	comp_schema = sos_schema_by_name(sos, "CompRef");
	if (!comp_schema) {
		printf("The 'CompRef' schema was not found.\n");
		return ENOENT;
	}
	sample_schema = sos_schema_by_name(sos, "Sample");
	if (!sample_schema) {
		printf("The 'Sample' schema was not found.\n");
		return ENOENT;
	}
	sos_schema_t job_schema = sos_schema_by_name(sos, "Job");
	if (!job_schema) {
		printf("Could not find the Job schema in the container.\n");
		return ENOENT;
	}
	sos_attr_t job_id_attr = sos_schema_attr_by_name(job_schema, "JobId");
	if (!job_id_attr) {
		printf("Could not find the JobId attribute in the schema.\n");
		return EINVAL;
	}
	sos_attr_t job_name_attr = sos_schema_attr_by_name(job_schema, "Name");
	if (!job_name_attr) {
		printf("Could not find the Name attribute in the schema.\n");
		return EINVAL;
	}
	sos_attr_t job_start_attr = sos_schema_attr_by_name(job_schema, "StartTime");
	if (!job_start_attr) {
		printf("Could not find the StartTime attribute in the schema.\n");
		return EINVAL;
	}
	sos_attr_t job_end_attr = sos_schema_attr_by_name(job_schema, "JobId");
	if (!job_end_attr) {
		printf("Could not find the EndTime attribute in the schema.\n");
		return EINVAL;
	}

	/* Start Time */
	time_iter = sos_iter_new(job_start_attr);
	if (!time_iter) {
		printf("Could not create the Time iterator.\n");
		return ENOMEM;
	}
	time_filt = sos_filter_new(time_iter);
	if (!time_filt) {
		printf("Could not create the Time filter.\n");
		return ENOMEM;
	}
	/* Start Time Condition */
	if (start_str) {
		start_value = sos_value_new(); assert(start_value);
		sos_value_init(start_value, NULL, time_attr);
		rc = sos_value_from_str(start_value, start_str, NULL);
		if (rc) {
			printf("Unable to set the start time from the string '%s'.\n",
			       start_str);
			usage(argc, argv);
		}
		rc = sos_filter_cond_add(time_filt, time_attr, SOS_COND_GE, start_value);
		if (rc) {
			printf("The start time specified, '%s', is invalid.\n", start_str);
			usage(argc, argv);
		}
	}
	/* End Time Condition */
	if (end_str) {
		end_value = sos_value_new(); assert(end_value);
		sos_value_init(end_value, NULL, time_attr);
		rc = sos_value_from_str(end_value, end_str, NULL);
		if (rc) {
			printf("Unable to set the  value from the string '%s'.\n", end_str);
			usage(argc, argv);
		}
		rc = sos_filter_cond_add(time_filt, time_attr, SOS_COND_LE, end_value);
		if (rc) {
			printf("The start time specified, '%s', is invalid.\n", end_str);
			usage(argc, argv);
		}
	}

	sos_attr_t comp_head_attr = sos_schema_attr_by_name(job_schema, "CompHead");
	sos_attr_t sample_head_attr = sos_schema_attr_by_name(comp_schema, "SampleHead");
	sos_attr_t next_comp_attr = sos_schema_attr_by_name(comp_schema, "NextComp");
	sos_attr_t next_sample_attr = sos_schema_attr_by_name(sample_schema, "NextSample");

	printf("%12s %22s %22s %7s %s\n", "JobId", "Start Time", "End Time", "User Id", "Name");
	printf("------------ ---------------------- ---------------------- ------- ---------\n");
	for (job_obj = sos_filter_begin(time_filt); job_obj;
	     job_obj = sos_filter_next(time_filt)) {
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
			comp_obj = sos_obj_from_value(sos, next_comp);
			sos_value_put(next_comp);
		}
		sos_obj_put(job_obj);
	}
	return 0;
}
