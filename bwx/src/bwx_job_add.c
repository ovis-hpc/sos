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
#include <string.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <assert.h>
#include <sos/sos.h>
#include <ods/ods_atomic.h>
#include <ods/rbt.h>
#include <bwx/bwx.h>

const char *short_options = "C:j:s:e:c:n:u:";

struct option long_options[] = {
	{"job-file",    required_argument,  0,  'j'},
	{"container",   required_argument,  0,  'C'},
	{0,             0,                  0,    0}
};

void usage(int argc, char *argv[])
{
	printf("usage: %s -C <container> <job_file>\n", argv[0]);
	printf("       <container>  The path to the container\n");
	printf("       <job_file>   File containing job information\n");
	exit(1);
}

sos_obj_t job_new(sos_t sos, char *id, char *start, char *end, char *uid, char *name)
{
	sos_schema_t job_schema = NULL;
	sos_attr_t attr = NULL;
	sos_iter_t job_id_iter = NULL;
	sos_obj_t job_obj = NULL;
	int rc;
	SOS_KEY(job_key);

	job_schema = sos_schema_by_name(sos, "Job");
	if (!job_schema) {
		printf("Could not find the Job schema in the container.\n");
		goto err;
	}
	/* Check to see if the job already exists */
	attr = sos_schema_attr_by_name(job_schema, "Id");
	job_id_iter = sos_attr_iter_new(attr);
	if (!job_id_iter) {
		printf("Could not create the Job iterator. The schema may be invalid\n");
		goto err;
	}
	rc = sos_attr_key_from_str(attr, job_key, id);
	if (rc) {
		printf("Error %d setting job key from %s. A JobId is an integer\n", rc, id);
		goto err;
	}
	rc = sos_iter_find(job_id_iter, job_key);
	if (!rc) {
		job_obj = sos_iter_obj(job_id_iter);
		return job_obj;
		printf("The specified job (%s) already exists. "
		       "Please use a different Job Id.\n", id);
		goto err;
	}
	/* Allocate a Job object and add it to the container */
	job_obj = sos_obj_new(job_schema);
	if (!job_obj) {
		printf("A Job object could not be allocated. Is the container full?\n");
		goto err;
	}
	/* Set the JobId */
	rc = sos_obj_attr_from_str(job_obj, attr, id, NULL);
	if (rc) {
		printf("Error %d setting Job Id with the string %s.\n", rc, id);
		goto err;
	}
	/* Set the StartTime */
	rc = sos_obj_attr_by_name_from_str(job_obj, "StartTime", start, NULL);
	if (rc) {
		printf("Error %d setting StartTime from the string %s.\n", rc, start);
		goto err;
	}
	/* Set the EndTime */
	rc = sos_obj_attr_by_name_from_str(job_obj, "EndTime", end, NULL);
	if (rc) {
		printf("Error %d setting EndTime from the string %s.\n", rc, end);
		goto err;
	}
	/* Set the UID */
	rc = sos_obj_attr_by_name_from_str(job_obj, "UserName", uid, NULL);
	if (rc) {
		printf("Error %d setting UID from the string %s.\n", rc, uid);
		goto err;
	}
	/* Set the Name */
	rc = sos_obj_attr_by_name_from_str(job_obj, "JobName", name, NULL);
	if (rc) {
		printf("Error %d setting Name from the string %s.\n", rc, name);
		goto err;
	}
	/* Add the new job to the index */
	rc = sos_obj_index(job_obj);
	if (rc) {
		printf("Error %d adding the new job to the index.\n", rc);
		goto err;
	}
	return job_obj;
 err:
	if (job_id_iter)
		sos_iter_free(job_id_iter);
	if (job_obj) {
		sos_obj_delete(job_obj);
		sos_obj_put(job_obj);
	}
	return NULL;
}

int main(int argc, char *argv[])
{
	sos_t sos;
	sos_obj_t job_obj;
	char *start_str = NULL;
	char *end_str = NULL;
	char *job_str = NULL;
	char *path = NULL;
	char *name_str = NULL;
	char *uid_str = NULL;
	FILE *comp_file = stdin;
	int rc, o;

	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'C':
			path = strdup(optarg);
			break;
		case 'j':
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

	char *s;
	char buf[128];
	s = fgets(buf, sizeof(buf), comp_file);
	if (!s)
		goto fmt_err;
	job_str = strdup(s);
	s = fgets(buf, sizeof(buf), comp_file);
	if (!s)
		goto fmt_err;
	uid_str = strdup(s);
	s = strchr(uid_str, '\n');
	if (s)
		*s = '\0';
	s = fgets(buf, sizeof(buf), comp_file);
	if (!s)
		goto fmt_err;
	start_str = strdup(s);
	s = fgets(buf, sizeof(buf), comp_file);
	if (!s)
		goto fmt_err;
	end_str = strdup(s);
	s = fgets(buf, sizeof(buf), comp_file);
	if (!s)
		goto fmt_err;
	name_str = strdup(s);
	s = strchr(name_str, '\n');
	if (s)
		*s = '\0';

	sos = sos_container_open(path, SOS_PERM_RW);
	if (!sos) {
		perror("sos_container_open: ");
		return errno;
	}

	sos_schema_t sample_s = sos_schema_by_name(sos, "Sample");
	if (!sample_s) {
		printf("The Sample schema is missing.\n");
		return 2;
	}
	sos_attr_t comp_time_attr = sos_schema_attr_by_name(sample_s, "comp_time");
	if (!comp_time_attr) {
		printf("The comp_time attribute is missing from the Sample schema.\n");
		return 2;
	}
	sos_iter_t comp_time_iter = sos_attr_iter_new(comp_time_attr);
	if (!comp_time_iter) {
		perror("sos_attr_iter_new");
		return 2;
	}
	job_obj = job_new(sos, job_str, start_str, end_str, uid_str, name_str);
	if (!job_obj)
		usage(argc, argv);

	job_t job = sos_obj_ptr(job_obj);
	while (NULL != (s = fgets(buf, sizeof(buf), comp_file))) {
		sos_obj_t obj;
		job_sample_t sample;
		uint32_t comp_id = strtoul(s, NULL, 0);
		SOS_KEY(comp_key);
		struct comp_time_key_s ct_val;

		ct_val.comp_id = comp_id;
		ct_val.secs = job->StartTime.secs;
		sos_key_set(comp_key, &ct_val, sizeof(ct_val));
		for (rc = sos_iter_sup(comp_time_iter, comp_key);
		     !rc; rc = sos_iter_next(comp_time_iter)) {

			obj = sos_iter_obj(comp_time_iter);
			sample = sos_obj_ptr(obj);

			if (sample->component_id == comp_id &&
			    sample->timestamp.secs <= job->EndTime.secs) {

				sample->job_id = job->job_id;
				sample->job_time.secs = sample->timestamp.secs;
				sample->job_time.job_id = job->job_id;
				/* Remove the object from the current indices */
				sos_obj_remove(obj);
				/* Now re-index the object with the new values */
				sos_obj_index(obj);
				sos_obj_put(obj);
			} else {
				sos_obj_put(obj);
				break;
			}
		}
	}
	return 0;
 fmt_err:
	printf("The component file has an invalid format\n");
	return 1;
}
