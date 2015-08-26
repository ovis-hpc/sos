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

struct Job {
	uint32_t id;
	struct sos_timestamp_s start_ts;
	struct sos_timestamp_s end_ts;
	uint64_t job_name_ref;
	uint64_t user_name_ref;
};

struct JobComp {
	uint64_t key;		/* CompId:StartTime */
	struct sos_timestamp_s end_time;
	uint32_t job_id;
};

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

	job_obj = job_new(sos, job_str, start_str, end_str, uid_str, name_str);
	if (!job_obj)
		usage(argc, argv);
	uint32_t job_id = strtoul(job_str, NULL, 0);

	sos_index_t comptime_idx = sos_index_open(sos, "CompTime");
	if (!comptime_idx) {
		rc = sos_index_new(sos, "CompTime", "BXTREE", "UINT64", "ORDER=5");
		if (rc) {
			perror("sos_index_new: ");
			exit(3);
		} else {
			comptime_idx = sos_index_open(sos, "CompTime");
		}
		if (!comptime_idx) {
			perror("sos_index_open: ");
			exit(3);
		}
	}
	sos_index_t jobcomp_idx = sos_index_open(sos, "JobComp");
	if (!jobcomp_idx) {
		rc = sos_index_new(sos, "JobComp", "BXTREE", "UINT64", "ORDER=5");
		if (rc) {
			perror("sos_index_new: ");
			exit(3);
		} else {
			jobcomp_idx = sos_index_open(sos, "JobComp");
		}
		if (!jobcomp_idx) {
			perror("sos_index_open: ");
			exit(3);
		}
	}
	struct Job *job = sos_obj_ptr(job_obj);
	while (NULL != (s = fgets(buf, sizeof(buf), comp_file))) {
		struct kvs {
			uint32_t secondary;
			uint32_t primary;
		} kv;
		uint32_t comp_id = strtoul(s, NULL, 0);
		SOS_KEY(comp_key);

		/* Add the Component:Time key */
		kv.primary = comp_id;
		kv.secondary = job->start_ts.secs;
		sos_key_set(comp_key, &kv, sizeof(kv));
		rc = sos_index_insert(comptime_idx, comp_key, job_obj);
		if (rc) {
			perror("sos_index_insert: ");
			exit(4);
		}
		/* Add the Job:Component key */
		kv.primary = job_id;
		kv.secondary = comp_id;
		sos_key_set(comp_key, &kv, sizeof(kv));
		rc = sos_index_insert(jobcomp_idx, comp_key, job_obj);
		if (rc) {
			perror("sos_index_insert: ");
			exit(4);
		}
	}
	sos_index_close(jobcomp_idx, SOS_COMMIT_SYNC);
	sos_index_close(comptime_idx, SOS_COMMIT_SYNC);
	return 0;
 fmt_err:
	printf("The component file has an invalid format\n");
	return 1;
}
