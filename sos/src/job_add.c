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

const char *short_options = "C:j:s:e:c:n:u:";

struct option long_options[] = {
	{"comp_file",   required_argument,  0,  'c'},
	{"container",   required_argument,  0,  'C'},
	{"job_id",      required_argument,  0,  'j'},
	{"start",       required_argument,  0,  's'},
	{"end",         required_argument,  0,  'e'},
	{"user_name",   required_argument,  0,  'u'},
	{"job_name",    required_argument,  0,  'n'},
	{0,             0,                  0,    0}
};

void usage(int argc, char *argv[])
{
	printf("usage: %s -C <container> -j <job_id> -s <start_time> "
	       "-e <end_time> -c <comp_file> -n <name> -u <uname>.\n", argv[0]);
	printf("       <container>  The path to the container\n"
	       "       <job_id>     A unique Job Id (64b integer)\n"
	       "       <start_time> The time the job started specified as follows:\n"
	       "                    \"YYYY/MM/DD HH:MM:SS\"\n"
	       "       <end_time>   The time the job ended formatted like <start_time>\n"
	       "       <comp_file>  A text file containing a newline separated list of\n"
	       "                    Component Id (64b integers)\n"
	       "       <name>       A text name for the Job\n"
	       "       <uname>      The user name\n");
	exit(1);
}

struct rbt comp_tree;
char fbuf[80];
struct component {
	uint64_t id;
	sos_obj_t comp_obj;
	sos_obj_t prev_sample;
	struct rbn rbn;
};

int comp_cmp(void *a, void *b)
{
	uint64_t _a = *(uint64_t *)a;
	uint64_t _b = *(uint64_t *)b;
	if (_a < _b)
		return -1;
	if (_a > _b)
		return 1;
	return 0;
}

struct add_arg {
	/* Objects */
	sos_obj_t job_obj;
	sos_obj_t prev_comp;

	/* Schemas */
	sos_schema_t comp_schema;

	/* Job attributes */
	sos_attr_t comp_head_attr;
	sos_attr_t comp_tail_attr;

	/* CompRef attributes */
	sos_attr_t comp_id_attr;
	sos_attr_t next_comp_attr;
	sos_attr_t prev_comp_attr;
};

int add_component(struct rbn *rbn, void *varg, int level)
{
	struct add_arg *arg = varg;
	struct component *comp = container_of(rbn, struct component, rbn);
	sos_obj_t comp_obj = sos_obj_new(arg->comp_schema);
	if (!comp_obj)
		return 1;
	sos_value_t id = sos_value(comp_obj, arg->comp_id_attr);
	id->data->prim.uint64_ = comp->id;
	comp->comp_obj = comp_obj;
	sos_value_put(id);

	if (!arg->prev_comp) {
		/* Initialize the component list for the job */
		arg->prev_comp = comp_obj;
		sos_value_t comp_head = sos_value(arg->job_obj, arg->comp_head_attr);
		sos_value_t comp_tail = sos_value(arg->job_obj, arg->comp_tail_attr);
		comp_head->data->prim.ref_ = sos_obj_ref(comp_obj).ref;
		comp_tail->data->prim.ref_ = sos_obj_ref(comp_obj).ref;
		sos_value_put(comp_head);
		sos_value_put(comp_tail);
		goto out;
	}
	sos_value_t next_comp = sos_value(arg->prev_comp, arg->next_comp_attr);
	sos_value_t prev_comp = sos_value(comp_obj, arg->prev_comp_attr);
	sos_value_t comp_tail = sos_value(arg->job_obj, arg->comp_tail_attr);
	next_comp->data->prim.ref_ = sos_obj_ref(comp_obj).ref;
	prev_comp->data->prim.ref_ = sos_obj_ref(arg->prev_comp).ref;
	comp_tail->data->prim.ref_ = sos_obj_ref(comp_obj).ref;
	arg->prev_comp = comp_obj;
 out:
	return 0;
}
uint64_t job_size = 0;
int build_comp_tree(FILE* comp_file, struct rbt* tree, struct add_arg *arg)
{
	char *s;
	while (s = fgets(fbuf, sizeof(fbuf), comp_file)) {
		struct component *comp;

		comp = calloc(1, sizeof *comp);
		if (!comp)
			return ENOMEM;
		job_size++;
		uint64_t id = strtoul(s, NULL, 0);
		comp->id = id;
		rbn_init(&comp->rbn, &comp->id);
		rbt_ins(&comp_tree, &comp->rbn);
	}
	return rbt_traverse(&comp_tree, add_component, arg);
}

struct component *job_component(uint64_t comp_id)
{
	struct rbn *rbn = rbt_find(&comp_tree, &comp_id);
	if (rbn)
		return container_of(rbn, struct component, rbn);
	return NULL;
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
	attr = sos_schema_attr_by_name(job_schema, "JobId");
	job_id_iter = sos_iter_new(attr);
	if (!job_id_iter) {
		printf("Could not create the Job iterator. The schema may be invalid\n");
		goto err;
	}
	rc = sos_key_from_str(attr, job_key, id);
	if (rc) {
		printf("Error %d setting job key from %s. A JobId is an integer\n", rc, id);
		goto err;
	}
	rc = sos_iter_find(job_id_iter, job_key);
	if (!rc) {
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
	if (job_schema)
		sos_schema_put(job_schema);
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
	sos_schema_t sample_schema;
	sos_schema_t comp_schema;
	sos_attr_t comp_attr;
	sos_iter_t time_iter;
	sos_filter_t time_filt;
	sos_value_t start_value;
	sos_value_t end_value;
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
	if (!path /* || !start_str || !end_str || !job_str || !uid_str || !name_str */)
		usage(argc, argv);

	sos = sos_container_open(path, SOS_PERM_RW);
	if (!sos) {
		perror("could not open container:");
		return errno;
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
	char *s;
	char buf[128];
	s = fgets(buf, sizeof(buf), comp_file);
	job_str = strdup(s);
	s = fgets(buf, sizeof(buf), comp_file);
	uid_str = strdup(s);
	s = strchr(uid_str, '\n');
	if (s)
		*s = '\0';
	s = fgets(buf, sizeof(buf), comp_file);
	start_str = strdup(s);
	s = fgets(buf, sizeof(buf), comp_file);
	end_str = strdup(s);
	s = fgets(buf, sizeof(buf), comp_file);
	name_str = strdup(s);
	s = strchr(name_str, '\n');
	if (s)
		*s = '\0';
	job_obj = job_new(sos, job_str, start_str, end_str, uid_str, name_str);
	if (!job_obj)
		usage(argc, argv);
	uint64_t job_id = strtoul(job_str, NULL, 0);

	/* Start Time */
	sos_attr_t ts_attr = sos_schema_attr_by_name(sample_schema, "Time");
	if (!ts_attr) {
		printf("The 'Time' attribute does not exist in the Sample schema.\n");
		usage(argc, argv);
	}
	start_value = sos_value_new(); assert(start_value);
	sos_value_init(start_value, NULL, ts_attr);
	rc = sos_value_from_str(start_value, start_str, NULL);
	if (rc) {
		printf("Unable to set the start time from the string '%s'.\n", start_str);
		usage(argc, argv);
	}
	/* End Time */
	end_value = sos_value_new(); assert(end_value);
	sos_value_init(end_value, NULL, ts_attr);
	rc = sos_value_from_str(end_value, end_str, NULL);
	if (rc) {
		printf("Unable to set the  value from the string '%s'.\n", end_str);
		usage(argc, argv);
	}
	comp_attr = sos_schema_attr_by_name(sample_schema, "CompId");
	if (!comp_attr) {
		printf("The 'JobId' attribute does not exist in the Sample schema.\n");
		usage(argc, argv);
	}
	time_iter = sos_iter_new(ts_attr);
	if (!time_iter) {
		printf("Could not create the Time iterator.\n");
		usage(argc, argv);
	}
	time_filt = sos_filter_new(time_iter);
	if (!time_filt) {
		printf("Could not create the Time filter.\n");
		usage(argc, argv);
	}
	rc = sos_filter_cond_add(time_filt, ts_attr, SOS_COND_GE, start_value);
	if (rc) {
		printf("The start time specified, '%s', is invalid.\n", start_str);
		usage(argc, argv);
	}
	rc = sos_filter_cond_add(time_filt, ts_attr, SOS_COND_LE, end_value);
	if (rc) {
		printf("The start time specified, '%s', is invalid.\n", end_str);
		usage(argc, argv);
	}
	rbt_init(&comp_tree, comp_cmp);
	struct add_arg arg;
	arg.job_obj = job_obj;
	arg.prev_comp = NULL;
	arg.comp_schema = comp_schema;
	arg.comp_head_attr = sos_schema_attr_by_name(sos_obj_schema(job_obj), "CompHead");
	arg.comp_tail_attr = sos_schema_attr_by_name(sos_obj_schema(job_obj), "CompTail");
	arg.comp_id_attr = sos_schema_attr_by_name(comp_schema, "CompId");
	arg.next_comp_attr = sos_schema_attr_by_name(comp_schema, "NextComp");
	arg.prev_comp_attr = sos_schema_attr_by_name(comp_schema, "PrevComp");
	rc = build_comp_tree(comp_file, &comp_tree, &arg);
	if (rc) {
		printf("Error %d building the component list.\n", rc);
		return rc;
	}
	sos_schema_t job_schema = sos_schema_by_name(sos, "Job");
	sos_attr_t job_size_attr = sos_schema_attr_by_name(job_schema, "JobSize");
	sos_value_t job_size_val = sos_value(job_obj, job_size_attr);
	job_size_val->data->prim.uint64_ = job_size;
	sos_value_put(job_size_val);
	sos_schema_put(job_schema);

	sos_attr_t sample_head_attr = sos_schema_attr_by_name(comp_schema, "SampleHead");
	sos_attr_t sample_tail_attr = sos_schema_attr_by_name(comp_schema, "SampleTail");
	sos_attr_t next_sample_attr = sos_schema_attr_by_name(sample_schema, "NextSample");
	sos_attr_t prev_sample_attr = sos_schema_attr_by_name(sample_schema, "PrevSample");
	sos_attr_t job_id_attr = sos_schema_attr_by_name(sample_schema, "JobId");
	for (sample_obj = sos_filter_begin(time_filt); sample_obj;
	     sample_obj = sos_filter_next(time_filt)) {
		struct component *comp;

		/* Check if the component for this sample is part of this job. */
		comp_value = sos_value(sample_obj, comp_attr);
		comp = job_component(comp_value->data->prim.uint64_);
		if (!comp) {
			sos_value_put(comp_value);
			sos_obj_put(sample_obj);
			continue;
		}
		sos_value_t sample_head = sos_value(comp->comp_obj, sample_head_attr);
		sos_value_t sample_tail = sos_value(comp->comp_obj, sample_tail_attr);
		sos_value_t job_id_val = sos_value(sample_obj, job_id_attr);
		job_id_val->data->prim.uint64_ = job_id;
		sample_tail->data->prim.ref_ = sos_obj_ref(sample_obj).ref;

		if (!comp->prev_sample) {
			sample_head->data->prim.ref_ = sos_obj_ref(sample_obj).ref;
			sos_value_t prev = sos_value(sample_obj, prev_sample_attr);
			prev->data->prim.ref_ = 0; /* Start the list */
			sos_value_put(prev);
		} else {
			sos_value_t next = sos_value(comp->prev_sample, next_sample_attr);
			sos_value_t prev = sos_value(sample_obj, prev_sample_attr);
			next->data->prim.ref_ = sos_obj_ref(sample_obj).ref;
			prev->data->prim.ref_ = sos_obj_ref(comp->prev_sample).ref;
			next = sos_value(sample_obj, next_sample_attr);
			next->data->prim.ref_ = 0; /* Terminate the list */
			sos_obj_put(comp->prev_sample);
			sos_value_put(next);
			sos_value_put(prev);
		}
		sos_value_put(sample_head);
		sos_value_put(sample_tail);
		sos_value_put(job_id_val);
		comp->prev_sample = sample_obj;
	}

	return 0;
}
