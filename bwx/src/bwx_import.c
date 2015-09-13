/*
 * Copyright (c) 2015 Open Grid Computing, Inc. All rights reserved.
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
#include <ctype.h>
#include <sos/sos.h>
#include <ods/ods_atomic.h>
#include <sos/sos_yaml.h>
#include <bwx/bwx.h>

const char *short_options = "I:M:C:S:T:";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"schema_name",	required_argument,  0,  'S'},
	{"csv",		required_argument,  0,  'I'},
	{"map",         required_argument,  0,  'M'},
	{"threads",	required_argument,  0,  'T'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos -C <container>\n");
	printf("    -C <path>      The path to the container. Required for all options.\n");
	printf("\n");
	printf("    -I <csv_file>  Import a CSV file into the container.\n");
	printf("       -S <schema> The schema for objects.\n");
	printf("       -M <map>    String that maps CSV columns to object attributes.\n");
	printf("\n");
	exit(1);
}

sos_index_t comptime_idx;
sos_index_t jobcomp_idx;

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
		cond_value->data->prim.timestamp_.fine.secs = (int)ts;
		uint32_t usecs = (uint32_t)((double)(ts - (int)ts) * 1.0e6);
		cond_value->data->prim.timestamp_.fine.usecs = usecs;
		rc = 0;
		break;
	default:
		break;
	}
	return rc;
}

int import_done = 0;
void *flush_proc(void *arg)
{
	sos_t sos = arg;
	while (!import_done) {
		sos_container_commit(sos, SOS_COMMIT_SYNC);
	}
	return NULL;
}

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
	char *tok, *next_tok;
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
		for (tok = work->buf; *tok != '\0'; tok = next_tok) {
			if (cols >= col_count) {
				printf("Warning: line contains more columns "
				       "than are in column map.\n\"%s\"",
				       work->buf);
				break;
			}
			int id = col_map[cols];
			if (id < 0) {
				while (*tok != ',' && *tok != '\0')
					tok++;
				if (*tok == ',')
					tok++;
				next_tok = tok;
				cols++;
				continue;
			}
			struct sos_value_s v_;
			sos_value_t v = sos_value_init(&v_, work->obj, attr_map[cols]);
			rc = value_from_str(attr_map[cols], v, tok, &next_tok);
			sos_value_put(v);
			if (rc) {
				printf("Warning: formatting error setting %s = %s.\n",
				       sos_attr_name(attr_map[cols]), tok);
			}
			next_tok++; /* skip ',' */
			cols++;
		}
		job_sample_t sample = sos_obj_ptr(work->obj);
		struct comp_time_key_s the_key;
		SOS_KEY(job_key);
		the_key.comp_id = sample->CompId;
		the_key.secs = sample->Time.secs;
		sos_key_set(job_key, &the_key, sizeof(the_key));
		sos_obj_t job_obj = sos_index_find_inf(comptime_idx, job_key);
		if (job_obj) {
			job_t job = sos_obj_ptr(job_obj);
			sample->JobTime.job_id = job->job_id;
			sample->JobTime.secs = sample->Time.secs;
			sample->CompTime.comp_id = sample->CompId;
			sample->CompTime.secs = sample->Time.secs;
			sos_obj_put(job_obj);
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
		work->obj = sos_obj_new(schema);
		if (!work->obj) {
			printf("Memory allocation failure!\n");
			break;
		}
		inp = fgets(work->buf, sizeof(work->buf), fp);
		if (!inp)
			break;

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
	import_done = 1;
	void *retval;
	printf("queued %d items...joining.\n", items_queued);
	for (i = 0; i < thread_count; i++)
		pthread_cond_signal(&work_queues[i].wait);
	for (i = 0; i < thread_count; i++)
		pthread_join(add_thread[i], &retval);
	printf("Added %d records.\n", records);
	return 0;
}

int main(int argc, char **argv)
{
	char *path = NULL;
	char *col_map = NULL;
	int o, rc;
	sos_t sos;
	char *schema_name = NULL;
	FILE *csv_file;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'C':
			path = strdup(optarg);
			break;
		case 'S':
			schema_name = strdup(optarg);
			break;
		case 'M':
			col_map = strdup(optarg);
			break;
		case 'I':
			csv_file = fopen(optarg, "r");
			if (!csv_file) {
					perror("Error opening CSV file: ");
					exit(9);
			}
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

	int mode = O_RDWR;
	sos = sos_container_open(path, mode);
	if (!sos) {
		printf("Error %d opening the container %s.\n",
		       errno, path);
		exit(1);
	}
	comptime_idx = sos_index_open(sos, "CompTime");
	if (!comptime_idx) {
		perror("sos_index_open: ");
		exit(2);
	}
	jobcomp_idx = sos_index_open(sos, "JobComp");
	if (!jobcomp_idx) {
		perror("sos_index_open: ");
		exit(2);
	}
	rc = import_csv(sos, csv_file, schema_name, col_map);
	return rc;
}
