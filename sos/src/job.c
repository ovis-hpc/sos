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
#include <pthread.h>
#include <errno.h>
#include <inttypes.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <sos/sos.h>
#include <sos/job.h>

struct work;
typedef void (*work_fn_t)(struct work *work);
struct work {
	work_fn_t work_fn;
	job_metric_vector_t mvec;
	job_iter_t job_iter;
	int start, end; /* comp_no = start, comp_no < end; */
	ods_atomic_t *sem;
	pthread_cond_t *cv;
	pthread_mutex_t *cv_lock;
	LIST_ENTRY(work) free_link;
	TAILQ_ENTRY(work) link;
};

struct worker {
	pthread_t thread;
	pthread_mutex_t work_q_lock;
	pthread_cond_t work_q_cv;
	TAILQ_HEAD(work_list, work) work_q;
};

struct component {
	uint64_t comp_id;
	sos_obj_t comp_obj;
	sos_obj_t sample_obj;
	sos_value_t sample_head;
};

typedef struct job_s {
	/* TODO? */
} *job_t;

struct job_iter_s {
	sos_t sos;
	job_metric_flags_t flags;
	job_iter_order_t order_by;
	double bin_width;

	uint64_t job_id;
	sos_obj_t job_obj;
	sos_iter_t id_iter;
	sos_iter_t time_iter;
	sos_attr_t job__id_attr;
	sos_attr_t job__time_attr;
	sos_attr_t job__size_attr;
	sos_attr_t job__comp_head_attr;
	sos_attr_t job__comp_tail_attr;

	sos_obj_t comp_obj;
	sos_attr_t comp__id_attr;
	sos_attr_t comp__next_comp_attr;
	sos_attr_t comp__prev_comp_attr;
	sos_attr_t comp__sample_head_attr;
	sos_attr_t comp__sample_tail_attr;

	sos_obj_t sample_obj;
	sos_attr_t sample__time_attr;
	sos_attr_t sample__job_id_attr;
	sos_attr_t sample__comp_id_attr;
	sos_attr_t sample__next_sample_attr;
	sos_attr_t sample__prev_sample_attr;

	size_t comp_count;
	struct component *components;
	struct job_metric_vector_s *mvec;
	struct job_attr_vector_s *avec;
};

static void __iter_reset(job_iter_t job_iter)
{
	if (job_iter->job_obj) {
		sos_obj_put(job_iter->job_obj);
		job_iter->job_obj = NULL;
	}
	if (job_iter->comp_obj) {
		sos_obj_put(job_iter->comp_obj);
		job_iter->comp_obj = NULL;
	}
	if (job_iter->sample_obj) {
		sos_obj_put(job_iter->sample_obj);
		job_iter->sample_obj = NULL;
	}
	if (job_iter->components) {
		int i;
		for (i = 0; i < job_iter->comp_count; i++) {
			struct component *c = &job_iter->components[i];
			if (c->comp_obj)
				sos_obj_put(c->comp_obj);
			if (c->sample_obj)
				sos_obj_put(c->comp_obj);
			if (c->sample_head)
				sos_value_put(c->sample_head);
		}
		free(job_iter->components);
		job_iter->comp_count = 0;
	}
	if (job_iter->mvec) {
		free(job_iter->mvec);
		job_iter->mvec = 0;
	}
	if (job_iter->avec) {
		free(job_iter->avec);
		job_iter->avec = 0;
	}
}

static job_attr_vector_t new_avec(size_t metric_count)
{
	job_attr_vector_t avec;
	size_t avec_size = sizeof(struct job_attr_vector_s)
		+ (metric_count * sizeof(sos_attr_t));
	avec = calloc(1, avec_size);
	if (avec)
		avec->count = metric_count;
	return avec;
}

static job_metric_vector_t new_mvec(size_t metric_count)
{
	job_metric_vector_t mvec;
	size_t mvec_size = sizeof(struct job_metric_vector_s)
		+ (metric_count * sizeof(struct job_metric_s));
	mvec = calloc(1, mvec_size);
	if (mvec)
		mvec->count = metric_count;
	return mvec;
}

void job_iter_mvec_del(job_metric_vector_t mvec)
{
	free(mvec);
}

job_metric_vector_t
job_iter_mvec_new(job_iter_t job_iter, size_t attr_count, const char *names[])
{
	int i;
	job_metric_vector_t mvec;
	sos_attr_t attr;
	sos_schema_t schema = sos_schema_by_name(job_iter->sos, "Sample");
	if (!schema)
		return NULL;

	/* The Time is automatically included and must be 0 */
	mvec = new_mvec(attr_count + 1);
	if (!mvec)
		goto out;

	mvec->vec[0].attr = job_iter->sample__time_attr;
	for (i = 0; i < attr_count; i++) {
		attr = sos_schema_attr_by_name(schema, names[i]);
		if (!attr)
			goto err;
		mvec->vec[i+1].attr = attr;
	}
	return mvec;
 err:
	free(mvec);
	mvec = NULL;
 out:
	sos_schema_put(schema);
	return mvec;
}

static void init_mvec(job_metric_vector_t mv)
{
	int i;
	for (i = 0; i < mv->count; i++) {
		job_metric_t m = &mv->vec[i];
		m->max_xi = DBL_MIN;
		m->min_xi = DBL_MAX;
		m->xi = 0.0;
		m->comp_i = 0.0;
		m->comp_mean_xi = 0.0;
		m->comp_sum_xi = 0.0;
		m->comp_sum_xi_sq = 0.0;
	}
}

static job_metric_vector_t dup_mvec(job_metric_vector_t mv)
{
	job_metric_vector_t nv = new_mvec(mv->count);
	int i;
	if (!nv)
		goto out;
	for (i = 0; i < mv->count; i++)
		nv->vec[i].attr = mv->vec[i].attr;
	init_mvec(mv);
 out:
	return nv;
}

#if 0
	attr_count = metric_count = sos_schema_attr_count(schema);
	/*
	 * The metric vector does not include the NextSample,
	 * PrevSample, or JobId attributes if present. If the iterator
	 * is ORDER_BY_TIME, the CompId is also skipped.
	 */
	switch (job_iter->order_by) {
	case JOB_ORDER_BY_TIME:
		metric_count -= 4;
		break;
	case JOB_ORDER_BY_COMPONENT:
		metric_count -= 3;
		break;
	default:
		assert(0);
	}
	job_iter->mvec = new_mvec(metric_count);
	if (!job_iter->mvec)
		goto err;
	job_iter->mvec->vec[0].attr = job_iter->sample__time_attr;
	job_iter->mvec->vec[0].max_xi = DBL_MIN;
	job_iter->mvec->vec[0].min_xi = DBL_MAX;
	metric_id = 1;
	for (attr_id = 0; attr_id < attr_count; attr_id++) {
		sos_attr_t attr = sos_schema_attr_by_id(schema, attr_id);
		assert(attr);
		if (attr == job_iter->sample__time_attr)
			continue;
		if (job_iter->order_by == JOB_ORDER_BY_TIME
		    && attr == job_iter->sample__comp_id_attr)
			continue;
		if (attr == job_iter->sample__next_sample_attr)
			continue;
		if (attr == job_iter->sample__prev_sample_attr)
			continue;
		if (attr == job_iter->sample__job_id_attr)
			continue;
		job_iter->mvec->vec[metric_id].attr = attr;
		job_iter->mvec->vec[metric_id].max_xi = DBL_MIN;
		job_iter->mvec->vec[metric_id].min_xi = DBL_MAX;
		metric_id++;
	}
#endif

static sos_obj_t __init_job(job_iter_t job_iter)
{
	sos_value_t value;
	sos_value_t head;
	sos_schema_t schema = NULL;
	size_t metric_count, attr_count;
	int metric_id, attr_id;

	head = sos_value(job_iter->job_obj, job_iter->job__comp_head_attr);
	job_iter->comp_obj = sos_obj_from_value(job_iter->sos, head);
	sos_value_put(head);
	if (!job_iter->comp_obj)
		goto err;

	value = sos_value(job_iter->job_obj, job_iter->job__size_attr);
	job_iter->comp_count = value->data->prim.uint64_;
	sos_value_put(value);

	return job_iter->job_obj;
 err:
	if (schema)
		sos_schema_put(schema);
	__iter_reset(job_iter);
	return NULL;
}

sos_obj_t job_iter_begin_job(job_iter_t job_iter)
{
	int rc = sos_iter_begin(job_iter->id_iter);
	if (rc)
		return NULL;

	/* Get rid of any leftovers */
	__iter_reset(job_iter);

	/* Position at first job */
	job_iter->job_obj = sos_iter_obj(job_iter->id_iter);
	if (!job_iter->job_obj)
		return NULL;

	return __init_job(job_iter);
}

sos_obj_t job_iter_end_job(job_iter_t job_iter)
{
	int rc = sos_iter_end(job_iter->id_iter);
	if (rc)
		return NULL;

	/* Get rid of any leftovers */
	__iter_reset(job_iter);

	/* Position at first job */
	job_iter->job_obj = sos_iter_obj(job_iter->id_iter);
	if (!job_iter->job_obj)
		return NULL;

	return __init_job(job_iter);
}

sos_obj_t job_iter_find_job_by_id(job_iter_t job_iter, long job_id)
{
	SOS_KEY(job_key);
	sos_key_set(job_key, &job_id, sizeof(job_id));
	int rc = sos_iter_find(job_iter->id_iter, job_key);
	if (rc)
		return NULL;

	/* Get rid of any leftovers */
	__iter_reset(job_iter);

	/* Position at first job */
	job_iter->job_obj = sos_iter_obj(job_iter->id_iter);
	if (!job_iter->job_obj)
		return NULL;

	return __init_job(job_iter);
}

sos_obj_t job_iter_next_job(job_iter_t job_iter)
{
	int rc = sos_iter_next(job_iter->id_iter);
	if (rc)
		return NULL;

	/* Position at next job */
	sos_obj_put(job_iter->job_obj);
	job_iter->job_obj = sos_iter_obj(job_iter->id_iter);
	if (!job_iter->job_obj)
		return NULL;

	return __init_job(job_iter);
}

sos_obj_t job_iter_prev_job(job_iter_t job_iter)
{
	int rc = sos_iter_prev(job_iter->id_iter);
	if (rc)
		return NULL;

	/* Position at previous job */
	sos_obj_put(job_iter->job_obj);
	job_iter->job_obj = sos_iter_obj(job_iter->id_iter);
	if (!job_iter->job_obj)
		return NULL;

	return __init_job(job_iter);
}

sos_obj_t job_iter_next_comp(job_iter_t job_iter)
{
	sos_value_t next_comp;
	sos_obj_t next_obj = NULL;
	next_comp = sos_value(job_iter->comp_obj, job_iter->comp__next_comp_attr);
	if (next_comp) {
		next_obj = sos_obj_from_value(job_iter->sos, next_comp);
		sos_value_put(next_comp);
		sos_obj_put(job_iter->comp_obj);
		job_iter->comp_obj = next_obj;
	}
	return next_obj;
}

sos_obj_t job_iter_prev_comp(job_iter_t job_iter)
{
	sos_value_t prev_comp;
	sos_obj_t prev_obj = NULL;
	prev_comp = sos_value(job_iter->comp_obj, job_iter->comp__prev_comp_attr);
	if (prev_comp) {
		prev_obj = sos_obj_from_value(job_iter->sos, prev_comp);
		sos_obj_put(job_iter->comp_obj);
		job_iter->comp_obj = prev_obj;
	}
	return prev_obj;
}

sos_obj_t job_iter_find_job_by_timestamp(job_iter_t job_iter, sos_value_t job_time)
{
	SOS_KEY(job_key);
	sos_key_set(job_key, sos_value_as_key(job_time),
		    sos_value_size(job_time));
	int rc = sos_iter_find(job_iter->time_iter, job_key);
	if (rc)
		return NULL;
	return sos_iter_obj(job_iter->time_iter);
}

#define BIN_WIDTH 1.0

job_iter_t job_iter_new(sos_t sos, job_iter_order_t order_by)
{
	job_iter_t job_iter = calloc(1, sizeof *job_iter);
	if (!job_iter)
		goto err_0;

	job_iter->sos = sos;
	job_iter->order_by = order_by;
	job_iter->bin_width = BIN_WIDTH;

	/* Get job attributes */
	sos_schema_t schema = sos_schema_by_name(sos, "Job");
	if (!schema)
		goto err_1;
	job_iter->job__id_attr = sos_schema_attr_by_name(schema, "JobId");
	job_iter->job__time_attr = sos_schema_attr_by_name(schema, "StartTime");
	job_iter->job__size_attr = sos_schema_attr_by_name(schema, "JobSize");
	job_iter->job__comp_head_attr = sos_schema_attr_by_name(schema, "CompHead");
	job_iter->job__comp_tail_attr = sos_schema_attr_by_name(schema, "CompTail");
	sos_schema_put(schema);
	if (!job_iter->job__size_attr || !job_iter->job__comp_head_attr
	    || !job_iter->job__time_attr || !job_iter->job__id_attr)
		goto err_1;

	/* Create the job iterator */
	job_iter->id_iter = sos_iter_new(job_iter->job__id_attr);
	if (!job_iter->id_iter)
		goto err_1;

	/* Create the time iterator */
	job_iter->time_iter = sos_iter_new(job_iter->job__time_attr);
	if (!job_iter->time_iter)
		goto err_1;

	/* Get component attributes */
	schema = sos_schema_by_name(sos, "CompRef");
	if (!schema)
		goto err_2;
	job_iter->comp__id_attr = sos_schema_attr_by_name(schema, "CompId");
	job_iter->comp__next_comp_attr = sos_schema_attr_by_name(schema, "NextComp");
	job_iter->comp__prev_comp_attr = sos_schema_attr_by_name(schema, "PrevComp");
	job_iter->comp__sample_head_attr = sos_schema_attr_by_name(schema, "SampleHead");
	job_iter->comp__sample_tail_attr = sos_schema_attr_by_name(schema, "SampleTail");
	sos_schema_put(schema);
	if (!job_iter->comp__id_attr || !job_iter->comp__next_comp_attr
	    || !job_iter->comp__sample_head_attr)
		goto err_3;

	schema = sos_schema_by_name(sos, "Sample");
	if (!schema)
		goto err_3;

	job_iter->sample__time_attr = sos_schema_attr_by_name(schema, "Time");
	job_iter->sample__job_id_attr = sos_schema_attr_by_name(schema, "JobId");
	job_iter->sample__comp_id_attr = sos_schema_attr_by_name(schema, "CompId");
	job_iter->sample__next_sample_attr = sos_schema_attr_by_name(schema, "NextSample");
	job_iter->sample__prev_sample_attr = sos_schema_attr_by_name(schema, "PrevSample");
	if (!job_iter->sample__time_attr  || !job_iter->sample__comp_id_attr
	    || !job_iter->sample__next_sample_attr || !job_iter->sample__prev_sample_attr)
		goto err_4;

	sos_schema_put(schema);
	return job_iter;

 err_4:
	sos_schema_put(schema);
 err_3:
	sos_iter_free(job_iter->time_iter);
 err_2:
	sos_iter_free(job_iter->id_iter);
 err_1:
	free(job_iter);
	job_iter = NULL;
 err_0:
	return job_iter;
}

void job_iter_free(job_iter_t job_iter)
{
	__iter_reset(job_iter);
	if (job_iter->id_iter)
		sos_iter_free(job_iter->id_iter);
	if (job_iter->time_iter)
		sos_iter_free(job_iter->time_iter);
	free(job_iter);
}

static void online_variance(job_metric_t m, double xi)
{
	double delta = xi - m->comp_mean_xi;
	m->xi = (m->xi * m->comp_i) + xi;
	m->comp_i += 1.0;
	m->xi /= m->comp_i;
	m->comp_mean_xi += delta/m->comp_i;
	m->comp_sum_xi += delta * (xi - m->comp_mean_xi);
}

static void update_metric(job_metric_t m, double xi)
{
	if (xi > m->max_xi)
		m->max_xi = xi;
	if (xi < m->min_xi)
		m->min_xi = xi;
#if 0
	m->comp_mean_xi = ((m->comp_i * m->comp_mean_xi) + xi) / (double)(m->comp_i + 1.0);
	m->comp_i += 1.0;
	m->comp_sum_xi += xi;
	m->comp_sum_xi_sq += xi * xi;
#else
	online_variance(m, xi);
#endif
}

static void update_time(job_metric_t m, double xi)
{
	double delta = xi - m->comp_mean_xi;
	m->comp_i += 1.0;
	m->comp_mean_xi += delta/m->comp_i;
	m->comp_sum_xi += delta * (xi - m->comp_mean_xi);
}

static double time_val(sos_obj_t sample, job_metric_t m, double *prev_time, double *time_diff)
{
	SOS_VALUE(val);
	double xi;
	if (sample) {
		double d;
		val = sos_value_init(val, sample, m->attr);
		xi = (double)val->data->prim.timestamp_.fine.secs
			+ (double)val->data->prim.timestamp_.fine.usecs / 1.0e6;
		assert(xi > 0.0);
		*time_diff = xi - *prev_time;
		*prev_time = xi;
		sos_value_put(val);
	} else {
		xi = round(*prev_time + *time_diff);
		*prev_time = xi;
	}
	return xi;
}

static double xi_val(sos_obj_t sample, job_metric_t m)
{
	double xi;
	SOS_VALUE(val);
	if (sample) {
		val = sos_value_init(val, sample, m->attr);
		xi = (double)val->data->prim.uint64_;
		sos_value_put(val);
	} else
		xi = m->comp_mean_xi;
	return xi;
}

pthread_mutex_t free_work_lock = PTHREAD_MUTEX_INITIALIZER;
LIST_HEAD(free_work_list, work) free_work_list = LIST_HEAD_INITIALIZER(free_work_list);

void del_work(struct work *w)
{
	pthread_mutex_lock(&free_work_lock);
	LIST_INSERT_HEAD(&free_work_list, w, free_link);
	pthread_mutex_unlock(&free_work_lock);
}

struct work *new_work(job_iter_t iter, work_fn_t work_fn,
		      ods_atomic_t *sem, pthread_mutex_t *lock, pthread_cond_t *cv)
{
	struct work *work;

	pthread_mutex_lock(&free_work_lock);
	if (!LIST_EMPTY(&free_work_list)) {
		work = LIST_FIRST(&free_work_list);
		LIST_REMOVE(work, free_link);
	} else {
		work = calloc(1, sizeof(*work));
		if (work) {
			work->mvec = dup_mvec(iter->mvec);
			if (!work->mvec) {
				free(work);
				work = NULL;
			}
		}
	}
	pthread_mutex_unlock(&free_work_lock);
	if (work) {
		work->job_iter = iter;
		work->work_fn = work_fn;
		work->sem = sem;
		work->cv = cv;
		work->cv_lock = lock;
		init_mvec(work->mvec);
	}
	return work;
}

static int work_done = 0;
void *work_proc(void *arg)
{
	struct worker *worker = arg;
	struct work *work;
	int rc;

	while (1) {
		pthread_mutex_lock(&worker->work_q_lock);
		while (TAILQ_EMPTY(&worker->work_q)) {
			rc = pthread_cond_wait(&worker->work_q_cv, &worker->work_q_lock);
			if (rc == EINTR)
				continue;
		}
		if (!TAILQ_EMPTY(&worker->work_q)) {
			work = TAILQ_FIRST(&worker->work_q);
			TAILQ_REMOVE(&worker->work_q, work, link);
		} else
			work = NULL;
		pthread_mutex_unlock(&worker->work_q_lock);
		if (!work)
			break;

		assert(work);
		work->work_fn(work);

		pthread_mutex_lock(work->cv_lock);
		ods_atomic_dec(work->sem);
		pthread_cond_signal(work->cv);
		pthread_mutex_unlock(work->cv_lock);
	}
	return NULL;
}

static double skip_to_sample(job_iter_t ji, struct component *c, double bin_start)
{
	double xi;
	SOS_VALUE(next);
	SOS_VALUE(xiv);

	next = sos_value_init(next, c->sample_obj,
			      ji->sample__next_sample_attr);
	sos_obj_put(c->sample_obj);
	c->sample_obj = sos_obj_from_value(ji->sos, next);
	while (c->sample_obj) {
		xiv = sos_value_init(xiv, c->sample_obj, ji->sample__time_attr);
		xi = (double)xiv->data->prim.timestamp_.fine.secs
			+ (double)xiv->data->prim.timestamp_.fine.usecs / 1.0e6;
		sos_value_put(xiv);
		if (abs(xi - bin_start) < ji->bin_width)
			return xi;

		if (xi > bin_start)
			/* Try in next bin */
			return xi;

		next = sos_value_init(next, c->sample_obj,
				      ji->sample__next_sample_attr);
		sos_obj_put(c->sample_obj);
		c->sample_obj = sos_obj_from_value(ji->sos, next);
		sos_value_put(next);
	}
	return DBL_MAX;
}

static void avg_over_comps(struct work *work)
{
	struct component *c;
	job_iter_t ji = work->job_iter;
	job_metric_vector_t mvec = work->mvec;
	size_t sample_count = 0;
	int comp_no, metric_no;
	double prev_time = 0.0;;
	double time_diff = 0.0;
	double xi;
	double bin_start = 0.0;
	for (comp_no = work->start, c = &ji->components[comp_no];
	     comp_no < work->end; comp_no++, c++) {
		job_metric_t m = &mvec->vec[0];
		double xi = time_val(c->sample_obj, m, &prev_time, &time_diff);
		if (bin_start == 0.0)
			bin_start = floor(xi);
		else if (abs(xi - bin_start) > ji->bin_width) {
			if (xi < bin_start) {
				xi = skip_to_sample(ji, c, bin_start);
				if (abs(xi - bin_start) > ji->bin_width)
					/* No luck, try in next bin */
					continue;
			} else
				/* Try this sample again in the next bin */
				continue;
		}
		update_metric(m, xi);
		for (metric_no = 1; metric_no < mvec->count; metric_no++) {
			m = &mvec->vec[metric_no];
			xi = xi_val(c->sample_obj, m);
			update_metric(m, xi);
		}
		if (!c->sample_obj)
			continue;
		SOS_VALUE(next);
		next = sos_value_init(next, c->sample_obj,
				      ji->sample__next_sample_attr);
		sos_obj_put(c->sample_obj);
		c->sample_obj = sos_obj_from_value(ji->sos, next);
		if (c->sample_obj)
			sample_count++;
		sos_value_put(next);
	}
	if (!sample_count)
		mvec->status = JOB_ITER_END;
	else
		mvec->status = JOB_ITER_OK;
}

#define THREAD_COUNT 16
static int thread_count = THREAD_COUNT;
static struct worker workers[THREAD_COUNT];

void job_iter_set_bin_width(job_iter_t job_iter, double width)
{
	job_iter->bin_width = width;
}

double job_iter_get_bin_width(job_iter_t job_iter)
{
	return job_iter->bin_width;
}


/*
 * The next_sample_by_time function must compute a number of
 * statistics across all components in the job for a particular
 * timestamp. A job of size 4096, that has 1024 samples over time, and
 * 200 metrics would have 4096 * 1024 * 200 = 838,860,800 individual
 * metrics to evaluate. On a single processor, this is too slow to
 * provide an interactive experience for the user as it requires 10's
 * of seconds to process the data.
 */
static int __next_sample_by_time(job_iter_t job_iter)
{
	struct component *c;
	int comp_no, metric_no;
	size_t sample_count, comp_count, remainder;
	double prev_time = 0.0;
	double time_diff = 0.0;
	ods_atomic_t sem;
	pthread_mutex_t lock;
	pthread_cond_t cv;
	struct work *work[THREAD_COUNT];

	pthread_mutex_init(&lock, NULL);
	pthread_cond_init(&cv, NULL);
	sem = thread_count;

	init_mvec(job_iter->mvec);
	comp_count = job_iter->comp_count / thread_count;
	remainder = job_iter->comp_count % thread_count;
	int t, start = 0;
	for (t = 0; t < thread_count; t++) {
		work[t] = new_work(job_iter, avg_over_comps, &sem, &lock, &cv);
		work[t]->start = start;
		work[t]->end = start + comp_count;
		if (remainder) {
			work[t]->end++;
			remainder--;
		}
		start = work[t]->end;
		pthread_mutex_lock(&workers[t].work_q_lock);
		TAILQ_INSERT_TAIL(&workers[t].work_q, work[t], link);
		pthread_mutex_unlock(&workers[t].work_q_lock);
		pthread_cond_signal(&workers[t].work_q_cv);
	}
	work_done = 1;
	pthread_mutex_lock(&lock);
	while (sem) {
		int rc = pthread_cond_wait(&cv, &lock);
		if (rc == EINTR)
			continue;
	}
	job_iter_status_t status = JOB_ITER_END;
	job_metric_t m;
	job_metric_t mw;
	for (t = 0; t < thread_count; t++) {
		struct work *w = work[t];
		if (w->mvec->status == JOB_ITER_OK)
			status = JOB_ITER_OK;
		for (comp_no = w->start, c = &job_iter->components[comp_no];
		     comp_no < w->end; comp_no++, c++) {

			m = &job_iter->mvec->vec[0];
			mw = &w->mvec->vec[0];

			for (metric_no = 0; metric_no < job_iter->mvec->count; metric_no++) {
				m = &job_iter->mvec->vec[metric_no];
				mw = &w->mvec->vec[metric_no];
				if (mw->max_xi > m->max_xi)
					m->max_xi = mw->max_xi;
				if (mw->min_xi < m->min_xi)
					m->min_xi = mw->min_xi;
				m->comp_mean_xi =
					((m->comp_i * m->comp_mean_xi) + (mw->comp_i * mw->comp_mean_xi))
					/ (m->comp_i + mw->comp_i);
				m->comp_i += mw->comp_i;
				m->comp_sum_xi += mw->comp_sum_xi;
			}
		}
		del_work(w);
	}
	job_iter->mvec->status = status;
	if (status == JOB_ITER_END)
		return ENOENT;

	/* Update the global stats */
	m = &job_iter->mvec->vec[0];
	double jitter = m->comp_mean_xi - floor(m->comp_mean_xi);
	m->time_mean_xi = ((m->time_i * m->time_mean_xi) + m->comp_mean_xi) / (m->time_i + 1.0);
	m->time_sum_xi += jitter;
	m->time_sum_xi_sq += jitter * jitter;
	m->time_i += 1.0;
	m->diff_xi = m->comp_mean_xi - m->prev_xi;
	m->prev_xi = m->comp_mean_xi;
	for (metric_no = 1; metric_no < job_iter->mvec->count; metric_no++) {
		m = &job_iter->mvec->vec[metric_no];
		m->time_mean_xi = ((m->time_i * m->time_mean_xi) + m->comp_mean_xi) / (m->time_i + 1.0);
		m->time_i += 1.0;
		m->diff_xi = m->comp_mean_xi - m->prev_xi;
		m->prev_xi = m->comp_mean_xi;
		m->time_sum_xi += m->time_mean_xi;
		m->time_sum_xi_sq += m->time_mean_xi * m->time_mean_xi;
	}
	return 0;
}

static int __begin_sample_by_time(job_iter_t job_iter)
{
	int i;
	sos_obj_t comp_obj = job_iter->comp_obj;
	sos_value_t next_comp;

	/* Build the component table */
	if (!job_iter->components)
		job_iter->components = calloc(job_iter->comp_count,
					      sizeof(struct component));
	if (!job_iter->components)
		goto empty;
	for (i = 0; i < job_iter->comp_count && comp_obj; i++) {
		struct component *c = &job_iter->components[i];
		struct sos_value_s val;
		sos_value_t comp_id_val = sos_value_init(&val, comp_obj, job_iter->comp__id_attr);
		c->comp_id = comp_id_val->data->prim.uint64_;
		sos_value_put(comp_id_val);
		c->comp_obj = comp_obj;
		c->sample_head = sos_value(comp_obj, job_iter->comp__sample_head_attr);
		c->sample_obj = sos_obj_from_value(job_iter->sos, c->sample_head);
		next_comp = sos_value_init(&val, comp_obj, job_iter->comp__next_comp_attr);
		comp_obj = sos_obj_from_value(job_iter->sos, next_comp);
		sos_value_put(next_comp);
	}
	return __next_sample_by_time(job_iter);
 empty:
	return ENOENT;
}

static int __next_sample_by_component(job_iter_t job_iter)
{
	SOS_VALUE(val);
	struct component *c = &job_iter->components[0];
	job_metric_vector_t mvec = job_iter->mvec;
	int metric_no;
	job_metric_t m;
	double xi, jitter;

	if (c->sample_obj) {
		mvec->status = JOB_ITER_END;
		return ENOENT;
	}

	init_mvec(job_iter->mvec);

	m = &job_iter->mvec->vec[0];
	val = sos_value_init(val, c->sample_obj, m->attr);
	xi = (double)val->data->prim.timestamp_.fine.secs
		+ (double)val->data->prim.timestamp_.fine.usecs / 1.0e6;
	sos_value_put(val);

	jitter = xi - floor(xi);
	if (jitter > m->max_xi)
		m->max_xi = jitter;
	if (jitter < m->min_xi)
		m->min_xi = jitter;
	m->time_mean_xi = ((m->time_i * m->time_mean_xi) + jitter) / (m->time_i + 1.0);
	m->comp_sum_xi += jitter;
	m->comp_sum_xi_sq += jitter * jitter;
	m->time_i += 1.0;
	m->comp_i += 1.0;

	for (metric_no = 1; metric_no < mvec->count; metric_no++) {
		m = &mvec->vec[metric_no];
		val = sos_value_init(val, c->sample_obj, m->attr);
		xi = (double)val->data->prim.uint64_;

		if (xi > m->max_xi)
			m->max_xi = xi;
		if (xi < m->min_xi)
			m->min_xi = xi;

		m->time_mean_xi = ((m->time_i * m->time_mean_xi) + xi) / (m->time_i + 1.0);
		m->comp_i += 1.0;
		m->comp_sum_xi += xi;
		m->comp_sum_xi_sq += xi * xi;
	}

	SOS_VALUE(next);
	next = sos_value_init(next, c->sample_obj,
			      job_iter->sample__next_sample_attr);
	sos_obj_put(c->sample_obj);
	c->sample_obj = sos_obj_from_value(job_iter->sos, next);
	if (c->sample_obj)
		mvec->status = JOB_ITER_OK;
	else
		mvec->status = JOB_ITER_END;

	return 0;
}

static int __begin_sample_by_component(job_iter_t job_iter)
{
	int i;
	sos_obj_t comp_obj = job_iter->comp_obj;

	/* Build the component table */
	if (!job_iter->components)
		job_iter->components = calloc(1, sizeof(struct component));
	if (!job_iter->components)
		goto empty;
	struct component *c = &job_iter->components[i];
	sos_value_t comp_id_val = sos_value(comp_obj, job_iter->comp__id_attr);
	c->comp_id = comp_id_val->data->prim.uint64_;
	sos_value_put(comp_id_val);
	c->comp_obj = comp_obj;
	c->sample_head = sos_value(comp_obj, job_iter->comp__sample_head_attr);
	c->sample_obj = sos_obj_from_value(job_iter->sos, c->sample_head);

	return __next_sample_by_component(job_iter);
 empty:
	return ENOENT;
}

int job_iter_begin_sample(job_iter_t job_iter, job_metric_flags_t flags,
			  job_metric_vector_t mvec)
{
	int rc;
	sos_obj_t job_obj;
	job_iter->flags = flags;
	if (!job_iter->job_obj) {
		job_obj = job_iter_begin_job(job_iter);
		if (!job_obj)
			return ENOENT;
		sos_obj_put(job_obj);
	}
	job_iter->mvec = mvec;
	if (job_iter->order_by == JOB_ORDER_BY_TIME)
		return __begin_sample_by_time(job_iter);
	else if (job_iter->order_by == JOB_ORDER_BY_COMPONENT)
		return __begin_sample_by_component(job_iter);
	assert(0);
	return EINVAL;
}

int job_iter_next_sample(job_iter_t job_iter)
{
	int rc;
	sos_obj_t job_obj;
	if (!job_iter->job_obj)
		return ENOENT;
	if (job_iter->order_by == JOB_ORDER_BY_TIME)
		return __next_sample_by_time(job_iter);
	else if (job_iter->order_by == JOB_ORDER_BY_COMPONENT)
		return __next_sample_by_component(job_iter);
	assert(0);
	return ENOENT;
}

sos_obj_t job_iter_job_first(job_iter_t job_iter)
{
	int rc = sos_iter_begin(job_iter->id_iter);
	if (rc)
		return NULL;

	/* Get rid of any leftovers */
	__iter_reset(job_iter);

	/* Position at first job */
	job_iter->job_obj = sos_iter_obj(job_iter->id_iter);
	if (!job_iter->job_obj)
		return NULL;

	return __init_job(job_iter);
}

sos_obj_t job_iter_comp_first(job_iter_t job_iter, sos_obj_t job_obj)
{
	sos_value_t comp_head;
	sos_obj_t comp_obj = NULL;
	comp_head = sos_value(comp_obj, job_iter->job__comp_head_attr);
	if (comp_head) {
		comp_obj = sos_obj_from_value(job_iter->sos, comp_head);
		sos_value_put(comp_head);
	}
	return comp_obj;
}

sos_obj_t job_iter_comp_last(job_iter_t job_iter, sos_obj_t job_obj)
{
	sos_value_t comp_tail;
	sos_obj_t comp_obj = NULL;
	comp_tail = sos_value(comp_obj, job_iter->job__comp_tail_attr);
	if (comp_tail) {
		comp_obj = sos_obj_from_value(job_iter->sos, comp_tail);
		sos_value_put(comp_tail);
	}
	return comp_obj;
}

sos_obj_t job_iter_comp_next(job_iter_t job_iter, sos_obj_t comp_obj)
{
	sos_value_t next_comp;
	sos_obj_t next_obj = NULL;
	next_comp = sos_value(comp_obj, job_iter->comp__next_comp_attr);
	if (next_comp) {
		next_obj = sos_obj_from_value(job_iter->sos, next_comp);
		sos_value_put(next_comp);
		sos_obj_put(comp_obj);
	}
	return next_obj;
}

sos_obj_t job_iter_comp_prev(job_iter_t job_iter, sos_obj_t comp_obj)
{
	sos_value_t prev_comp;
	sos_obj_t prev_obj = NULL;
	prev_comp = sos_value(comp_obj, job_iter->comp__prev_comp_attr);
	if (prev_comp) {
		prev_obj = sos_obj_from_value(job_iter->sos, prev_comp);
		sos_value_put(prev_comp);
		sos_obj_put(comp_obj);
	}
	return prev_obj;
}

sos_obj_t job_iter_sample_first(job_iter_t job_iter, sos_obj_t comp_obj)
{
	sos_value_t sample_head;
	sos_obj_t sample_obj = NULL;
	sample_head = sos_value(comp_obj, job_iter->comp__sample_head_attr);
	if (sample_head) {
		sample_obj = sos_obj_from_value(job_iter->sos, sample_head);
		sos_value_put(sample_head);
	}
	return sample_obj;
}

sos_obj_t job_iter_sample_last(job_iter_t job_iter, sos_obj_t comp_obj)
{
	sos_value_t sample_tail;
	sos_obj_t sample_obj = NULL;
	sample_tail = sos_value(comp_obj, job_iter->comp__sample_tail_attr);
	if (sample_tail) {
		sample_obj = sos_obj_from_value(job_iter->sos, sample_tail);
		sos_value_put(sample_tail);
	}
	return sample_obj;
}

sos_obj_t job_iter_sample_next(job_iter_t job_iter, sos_obj_t sample_obj)
{
	sos_value_t next_sample;
	sos_obj_t next_obj = NULL;
	next_sample = sos_value(sample_obj, job_iter->sample__next_sample_attr);
	if (next_sample) {
		next_obj = sos_obj_from_value(job_iter->sos, next_sample);
		sos_value_put(next_sample);
		sos_obj_put(sample_obj);
	}
	return next_obj;
}

sos_obj_t job_iter_sample_prev(job_iter_t job_iter, sos_obj_t sample_obj)
{
	sos_value_t prev_sample;
	sos_obj_t prev_obj = NULL;
	prev_sample = sos_value(sample_obj, job_iter->sample__prev_sample_attr);
	if (prev_sample) {
		prev_obj = sos_obj_from_value(job_iter->sos, prev_sample);
		sos_value_put(prev_sample);
		sos_obj_put(sample_obj);
	}
	return prev_obj;
}

#if 0
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
#endif

static void __attribute__ ((constructor)) job_lib_init(void)
{
	int i, rc;
	char *threads = getenv("JOB_THREADS");
	if (threads) {
		int i = atoi(threads);
		if (i < THREAD_COUNT)
			thread_count = i;
	}
	for (i = 0; i < thread_count; i++) {
		pthread_mutex_init(&workers[i].work_q_lock, NULL);
		pthread_cond_init(&workers[i].work_q_cv, NULL);
		TAILQ_INIT(&workers[i].work_q);
		rc = pthread_create(&workers[i].thread, NULL, work_proc, &workers[i]);
	}
}

static void __attribute__ ((destructor)) job_lib_term(void)
{
}
