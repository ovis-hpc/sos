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
#ifndef _JOB_H_
#define _JOB_H_

typedef struct job_attr_vector_s {
	size_t count;
	sos_attr_t attr[0];
} *job_attr_vector_t;

typedef struct job_metric_s {
	sos_attr_t attr;

	double xi;
	double max_xi;
	double min_xi;
	double prev_xi;
	double diff_xi;

	double comp_i;
	double comp_mean_xi;
	double comp_sum_xi;
	double comp_sum_xi_sq;

	double time_i;
	double time_mean_xi;
	double time_sum_xi;
	double time_sum_xi_sq;
} *job_metric_t;

typedef enum job_iter_status_e {
	JOB_ITER_OK = 0,
	JOB_ITER_END,
} job_iter_status_t;

typedef struct job_metric_vector_s {
	size_t count;
	job_iter_status_t status;
	struct job_metric_s vec[0];
} *job_metric_vector_t;

typedef enum job_iter_order_e {
	/*! Samples are returned in time order where each metric is an
	 * average of the value for the metric recorded by each component.
	 * T(1){C(1)...C(m)}, T(1){avg(C(1)...C(m))}, ... T(N){avg(C(1)...C{m})}
	 * The iterator will return (N) sample records
	 */
	JOB_ORDER_BY_TIME,
	/*! Metric values are returned in time order by component:
	 * C(1)[T(1)...T(N)], C(1)[T(1)...T(N)]...C(M)[T(1)...T(N)]
	 * No averaging is done on the samples. The iterator will
	 * return (N * M) sample records.
	 */
	JOB_ORDER_BY_COMPONENT,
} job_iter_order_t;

typedef enum job_metric_flags_e {
	/*! The sample measurement is an instantaneous measurement, it
	 *  is not cumulative */
	JOB_METRIC_VAL,
	/*! The sample measurement is cumulative, each sample is
            incrementally larger than the previous sample. */
	JOB_METRIC_CUM,
} job_metric_flags_t;

typedef struct job_iter_s *job_iter_t;

job_iter_t job_iter_new(sos_t sos, job_iter_order_t order_by);
void job_iter_set_bin_width(job_iter_t job_iter, double width);
double job_iter_get_bin_width(job_iter_t job_iter);
void job_iter_free(job_iter_t job_iter);
sos_obj_t job_iter_begin_job(job_iter_t job_iter);
sos_obj_t job_iter_next_job(job_iter_t job_iter);
sos_obj_t job_iter_prev_job(job_iter_t job_iter);
sos_obj_t job_iter_end_job(job_iter_t job_iter);
sos_obj_t job_iter_find_job_by_id(job_iter_t job_iter, long job_id);
sos_obj_t job_iter_find_job_by_timestamp(job_iter_t job_iter, sos_value_t job_time);
sos_obj_t job_iter_next_comp(job_iter_t job_iter);

job_metric_vector_t
job_iter_mvec_new(job_iter_t job_iter, size_t attr_count, const char **names);
void job_iter_mvec_del(job_metric_vector_t mvec);

int job_iter_begin_sample(job_iter_t job_iter,
			  job_metric_flags_t flags,
			  job_metric_vector_t mvec);
int job_iter_next_sample(job_iter_t job_iter);

sos_obj_t job_iter_sample_first(job_iter_t job_iter, sos_obj_t comp_obj);
sos_obj_t job_iter_sample_last(job_iter_t job_iter, sos_obj_t comp_obj);
sos_obj_t job_iter_sample_next(job_iter_t job_iter, sos_obj_t sample_obj);
sos_obj_t job_iter_sample_prev(job_iter_t job_iter, sos_obj_t sample_obj);

sos_obj_t job_iter_comp_first(job_iter_t job_iter, sos_obj_t job_obj);
sos_obj_t job_iter_comp_last(job_iter_t job_iter, sos_obj_t job_obj);
sos_obj_t job_iter_comp_next(job_iter_t job_iter, sos_obj_t comp_obj);
sos_obj_t job_iter_comp_prev(job_iter_t job_iter, sos_obj_t comp_obj);

#endif
