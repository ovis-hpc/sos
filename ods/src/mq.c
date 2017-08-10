/* -*- c-basic-offset: 8 -*-
 * Copyright (c) 2016 Open Grid Computing, Inc. All rights reserved.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "mq.h"

int mq_is_empty(mq_t mq)
{
	return (mq->mq_cons == mq->mq_prod) && (mq->mq_cons_g == mq->mq_prod_g);
}

int mq_is_full(mq_t mq)
{
	return (mq->mq_cons == mq->mq_prod) && (mq->mq_cons_g != mq->mq_prod_g);
}

mq_msg_t mq_get_cons_msg(mq_t mq)
{
	if (mq_is_empty(mq))
		return NULL;
	return mq->mq_q[mq->mq_cons];
}

mq_msg_t mq_get_cons_msg_wait(mq_t mq)
{
	if (mq->block)
		pthread_mutex_lock(&mq->mq_lock);
	while (mq_is_empty(mq)) {
		if (mq->block)
			pthread_cond_wait(&mq->mq_cv, &mq->mq_lock);
		else
			pthread_testcancel();
	}
	if (mq->block)
		pthread_mutex_unlock(&mq->mq_lock);
	return mq->mq_q[mq->mq_cons];
}

void mq_post_cons_msg(mq_t mq)
{
	if (mq->block)
		pthread_mutex_lock(&mq->mq_lock);
	mq->mq_cons += 1;
	if (mq->mq_cons >= mq->mq_depth) {
		mq->mq_cons = 0;
		mq->mq_cons_g = !mq->mq_cons_g;
	}
	if (mq->block) {
		pthread_cond_broadcast(&mq->mq_cv);
		pthread_mutex_unlock(&mq->mq_lock);
	}
}

mq_msg_t mq_get_prod_msg(mq_t mq)
{
	if (mq_is_full(mq))
		return NULL;
	return mq->mq_q[mq->mq_prod];
}

mq_msg_t mq_get_prod_msg_wait(mq_t mq)
{
	if (mq->block)
		pthread_mutex_lock(&mq->mq_lock);
	while (mq_is_full(mq)) {
		if (mq->block)
			pthread_cond_wait(&mq->mq_cv, &mq->mq_lock);
		else
			pthread_testcancel();
	}
	int prod = mq->mq_prod;
	if (mq->block)
		pthread_mutex_unlock(&mq->mq_lock);
	return mq->mq_q[prod];
}

int mq_post_prod_msg(mq_t mq)
{
	int rc;
	if (mq->block)
		pthread_mutex_lock(&mq->mq_lock);
	rc = mq->mq_prod;
	mq->mq_prod += 1;
	if (mq->mq_prod >= mq->mq_depth) {
		mq->mq_prod = 0;
		mq->mq_prod_g = !mq->mq_prod_g;
	}
	if (mq->block) {
		pthread_cond_broadcast(&mq->mq_cv);
		pthread_mutex_unlock(&mq->mq_lock);
	}
	return rc;
}

mq_t mq_new(size_t q_depth, size_t max_msg_size, int blocking)
{
	int i;
	mq_t mq;
	size_t msg_size = (max_msg_size * q_depth);
	mq = malloc(sizeof(*mq));
	if (!mq)
		goto out;
	mq->block = blocking;
	mq->mq_msg_mem = malloc(msg_size);
	if (!mq->mq_msg_mem)
		goto err_0;
	mq->mq_q = malloc(q_depth * sizeof(mq_msg_t));
	if (!mq->mq_q)
		goto err_1;
	mq->mq_depth = q_depth;
	mq->mq_msg_max = max_msg_size;
	mq->mq_prod = mq->mq_cons = mq->mq_prod_g = mq->mq_cons_g = 0;
	/* Populate msg_id (1 based) */
	for (i = 0; i < mq->mq_depth; i++) {
		mq->mq_q[i] = (mq_msg_t)&mq->mq_msg_mem[i * max_msg_size];
		mq->mq_q[i]->msg_id = i + 1;
	}
	pthread_mutex_init(&mq->mq_lock, NULL);
	pthread_cond_init(&mq->mq_cv, NULL);
 out:
	return mq;
 err_1:
	free(mq->mq_msg_mem);
 err_0:
	free(mq);
	return NULL;
}

void mq_finish(mq_t mq)
{
	pthread_mutex_unlock(&mq->mq_lock);
}
