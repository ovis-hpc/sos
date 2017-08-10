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
#ifndef _MQ_H_
#define _MQ_H_
#include <inttypes.h>
struct mq_s;
typedef struct mq_msg_s {
	uint32_t msg_id;
	uint32_t msg_type;
	uint32_t msg_size;
	int (*msg_work_fn)(struct mq_s *, struct mq_msg_s *);
	uint8_t msg_data[0];
} *mq_msg_t;

typedef struct mq_s {
	uint32_t mq_prod;
	uint32_t mq_cons;
	uint32_t mq_prod_g;
	uint32_t mq_cons_g;
	uint32_t mq_depth;
	uint32_t mq_msg_max;
	uint8_t *mq_msg_mem;
	mq_msg_t *mq_q;
	uint32_t block;
	pthread_mutex_t mq_lock;
	pthread_cond_t mq_cv;
} *mq_t;

/**
 * \brief Check if the queue is empty
 *
 * Return True (!0) if the queue is empty.
 *
 * \param mq The message queue handle
 * \retval 1 The queue is empty
 * \retval 0 The queue is not empty
 */
int mq_is_empty(mq_t mq);

/**
 * \brief Check if the message queue is full
 *
 * Return true (!0) if the queue is full
 *
 * \retval mq The message queue handle
 * \retval 1 The queue is full
 * \retval 0 The queue is not full
 */
int mq_is_full(mq_t mq);

/**
 * \brief Return the next available producer message buffer.
 *
 * The caller prepares the contents of the message and then posts
 * it to the queue for the consumer by calling mq_post_prod_msg().
 *
 * If the queue is full, this function returns NULL. Use the
 * mq_is_empty() function to determine if there are producer messages
 * available.
 *
 * Use the mq_get_prod_msg_wait() function to block and wait until a
 * producer message buffer is avaialbe in the queue.
 *
 * \param mq The message queue handle
 * \retval !NULL Pointer to the message buffer
 * \retval NULL The queue is full
 */
mq_msg_t mq_get_prod_msg(mq_t mq);
/**
 * \brief Post the producer message and make it available to the consumer
 *
 * This message cannot fail, i.e. it will never return 0.
 *
 * \param mq The message queue handle
 * \returns The message id of the produced message.
 */
int mq_post_prod_msg(mq_t mq);

/**
 * \brief Wait until a producer message is avaialbe and then return.
 *
 * If the producer queue is full, this function blocks waits until there
 * is a producer message buffer available.
 *
 * \returns Pointer to the message buffer.
 */
mq_msg_t mq_get_prod_msg_wait(mq_t mq);

/**
 * \brief Return the next available consumer message buffer.
 *
 * When the consumer is finished processing the message, it must
 * be returned to the queue with the mq_post_cons_msg() function.
 *
 * If the queue is empty, the function returns NULL. Use the
 * mq_is_empty() function to determine if there are messages
 * available for processing.
 *
 * Use the mq_get_cons_msg_wait() function to block and wait until a
 * message is available.
 *
 * \retval !NULL Pointer to the message buffer
 * \retval NULL The queue is empty
 */
mq_msg_t mq_get_cons_msg(mq_t mq);

/**
 * \brief Return the next avaialable message
 *
 * If the consumer queue is empty, this function waits until there
 * is an available message. When the consumer is finished processing
 * the message, it must be returned to the queue-pair with the
 * mq_post_cons_msg() function.
 *
 * \retval !NULL Pointer to the message buffer
 */
mq_msg_t mq_get_cons_msg_wait(mq_t mq);

/**
 * \brief Return the message to the producer
 *
 * After consumer is finished processing the message, call
 * this function to return the message buffer to the producer.
 */
void mq_post_cons_msg(mq_t mq);

/**
 * \brief Create a producer/consumer queue-pair
 *
 * \param q_depth The depth of the queue-pair
 * \param max_msg_size The maximum size of a message
 * \param blocking Set to a non-zero value for a blocking MQ
 * \retval !NULL Pointer to the message buffer
 */
mq_t mq_new(size_t q_depth, size_t max_msg_size, int blocking);

/**
 * \brief Notify consumer that the producer is no longer using the MQ
 * \param mq The message queue
 */
void mq_finish(mq_t mq);

#endif
