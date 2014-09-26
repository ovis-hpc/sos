/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
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

/*
 * Author: Tom Tucker tom at ogc dot us
 */

#ifndef _ODS_ATOMIC_H_
#define _ODS_ATOMIC_H_

typedef uint32_t ods_atomic_t;

/*
 * Atomic increment/decrement
 */
static inline ods_atomic_t ods_atomic_inc(ods_atomic_t *a) {
	return __sync_add_and_fetch(a, 1);
}

static inline ods_atomic_t ods_atomic_dec(ods_atomic_t *a) {
	return __sync_sub_and_fetch(a, 1);
}

/*
 * Atomic add/subtract
 */
static inline ods_atomic_t ods_atomic_add(ods_atomic_t *a, int v) {
	return __sync_add_and_fetch(a, v);
}

static inline ods_atomic_t ods_atomic_sub(ods_atomic_t *a, int v) {
	return __sync_sub_and_fetch(a, v);
}

/*
 * Atomic bitwise operations
 */
static inline ods_atomic_t ods_atomic_and(ods_atomic_t *a, int m) {
	return __sync_and_and_fetch(a, m);
}

static inline ods_atomic_t ods_atomic_nand(ods_atomic_t *a, int m) {
	return __sync_nand_and_fetch(a, m);
}

static inline ods_atomic_t ods_atomic_or(ods_atomic_t *a, int m) {
	return __sync_or_and_fetch(a, m);
}

static inline ods_atomic_t ods_atomic_xor(ods_atomic_t *a, int m) {
	return __sync_xor_and_fetch(a, m);
}


#endif
