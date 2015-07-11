/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2014 Sandia Corporation. All rights reserved.
 *
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

/*
 * Author: Tom Tucker tom at ogc dot us
 */

#include <sys/queue.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

/** \addtogroup keys
 * @{
 */

/**
 * \brief Set the value of a key
 *
 * Sets the value of the key to 'value'. The \c value parameter is of
 * type void* to make it convenient to use values of arbitrary
 * types. The minimum of 'sz' and the maximum key length is
 * copied. The number of bytes copied into the key is returned.
 *
 * \param key	The key
 * \param value	The value to set the key to
 * \param sz	The size of value in bytes
 * \returns The number of bytes copied
 */
size_t sos_key_set(sos_key_t key, void *value, size_t sz)
{
	return ods_key_set(key, value, sz);
}

/**
 * \brief Set the value of a key from a string
 *
 * \param attr	The attribute handle
 * \param key	The key
 * \param str	Pointer to a string
 * \retval 0	if successful
 * \retval -1	if there was an error converting the string to a value
 */
int sos_key_from_str(sos_attr_t attr, sos_key_t key, const char *str)
{
	sos_idx_part_t part = __sos_active_idx_part(attr);
	return ods_key_from_str(part->index, key, str);
}

/**
 * \brief Return a string representation of the key value
 *
 * \param attr	The attribute handle
 * \param key	The key
 * \return A const char * representation of the key value.
 */
const char *sos_key_to_str(sos_attr_t attr, sos_key_t key)
{
	sos_idx_part_t part = __sos_active_idx_part(attr);
	char *keystr = malloc(ods_idx_key_str_size(part->index));
	return ods_key_to_str(part->index, key, keystr);
}

/**
 * \brief Compare two keys using the attribute index's compare function
 *
 * \param attr	The attribute handle
 * \param a	The first key
 * \param b	The second key
 * \return <0	a < b
 * \return 0	a == b
 * \return >0	a > b
 */
int sos_key_cmp(sos_attr_t attr, sos_key_t a, sos_key_t b)
{
	sos_idx_part_t part = __sos_active_idx_part(attr);
	return ods_key_cmp(part->index, a, b);
}

/**
 * \brief Return the size of the attribute index's key
 *
 * Returns the native size of the attribute index's key values. If the
 * key value is variable size, this function returns -1. See the sos_key_len()
 * and sos_key_size() functions for the current size of the key's
 * value and the size of the key's buffer respectively.
 *
 * \return The native size of the attribute index's keys in bytes
 */
size_t sos_attr_key_size(sos_attr_t attr)
{
	sos_idx_part_t part = __sos_active_idx_part(attr);
	return ods_idx_key_size(part->index);
}

/**
 * \brief Return the maximum size of this key's value
 *
 * \returns The size in bytes of this key's value buffer.
 */
size_t sos_key_size(sos_key_t key)
{
	return ods_key_size(key);
}

/**
 * \brief Return the length of the key's value
 *
 * Returns the current size of the key's value.
 *
 * \param key	The key
 * \returns The size of the key in bytes
 */
size_t sos_key_len(sos_key_t key)
{
	ods_key_value_t kv = ods_key_value(key);
	return kv->len;
}

/**
 * \brief Return the value of a key
 *
 * \param key	The key
 * \returns Pointer to the value of the key
 */
unsigned char* sos_key_value(sos_key_t key)
{
	ods_key_value_t kv = ods_key_value(key);
	return kv->value;
}
/** @} */
