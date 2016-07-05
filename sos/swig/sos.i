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
%module sos
%include "cpointer.i"
%include "cstring.i"
%{
#include <stdio.h>
#include <sys/queue.h>
#include <sos/sos.h>
#include "sos_priv.h"

#pragma GCC diagnostic ignored "-Wignored-qualifiers"

const char *py_value_as_str(sos_value_t value)
{
	static char buff[1024];
	if (!value)
		return "";
	return sos_value_to_str(value, buff, sizeof(buff));
}

int py_pos_from_str(sos_pos_t pos, const char *str)
{
	const char *src = str;
	int i;
	for (i = 0; i < sizeof(pos->data); i++) {
		int rc = sscanf(src, "%02hhX", &pos->data[i]);
		if (rc != 1)
			return EINVAL;
		src += 2;
	}
	return 0;
}

const char *py_pos_to_str(sos_pos_t pos)
{
	static char str[258];
	char *dst = str;
	int i;
	for (i = 0; i < sizeof(pos->data); i++) {
		sprintf(dst, "%02hhX", pos->data[i]);
		dst += 2;
	}
	return str;
}

%}
/* %include <inttypes.h> */
typedef short int16_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
%include <sys/queue.h>
%include "sos_priv.h"
%include <sos/sos.h>

const char *py_value_as_str(sos_value_t value);
int py_pos_from_str(sos_pos_t pos, const char *str);
const char *py_pos_to_str(sos_pos_t pos);

%extend sos_obj_ref_s {
	inline uint64_t ods() {
		return self->ref.ods;
	}
	inline uint64_t obj() {
		return self->ref.obj;
	}
}

%pythoncode %{
%}
