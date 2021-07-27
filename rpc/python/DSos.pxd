# Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
# Copyright (c) 2019 NTESS Corporation. All rights reserved.
# Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
# license for use of this work by or on behalf of the U.S. Government.
# Export of this program may require a license from the United States
# Government.
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the BSD-type
# license below:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#      Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#      Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
#      Neither the name of NTESS Corporation, Open Grid Computing nor
#      the names of any contributors may be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
#
#      Modified source versions must be plainly marked as such, and
#      must not be misrepresented as being the original software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cython
cimport numpy as np
from libc.stdint cimport *
from libc.string cimport *
from Sos cimport *
cdef extern from "dsos.h":

    cdef enum dsos_error:
        DSOS_ERR_OK,
        DSOS_ERR_MEMORY,
        DSOS_ERR_CLIENT,
        DSOS_ERR_SCHEMA,
        DSOS_ERR_ATTR,
        DSOS_ERR_ITER,
        DSOS_ERR_ITER_EMPTY,
        DSOS_ERR_QUERY_ID,
        DSOS_ERR_QUERY_EMPTY,
        DSOS_ERR_QUERY_BAD_SELECT,
        DSOS_ERR_PARAMETER,
        DSOS_ERR_TRANSPORT

    cdef struct dsos_result_s:
        int count
        dsos_error any_err
        dsos_error res[16]
    ctypedef dsos_result_s dsos_res_t

    cdef struct dsos_container_s:
        pass
    cdef struct dsos_session_s:
        pass
    cdef struct dsos_schema_s:
        pass
    cdef struct dsos_iter_s:
        pass
    cdef struct dsos_query_s:
        pass
    cdef struct dsos_session_s:
        pass
    ctypedef dsos_session_s *dsos_session_t
    ctypedef dsos_container_s *dsos_container_t
    ctypedef dsos_schema_s *dsos_schema_t
    ctypedef dsos_iter_s *dsos_iter_t
    ctypedef dsos_query_s *dsos_query_t
    cdef struct dsos_name_array_s:
        int count
        char **names
    ctypedef dsos_name_array_s *dsos_name_array_t

    void dsos_session_close(dsos_session_t sess)
    dsos_session_t dsos_session_open(const char *config_file)
    void dsos_container_close(dsos_container_t cont)
    void dsos_container_commit(dsos_container_t cont)
    dsos_container_t dsos_container_open(dsos_session_t sess, const char *path, sos_perm_t perm, int mode)
    dsos_schema_t dsos_schema_create(dsos_container_t cont, sos_schema_t schema, dsos_res_t *res)
    dsos_schema_t dsos_schema_by_name(dsos_container_t cont, const char *name, dsos_res_t *res)
    dsos_schema_t dsos_schema_by_uuid(dsos_container_t cont, uuid_t uuid, dsos_res_t *res)
    void dsos_transaction_begin(dsos_container_t cont, dsos_res_t *res)
    void dsos_transaction_end(dsos_container_t cont, dsos_res_t *res)
    void dsos_obj_create(dsos_container_t cont, dsos_schema_t schema, sos_obj_t obj, dsos_res_t *res)
    dsos_iter_t dsos_iter_create(dsos_container_t cont, dsos_schema_t schema, const char *attr_name)
    sos_obj_t dsos_iter_begin(dsos_iter_t iter)
    sos_obj_t dsos_iter_end(dsos_iter_t iter)
    sos_obj_t dsos_iter_next(dsos_iter_t iter)
    sos_obj_t dsos_iter_prev(dsos_iter_t iter)
    int dsos_iter_find_glb(dsos_iter_t iter, sos_key_t key)
    int dsos_iter_find_lub(dsos_iter_t iter, sos_key_t key)
    int dsos_iter_find(dsos_iter_t iter, sos_key_t key)
    int dsos_attr_value_min(dsos_container_t cont, sos_attr_t attr)
    int dsos_attr_value_max(dsos_container_t cont, sos_attr_t attr)
    dsos_query_t dsos_query_create(dsos_container_t cont)
    void dsos_query_destroy(dsos_query_t query)
    int dsos_query_select(dsos_query_t query, const char *clause)
    sos_obj_t dsos_query_next(dsos_query_t query)
    sos_schema_t dsos_query_schema(dsos_query_t query)


