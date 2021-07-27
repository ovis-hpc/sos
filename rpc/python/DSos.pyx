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
from builtins import str
from cpython cimport PyObject, Py_INCREF
from libc.stdint cimport *
from libc.stdlib cimport calloc, malloc, free, realloc
import datetime as dt
import numpy as np
import struct
import sys
import copy
import binascii
import uuid
import pandas as pd
cimport numpy as np
# cimport Sos
from sosdb import Sos

np.import_array()

cdef class Session:
    cdef dsos_session_t c_session
    def __init__(self, config_file):
        self.c_session = dsos_session_open(config_file.encode())
        if self.c_session == NULL:
            raise ValueError("The cluster defined in {0} " \
                             "could not be attached.".format(config_file))

    def open(self, path, o_perm=SOS_PERM_RW, o_mode=0660):
        cdef dsos_container_t c_cont
        c_cont = dsos_container_open(<dsos_session_t>self.c_session,
                                     path.encode('utf-8'),
                                     <sos_perm_t>o_perm, o_mode)
        if c_cont == NULL:
            raise ValueError(f"The container {path} could not be opened")

        cont = Container(self)
        cont.assign(c_cont)
        cont.assign_session(self.c_session)
        return cont

cdef class Container:
    cdef object session
    cdef dsos_container_t c_cont
    cdef dsos_session_t c_sess
    cdef object path_
    cdef sos_perm_t o_perm
    cdef int o_mode

    cdef assign(self, dsos_container_t c_cont):
        self.c_cont = c_cont
        return self

    cdef assign_session(self, dsos_session_t c_sess):
        self.c_sess = c_sess
        return self

    def __init__(self, session):
        self.session = <Session>session

    def path(self):
        return self.path_

    def open(self, path, o_perm=SOS_PERM_RW, o_mode=0660, create=False,
             backend=SOS_BE_MMOS):
        """Open the container

        If the container cannot be opened (or created) an Exception
        is thrown with an error string based on the errno. Note that if
        the container does not exist and PERM_CREAT is not specified, the
        exception will indicate File Not Found.

        Positional Parameters:

        - The path to the container

        Keyword Parameters:

        o_perm - The permisions, one of SOS_PERM_RW or SOS_PERM_RO
        o_mode - The file creation mode if o_perm includes SOS_PERM_CREAT
        create - True to create the container if it does not exist
        backend - One of BE_MMOS (Memory Mapped Object Store) or
                  BE_LSOS (Log Structured Object Store)
        """
        if self.c_cont != NULL:
            self.abort(EBUSY)
        self.path_ = path
        if create:
            o_perm |= SOS_PERM_CREAT
        if backend == SOS_BE_LSOS:
            o_perm |= SOS_BE_LSOS
        self.o_perm = o_perm | backend
        self.o_mode = o_mode
        self.c_cont = dsos_container_open(<dsos_session_t>self.session.c_session,
                                          path.encode('utf-8'),
                                          <sos_perm_t>o_perm, o_mode)
        if self.c_cont == NULL:
            raise self.abort(errno)

    def create(self, path, o_mode=0660, backend=SOS_BE_MMOS):
        """Create the container

        This is a convenience method that calls open with
        o_perm |= SOS_PERM_CREAT. See the open() method for
        more information.

        Positional Parameters:

        - The path to the container

        Keyword Parameters:

        o_mode - The file creation mode, default is 0660
        """
        cdef dsos_container_t c_cont
        cdef int c_perm = SOS_PERM_CREAT | SOS_PERM_RW
        cdef int c_mode = o_mode
        if self.c_cont != NULL:
            self.abort(EBUSY)
        self.path_ = path
        if backend == SOS_BE_LSOS:
            c_perm |= SOS_BE_LSOS
        c_cont = dsos_container_open(<dsos_session_t>self.session.c_session,
                                     path.encode(),
                                     <sos_perm_t>c_perm, c_mode)
        if c_cont == NULL:
            raise self.abort(errno)
        dsos_container_close(c_cont)

    def close(self):
        """Close a container

        Closes the container. If the container is not open, an
        exception is thrown.

        if the 'commit' keyword parameter is set to SOS_COMMIT_ASYNC,
        the method will not return until all outstanding data has been
        written to stable storage.

        Keyword Parameters:

        commit - SOS_COMMIT_SYNC or SOS_COMMIT_ASYNC, The default is
                 SOS_COMMIT_ASYNC.

        """
        if self.c_cont == NULL:
            self.abort(EINVAL)
        dsos_container_close(self.c_cont)
        self.c_cont = NULL

    def commit(self):
        """Commit objects in memory to storage

        Commits outstanding data to stable storage. If the 'commit'
        keyword parameter is set to SOS_COMMIT_SYNC, the method will
        not return until all oustanding data has been written to
        storage.

        Keyword Parameters:

        commit - SOS_COMMIT_ASYNC (default) or SOS_COMMIT_SYNC
        """
        dsos_container_commit(self.c_cont)

    def schema_by_name(self, name):
        """Return the named schema

        Positional Parameters:

        - The name of the partition

        Returns:

        A Schema object, or None if the named schema does not exist in
        the container.

        """
        cdef dsos_res_t c_res
        cdef dsos_schema_t c_schema = \
            dsos_schema_by_name(<dsos_container_t>self.c_cont, name.encode(), &c_res)
        # if c_schema != NULL:
        #     s = Sos.Schema()
        #     s.assign(c_schema)
        #     return s
        return None

    def schema_by_uuid(self, uuid_):
        """Return the Schema with the specified 'id'
        Every schema has a unique 16B identifier that is stored with
        every Object of that Schema. See uuid_generate(3).

        Positional Parameters:

        - The unique schema id.

        Returns:

        The Schema with the specified id, or None if no schema with
        that id exists.

        """
        cdef dsos_res_t c_res
        cdef dsos_schema_t c_schema = \
            dsos_schema_by_uuid(<dsos_container_t>self.c_cont, uuid_.bytes, &c_res)
        # if c_schema != NULL:
        #     s = Schema()
        #     s.assign(c_schema)
        #     return s
        return None

    def query(self, sql, options=None):
        cdef dsos_query_t c_query = dsos_query_create(self.c_cont)
        cdef sos_schema_t c_schema
        cdef sos_obj_t c_obj
        cdef int c_rc
        cdef int c_rec_count
        cdef sos_attr_t c_attr;
        cdef char *c_str = <char *>malloc(1024)
        c_rc = dsos_query_select(c_query, sql.encode())
        if c_rc != 0:
                print(f"Error {c_rc} returned by select clause '{sql}'")
                return c_rc
        c_schema = dsos_query_schema(c_query)
        col_name_list = []
        col_id_list = []
        c_attr = sos_schema_attr_first(c_schema)
        while c_attr != NULL:
            if sos_attr_type(c_attr) != SOS_TYPE_JOIN:
                col_name_list.append(sos_attr_name(c_attr))
                col_id_list.append(sos_attr_id(c_attr))
            c_attr = sos_schema_attr_next(c_attr)

        c_rec_count = 0
        c_obj = dsos_query_next(c_query)
        c_rc = 0
        while c_obj != NULL:
            for i in col_id_list:
                sos_obj_attr_by_id_to_str(c_obj, <int>col_id_list[i], c_str, 1024)
                print(f"{c_str.decode()} ", end="")
            print("")
            sos_obj_put(c_obj)
            c_rec_count += 1
            c_obj = dsos_query_next(c_query)

        free(c_str)
        return 0;

ctypedef void (*nda_setter_fn_t)(np.ndarray nda, int idx, sos_value_t v)
ctypedef void (*nda_resample_fn_t)(np.ndarray nda, int idx, sos_value_t v,
                                   double bin_samples, double bin_width)
cdef struct nda_setter_opt_s:
    sos_attr_t attr             # attribute in schema
    int idx                     # index of this object in objects[]
    nda_setter_fn_t setter_fn
ctypedef nda_setter_opt_s *nda_setter_opt

cdef void int16_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    nda[idx] = v.data.prim.int16_

cdef void int32_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    nda[idx] = v.data.prim.int32_

cdef void int64_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    nda[idx] = v.data.prim.int64_

cdef void uint16_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    # have to force unsigned to double
    nda[idx] = <double>v.data.prim.uint16_

cdef void uint32_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    # have to force unsigned to double
    nda[idx] = <double>v.data.prim.uint32_

cdef void uint64_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    nda[idx] = <double>v.data.prim.uint64_

cdef void float_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    nda[idx] = v.data.prim.float_

cdef void double_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    nda[idx] = v.data.prim.double_

cdef void timestamp_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    # destination numpy data type is assumed to be datetime64[us]
    nda[idx] = (v.data.prim.timestamp_.tv.tv_sec * 1000000L) + \
            v.data.prim.timestamp_.tv.tv_usec

cdef void struct_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    nda[idx] = bytearray(v.data.prim.struc_[:sos_attr_size(v.attr)])

cdef void uint8_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = v.data.array.data.byte_[i]

cdef void int8_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    nda[idx] = v.data.array.data.char_[:v.data.array.count]

cdef void int16_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = v.data.array.data.int16_[i]

cdef void int32_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = v.data.array.data.int32_[i]

cdef void int64_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = v.data.array.data.int64_[i]

cdef void uint16_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = <double>v.data.array.data.uint16_[i]

cdef void uint32_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = <double>v.data.array.data.uint32_[i]

cdef void uint64_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = <double>v.data.array.data.uint64_[i]

cdef void float_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = v.data.array.data.float_[i]

cdef void double_array_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = v.data.array.data.double_[i]

cdef void unsupp_nda_setter(np.ndarray nda, int idx, sos_value_t v):
    raise ValueError("The attribute {0} has an unsupported data type."
                     .format(str(sos_attr_name(v.attr))))

cdef nda_setter_fn_t *dsos_nda_setters = [
    int16_nda_setter,
    int32_nda_setter,
    int64_nda_setter,
    uint16_nda_setter,
    uint32_nda_setter,
    uint64_nda_setter,
    float_nda_setter,
    double_nda_setter,
    unsupp_nda_setter,          # long double
    timestamp_nda_setter,
    unsupp_nda_setter,          # obj
    struct_nda_setter,          # struct
    unsupp_nda_setter,          # join
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,     # twenty zeros to get to 32, i.e. SOS_TYPE_BYTE_ARRAY
    uint8_array_nda_setter,     # byte_array
    int8_array_nda_setter,      # char_array
    int16_array_nda_setter,
    int32_array_nda_setter,
    int64_array_nda_setter,
    uint16_array_nda_setter,
    uint32_array_nda_setter,
    uint64_array_nda_setter,
    float_array_nda_setter,
    double_array_nda_setter,
    unsupp_nda_setter,          # long double
    unsupp_nda_setter,          # obj array
]

cdef class Query:
    DEFAULT_ARRAY_LIMIT = 256
    cdef Container cont
    cdef int c_start
    cdef int c_row_limit
    cdef int c_col_count
    cdef int c_row_count
    cdef int c_max_array
    cdef object schema
    cdef object result
    cdef sos_obj_t *c_objects
    cdef sos_schema_t c_schema
    cdef dsos_query_t c_query
    cdef nda_setter_opt c_res_acc

    def __init__(self, Container cont, int row_limit, int max_array=256):
        self.cont = cont
        self.c_row_limit = row_limit
        self.c_col_count = 0
        self.c_row_count = 0
        self.c_max_array = max_array
        self.c_res_acc = NULL
        self.c_query = dsos_query_create(<dsos_container_t>self.cont.c_cont)
        if self.c_query == NULL:
            raise ValueError("The query could not be created.")

    def select(self, sql, options=None):
        cdef int c_rc
        cdef sos_attr_t c_attr;
        cdef int c_col_no

        c_rc = dsos_query_select(<dsos_query_t>self.c_query, sql.encode())
        if c_rc != 0:
            raise ValueError(f"Error {c_rc} returned by select clause '{sql}'")
        self.c_schema = dsos_query_schema(self.c_query)
        if self.c_schema == NULL:
            raise ValueError("The select returned no schema")

        self.c_col_count = 0
        c_col_no = 0
        c_attr = sos_schema_attr_first(self.c_schema)
        while c_attr != NULL:
            if sos_attr_type(c_attr) != SOS_TYPE_JOIN:
                c_col_no += 1
            c_attr = sos_schema_attr_next(c_attr)
        if c_col_no == 0:
            raise ValueError("The query schema contains no attributes")
        self.c_col_count = c_col_no
        self.c_res_acc = <nda_setter_opt>malloc(sizeof(nda_setter_opt_s) * self.c_col_count)
        if self.c_res_acc == NULL:
            raise MemoryError("Insufficient memory")

        self.c_objects = <sos_obj_t *>malloc(self.c_row_limit * sizeof(sos_obj_t))
        if self.c_objects == NULL:
            raise MemoryError("Insufficient memory")

    def next(self):
        """Return a dataframe for the next batch of query results"""
        cdef sos_obj_t c_o
        cdef sos_value_s v_
        cdef sos_value_t v
        cdef int col_no
        cdef int row_idx

        # read up to c_row_count objects into c_objects
        self.c_row_count = 0
        for row_idx in range(0, self.c_row_limit):
            c_o = dsos_query_next(self.c_query)
            if c_o == NULL:
                break
            self.c_objects[row_idx] = c_o
        self.c_row_count = row_idx
        if self.c_row_count == 0:
            return None
        self.result = []
        c_attr = sos_schema_attr_first(self.c_schema)
        c_col_no = 0
        while c_attr != NULL:
            atyp = <int>sos_attr_type(c_attr)
            if atyp == SOS_TYPE_JOIN:
                c_attr = sos_schema_attr_next(c_attr)
                continue
            self.c_res_acc[c_col_no].attr = c_attr
            self.c_res_acc[c_col_no].setter_fn = dsos_nda_setters[atyp]
            if atyp == SOS_TYPE_TIMESTAMP:
                data = np.zeros([ self.c_row_count ], dtype=np.dtype('datetime64[us]'))
            elif atyp == SOS_TYPE_STRUCT:
                data = np.zeros([ self.c_row_count, sos_attr_size(self.c_res_acc[c_col_no].attr) ],
                                dtype=np.dtype(np.uint8))
            elif atyp == SOS_TYPE_FLOAT:
                data = np.zeros([ self.c_row_count ], dtype=np.dtype(np.float32))
            elif atyp == SOS_TYPE_DOUBLE:
                data = np.zeros([ self.c_row_count ], dtype=np.dtype(np.float64))
            elif atyp == SOS_TYPE_UINT64:
                data = np.zeros([ self.c_row_count ], dtype=np.dtype(np.uint64))
            elif atyp == SOS_TYPE_UINT32:
                data = np.zeros([ self.c_row_count ], dtype=np.dtype(np.uint32))
            elif atyp == SOS_TYPE_INT64:
                data = np.zeros([ self.c_row_count ], dtype=np.dtype(np.int64))
            elif atyp == SOS_TYPE_INT32:
                data = np.zeros([ self.c_row_count ], dtype=np.dtype(np.int32))
            elif atyp == SOS_TYPE_BYTE_ARRAY:
                data = np.zeros([ self.c_row_count ],
                                dtype=np.dtype('U{0}'.format(self.c_max_array)))
            elif atyp == SOS_TYPE_CHAR_ARRAY:
                data = np.zeros([ self.c_row_count ],
                                dtype=np.dtype('U{0}'.format(self.c_max_array)))
            elif atyp == SOS_TYPE_INT16_ARRAY:
                data = np.zeros([ self.c_row_count, self.c_max_array ],
                                dtype=np.dtype(np.int16))
            elif atyp == SOS_TYPE_INT32_ARRAY:
                data = np.zeros([ self.c_row_count, self.c_max_array ],
                                dtype=np.dtype(np.int32))
            elif atyp == SOS_TYPE_INT64_ARRAY:
                data = np.zeros([ self.c_row_count, self.c_max_array ],
                                dtype=np.dtype(np.int64))
            elif atyp == SOS_TYPE_UINT16_ARRAY:
                data = np.zeros([ self.c_row_count, self.c_max_array ],
                                dtype=np.dtype(np.uint16))
            elif atyp == SOS_TYPE_UINT32_ARRAY:
                data = np.zeros([ self.c_row_count, self.c_max_array ],
                                dtype=np.dtype(np.uint32))
            elif atyp == SOS_TYPE_UINT64_ARRAY:
                data = np.zeros([ self.c_row_count, self.c_max_array ],
                                dtype=np.dtype(np.uint64))
            elif atyp == SOS_TYPE_FLOAT_ARRAY:
                data = np.zeros([ self.c_row_count, self.c_max_array ],
                                dtype=np.dtype(np.float32))
            elif atyp == SOS_TYPE_DOUBLE_ARRAY:
                data = np.zeros([ self.c_row_count, self.c_max_array ],
                                dtype=np.dtype(np.float64))
            else:
                continue
                # raise ValueError(f"Invalid attribute type {atyp} for DataFrame encoding")
            self.result.append(data)
            c_attr = sos_schema_attr_next(c_attr)
            c_col_no += 1

        for row_idx in range(0, self.c_row_count):
            for col_no in range(0, self.c_col_count):
                v = sos_value_init(&v_,
                                   self.c_objects[row_idx],
                                   self.c_res_acc[col_no].attr)
                self.c_res_acc[col_no].setter_fn(self.result[col_no], row_idx, v)
                sos_value_put(v)
            sos_obj_put(self.c_objects[row_idx])
            self.c_objects[row_idx] = NULL
            self.c_row_count += 1
        pdres = {}
        df_idx = None
        for col_no in range(0, self.c_col_count):
            col_name = sos_attr_name(self.c_res_acc[col_no].attr).decode()
            pdres[col_name] = self.result[col_no]
            if 'timestamp' == col_name:
                df_idx = pd.DatetimeIndex(self.result[col_no])
        if df_idx is not None:
            res = pd.DataFrame(pdres, index=df_idx)
        else:
            res = pd.DataFrame(pdres)
        return res


    @property
    def capacity(self):
        """Return the row capacity of the result"""
        return self.c_row_limit

    @property
    def count(self):
        """Return the number of rows in the result"""
        return self.c_row_count

    def __len__(self):
        """Return the number of rows in the result"""
        return self.c_row_count

    def __dealloc__(self):
        free(self.c_res_acc)
