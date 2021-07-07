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
from sosdb.DataSet import DataSet
# cimport numpy as np
from sosdb import Sos

cdef class Session(Sos.SosObject):
    cdef dsos_session_t c_session
    def __init__(self, config_file):
        self.c_session = dsos_session_open(config_file)

cdef class Container(SosObject):
    cdef dsos_container_t c_cont
    cdef object path_
    cdef sos_perm_t o_perm
    cdef int o_mode

    def __init__(self, session, path=None, o_perm=SOS_PERM_RW, o_mode=0660):
        SosObject.__init__(self)
        self.o_perm = o_perm
        self.o_mode = o_mode
        self.c_cont = NULL
        self.session = session
        if path:
            self.open(path, o_perm=o_perm, o_mode=o_mode)

    def path(self):
        return self.path_

    def version(self):
        """Return the container version information"""
        if self.c_cont != NULL:
            return sos_container_version(self.c_cont)
        return None

    def open(self, path, o_perm=SOS_PERM_RW, o_mode=0660, create=False, backend=SOS_BE_MMOS):
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
        self.c_cont = dsos_container_open(session.c_session, path.encode('utf-8'), <sos_perm_t>o_perm, o_mode)
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
        cdef int c_perm = SOS_PERM_CREAT | SOS_PERM_RW
        cdef int c_mode = o_mode
        if self.c_cont != NULL:
            self.abort(EBUSY)
        self.path_ = path
        if backend == SOS_BE_LSOS:
            c_perm |= SOS_BE_LSOS
        c_cont = dsos_container_open(self.c_session, path.encode(), <sos_perm_t>c_perm, c_mode)
        if c_cont == NULL:
            raise self.abort(errno)
        sos_container_close(c_cont, SOS_COMMIT_SYNC)

    def close(self, commit=SOS_COMMIT_ASYNC):
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
        sos_container_close(self.c_cont, commit)
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
        cdef int rc
        rc = dsos_container_commit(self.c_cont)
        if rc != 0:
            self.abort(rc)

    def schema_by_name(self, name):
        """Return the named schema

        Positional Parameters:

        - The name of the partition

        Returns:

        A Schema object, or None if the named schema does not exist in
        the container.

        """
        cdef dsos_schema_t c_schema = dsos_schema_by_name(self.c_cont, name.encode())
        if c_schema != NULL:
            s = Sos.Schema()
            s.assign(c_schema)
            return s
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
        cdef dsos_schema_t c_schema = dsos_schema_by_uuid(self.c_cont, uuid_.bytes)
        if c_schema != NULL:
            s = Schema()
            s.assign(c_schema)
            return s
        return None

cdef class Query:
    QUERY_RESULT_LIMIT = 4096
    cdef Container cont     # The container
    cdef dsos_query_t c_query;

    def __init__(self, container):
        """Implements a Query interface to the SOS container.

        The Query API roughly mirrors an SQL select statement. There
        are API for each of the fundamental clauses in an SQL select
        statement, i.e. select(), where(), from_(), and order_by().

        select() - identifies which attributes (columns) will be returned
        from_() - specifies which schema (tables) are being queried
        where()  - specifies the object(row) selection criteria
        order_by() - specifies the index being searched

        Positional Parameters:
        container -- The Container to query
        """
        self.cont = container
        self.c_query = dsos_query_create(self.cont.c_session);

    def select(self, sql):
        cdef dsos_res_t res;
        self.c_query = dsos_query_select(self.c_query, sql.encode('utf-8'));
