# Copyright (c) 2020 Open Grid Computing, Inc. All rights reserved.
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
cimport numpy as np
import pandas as pd

#
# Initialize the numpy array support. Numpy arrays are used
# for all SOS arrays. The array data is therefore not copied
# when accessed from Python
#
np.import_array()

libc_errno_str = {
    E2BIG : "Argument list too long",
    EACCES : "Permission denied",
    EADDRINUSE : "Address already in use",
    EADDRNOTAVAIL : "Address not available",
    EAFNOSUPPORT : "Address family not supported",
    EAGAIN : "Resource temporarily unavailable",
    EALREADY : "Connection already in progress",
    EBADE : "Invalid exchange",
    EBADF : "Bad file descriptor",
    EBADFD : "File descriptor in bad state",
    EBADMSG : "Bad message",
    EBADR : "Invalid request descriptor",
    EBADRQC : "Invalid request code",
    EBADSLT : "Invalid slot",
    EBUSY : "Device or resource busy",
    ECHILD : "No child processes",
    ECHRNG : "Channel number out of range",
    ECOMM : "Communication error on send",
    ECONNABORTED : "Connection aborted",
    ECONNREFUSED : "Connection refused",
    ECONNRESET : "Connection reset",
    EDEADLK : "Resource deadlock avoided",
    EDEADLOCK : "Synonym for EDEADLK",
    EDESTADDRREQ : "Destination address required",
    EDOM : "Mathematics argument out of domain of function",
    EDQUOT : "Disk quota exceeded",
    EEXIST : "File exists",
    EFAULT : "Bad address",
    EFBIG : "File too large",
    EHOSTDOWN : "Host is down",
    EHOSTUNREACH : "Host is unreachable",
    EIDRM : "Identifier removed",
    EILSEQ : "Illegal byte sequence (POSIX.1, C99)",
    EINPROGRESS : "Operation in progress",
    EINTR : "Interrupted function call ; see signal(7).",
    EINVAL : "Invalid argument",
    EIO : "Input/output error",
    EISCONN : "Socket is connected",
    EISDIR : "Is a directory",
    EISNAM : "Is a named type file",
    ELOOP : "Too many levels of symbolic links",
    EMFILE : "Too many open files",
    EMLINK : "Too many links",
    EMSGSIZE : "Message too long",
    EMULTIHOP : "Multihop attempted",
    ENAMETOOLONG : "Filename too long",
    ENETDOWN : "Network is down",
    ENETRESET : "Connection aborted by network",
    ENETUNREACH : "Network unreachable",
    ENFILE : "Too many open files in system",
    ENOBUFS : "No buffer space available",
    ENODATA : "No message is available on the STREAM head read queue",
    ENODEV : "No such device",
    ENOENT : "No such file or directory",
    ENOEXEC : "Exec format error",
    ENOLCK : "No locks available",
    ENOLINK : "Link has been severed",
    ENOMEM : "Not enough space",
    ENOMSG : "No message of the desired type",
    ENONET : "Machine is not on the network",
    ENOPKG : "Package not installed",
    ENOPROTOOPT : "Protocol not available",
    ENOSPC : "No space left on device",
    ENOSR : "No STREAM resources (POSIX.1 (XSI STREAMS option))",
    ENOSTR : "Not a STREAM (POSIX.1 (XSI STREAMS option))",
    ENOSYS : "Function not implemented",
    ENOTBLK : "Block device required",
    ENOTCONN : "The socket is not connected",
    ENOTDIR : "Not a directory",
    ENOTEMPTY : "Directory not empty",
    ENOTSOCK : "Not a socket",
    ENOTSUP : "Operation not supported",
    ENOTTY : "Inappropriate I/O control operation",
    ENOTUNIQ : "Name not unique on network",
    ENXIO : "No such device or address",
    EOPNOTSUPP : "Operation not supported on socket",
    EOVERFLOW : "Value too large to be stored in data type",
    EPERM : "Operation not permitted",
    EPFNOSUPPORT : "Protocol family not supported",
    EPIPE : "Broken pipe",
    EPROTO : "API or Protocol version error",
    EPROTONOSUPPORT : "Protocol not supported",
    EPROTOTYPE : "Protocol wrong type for socket",
    ERANGE : "Result too large (POSIX.1, C99)",
    EREMCHG : "Remote address changed",
    EREMOTE : "Object is remote",
    EREMOTEIO : "Remote I/O error",
    ERESTART : "Interrupted system call should be restarted",
    EROFS : "Read-only filesystem",
    ESHUTDOWN : "Cannot send after transport endpoint shutdown",
    ESPIPE : "Invalid seek",
    ESOCKTNOSUPPORT : "Socket type not supported",
    ESRCH : "No such process",
    ESTALE : "Stale file handle",
    ESTRPIPE : "Streams pipe error",
    ETIME : "Timer expired",
    ETIMEDOUT : "Connection timed out",
    EUSERS : "Too many users",
    EWOULDBLOCK : "Operation would block",
    EXFULL : "Exchange full"
}

cdef class SosObject:
    cdef int error
    def __init__(self):
        self.error = 0
    def errno(self):
        return self.error
    def errstr(self):
        return libc_errno_str(self.error)
    def abort(self, error=None):
        if error:
            self.error = error
        else:
            self.error = errno
        if self.error in libc_errno_str:
            raise Exception(libc_errno_str[self.error])
        else:
            raise Exception("Error {0}".format[self.error])

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

        cont = DsosContainer(self)
        cont.assign(c_cont)
        cont.assign_session(self.c_session)
        return cont

cdef class DsosContainer:
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

        o_perm - The permisions, one of SOS_PERM_RW or SOS_PERM_RD
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

    def schema_query(self):
        cdef int i
        cdef dsos_name_array_t schemas = dsos_schema_query(self.c_cont);
        names = []
        for i in range(0, schemas.count):
            name = schemas.names[i]
            name = name.encode()
            names.append(name)
        dsos_name_array_free(schemas);
        return names

    def schema_by_name(self, name):
        """Return the named schema

        Positional Parameters:

        - The name of the schema

        Returns:

        A Schema object, or None if the named schema does not exist in
        the container.

        """
        cdef dsos_schema_t c_schema = \
            dsos_schema_by_name(<dsos_container_t>self.c_cont, name.encode())
        if c_schema != NULL:
            s = Schema()
            s.c_schema = dsos_schema_local(c_schema)
            return s

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
        cdef dsos_schema_t c_schema = \
            dsos_schema_by_uuid(<dsos_container_t>self.c_cont, uuid_.bytes)
        if c_schema != NULL:
            s = Schema()
            s.c_schema = dsos_schema_local(c_schema)
            return s
        return None

    def part_query(self):
        """Query the list of partition names in the container

        Returns the list of names of partitions of which the user has at
        least read permission.

        Use the part_by_name() interface to get detailed information for
        each partition.
        """
        cdef int i
        cdef dsos_name_array_t parts = dsos_part_query(self.c_cont);
        names = []
        for i in range(0, parts.count):
            name = parts.names[i]
            name = name.encode()
            names.append(name)
        dsos_name_array_free(parts);
        return names

    def part_by_name(self, name):
        """Query the container for a partition

        If the user has read-access to the partition, a DsosPartition
        object is returned containing the detail informaiton for the
        partition.

        Parameters:
            - The name of the partition

        Returns:
            DsosParition objet or None if the partition does not exist, or
            for which the does not have read permission
        """
        cdef dsos_part_t c_part = \
            dsos_part_by_name(<dsos_container_t>self.c_cont, name.encode())
        if c_part != NULL:
            p = DsosPartition()
            p.c_part = c_part
            return p
        return None

    def part_by_uuid(self, name):
        """Query the container for a partition by UUID

        If the user has read-access to the partition, a DsosPartition
        object is returned containing the detail informaiton for the
        partition.

        Parameters:
            - The Universally Unique ID (UUID) of the partition

        Returns:
            DsosParition objet or None if the partition does not exist, or
            for which the does not have read permission
        """
        cdef dsos_part_t c_part = \
            dsos_part_by_uuid(<dsos_container_t>self.c_cont, uuid.bytes)
        if c_part != NULL:
            p = DsosPartition()
            p.c_part = c_part
            return p
        return None

    def query(self, sql, options=None):
        cdef dsos_query_t c_query = dsos_query_create(self.c_cont)
        cdef sos_schema_t c_schema
        cdef sos_obj_t c_obj
        cdef int c_rc
        cdef int c_rec_count
        cdef sos_attr_t c_attr
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
        return 0

cdef class DsosPartition:
    cdef dsos_part_t c_part
    def __init__(self):
        self.c_part = NULL

    def name(self):
        """Return the partition name"""
        return dsos_part_name(self.c_part)

    def desc(self):
        """Return the parition description"""
        return dsos_part_desc(self.c_part)

    def path(self):
        """Return the path to the paritition"""
        return dsos_part_path(self.c_part)

    def uid(self):
        """Returns the partition user-id"""
        return dsos_part_uid(self.c_part)

    def gid(self):
        """Returns the partition group-id"""
        return dsos_part_gid(self.c_part)

    def perm(self):
        """Returns the partition's permission bits"""
        return dsos_part_perm(self.c_part)

cdef class SchemaIter(SosObject):
    """Implements a Schema iterator

    Example:

    # print names of all the in the container
    for schema in container.schema_iter():
        print(schema.name())

    """
    cdef sos_schema_t c_next_schema
    def __init__(self, Container cont):
        self.c_next_schema = sos_schema_first(cont.c_cont)

    def __iter__(self):
        return self

    def __next__(self):
        if self.c_next_schema == NULL:
            raise StopIteration
        s = Schema()
        s.c_schema = self.c_next_schema # s.assign(self.c_next_schema)
        self.c_next_schema = sos_schema_next(self.c_next_schema)
        return s

cdef class PartIter(SosObject):
    """Implements a Partition iterator

    Example:

    # print names of all the partitions
    for part in container.part_iter():
        print(part.name())

    """
    cdef sos_part_iter_t c_iter
    cdef sos_part_t c_part
    def __init__(self, Container cont):
        self.c_iter = sos_part_iter_new(cont.c_cont)
        if self.c_iter == NULL:
            raise MemoryError("Could not allocate a new partition iterator.")
        self.c_part = NULL

    def __iter__(self):
        self.c_part = sos_part_first(self.c_iter)
        return self

    def __next__(self):
        if self.c_part == NULL:
            raise StopIteration
        p = Partition()
        p.assign(self.c_part)
        self.c_part = sos_part_next(self.c_iter)
        return p

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        if self.c_iter != NULL:
            sos_part_iter_free(self.c_iter)
            self.c_iter = NULL

cdef class IndexIter(SosObject):
    """Implements an Index iterator

    Example:

    # print names of all the indices in the container
    for idx in container.index_iter():
        print(idx.name())

    """
    cdef sos_container_index_iter_t c_iter
    cdef sos_index_t c_idx
    def __init__(self, Container cont):
        self.c_iter = sos_container_index_iter_new(cont.c_cont)
        if self.c_iter == NULL:
            raise MemoryError("Could not allocate a new container iterator.")
        self.c_idx = NULL

    def __iter__(self):
        self.c_idx = sos_container_index_iter_first(self.c_iter)
        return self

    def __next__(self):
        if self.c_idx == NULL:
            raise StopIteration
        idx = Index()
        idx.assign(self.c_idx)
        self.c_idx = sos_container_index_iter_next(self.c_iter)
        return idx

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        if self.c_iter != NULL:
            sos_container_index_iter_free(self.c_iter)
            self.c_iter = NULL

cdef class Container(SosObject):
    cdef sos_t c_cont
    cdef object path_
    cdef sos_perm_t o_perm
    cdef int o_mode

    def __init__(self, path=None, o_perm=SOS_PERM_RW, o_mode=0660):
        SosObject.__init__(self)
        self.o_perm = o_perm
        self.o_mode = o_mode
        self.c_cont = NULL
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

        o_perm - The permisions, one of SOS_PERM_RW or SOS_PERM_RD
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
        self.c_cont = sos_container_open(path.encode('utf-8'), <sos_perm_t>o_perm, o_mode)
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
        c_cont = sos_container_open(path.encode(), <sos_perm_t>c_perm, c_mode)
        if c_cont == NULL:
            raise self.abort(errno)
        sos_container_close(c_cont, SOS_COMMIT_SYNC)

    def clone(self, clone_path, part_list=None, o_mode=0660):
        """Clone the schema and selected partitions into a new container

        If an error occurs, an exception is thrown.

        Positional Parameters:

        - The path to the new container

        Keyword Parameters:

        part_list - A list of partition names to clone into the new
            container. An empty list indicates that the new container
            is to be left empty. The value None (default) indicates
            that all partitions are to be cloned.
        o_mode - The file creation mode, default is 0660
        """
        if self.c_cont == NULL:
            self.abort(ENOENT)
        clone = Container()
        clone.open(clone_path, o_perm=SOS_PERM_CREAT|SOS_PERM_RW,o_mode=0660)
        for schema in self.schema_iter():
            dup = schema.dup()
            dup.add(clone)
        return clone

    def delete(self):
        """Delete a container

        Removes the container from the filesystem. If the container
        cannot be deleted an Exception is thrown with a message based
        on the errno.
        """
        cdef int rc
        if self.c_cont == NULL:
            self.abort(EINVAL)
        rc = sos_container_delete(self.c_cont)
        if rc != 0:
            self.abort(rc)

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

    def commit(self, commit=SOS_COMMIT_ASYNC):
        """Commit objects in memory to storage

        Commits outstanding data to stable storage. If the 'commit'
        keyword parameter is set to SOS_COMMIT_SYNC, the method will
        not return until all oustanding data has been written to
        storage.

        Keyword Parameters:

        commit - SOS_COMMIT_ASYNC (default) or SOS_COMMIT_SYNC
        """
        cdef int rc
        rc = sos_container_commit(self.c_cont, commit)
        if rc != 0:
            self.abort(rc)

    def part_create(self, name, desc=None, path=None):
        """Create a new partition and attach it to the container

        Positional Parameters:
        - The name to give the new partition in the container

        Keyword Parameters:
        desc - A description to assign to the new partition. This defaults to
               the partition name if not specified
        path - The path to the new partition. By default it is
               the container path '/' name
        """
        cdef int c_rc
        cdef sos_part_t c_part
        if desc is None:
            desc = name
        if path is None:
            path = self.path_ + '/' + name
        c_part = sos_part_open(path.encode(),
                            self.o_perm | SOS_PERM_CREAT,
                            self.o_mode,
                            desc.encode())
        if c_part == NULL:
            self.abort(errno)
        sos_part_put(c_part)
        c_rc = sos_part_attach(self.c_cont, name.encode(), path.encode())
        if c_rc != 0:
            self.abort(c_rc)

    def part_attach(self, name, path=None):
        """Attach a partition to the container

        Positional Parameters:
        - The name to give the new partition in the container

        Keyword Parameters:
        path - The path to the container. If not specified, the path
             is self.path + '/' + name
        """
        cdef int c_rc
        if path is None:
            path = self.path_ + '/' + name
        c_rc = sos_part_attach(self.c_cont, name.encode(), path.encode())
        if c_rc != 0:
            self.abort(c_rc)

    def part_detach(self, name):
        """Detach partition from the container"""
        cdef int c_rc = sos_part_detach(self.c_cont, name.encode())
        if c_rc != 0:
            self.abort(c_rc)

    def part_by_name(self, name):
        """Return the named partition

        Positional Parameters:

        - The name of the partition

        Returns:

        A Partition object, or None if the partition does not exist.
        """
        cdef sos_part_t c_part = sos_part_by_name(self.c_cont, name.encode())
        if c_part != NULL:
            p = Partition()
            p.assign(c_part)
            return p
        return None

    def part_by_path(self, path):
        """Return the partition at the specified path

        Positional Parameters:

        - The path to the partition

        Returns:

        A Partition object, or None if the partition does not exist.
        """
        cdef sos_part_t c_part = sos_part_by_path(self.c_cont, path.encode())
        if c_part != NULL:
            p = Partition()
            p.assign(c_part)
            return p
        return None

    def part_iter(self):
        """Return a PartIter iterator for the container"""
        return PartIter(self)

    def index_iter(self):
        """Return an IndexIter iterator for the container"""
        return IndexIter(self)

    def schema_by_name(self, name):
        """Return the named schema

        Positional Parameters:

        - The name of the partition

        Returns:

        A Schema object, or None if the named schema does not exist in
        the container.

        """
        cdef sos_schema_t c_schema = sos_schema_by_name(self.c_cont, name.encode())
        if c_schema != NULL:
            s = Schema()
            s.c_schema = c_schema #  s.assign(c_schema)
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
        cdef sos_schema_t c_schema = sos_schema_by_uuid(self.c_cont, uuid_.bytes)
        if c_schema != NULL:
            s = Schema()
            s.c_schema = c_schema # s.assign(c_schema)
            return s
        return None

    def schema_iter(self):
        """Return a SchemaIter iterator for the container"""
        return SchemaIter(self)

PART_STATE_OFFLINE = SOS_PART_STATE_OFFLINE
PART_STATE_PRIMARY = SOS_PART_STATE_PRIMARY
PART_STATE_ACTIVE = SOS_PART_STATE_ACTIVE
PART_STATE_BUSY = SOS_PART_STATE_BUSY

cdef class PartState(object):
    cdef int c_state
    def __init__(self, int state):
        self.c_state = state

    def __int__(self):
        return self.c_state

    def __long__(self):
        return self.c_state

    def __str__(self):
        if self.c_state == SOS_PART_STATE_OFFLINE:
            return "OFFLINE"
        elif self.c_state == SOS_PART_STATE_ACTIVE:
            return "ACTIVE"
        elif self.c_state == SOS_PART_STATE_PRIMARY:
            return "PRIMARY"
        elif self.c_state == SOS_PART_STATE_BUSY:
            return "BUSY"
        elif self.c_state == SOS_PART_STATE_DETACHED:
            return "DETACHED"
        raise ValueError("{0} is an invalid partition state".format(self.c_state))

cdef class PartStat(object):
    cdef sos_part_stat_s c_stat
    def __init__(self, Partition part):
        cdef int rc = sos_part_stat(part.c_part, &self.c_stat)

    @property
    def accessed(self):
        return self.c_stat.accessed

    @property
    def changed(self):
        return self.c_stat.changed

    @property
    def modified(self):
        return self.c_stat.modified

    @property
    def size(self):
        return self.c_stat.size

    def __str__(self):
        return str(self.c_stat)

cdef class Partition(SosObject):
    cdef sos_part_t c_part
    cdef object path_
    cdef object desc_

    def __init__(self):
        self.c_part = NULL
        self.path_ = None
        self.desc_ = None

    cdef assign(self, sos_part_t c_part):
        self.c_part = c_part
        self.path_ = sos_part_name(c_part)
        self.desc_ = sos_part_desc(c_part)
        return self

    def name(self):
        """Returns the partition name"""
        return sos_part_name(self.c_part).decode('utf-8')

    def path(self):
        """Returns the partition path"""
        return sos_part_path(self.c_part).decode('utf-8')

    def desc(self):
        """Returns the parition description"""
        return sos_part_desc(self.c_part).decode('utf-8')

    def uuid(self):
        """Returns the parition UUID"""
        cdef uuid_t c_uuid
        sos_part_uuid(self.c_part, c_uuid)
        return uuid.UUID(bytes=c_uuid)

    def desc_set(self, desc):
        """Set the paritition's description."""
        sos_part_desc_set(self.c_part, desc.encode())

    def chown(self, uid, gid):
        """Change the user-id and group-id for the partition"""
        sos_part_chown(self.c_part, <int>uid, <int>gid)

    def chmod(self, perm):
        """Change the access permissions to the container"""
        sos_part_chmod(self.c_part, <int>perm)

    def uid(self):
        """Return the container's user-id"""
        return sos_part_uid(self.c_part)

    def gid(self):
        """Return the container's group-id"""
        return sos_part_gid(self.c_part)

    def perm(self):
        """Return the container's access rights"""
        return sos_part_perm(self.c_part)

    def state(self):
        """Returns the partition state"""
        return PartState(sos_part_state(self.c_part))

    def stat(self):
        """Returns the partition PartStat (size, access time, etc...) information"""
        return PartStat(self)

    def attach(self, Container cont, name):
        """Attach partition to a container

        Adds the partition as \c name to the container \c cont. The
        partition state is OFFLINE once attached. The schema in the
        container must define all objects stored in the partition.

        Positional Parameters:
        - The container handle
        - The name for the partition in the container
        """
        cdef int c_rc
        if self.c_part == NULL:
            raise ValueError("The partition is not open")
        c_rc = sos_part_attach(cont.c_cont, name.encode(), self.path_.encode())
        if c_rc != 0:
            self.abort(c_rc)
        self.release()
        self.c_part =  sos_part_by_name(cont.c_cont, name.encode())
        if self.c_part == NULL:
            self.abort(errno)

    def open(self, path, o_perm=SOS_PERM_RW, o_mode=0660, desc=None, backend=SOS_BE_MMOS):
        """Open the partition at path

        Positional Arguments:
        -- The path to the partition

        Keyword Arguments:
        o_perm - The SOS access mode, the default is SOS_PERM_RW
        o_mode - The partition access rights, default is 0660. Only used
                if o_perm includes SOS_PART_CREAT and the partition does
                not already exist.
        desc   - If o_perm contains SOS_PERM_CREAT, this is the description
                given to the new partition
        """
        cdef int c_perm
        cdef int c_mode
        cdef sos_part_t c_part
        self.path_ = path
        if desc is None:
            self.desc_ = "".encode()
        else:
            self.desc_ = desc.encode()
        c_mode = o_mode
        c_perm = o_perm
        if backend == BE_LSOS:
            c_perm |= SOS_BE_LSOS
        c_part = sos_part_open(path.encode(), c_perm, o_mode, self.desc_)
        if c_part == NULL:
            self.abort(errno)
        self.c_part = c_part;

    def state_set(self, new_state):
        """Set the partition state"""
        cdef int rc
        cdef sos_part_state_t state
        if self.c_part == NULL:
            raise ValueError("The partition is not open")
        if new_state.upper() == "PRIMARY":
            state = SOS_PART_STATE_PRIMARY
        elif new_state.upper() == "ACTIVE":
            state = SOS_PART_STATE_ACTIVE
        elif new_state.upper() == "OFFLINE":
            state = SOS_PART_STATE_OFFLINE
        else:
            raise ValueError("Invalid partition state name {0}".format(new_state))
        rc = sos_part_state_set(self.c_part, state)
        if rc != 0:
            self.abort(rc)

    def release(self):
        if self.c_part:
            sos_part_put(self.c_part)
            self.c_part = NULL

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        self.release()

sos_type_strs = {
     SOS_TYPE_INT16 : "INT16",
     SOS_TYPE_INT32 : "INT32",
     SOS_TYPE_INT64 : "INT64",
     SOS_TYPE_UINT16 : "UINT16",
     SOS_TYPE_UINT32 : "UINT32",
     SOS_TYPE_UINT64 : "UINT64",
     SOS_TYPE_FLOAT : "FLOAT",
     SOS_TYPE_DOUBLE : "DOUBLE",
     SOS_TYPE_LONG_DOUBLE : "LONG_DOUBLE",
     SOS_TYPE_TIMESTAMP : "TIMESTAMP",
     SOS_TYPE_OBJ : "OBJ",
     SOS_TYPE_STRUCT : "STRUCT",
     SOS_TYPE_JOIN : "JOIN",
     SOS_TYPE_BYTE_ARRAY : "BYTE_ARRAY",
     SOS_TYPE_CHAR_ARRAY : "CHAR_ARRAY",
     SOS_TYPE_CHAR_ARRAY : "STRING",
     SOS_TYPE_INT16_ARRAY : "INT16_ARRAY",
     SOS_TYPE_INT32_ARRAY : "INT32_ARRAY",
     SOS_TYPE_INT64_ARRAY : "INT64_ARRAY",
     SOS_TYPE_UINT16_ARRAY : "UINT16_ARRAY",
     SOS_TYPE_UINT32_ARRAY : "UINT32_ARRAY",
     SOS_TYPE_UINT64_ARRAY : "UINT64_ARRAY",
     SOS_TYPE_FLOAT_ARRAY : "FLOAT_ARRAY",
     SOS_TYPE_DOUBLE_ARRAY : "DOUBLE_ARRAY",
     SOS_TYPE_LONG_DOUBLE_ARRAY : "LONG_DOUBLE_ARRAY",
     SOS_TYPE_OBJ_ARRAY : "OBJ_ARRAY",
}

sos_attr_types = {
    "INT16" : SOS_TYPE_INT16,
    "INT32" : SOS_TYPE_INT32,
    "INT64" : SOS_TYPE_INT64,
    "UINT16" : SOS_TYPE_UINT16,
    "UINT32" : SOS_TYPE_UINT32,
    "UINT64" : SOS_TYPE_UINT64,
    "FLOAT" : SOS_TYPE_FLOAT,
    "DOUBLE" : SOS_TYPE_DOUBLE,
    "LONG_DOUBLE" : SOS_TYPE_LONG_DOUBLE,
    "TIMESTAMP" : SOS_TYPE_TIMESTAMP,
    "OBJ" : SOS_TYPE_OBJ,
    "STRUCT" : SOS_TYPE_STRUCT,
    "JOIN" : SOS_TYPE_JOIN,
    "BYTE_ARRAY" : SOS_TYPE_BYTE_ARRAY,
    "CHAR_ARRAY" : SOS_TYPE_CHAR_ARRAY,
    "STRING" : SOS_TYPE_CHAR_ARRAY,
    "INT16_ARRAY" : SOS_TYPE_INT16_ARRAY,
    "INT32_ARRAY" : SOS_TYPE_INT32_ARRAY,
    "INT64_ARRAY" : SOS_TYPE_INT64_ARRAY,
    "UINT16_ARRAY" : SOS_TYPE_UINT16_ARRAY,
    "UINT32_ARRAY" : SOS_TYPE_UINT32_ARRAY,
    "UINT64_ARRAY" : SOS_TYPE_UINT64_ARRAY,
    "FLOAT_ARRAY" : SOS_TYPE_FLOAT_ARRAY,
    "DOUBLE_ARRAY" : SOS_TYPE_DOUBLE_ARRAY,
    "LONG_DOUBLE_ARRAY" : SOS_TYPE_LONG_DOUBLE_ARRAY,
    "OBJ_ARRAY" : SOS_TYPE_OBJ_ARRAY
}

cdef class Schema(SosObject):
    cdef sos_schema_t c_schema
    cdef sos_attr_t c_next_attr

    def __init__(self):
        SosObject.__init__(self)
        self.c_schema = NULL

    def attr_iter(self):
        return self.__iter__()

    def __iter__(self):
        self.c_next_attr = sos_schema_attr_first(self.c_schema)
        return self

    def __next__(self):
        if self.c_next_attr == NULL:
            raise StopIteration
        a = self.attr_by_id(sos_attr_id(self.c_next_attr))
        self.c_next_attr = sos_schema_attr_next(self.c_next_attr)
        return a

    def from_template(self, template):
        """Create a schema from a template specification

        The template parameter defines a SOS schema. The format is
        as follows:

        {
          "name" : "<schema-name>",
          "uuid" : "<uuid>",
          "attrs" : [ <attr-def ]
        }

        The <schema-name> must be unique within the container. The
        <uuid> is a globally unique identifier for the schema. The
        uuidgen command can be used to generate these values. Each
        <attr-def> is a dictionary defining a SOS schema attribute as
        follows:

        {
            "name" : "<attr-name>",
            "type" : "<attr-type>",
            "index" : { "type": "<idx-type>", },
            "join_attrs" : [ "<attr-name>", "<attr-name>", ... ]
        }

        The name entry specifies the name of the attribute in the
        schema and must be unique within the schema.  The valid type
        names are as follows:
            - "INT16"
            - "INT32"
            - "INT64"
            - "UINT16"
            - "UINT32"
            - "UINT64"
            - "FLOAT"
            - "DOUBLE",
            - "LONG_DOUBLE"
            - "TIMESTAMP"
            - "OBJ"
            - "STRUCT"
            - "JOIN"
            - "BYTE_ARRAY"
            - "CHAR_ARRAY"
            - "STRING"
            - "INT16_ARRAY"
            - "INT32_ARRAY"
            - "INT64_ARRAY"
            - "UINT16_ARRAY"
            - "UINT32_ARRAY"
            - "UINT64_ARRAY"
            - "FLOAT_ARRAY"
            - "DOUBLE_ARRAY"
            - "LONG_DOUBLE_ARRAY"
            - "OBJ_ARRAY"
        Type names are not case sensitive.

        If the type name is "JOIN", an additional attribute
        "join_attrs" must be specified that indicates which attributes
        are going to be joined. Attributes that are to be joined must
        have been previously defined in the template.

        The index entry is optional but if present, an index will be
        associated with the attribute value. The contents of the
        dictionary object argument to the index attribute specifies
        optional features of the index. If it is empty, i.e. {}, the
        defaults are used for the index.

        An example template:

        {
          "name" : "papi-flits-stalls",
          "uuid" : "579be18a-a4c5-4dca-b541-5d813e8c15b3",
          "attrs" : [
            { "name" : "timestamp", "type" : "timestamp", "index" : {} },
            { "name" : "component_id", "type" : "uint64" },
            { "name" : "flits", "type" : "double" },
            { "name" : "stalls", "type" : "double" },
            { "name" : "comp_time", "type" : "join",
              "join_attrs" : [ "component_id", "timestamp" ],
              "index" : {}
            }
          ]
        }

        Positional Arguments:
        -- The schema name
        -- The schema template

        """
        cdef int rc
        cdef const char *idx_type = NULL
        cdef const char *idx_key = NULL
        cdef const char *idx_args = NULL
        cdef int join_count
        cdef char **join_args
        cdef uuid_t uuid

        if 'name' not in template:
            raise ValueError("'name' is missing from the template")
        if 'uuid' not in template:
            raise ValueError("'uuid' is missing from the template")
        if 'attrs' not in template:
            raise ValueError("'attrs' is missing from the template")

        uuid_str = template['uuid']
        uuid_parse(uuid_str.encode(), uuid)
        schema_name = template['name']
        attrs = template['attrs']

        self.c_schema = sos_schema_create(schema_name.encode(), uuid)
        if self.c_schema == NULL:
            self.abort(ENOMEM)
        for attr in attrs:
            if 'name' not in attr:
                raise ValueError("The 'name' is missing from the attribute")

            if 'type' not in attr:
                raise ValueError("The 'type' is missing from the attribute")

            n = attr['type'].upper()
            if n in sos_attr_types:
                t = sos_attr_types[n]
            else:
                raise ValueError("Invalid attribute type {0}.".format(n))

            if t == SOS_TYPE_JOIN:
                try:
                    join_attrs = []
                    for attr_name in attr['join_attrs']:
                        join_attrs.append(attr_name.encode())
                except:
                    raise ValueError("The 'join_attrs' attribute is required for the 'join' attribute type.")

                join_count = len(join_attrs)
                join_args = <char **>malloc(join_count * 8)
                for i in range(len(join_attrs)):
                    join_args[i] = <char *>join_attrs[i]
                rc = sos_schema_attr_add(self.c_schema, attr['name'].encode(),
                                         t, <size_t>join_count, join_args)
            elif t == SOS_TYPE_STRUCT:
                if 'size' not in attr:
                    raise ValueError("The type {0} must have a 'size'.".format(n))
                sz = attr['size']
                rc = sos_schema_attr_add(self.c_schema, attr['name'].encode(), t, <size_t>sz)
            else:
                rc = sos_schema_attr_add(self.c_schema, attr['name'].encode(), t, 0)

            if rc != 0:
                raise ValueError("The attribute named {0} resulted in error {1}". \
                                 format(attr['name'], rc))

            if 'index' in attr:
                rc = sos_schema_index_add(self.c_schema, attr['name'].encode())
                if rc != 0:
                    self.abort(rc)

                idx_type = "BXTREE"
                # The index modifiers are optional
                idx = attr['index']
                if 'type' in idx:
                    t = idx['type'].encode()
                    idx_type = t
                if 'key' in idx:
                    k = idx['key'].encode()
                    idx_key = k
                if 'args' in idx:
                    a = idx['args'].encode()
                    idx_args = a
                rc = sos_schema_index_modify(self.c_schema,
                                             attr['name'].encode(),
                                             idx_type,
                                             idx_key,
                                             <const char *>idx_args)
                if rc != 0:
                    self.abort(rc)
        return self

    def add(self, Container cont):
        """Add the schema to the container

        Positional Arguments:
        -- The container handle
        """
        cdef int rc
        rc = sos_schema_add(cont.c_cont, self.c_schema)
        if rc != 0:
            self.abort(rc)

    def attr_by_name(self, name):
        """Returns the attribute from the schema

        Positional Arguments:
        -- The name of the attribute
        """
        return Attr(self, attr_name=name)

    def attr_by_id(self, attr_id):
        """Returns the attribute from the schema

        Positional Arguments:
        -- The attribute id
        """
        return Attr(self, attr_id=attr_id)

    def attr_count(self):
        """Returns the number of attributes in the schema"""
        return sos_schema_attr_count(self.c_schema)

    def uuid(self):
        """Returns the unique schema id encoded as a byte array"""
        cdef uuid_t c_uuid
        sos_schema_uuid(self.c_schema, c_uuid)
        pyobj = c_uuid
        return uuid.UUID(bytes=pyobj[0:16])

    def name(self):
        """Returns the name of the schema"""
        return sos_schema_name(self.c_schema).decode('utf-8')

    def alloc(self):
        """Allocate a new object of this type in the container"""
        cdef sos_obj_t c_obj = sos_obj_new(self.c_schema)
        if c_obj == NULL:
            self.abort()
        o = Object()
        return o.assign(c_obj)

    def dup(self):
        cdef sos_schema_t c_schema = sos_schema_dup(self.c_schema)
        s = Schema()
        s.c_schema = c_schema
        return s

    def __getitem__(self, attr_id):
        if type(attr_id) == int:
            return Attr(self, attr_id=attr_id)
        elif type(attr_id) == str:
            return Attr(self, attr_name=attr_id)
        raise ValueError("The index must be a string or an integer.")

    def index_add(self, attr):
        rc = sos_schema_index_add(self.c_schema, attr['name'].encode())
        if rc != 0:
            self.abort(rc)

    def __str__(self):
        cdef int i
        cdef sos_attr_t c_attr
        cdef sos_index_t c_idx
        s = '{{ "name" : "{0}",\n  "attrs" : ['.format(sos_schema_name(self.c_schema))
        for i in range(sos_schema_attr_count(self.c_schema)):
            c_attr = sos_schema_attr_by_id(self.c_schema, i)
            if i > 0:
                s += ","
            s += "\n"
            s += '    {{ "name" : "{0}", "type" : "{1}", "size" : {2}'.format(
                sos_attr_name(c_attr), sos_type_strs[sos_attr_type(c_attr)],
                sos_attr_size(c_attr))
            c_idx = sos_attr_index(c_attr)
            if c_idx != NULL:
                s += ', "indexed" : "true"'
            s += "}"
        s += "\n  ]\n}"
        return s

cdef class Key(object):
    cdef sos_key_t c_key
    cdef size_t c_size
    cdef sos_type_t sos_type
    cdef Attr attr
    cdef object str_fmt

    def __init__(self, size=None, sos_type=None, attr=None):
        self.attr = attr
        if attr:
            if not size:
                size = attr.size()
            self.sos_type = attr.type()
        if sos_type:
            if attr:
                raise ValueError("sos_type and attr are mutually exclusive")
            if sos_type < 0 or sos_type > SOS_TYPE_LAST:
                raise ValueError("{0} is an invalid Sos type")
            self.sos_type = sos_type
            if not size and sos_type >= SOS_TYPE_BYTE_ARRAY:
                raise ValueError("size must be specified if the key type is an array")
        # if not sos_type and not attr:
        #    raise ValueError("Either attr or sos_type must be specified.")
        if not size:
            size = type_sizes[<int>self.sos_type](None)
        self.c_key = sos_key_new(size)
        self.c_size = size
        self.str_fmt = "{0}"

    def __len__(self):
        return self.c_size

    def __str__(self):
        cdef sos_value_data_t v = <sos_value_data_t>sos_key_value(self.c_key)
        if self.sos_type == SOS_TYPE_UINT64:
            return self.str_fmt.format(v.prim.uint64_)
        elif self.sos_type == SOS_TYPE_INT64:
            return self.str_fmt.format(v.prim.int64_)
        elif self.sos_type == SOS_TYPE_TIMESTAMP:
            return "({0}, {1})".format(v.prim.timestamp_.tv.tv_sec,
                                       v.prim.timestamp_.tv.tv_usec)
        elif self.sos_type == SOS_TYPE_DOUBLE:
            return self.str_fmt.format(v.prim.double_)
        elif self.sos_type == SOS_TYPE_INT32:
            return self.str_fmt.format(v.prim.int32_)
        elif self.sos_type == SOS_TYPE_UINT32:
            return self.str_fmt.format(v.prim.uint32_)
        elif self.sos_type == SOS_TYPE_FLOAT:
            return self.str_fmt.format(v.prim.float_)
        elif self.sos_type == SOS_TYPE_INT16:
            return self.str_fmt.format(v.prim.int16_)
        elif self.sos_type == SOS_TYPE_UINT16:
            return self.str_fmt.format(v.prim.uint16_)
        elif self.sos_type == SOS_TYPE_LONG_DOUBLE:
            return self.str_fmt.format(v.prim.long_double_)
        elif self.sos_type == SOS_TYPE_JOIN:
            return self.str_fmt.format(self.split())
        elif self.sos_type == SOS_TYPE_STRUCT:
            return self.str_fmt.format(bytearray(v.prim.struc_[:self.c_size]))
        elif self.sos_type == SOS_TYPE_BYTE_ARRAY:
            return bytearray(v.array.char_[:v.array.count])
        elif self.sos_type == SOS_TYPE_CHAR_ARRAY:
            return str(v.array.char_[:v.array.count])
        elif self.sos_type == SOS_TYPE_UINT64_ARRAY:
            s = ""
            for j in range(v.array.count):
                if len(s) > 0:
                    s += ","
                s += self.str_fmt.format(v.prim.uint64_)
            return s
        elif self.sos_type == SOS_TYPE_INT64_ARRAY:
            pass
        elif self.sos_type == SOS_TYPE_DOUBLE_ARRAY:
            pass
        elif self.sos_type == SOS_TYPE_UINT32_ARRAY:
            pass
        elif self.sos_type == SOS_TYPE_INT32_ARRAY:
            pass
        elif self.sos_type == SOS_TYPE_FLOAT_ARRAY:
            pass
        elif self.sos_type == SOS_TYPE_UINT16_ARRAY:
            pass
        elif self.sos_type == SOS_TYPE_INT16_ARRAY:
            pass
        elif self.sos_type == SOS_TYPE_LONG_DOUBLE_ARRAY:
            pass
        else:
            raise ValueError("Invalid type {0} found in key.".format(self.sos_type))

    def join(self, *args):
        cdef int i, j, count, typ, sz
        cdef sos_comp_key_spec_t specs
        cdef sos_key_t c_key

        count = len(args) // 2
        if count * 2 != len(args):
            raise ValueError("The argument list must consist of a pairs of type, value")

        specs = <sos_comp_key_spec_t>malloc(count * sizeof(sos_comp_key_spec))
        if specs == NULL:
            raise MemoryError("Could not allocate the component key spec list.")
        i = 0
        j = 0
        while i < len(args):
            typ = <int>args[i]
            if typ >= SOS_TYPE_LAST:
                raise ValueError("Invalid value type {0} specifed".format(typ))

            if typ >= TYPE_IS_ARRAY:
                sz = len(args[i+1])
            else:
                sz = 0
            specs[j].data = sos_value_data_new(<sos_type_t>typ, sz)
            specs[j].type = typ
            type_setters[typ](NULL, specs[j].data, args[i+1])
            i += 2
            j += 1

        i = sos_comp_key_set(self.c_key, count, specs);
        if i != 0:
            raise ValueError("Error encoding the composite key")
        for i in range(0, count):
            sos_value_data_del(specs[i].data)
        free(specs)
        return self

    def split(self):
        """Split a join key into its component parts"""
        cdef int rc, i, j
        cdef int typ
        cdef size_t count
        cdef sos_comp_key_spec_t specs
        cdef sos_key_t c_key
        cdef uint32_t secs
        cdef uint32_t usecs

        specs = sos_comp_key_get(self.c_key, &count)
        if specs == NULL:
            raise ValueError("Error decoding component key")

        res = []
        for i in range(count):
            typ = specs[i].type
            res.append(typ)
            if typ == SOS_TYPE_UINT64:
                res.append(specs[i].data.prim.uint64_)
            elif typ == SOS_TYPE_INT64:
                res.append(specs[i].data.prim.int64_)
            elif typ == SOS_TYPE_DOUBLE:
                res.append(specs[i].data.prim.double_)
            elif typ == SOS_TYPE_INT32:
                res.append(specs[i].data.prim.int32_)
            elif typ == SOS_TYPE_UINT32:
                res.append(specs[i].data.prim.uint32_)
            elif typ == SOS_TYPE_FLOAT:
                res.append(specs[i].data.prim.float_)
            elif typ == SOS_TYPE_INT16:
                res.append(specs[i].data.prim.int16_)
            elif typ == SOS_TYPE_UINT16:
                res.append(specs[i].data.prim.uint16_)
            elif typ == SOS_TYPE_LONG_DOUBLE:
                res.append(specs[i].data.prim.long_double_)
            elif typ == SOS_TYPE_TIMESTAMP:
                secs = specs[i].data.prim.timestamp_.tv.tv_sec
                usecs = specs[i].data.prim.timestamp_.tv.tv_usec
                res.append(( secs, usecs ))
            elif typ == SOS_TYPE_STRUCT:
                res.append(bytearray(specs[i].data.array.data.char_[:specs[i].data.array.count]))
            elif typ == SOS_TYPE_BYTE_ARRAY:
                res.append(bytearray(specs[i].data.array.data.char_[:specs[i].data.array.count]))
            elif typ == SOS_TYPE_CHAR_ARRAY:
                res.append(specs[i].data.array.data.char_[:specs[i].data.array.count].decode())
            elif typ == SOS_TYPE_UINT64_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.uint64_[j])
                res.append(a)
            elif typ == SOS_TYPE_INT64_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.int64_[j])
                res.append(a)
            elif typ == SOS_TYPE_DOUBLE_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.double_[j])
                res.append(a)
            elif typ == SOS_TYPE_UINT32_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.uint32_[j])
                res.append(a)
            elif typ == SOS_TYPE_INT32_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.int32_[j])
                res.append(a)
            elif typ == SOS_TYPE_FLOAT_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.float_[j])
                res.append(a)
            elif typ == SOS_TYPE_UINT16_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.uint16_[j])
                res.append(a)
            elif typ == SOS_TYPE_INT16_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.int16_[j])
                res.append(a)
            elif typ == SOS_TYPE_LONG_DOUBLE_ARRAY:
                a = []
                for j in range(specs[i].data.array.count):
                    a.append(specs[i].data.array.data.long_double_[j])
                res.append(a)
            else:
                free(specs)
                raise ValueError("Invalid type {0} found in key.".format(typ))
        for i in range(count):
            sos_value_data_del(specs[i].data)
        free(specs)
        return res

    cdef assign(self, sos_key_t c_key):
        if c_key == NULL:
            raise ValueError("key argument cannot be NULL")
        if self.c_key:
            sos_key_put(self.c_key)
        self.c_key = c_key
        return self

    def release(self):
        if self.c_key:
            sos_key_put(self.c_key)
            self.c_key = NULL

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        self.release()

    def get_attr(self):
        return self.attr

    def set_value(self, value):
        """Set the value of a key.

        Positional Parameters:
        - The value to assign to the key

        The value parameter can be a string or a value of the type
        appropriate for the attribute. If the value is a string, an
        attempt will be made to convert the string to the type.

        Integers, i.e. SOS_TYPE_INT16 ... SOS_TYPE_UINT64 are converted
        with the Python int() functions. The values will be truncated as
        necessary to fit in the target value.

        Floating point, i.e. SOS_TYPE_FLOAT ... SOS_TYPE_DOUBLE will
        be converted using the Python float().

        Arrays, with the exception of SOS_TYPE_CHAR_ARRAY and
        SOS_TYPE_BYTE_ARRAY, are expected to be a list or tuple.

        If the attribute is a SOS_TYPE_CHAR_ARRAY the value is
        expected to be a Python string. If the attribute is a
        SOS_TYPE_BYTE_ARRAY, the value is expected to be a Python
        bytearray.

        Finally, if the type is SOS_TYPE_TIMESTAMP, three value types are accepted:

        -- A tuple as follows: ( seconds, microseconds ), representing
           the number of seconds and microseconds since the Epoch.
           This is the preferred method since the value can be
           exactly specified with no loss of precision.

        -- A floating point value representing the number of seconds
           since the Epoch. This is equivalent to specifying;
              ( int(value), int( (value - int(value)) * 1.0e6) )
           Note that there is insufficient precision in a double
           precision floating point number to accurately represent the
           number of seconds and microseconds since the Epoch. This
           can lead to confusion when searching for a particular
           timestamp if the value cannot be exactly represented by a
           double.

        -- An integer representing the number of seconds since the
           Epoch. This is equivalent to specifying ( seconds, 0 ).

        """
        cdef ods_key_value_t kv
        cdef sos_value_data_t data
        kv = self.c_key.as.key
        key_setters[<int>self.sos_type](self.attr.c_attr, kv, value)

    def __int__(self):
        cdef int typ
        if self.attr:
            typ = self.attr.type()
            if typ == SOS_TYPE_UINT64:
                return self.get_uint64()
            elif typ == SOS_TYPE_INT64:
                return self.get_int64()
            if typ == SOS_TYPE_UINT32:
                return self.get_uint32()
            elif typ == SOS_TYPE_INT32:
                return self.get_int32()
            if typ == SOS_TYPE_UINT16:
                return self.get_uint16()
            elif typ == SOS_TYPE_INT16:
                return self.get_int16()
            if typ == SOS_TYPE_FLOAT:
                return self.get_float()
            elif typ == SOS_TYPE_DOUBLE:
                return self.get_double()
        raise TypeError("The base type cannot be converted to an integer.")

    def __long__(self):
        cdef int typ
        if self.attr:
            typ = self.attr.type()
            if typ == SOS_TYPE_UINT64:
                return self.get_uint64()
            elif typ == SOS_TYPE_INT64:
                return self.get_int64()
            if typ == SOS_TYPE_UINT32:
                return self.get_uint32()
            elif typ == SOS_TYPE_INT32:
                return self.get_int32()
            if typ == SOS_TYPE_UINT16:
                return self.get_uint16()
            elif typ == SOS_TYPE_INT16:
                return self.get_int16()
            if typ == SOS_TYPE_FLOAT:
                return self.get_float()
            elif typ == SOS_TYPE_DOUBLE:
                return self.get_double()
        raise TypeError("The base type cannot be converted to a long.")

    def __float__(self):
        cdef int typ
        if self.attr:
            typ = self.attr.type()
            if typ == SOS_TYPE_UINT64:
                return float(self.get_uint64())
            elif typ == SOS_TYPE_INT64:
                return float(self.get_int64())
            if typ == SOS_TYPE_UINT32:
                return float(self.get_uint32())
            elif typ == SOS_TYPE_INT32:
                return float(self.get_int32())
            if typ == SOS_TYPE_UINT16:
                return float(self.get_uint16())
            elif typ == SOS_TYPE_INT16:
                return float(self.get_int16())
            if typ == SOS_TYPE_FLOAT:
                return self.get_float()
            elif typ == SOS_TYPE_DOUBLE:
                return self.get_double()
        raise TypeError("The base type cannot be converted to an float.")

    def get_uint64(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.uint64_

    def get_uint32(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.uint32_

    def get_uint16(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.uint16_

    def get_byte(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.byte_

    def get_int64(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.int64_

    def get_int32(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.int32_

    def get_int16(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.int16_

    def get_float(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.float_

    def get_double(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.double_

cdef class AttrIter(SosObject):
    """Implements a non-Python iterator on a Schema attribute

    This class implements a database-style iterator. It is not a
    Python iterable and does not employ the Python iterator
    syntax. Calling iter(AttrIter) will genarate an exception,
    An AttrIter can be contructed directly as follows:

    ```python
    schema = db.schema_by_name('vmstat')
    ts = schema.attr_by_name('timestamp')
    it = Sos.AttrIter(ts)
    ```

    There is also a convenience method in the Attr class Attr.iter(),
    for example:

    ```
    it = ts.attr_iter()
    ```
    The AttrIter implements begin(), end(), prev(), and __next__() to
    iterate through objects in the index. Each of these methods
    returns True if there is an object at the current iterator
    position or False otherwise. For example:

    ```python
    b = it.begin()
    while b:
        o = it.item()
        # do something with the object
        b = next(it)
    ```
    There are also methods that take a Key as an argument to position
    the iterator at an object with the specified key. See find(),
    find_sup(), and find_inf() for documentation on these methods.
    """
    cdef Attr attr
    cdef sos_iter_t c_iter

    def __init__(self, Attr attr, unique=False):
        """Instantiate an AttrIter object

        Positional Arguments:
        attr    The Attr with the Index on which the iterator is being
                created.

        Keyword Arguments:
        unique  If True, the iterator will skip duplicates in the index
        """
        self.c_iter = sos_attr_iter_new(attr.c_attr)
        self.attr = attr
        if self.c_iter == NULL:
            raise ValueError("The {0} attribute is not indexed".format(self.attr.name()))
        if unique:
            sos_iter_flags_set(self.c_iter, SOS_ITER_F_UNIQUE)

    def prop_set(self, prop_name, b):
        cdef sos_iter_flags_t f = sos_iter_flags_get(self.c_iter)
        prop_names = {
            "unique" : SOS_ITER_F_UNIQUE,
            "inf_last_dup" : SOS_ITER_F_INF_LAST_DUP,
            "sup_last_dup" : SOS_ITER_F_SUP_LAST_DUP
        }
        bit = prop_names[prop_name]
        if b:
            f  = f | bit
        else:
            f = f & ~bit
        sos_iter_flags_set(self.c_iter, f)

    def item(self):
        """Return the Object at the current iterator position"""
        cdef sos_obj_t c_obj
        c_obj = sos_iter_obj(self.c_iter)
        if c_obj:
            o = Object()
            o.assign(c_obj)
            return o
        return None

    def key(self):
        """Return the Key at the current iterator position"""
        cdef sos_key_t c_key
        c_key = sos_iter_key(self.c_iter)
        if c_key:
            k = Key(size=sos_key_len(c_key), attr=self.attr)
            k.assign(c_key)
            return k
        return None

    def begin(self):
        """Position the iterator at the first object in the index"""
        cdef int rc = sos_iter_begin(self.c_iter)
        if rc == 0:
            return True
        return False

    def end(self):
        """Position the iterator at the last object in the index"""
        cdef int rc = sos_iter_end(self.c_iter)
        if rc == 0:
            return True
        return False

    def next(self):
        return next(self)

    def __next__(self):
        """Move the iterator position to the next object in the index
        Returns:
        True    There is an object at the new iterator position
        False   There is no object at the new iterator position
        """
        cdef int rc
        rc = sos_iter_next(self.c_iter)
        if rc == 0:
            return True
        return False

    def prev(self):
        """Move the iterator position to the previous object in the index
        Returns:
        True    There is an object at the new iterator position
        False   There is no object at the new iterator position
        """
        cdef int rc = sos_iter_prev(self.c_iter)
        if rc == 0:
            return True
        return False

    def find(self, Key key):
        """Position the iterator at the object with the specified key

        Find the index entry with a key that is equal the specified input Key

        Positional arguments:
        -- The key to search for

        Returns:
        True   There is an object with the specified key
        False  There was no object with the specified key
        """
        cdef int rc = sos_iter_find(self.c_iter, key.c_key)
        if rc == 0:
            return True
        return False

    def find_sup(self, Key key):
        """Find the key in the index greater-or-equal to the input key

        Find the index entry with a key that is greator-or-equal the
        specified input Key

        Positional arguments:
        -- The key to search for

        Returns:
        True    Found
        False   Not found

        """
        cdef int rc = sos_iter_sup(self.c_iter, key.c_key)
        if rc == 0:
            return True
        return False

    def find_inf(self, Key key):
        """Find the key in the index less-or-equal to the input key

        Find the index entry with a key that is less-or-equal the
        specified input Key

        Returns:
        True    Found
        False   Not found
        """
        cdef int rc = sos_iter_inf(self.c_iter, key.c_key)
        if rc == 0:
            return True
        return False

    def release(self):
        """Release resources and references associated with the iterator

        Although this is done automatically when the object is deleted, Python's
        lazy deallocation can defer freeing these resources indefinitely
        """
        if self.c_iter != NULL:
            sos_iter_free(self.c_iter)
            self.c_iter = NULL

    def __dealloc__(self):
        self.release()

TYPE_INT16 = SOS_TYPE_INT16
TYPE_INT32 = SOS_TYPE_INT32
TYPE_INT64 = SOS_TYPE_INT64
TYPE_UINT16 = SOS_TYPE_UINT16
TYPE_UINT32 = SOS_TYPE_UINT32
TYPE_UINT64 = SOS_TYPE_UINT64
TYPE_FLOAT = SOS_TYPE_FLOAT
TYPE_DOUBLE = SOS_TYPE_DOUBLE
TYPE_LONG_DOUBLE = SOS_TYPE_LONG_DOUBLE
TYPE_TIMESTAMP = SOS_TYPE_TIMESTAMP
TYPE_OBJ = SOS_TYPE_OBJ
TYPE_STRUCT = SOS_TYPE_STRUCT
TYPE_JOIN = SOS_TYPE_JOIN
TYPE_IS_ARRAY = SOS_TYPE_BYTE_ARRAY
TYPE_BYTE_ARRAY = SOS_TYPE_BYTE_ARRAY
TYPE_CHAR_ARRAY = SOS_TYPE_CHAR_ARRAY
TYPE_STRING = SOS_TYPE_CHAR_ARRAY
TYPE_INT16_ARRAY = SOS_TYPE_INT16_ARRAY
TYPE_INT32_ARRAY = SOS_TYPE_INT32_ARRAY
TYPE_INT64_ARRAY = SOS_TYPE_INT64_ARRAY
TYPE_UINT16_ARRAY = SOS_TYPE_UINT16_ARRAY
TYPE_UINT32_ARRAY = SOS_TYPE_UINT32_ARRAY
TYPE_UINT64_ARRAY = SOS_TYPE_UINT64_ARRAY
TYPE_FLOAT_ARRAY = SOS_TYPE_FLOAT_ARRAY
TYPE_DOUBLE_ARRAY = SOS_TYPE_DOUBLE_ARRAY
TYPE_LONG_DOUBLE_ARRAY = SOS_TYPE_LONG_DOUBLE_ARRAY
TYPE_OBJ_ARRAY = SOS_TYPE_OBJ_ARRAY

PERM_RW = SOS_PERM_RW
PERM_RD = SOS_PERM_RD
PERM_WR = SOS_PERM_WR
PERM_CREAT = SOS_PERM_CREAT
BE_LSOS = SOS_BE_LSOS
BE_MMOS = SOS_BE_MMOS
PERM_USER = SOS_PERM_USER

VERS_MAJOR = SOS_VERS_MAJOR
VERS_MINOR = SOS_VERS_MINOR
VERS_FIX = SOS_VERS_FIX
GIT_COMMIT_ID = ODS_COMMIT_ID

cdef class AttrJoinIter(object):
    cdef id_list
    cdef Schema schema
    cdef int next_idx

    def __init__(self, Attr attr):
        if attr.type() != SOS_TYPE_JOIN:
            raise ValueError("The attribute type must be SOS_TYPE_JOIN")
        self.schema = attr.schema()
        self.id_list = attr.join_list()
        self.next_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_idx >= len(self.id_list):
            raise StopIteration
        attr_id = self.id_list[self.next_idx]
        self.next_idx += 1
        return self.schema.attr_by_id(attr_id)

cdef class Attr(SosObject):
    cdef sos_schema_t c_schema
    cdef sos_attr_t c_attr

    def __init__(self, Schema schema, attr_id=None, attr_name=None):
        SosObject.__init__(self)
        self.c_schema = schema.c_schema
        self.c_attr = NULL
        if attr_id is not None:
            self.c_attr = sos_schema_attr_by_id(schema.c_schema, attr_id)
        elif attr_name is not None:
            self.c_attr = sos_schema_attr_by_name(schema.c_schema, attr_name.encode())
        if self.c_attr == NULL:
            if attr_id:
                name = attr_id
            elif attr_name:
                name = attr_name
            else:
                name = "unspecified"
            raise ValueError("The attribute '{0}' is not present in schema '{1}'".
                             format(name, schema.name()))

    def __richcmp__(self, b, op):
        if type(b) == type(self):
            if op == 0:   # <
                return self.attr_id() < b.attr_id()
            elif op == 2: # ==
                return self.attr_id() == b.attr_id()
            elif op == 4: # >
                return self.attr_id() > b.attr_id()
            elif op == 1: # <=
                return self.attr_id() <= b.attr_id()
            elif op == 3: # !=
                return self.attr_id() != b.attr_id()
            elif op == 5: # >=
                return self.attr_id() >= b.attr_id()
        elif type(b) == str:
            if op == 0:   # <
                return self.name() < b
            elif op == 2: # ==
                return self.name() == b
            elif op == 4: # >
                return self.name() > b
            elif op == 1: # <=
                return self.name() <= b
            elif op == 3: # !=
                return self.name() != b
            elif op == 5: # >=
                return self.name() >= b
        else:
            raise ValueError("Comparison not supported for type {0}".format(type(b)))

    def schema(self):
        """Returns the schema for which this attribute is a member"""
        s = Schema()
        s.c_schema = sos_attr_schema(self.c_attr)
        return s

    def attr_id(self):
        """Returns the attribute id"""
        return sos_attr_id(self.c_attr)

    def is_array(self):
        """Return True if the attribute is an array"""
        return (0 != sos_attr_is_array(self.c_attr))

    def name(self):
        """Returns the attribute name"""
        return sos_attr_name(self.c_attr).decode('utf-8')

    def type(self):
        """Returns the attribute type"""
        return sos_attr_type(self.c_attr)

    def type_name(self):
        """Returns the type name of this attribute"""
        return sos_type_strs[sos_attr_type(self.c_attr)]

    def is_indexed(self):
        """Returns True if the attribute has an index"""
        if sos_attr_index(self.c_attr) != NULL:
            return True
        return False

    def size(self):
        """Returns the size of the attribute data in bytes"""
        return sos_attr_size(self.c_attr)

    def index(self):
        """Returns an Index() object for this attribute"""
        return Index(self)

    def attr_iter(self):
        """Return an AttrIter instance for this attribute"""
        return AttrIter(self)

    def filter(self):
        """Return a Filter instance for this attribute"""
        return Filter(self)

    def join_list(self):
        cdef sos_array_t array = sos_attr_join_list(self.c_attr)
        if array == NULL:
            return None
        res = np.ndarray([ array.count ], dtype=np.dtype('uint32'), order="C")
        for i in range(0, array.count):
            res[i] = array.data.uint32_[i]
        return res

    def join_iter(self):
        return AttrJoinIter(self)

    def key(self, *args):
        """Construct a key for this attribute from the input arguments

        The input argument is expected to be capable of being
        converted to the attribute type. If the attribute is a join,
        then each argument corresponds to each element of the
        join. For example, if the attribute consists of three
        components: Sos.TYPE_UINT16, Sos.STRING, Sos.TYPE_UINT32, then
        one would call this function as follows:

            key = attr.key(25, "the string value", 3423513)

        Passing a value that is inappropriate for the attribute type
        will result in a ValueError exception.
        """
        cdef sos_array_t attrs
        cdef size_t size
        cdef sos_comp_key_spec_t specs
        cdef size_t specs_len
        cdef int i, j, typ
        cdef sos_attr_t attr

        typ = sos_attr_type(self.c_attr)
        if typ == SOS_TYPE_JOIN:
            # create an argument list to use with Key.join()
            attrs = sos_attr_join_list(self.c_attr)
            specs_len = len(args)
            if specs_len != attrs.count:
                raise ValueError("A Join key was specified with fewer "
                                 "than the required number of values. "
                                 "Expecting {0} got {1}".format(attrs.count, specs_len))
            specs = <sos_comp_key_spec_t>malloc(specs_len * sizeof(sos_comp_key_spec))
            if specs == NULL:
                raise MemoryError("Could not allocate the component key spec list.")
            for i in range(0, specs_len):
                attr = sos_schema_attr_by_id(self.c_schema, attrs.data.uint32_[i])
                typ = sos_attr_type(attr)
                arg = args[i]
                specs[i].type = typ
                if typ >= TYPE_IS_ARRAY:
                    j = len(arg)
                else:
                    j = 0;
                specs[i].data = sos_value_data_new(<sos_type_t>typ, j)
                type_setters[typ](attr, specs[i].data, arg)
            size = sos_comp_key_size(specs_len, specs)
            key = Key(size=size, sos_type=SOS_TYPE_JOIN)
            i = sos_comp_key_set(key.c_key, specs_len, specs)
            for j in range(0, specs_len):
                sos_value_data_del(specs[j].data)
            free(specs)
            if i != 0:
                raise ValueError("Error {0} encoding the key.".format(i))
        elif typ < SOS_TYPE_ARRAY:
            key = Key(attr=self)
            key.set_value(args[0])
        else:
            key = Key(size=type_sizes[typ](args[0]), sos_type=typ)
            key.set_value(args[0])
        return key

    def find(self, Key key):
        cdef sos_index_t c_index = sos_attr_index(self.c_attr)
        cdef sos_obj_t c_obj = sos_index_find(c_index, key.c_key)
        if c_obj == NULL:
            return None
        o = Object()
        o.assign(c_obj)
        return o

    def max(self):
        """Return the maximum value of this attribute in the container"""
        cdef sos_obj_t c_obj
        cdef sos_obj_t c_arr_obj
        cdef sos_value_data_t c_data
        cdef sos_key_t c_key
        cdef int t

        if not self.is_indexed():
            return None

        c_obj = sos_index_find_max(sos_attr_index(self.c_attr), &c_key)
        if c_obj == NULL:
            return None
        c_data = sos_obj_attr_data(c_obj, self.c_attr, &c_arr_obj)

        t = sos_attr_type(self.c_attr)
        v = <object>type_getters[<int>t](c_obj, c_data, self.c_attr)

        sos_obj_put(c_obj)
        sos_key_put(c_key)
        if c_arr_obj != NULL:
            sos_obj_put(c_arr_obj)
        return v

    def min(self):
        """Return the minimum value of this attribute in the container"""
        cdef sos_obj_t c_obj
        cdef sos_obj_t c_arr_obj
        cdef sos_value_data_t c_data
        cdef sos_key_t c_key
        cdef int t

        if not self.is_indexed():
            return None

        c_obj = sos_index_find_min(sos_attr_index(self.c_attr), &c_key)
        if c_obj == NULL:
            return None
        c_data = sos_obj_attr_data(c_obj, self.c_attr, &c_arr_obj)

        t = sos_attr_type(self.c_attr)
        v = <object>type_getters[<int>t](c_obj, c_data, self.c_attr)

        sos_obj_put(c_obj)
        sos_key_put(c_key);
        if c_arr_obj != NULL:
            sos_obj_put(c_arr_obj)
        return v

    def __str__(self):
        cdef sos_index_t c_idx
        s = '{{ "name" : "{0}", "schema" : "{1}", "type" : "{2}", "size" : {3}'.format(
            sos_attr_name(self.c_attr), sos_schema_name(sos_attr_schema(self.c_attr)),
            sos_type_strs[sos_attr_type(self.c_attr)],
            sos_attr_size(self.c_attr))
        c_idx = sos_attr_index(self.c_attr)
        if c_idx != NULL:
            s += ', "indexed" : "true"'
        s += '}'
        return s

COND_LT = SOS_COND_LT
COND_LE = SOS_COND_LE
COND_EQ = SOS_COND_EQ
COND_GE = SOS_COND_GE
COND_GT = SOS_COND_GT
COND_NE = SOS_COND_NE

ctypedef shape_opt_s *shape_opt

cdef double uint64_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.uint64_

cdef double uint32_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.uint32_

cdef double uint16_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.uint16_

cdef double int64_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.uint64_

cdef double int32_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.uint32_

cdef double int16_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.uint16_

cdef double double_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.double_

cdef double float_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.float_

cdef double timestamp_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.prim.timestamp_.tv.tv_sec + \
        (<double>v.data.prim.timestamp_.tv.tv_usec / 1.0e6)

# uint64 array ops
cdef double uint64_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.uint64_[idx]

cdef double uint64_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.uint64_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.uint64_[idx]:
            max_ = v.data.array.data.uint64_[idx]
    return max_

cdef double uint64_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.uint64_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.uint64_[idx]:
            min_ = v.data.array.data.uint64_[idx]
    return min_

cdef double uint64_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += v.data.array.data.uint64_[idx]
    return avg_ / <double>v.data.array.count

cdef double uint64_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += v.data.array.data.uint64_[idx]
    return sum_

cdef double uint64_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.uint64_[s.idx]

# int64 array ops
cdef double int64_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.int64_[idx]

cdef double int64_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.int64_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.int64_[idx]:
            max_ = v.data.array.data.int64_[idx]
        first = 0
    return max_

cdef double int64_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.int64_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.int64_[idx]:
            min_ = v.data.array.data.int64_[idx]
    return min_

cdef double int64_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += v.data.array.data.int64_[idx]
    return avg_ / <double>v.data.array.count

cdef double int64_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += v.data.array.data.int64_[idx]
    return sum_

cdef double int64_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.int64_[s.idx]

# uint32 array ops
cdef double uint32_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.uint32_[idx]

cdef double uint32_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.uint32_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.uint32_[idx]:
            max_ = v.data.array.data.uint32_[idx]
        first = 0
    return max_

cdef double uint32_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.uint32_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.uint32_[idx]:
            min_ = v.data.array.data.uint32_[idx]
        first = 0
    return min_

cdef double uint32_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += v.data.array.data.uint32_[idx]
    return avg_ / <double>v.data.array.count

cdef double uint32_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += <double>v.data.array.data.uint32_[idx]
    return sum_

cdef double uint32_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.uint32_[s.idx]

# int32 array ops
cdef double int32_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.int32_[idx]

cdef double int32_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = <double>v.data.array.data.int32_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if first == 1 or max_ < v.data.array.data.int32_[idx]:
            max_ = <double>v.data.array.data.int32_[idx]
        first = 0
    return max_

cdef double int32_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = <double>v.data.array.data.int32_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.int32_[idx]:
            min_ = <double>v.data.array.data.int32_[idx]
        first = 0
    return min_

cdef double int32_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += <double>v.data.array.data.int32_[idx]
    return avg_ / <double>v.data.array.count

cdef double int32_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += <double>v.data.array.data.int32_[idx]
    return sum_

cdef double int32_pick_array_acc(sos_value_t v, int idx, shape_opt s):
    return v.data.array.data.int32_[s.idx]

# uint16 array ops
cdef double uint16_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.uint16_[idx]

cdef double uint16_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.uint16_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.uint16_[idx]:
            max_ = v.data.array.data.uint16_[idx]
    return max_

cdef double uint16_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.uint16_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.uint16_[idx]:
            min_ = v.data.array.data.uint16_[idx]
        first = 0
    return min_

cdef double uint16_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += v.data.array.data.uint16_[idx]
    return avg_ / <double>v.data.array.count

cdef double uint16_sum_array_acc(sos_value_t v, int idx, shape_opt s):
    cdef double sum_ = 0
    for idx in range(0, v.data.array.count):
        sum_ += v.data.array.data.uint16_[idx]
    return sum_

cdef double uint16_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.uint16_[s.idx]

# int16 array ops
cdef double int16_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.int16_[idx]

cdef double int16_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.int16_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.int16_[idx]:
            max_ = v.data.array.data.int16_[idx]
    return max_

cdef double int16_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.int16_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.int16_[idx]:
            min_ = v.data.array.data.int16_[idx]
    return min_

cdef double int16_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += v.data.array.data.int16_[idx]
    return avg_ / <double>v.data.array.count

cdef double int16_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += v.data.array.data.int16_[idx]
    return sum_

cdef double int16_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.int16_[s.idx]

# uint8 array ops
cdef double uint8_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.byte_[idx]

cdef double uint8_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.byte_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.byte_[idx]:
            max_ = v.data.array.data.byte_[idx]
    return max_

cdef double uint8_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.byte_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.byte_[idx]:
            min_ = v.data.array.data.byte_[idx]
    return min_

cdef double uint8_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += <double>v.data.array.data.byte_[idx]
    return avg_ / <double>v.data.array.count

cdef double uint8_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += v.data.array.data.byte_[idx]
    return sum_

cdef double uint8_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.byte_[s.idx]

# int8 array ops
cdef double int8_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.char_[idx]

cdef double int8_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.char_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.char_[idx]:
            max_ = v.data.array.data.char_[idx]
    return max_

cdef double int8_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.char_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.char_[idx]:
            min_ = v.data.array.data.char_[idx]
    return min_

cdef double int8_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += <double>v.data.array.data.char_[idx]
    return avg_ / <double>v.data.array.count

cdef double int8_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += v.data.array.data.char_[idx]
    return sum_

cdef double int8_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.char_[s.idx]

# double array ops
cdef double double_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return v.data.array.data.double_[idx]

cdef double double_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.double_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.double_[idx]:
            max_ = v.data.array.data.double_[idx]
    return max_

cdef double double_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.double_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.double_[idx]:
            min_ = v.data.array.data.double_[idx]
    return min_

cdef double double_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += v.data.array.data.double_[idx]
    return avg_ / <double>v.data.array.count

cdef double double_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += v.data.array.data.double_[idx]
    return sum_

cdef double double_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.double_[s.idx]

# float array ops
cdef double float_inline_array_acc(sos_value_t v, int idx, shape_opt s):
    return <double>v.data.array.data.float_[idx]

cdef double float_max_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double max_ = v.data.array.data.float_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if max_ < v.data.array.data.float_[idx]:
            max_ = v.data.array.data.float_[idx]
    return max_

cdef double float_min_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double min_ = v.data.array.data.float_[0]
    cdef int idx
    for idx in range(1, v.data.array.count):
        if min_ > v.data.array.data.float_[idx]:
            min_ = v.data.array.data.float_[idx]
    return min_

cdef double float_avg_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double avg_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        avg_ += v.data.array.data.float_[idx]
    return avg_ / <double>v.data.array.count

cdef double float_sum_array_acc(sos_value_t v, int unused, shape_opt s):
    cdef double sum_ = 0
    cdef int idx
    for idx in range(0, v.data.array.count):
        sum_ += v.data.array.data.float_[idx]
    return sum_

cdef double float_pick_array_acc(sos_value_t v, int unused, shape_opt s):
    return v.data.array.data.float_[s.idx]

ctypedef double (*accessor_fn_t)(sos_value_t v, int idx, shape_opt)
cdef struct shape_opt_s:
    accessor_fn_t acc_fn
    int op                      # array handling op
    int idx                     # index if the acc_fn is 'pick'
    int len                     # truncate the len of the array

cdef accessor_fn_t *accessors = [
    int16_acc,
    int32_acc,
    int64_acc,
    uint16_acc,
    uint32_acc,
    uint64_acc,
    float_acc,
    double_acc,
    NULL,                       # long double
    timestamp_acc,
    NULL,                       # obj
    NULL,                       # struct
    NULL,                       # join
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,     # twenty zeros to get to 32, i.e. SOS_TYPE_BYTE_ARRAY
    uint8_inline_array_acc,     # byte_array
    int8_inline_array_acc,      # char_array
    int16_inline_array_acc,
    int32_inline_array_acc,
    int64_inline_array_acc,
    uint16_inline_array_acc,
    uint32_inline_array_acc,
    uint64_inline_array_acc,
    float_inline_array_acc,
    double_inline_array_acc,
    NULL,                       # long double
    NULL,                       # obj array
    # min
    uint8_min_array_acc,
    int8_min_array_acc,
    int16_min_array_acc,
    int32_min_array_acc,
    int64_min_array_acc,
    uint16_min_array_acc,
    uint32_min_array_acc,
    uint64_min_array_acc,
    float_min_array_acc,
    double_min_array_acc,
    # max
    uint8_max_array_acc,
    int8_max_array_acc,
    int16_max_array_acc,
    int32_max_array_acc,
    int64_max_array_acc,
    uint16_max_array_acc,
    uint32_max_array_acc,
    uint64_max_array_acc,
    float_max_array_acc,
    double_max_array_acc,
    # avg
    uint8_avg_array_acc,
    int8_avg_array_acc,
    int16_avg_array_acc,
    int32_avg_array_acc,
    int64_avg_array_acc,
    uint16_avg_array_acc,
    uint32_avg_array_acc,
    uint64_avg_array_acc,
    float_avg_array_acc,
    double_avg_array_acc,
    # sum
    uint8_sum_array_acc,
    int8_sum_array_acc,
    int16_sum_array_acc,
    int32_sum_array_acc,
    int64_sum_array_acc,
    uint16_sum_array_acc,
    uint32_sum_array_acc,
    uint64_sum_array_acc,
    float_sum_array_acc,
    double_sum_array_acc,
    # pick
    uint8_pick_array_acc,
    int8_pick_array_acc,
    int16_pick_array_acc,
    int32_pick_array_acc,
    int64_pick_array_acc,
    uint16_pick_array_acc,
    uint32_pick_array_acc,
    uint64_pick_array_acc,
    float_pick_array_acc,
    double_pick_array_acc,
]

#min
cdef int UINT8_MIN_ARRAY_ACC  = SOS_TYPE_LAST + 1
cdef int INT8_MIN_ARRAY_ACC   = SOS_TYPE_LAST + 2
cdef int INT16_MIN_ARRAY_ACC  = SOS_TYPE_LAST + 3
cdef int INT32_MIN_ARRAY_ACC  = SOS_TYPE_LAST + 4
cdef int INT64_MIN_ARRAY_ACC  = SOS_TYPE_LAST + 5
cdef int UINT16_MIN_ARRAY_ACC = SOS_TYPE_LAST + 6
cdef int UINT32_MIN_ARRAY_ACC = SOS_TYPE_LAST + 7
cdef int UINT64_MIN_ARRAY_ACC = SOS_TYPE_LAST + 8
cdef int FLOAT_MIN_ARRAY_ACC  = SOS_TYPE_LAST + 9
cdef int DOUBLE_MIN_ARRAY_ACC = SOS_TYPE_LAST + 10
# max
cdef int UINT8_MAX_ARRAY_ACC  = SOS_TYPE_LAST + 11
cdef int INT8_MAX_ARRAY_ACC   = SOS_TYPE_LAST + 12
cdef int INT16_MAX_ARRAY_ACC  = SOS_TYPE_LAST + 13
cdef int INT32_MAX_ARRAY_ACC  = SOS_TYPE_LAST + 14
cdef int INT64_MAX_ARRAY_ACC  = SOS_TYPE_LAST + 15
cdef int UINT16_MAX_ARRAY_ACC = SOS_TYPE_LAST + 16
cdef int UINT32_MAX_ARRAY_ACC = SOS_TYPE_LAST + 17
cdef int UINT64_MAX_ARRAY_ACC = SOS_TYPE_LAST + 18
cdef int FLOAT_MAX_ARRAY_ACC  = SOS_TYPE_LAST + 19
cdef int DOUBLE_MAX_ARRAY_ACC = SOS_TYPE_LAST + 20
# avg
cdef int UINT8_AVG_ARRAY_ACC  = SOS_TYPE_LAST + 21
cdef int INT8_AVG_ARRAY_ACC   = SOS_TYPE_LAST + 22
cdef int INT16_AVG_ARRAY_ACC  = SOS_TYPE_LAST + 23
cdef int INT32_AVG_ARRAY_ACC  = SOS_TYPE_LAST + 24
cdef int INT64_AVG_ARRAY_ACC  = SOS_TYPE_LAST + 25
cdef int UINT16_AVG_ARRAY_ACC = SOS_TYPE_LAST + 26
cdef int UINT32_AVG_ARRAY_ACC = SOS_TYPE_LAST + 27
cdef int UINT64_AVG_ARRAY_ACC = SOS_TYPE_LAST + 28
cdef int FLOAT_AVG_ARRAY_ACC  = SOS_TYPE_LAST + 29
cdef int DOUBLE_AVG_ARRAY_ACC = SOS_TYPE_LAST + 30
# sum
cdef int UINT8_SUM_ARRAY_ACC  = SOS_TYPE_LAST + 31
cdef int INT8_SUM_ARRAY_ACC   = SOS_TYPE_LAST + 32
cdef int INT16_SUM_ARRAY_ACC  = SOS_TYPE_LAST + 33
cdef int INT32_SUM_ARRAY_ACC  = SOS_TYPE_LAST + 34
cdef int INT64_SUM_ARRAY_ACC  = SOS_TYPE_LAST + 35
cdef int UINT16_SUM_ARRAY_ACC = SOS_TYPE_LAST + 36
cdef int UINT32_SUM_ARRAY_ACC = SOS_TYPE_LAST + 37
cdef int UINT64_SUM_ARRAY_ACC = SOS_TYPE_LAST + 38
cdef int FLOAT_SUM_ARRAY_ACC  = SOS_TYPE_LAST + 39
cdef int DOUBLE_SUM_ARRAY_ACC = SOS_TYPE_LAST + 40

cdef int INLINE_ARRAY = 0
cdef int MIN_ARRAY    = 1
cdef int MAX_ARRAY    = 11
cdef int AVG_ARRAY    = 21
cdef int SUM_ARRAY    = 31
cdef int PICK_ARRAY   = 41

cdef accessor_fn_t get_accessor_for_type(int sos_type, int arr_op):
    cdef int idx
    if sos_type < SOS_TYPE_ARRAY or arr_op == INLINE_ARRAY:
        return accessors[sos_type]
    idx = SOS_TYPE_LAST + (sos_type - SOS_TYPE_ARRAY) + arr_op
    return accessors[idx]

cdef int array_op_to_offset(name):
    if name == "inline":
        return MIN_ARRAY
    if name == "min":
        return MIN_ARRAY
    if name == "max":
        return MAX_ARRAY
    if name == "avg":
        return AVG_ARRAY
    if name == "sum":
        return SUM_ARRAY
    if name == "pick":
        return PICK_ARRAY
    raise ValueError("{0} is an invalid 'op' value.".format(name))

cdef class Filter(object):
    """Implements a non-Python iterator on a Schema object

    This class implements a database-style iterator. It is not a
    Python iterable and does not employ the Python iterator
    syntax. Calling iter(Filter) will generate an exception,
    A Filter can be constructed directly as follows:

    ```python
    schema = db.schema_by_name('vmstat')
    ts = schema.attr_by_name('timestamp')
    it = Sos.Filter(ts)
    ```

    A Filter iterates through an index and will skip objects that do
    not match the conditions specified by the add_condition() method.

    ```python
    it = ts.obj_iter()
    ```

    The Filter implements begin(), end(), __next__, and prev() to
    iterate through objects in the index. Each of these methods
    returns an Object or None, if there is no object at the iterator
    position. The rational for the difference of return value between
    the AttrIter and Filter is that the object has to be instantiated
    in memory in order to evaluate the match conditions whereas for an
    AttrIter only the key (which is part of the Index) has to be
    instantiated. For performance reasons, AttrIter does not
    automatically instantiate the object while Filter does.

    ```python
    o = it.begin()
    while o:
        # do something with the object
        o = next(it)
    ```
    """
    cdef Attr attr
    cdef sos_filter_t c_filt
    cdef sos_obj_t c_obj
    cdef double start_us
    cdef double end_us

    def __init__(self, Attr attr):
        """Positional Parameters:
        -- The primary filter attribute
        """
        cdef sos_iter_t c_iter
        self.attr = attr
        c_iter = sos_attr_iter_new(attr.c_attr)
        if c_iter == NULL:
            raise ValueError("The attribute {0} must be indexed.".format(attr.name()))

        self.c_filt = sos_filter_new(c_iter)
        self.start_us = 0.0
        self.end_us = 0.0

    def get_attr(self):
        """Return the iterator attribute for this filter"""
        return self.attr

    def attr_by_name(self, name):
        """Return the attribute with this name"""
        return self.attr.schema[name]

    def add_condition(self, Attr cond_attr, cond, value):
        """Add a filter condition on the iterator

        Adds a condition on objects returned by the iterator. Objects
        that do not match all of the conditions are skipped by the
        iterator.

        Conditions are evaluated in order based on the comparison type
        as follows: FIRST, LAST, GT, GE, NE, EQ, LE, LT. This is
        because objects are ordered from smallest to largest on the
        specified index attribute.

        The first condition on the index attribute is used to set the
        initial position for the iteration.  Note that if SOS_COND_EQ,
        SOS_COND_LT or SOS_COND_LE is specified alone, iteration is
        only possible backwards. If the intent is to return all
        objects less than a particular value, then use
        SOS_COND_GT/SOS_COND_GE with the desired start value.

        Positional parameters:
        -- The attribute whose value is being compared
        -- The condition, which is one of:
           SOS_COND_LE    less-or-equal
           SOS_COND_LT    less-than
           SOS_COND_EQ    equal
           SOS_COND_NE    not-equal
           SOS_COND_GE    greater-or-equal
           SOS_COND_GT    greater-than
        -- The value to compare to the object attribute value

        The value parameter can be a string or a value of a type
        appropriate for the attribute. If the value is a string, an
        attempt will be made to convert the string to the appropriate
        type.

        Integers, i.e. SOS_TYPE_INT16 ... SOS_TYPE_UINT64 are converted
        with the C strtol()/strtoul() functions. The values will be truncated as
        necessary to fit in the target value.

        Floating point, i.e. SOS_TYPE_FLOAT ... SOS_TYPE_DOUBLE will
        be converted using the strtod() function.

        Arrays, with the exception of SOS_TYPE_CHAR_ARRAY and
        SOS_TYPE_BYTE_ARRAY, are expected to be of the form
        "value,value,...,value".  The values themselves are converted
        as described above.

        If the attribute is a SOS_TYPE_CHAR_ARRAY the value is
        expected to be a Python string. If the attribute is a
        SOS_TYPE_BYTE_ARRAY, the value is expected to be a Python
        bytearray.

        Finally, if the type is SOS_TYPE_TIMESTAMP, three value types are accepted:

        -- A tuple as follows: ( seconds, microseconds ), representing
           the number of seconds and microseconds since the Epoch.
           This is the preferred method since the value can be
           exactly specified with no loss of precision.

        -- A floating point value representing the number of seconds
           since the Epoch. This is equivalent to specifying;
              ( int(value), int( (value - int(value)) *1.0e6) )
           Note that there is insufficient precision in a double
           precision floating point number to accurately represent the
           number of seconds and microseconds since the Epoch. This
           can lead to confusion when searching for a particular
           timestamp if the value cannot be exactly represented by a
           double.

        -- An integer representing the number of seconds since the
           Epoch. This is equivalent to specifying ( seconds, 0 ).

        """
        cdef int rc
        cdef int typ
        cdef int typ_is_array
        cdef int count
        cdef sos_value_t cond_v

        typ = <int>sos_attr_type(cond_attr.c_attr)
        typ_is_array = sos_attr_is_array(cond_attr.c_attr)

        if type(value) == str:
            # strip embedded '"' from value if present
            value = value.replace('"', '')
            if typ_is_array:
                count = value.count(',') + 1
        else:
            if typ_is_array:
                count = len(value)

        cond_v = sos_value_new()
        if typ_is_array != 0:
            cond_v = sos_array_new(cond_v, cond_attr.c_attr, NULL, count)
            if cond_v == NULL:
                raise MemoryError()
        else:
            cond_v = sos_value_init(cond_v, NULL, cond_attr.c_attr)

        if not cond_v:
            raise ValueError("The attribute value for {0} "
                             "could not be created.".format(cond_attr.name()))

        if typ == SOS_TYPE_STRUCT:
            # truncate the value to avoid overflowing the struct
            value = value[:sos_attr_size(cond_attr.c_attr)]

        if type(value) != str:
            type_setters[typ](cond_attr.c_attr, cond_v.data, value)
        else:
            rc = sos_value_from_str(cond_v, value.encode(), NULL)
            if rc != 0:
                raise ValueError("The value {0} is invalid for the {1} attribute."
                                 .format(value, cond_attr.name()))

        if typ == SOS_TYPE_TIMESTAMP:
            # this is to support as_timeseries
            if cond == SOS_COND_GT or cond == SOS_COND_GE:
                self.start_us = <double>cond_v.data.prim.timestamp_.tv.tv_sec * 1.0e6 \
                                + <double>cond_v.data.prim.timestamp_.tv.tv_usec
            elif cond == SOS_COND_LE or cond == SOS_COND_LT:
                self.end_us = <double>cond_v.data.prim.timestamp_.tv.tv_sec * 1.0e6 \
                              + <double>cond_v.data.prim.timestamp_.tv.tv_usec

        rc = sos_filter_cond_add(self.c_filt, cond_attr.c_attr,
                                 cond, cond_v)
        sos_value_put(cond_v)
        sos_value_free(cond_v)
        if rc != 0:
            raise ValueError("Invalid filter condition, error {0}".format(rc))

    def unique(self):
        """Return unique values

        If there are duplicate keys, the first matching object will be
        returned for each unique key.

        """
        cdef sos_iter_flags_t flags
        flags = sos_filter_flags_get(self.c_filt)
        flags |= SOS_ITER_F_UNIQUE
        sos_filter_flags_set(self.c_filt, flags)

    def attr_iter(self):
        """Return the AttrIter for the primary attribute underlying this Filter"""
        return AttrIter(self.attr)

    def begin(self):
        """Set the filter at the first object that matches all of the input conditions"""
        cdef sos_obj_t c_obj = sos_filter_begin(self.c_filt)
        if c_obj == NULL:
            return None
        o = Object()
        return o.assign(c_obj)

    def end(self):
        """Set the filter at the last object that matches all of the input conditions"""
        cdef sos_obj_t c_obj = sos_filter_end(self.c_filt)
        if c_obj == NULL:
            return None
        o = Object()
        return o.assign(c_obj)

    def __next__(self):
        """Set the filter at the next object that matches all of the input conditions"""
        cdef sos_obj_t c_obj = sos_filter_next(self.c_filt)
        if c_obj == NULL:
            return None
        o = Object()
        return o.assign(c_obj)

    def prev(self):
        """Set the filter at the previous object that matches all of the input conditions"""
        cdef sos_obj_t c_obj = sos_filter_prev(self.c_filt)
        if c_obj == NULL:
            return None
        o = Object()
        return o.assign(c_obj)

    def miss_count(self):
        """Return the filter miss compare count"""
        return sos_filter_miss_count(self.c_filt)

    def count(self):
        """Return the number of objects matching all conditions"""
        cdef size_t count = 0
        cdef sos_obj_t c_o = sos_filter_begin(self.c_filt)
        if c_o == NULL:
            return count
        while c_o != NULL:
            count += 1
            sos_obj_put(c_o)
            c_o = sos_filter_next(self.c_filt)
        return count

    def obj(self):
        """Return the object at the currrent filter position"""
        cdef sos_obj_t c_obj = sos_filter_obj(self.c_filt)
        if c_obj:
            o = Object()
            return o.assign(c_obj)
        return None

    def as_ndarray(self, size_t count, shape=None, order='attribute', cont=False):
        """Return filter data as a Numpy array

        The keyword parameter 'shape' is used to specify the object
        attributes that will comprise the returned array. Each element
        of the array should be either a string or a dictionary object
        as follows:

            {
                "name" : <str attr-name>,
                "op" : <str>,
                "idx" : <int>
            }

        where name is the name of the object attribute and "op", and
        "idx" are used to control how array attributes are handled.

             [ self.attr.name() ]

        If the len(shape) is 1, the array will be simplified to a singly
        dimensioned array of the 'attr' attribute of each object.

        If the number of objects defined by the Filter is less than 'count',
        the array will be padded with zeros.

        The return value is a tuple containing the number of elements written
        to the array and the array itself. For example,

             schema = db.schema_by_name('Sample')
             tstamp = schema.attr_by_name('timestamp')
             f = Filter(tstamp)
             count, array = f.as_ndarray(1024, shape=['timestamp', 'current_freemem'],
                                         order='attribute')

        will return the number of elements actually written to 'array' of
        size 1024 elements. The array is an Numpy.ndarray as follows:

             [ [ 1425362400.0, 1425362460.0, ... ], [ 62453912.0, 6553912.0, ... ] ]

        which is an 'attribute' ordering of attribute values. This
        ordering is more natural for numerical analysis as each
        dimension contains data of the same type.

        For applications such as graphing, it is often preferred to have the
        attribute values grouped by index. Set the 'order' keyword argument to
        'index', and the array data will be ordered as follows:

             [ [ 1425362400.0, 62453912.0 ], [ 1425362460.0, 6553912.0 ], ... ]

        Columns in an ndarray must all be the same order, therefore
        when mixing scalars with arrays in the output, additional
        fields in the shape column argument are used to specify how
        the array should be handled;

        - If the 'op' attribute equals the string 'inline', or the shape
          argument is only a string containing the name of the array attribute,
          the output array contains additional columns for each array element in
          the attribute value. Assume that:

               shape = [ 'timestamp', 'a_array', 'b_array' ]

          and that a_array has a length of 2, b_array has a length of 4, and
          order = 'index', then the output array would contain the following:

          [ [ t0,
              a_array[0], a_array[1],
              b_array[0], b_array[1], b_array[2], b_array[3] ],
            ...
          ]

        - If the shape array element dictionary keyword "op" equals 'sum',
          'avg', 'max', or 'min', the output is a single scalar value that is
          either the sum, average, maximum, or minimum of the array elements for
          that column. For example:

              shape = [ 'timestamp', { 'name' : 'power', 'op' : 'avg' } ]

          would result in column 2 containing a single value that is the average
          of the values in the power array.

        - If the shape array element dictionary keyword "op" equals 'pick', the
          output is a single scalar value that is chosen from the array based on
          the value of the element dictionary keyword "idx" in the shape array
          element dictionary. For example:

              shape = [ 'timestamp', { 'name' : 'power', 'op' : 'pick', 'idx' : 5 } ]

          would result in column 2 containing a single value that corresponds to
          the 6-th element of the power array, i.e. power[5].

        Positional Parameters:
        -- The maximum number of rows in the output array. If this parameter is zero,
           the output size will be the size necessary to hold all matching
           data.

        Keyword Parameters:
        shape       -- Shape is an array/tuple specifies the contents of each column
                       in the output.
        order       -- One of 'attribute' (default) or 'index' as described
                       above
        cont        -- If true, the filter will continue where it left off,
                       i.e. processing will not begin at the first matching
                       key.
        array       -- If specified, one of INLINE_ARRAY, MAX_ARRAY, MIN_ARRAY, AVG_ARRAY,
                       or SUM_ARRAY
        array_idx   -- If array=='pick', used to specify which element of the
                       array is returned in the result.
        Returns:
        -- A two element tuple containing the number of samples written to
           the array and the array itself as follows: (count, array)

        """
        cdef sos_obj_t c_o
        cdef sos_value_s v_, t_
        cdef sos_value_t v, t
        cdef int el_idx
        cdef int el_count
        cdef int idx
        cdef int attr_idx
        cdef int res_idx
        cdef int val_idx
        cdef int atype
        cdef int nattr
        cdef int assign
        cdef Schema schema = self.attr.schema()
        cdef Attr attr
        cdef sos_attr_t c_attr
        cdef sos_attr_t *res_attr
        cdef int *res_dim
        cdef int dim
        cdef int *res_type
        cdef shape_opt res_acc
        cdef int type_id

        if shape == None:
            shape = [ self.attr.name() ]
        nattr = len(shape)

        res_attr = <sos_attr_t *>malloc(sizeof(sos_attr_t) * nattr)
        if res_attr == NULL:
            raise MemoryError("Insufficient memory to allocate dimension array")
        res_type = <int *>malloc(sizeof(uint64_t) * nattr)
        if res_type == NULL:
            free(res_attr)
            raise MemoryError("Insufficient memory to allocate type array")
        res_acc = <shape_opt>malloc(sizeof(shape_opt_s) * nattr)
        if res_acc == NULL:
            free(res_attr)
            free(res_type)
            raise MemoryError("Insufficient memory to allocate type array")
        res_dim = <int *>malloc(sizeof(uint64_t) * nattr)
        if res_dim == NULL:
            free(res_attr)
            free(res_type)
            free(res_acc)
            raise MemoryError("Insufficient memory to allocate type array")

        try:
            idx = 0
            for opt in shape:
                if type(opt) == str:
                    aname = opt
                elif type(opt) == dict:
                    aname = opt['name']
                else:
                    raise ValueError("shape elements must be either <str> or <dict>.")
                attr = schema.attr_by_name(aname)
                res_attr[idx] = attr.c_attr
                res_type[idx] = sos_attr_type(attr.c_attr)
                # set access defaults
                res_acc[idx].idx = 0
                res_acc[idx].len = 0
                res_acc[idx].op = INLINE_ARRAY
                if type(opt) == dict:
                    # override defaults
                    res_acc[idx].op = array_op_to_offset(opt['op'])
                    if 'idx' in opt:
                        res_acc[idx].idx = <int>opt['idx']
                    if 'len' in opt:
                        res_acc[idx].len = <int>opt['len']
                res_acc[idx].acc_fn = get_accessor_for_type(res_type[idx], res_acc[idx].op)
                idx += 1
        except Exception as e:
                free(res_attr)
                free(res_type)
                free(res_acc)
                free(res_dim)
                raise ValueError("Error '{0}' processing the shape keyword parameter".format(str(e)))

        if cont:
            c_o = sos_filter_next(self.c_filt)
        else:
            c_o = sos_filter_begin(self.c_filt)

        # determine the result depth in the 'attribute' dimension
        dim = 0
        if c_o != NULL:
            for attr_idx in range(0, nattr):
                # expand the attribute dimension if there are arrays
                if sos_attr_is_array(res_attr[attr_idx]):
                    v = sos_value_init(&v_, c_o, res_attr[attr_idx])
                    res_dim[attr_idx] = sos_array_count(v)
                    sos_value_put(v)
                else:
                    res_dim[attr_idx] = 1
                if res_acc[attr_idx].op == INLINE_ARRAY:
                    dim += res_dim[attr_idx]
                else:
                    dim += 1

        if nattr > 1:
            if order == 'index':
                ndshape = [ count, dim ]
            elif order == 'attribute':
                ndshape = [ dim, count ]
            else:
                raise ValueError("The 'order' keyword parameter must be one " \
                                 "of 'index' or 'attribute'")
        else:
            # if dim == nattr, there are no arrays in the result
            if dim > nattr:
                ndshape = [ count, dim ]
            else:
                ndshape = [ count ]

        result = np.zeros(ndshape, dtype=np.float64, order='C')

        if nattr == 1 and dim == nattr:
            assign = 0
        else:
            if order == 'index':
                assign = 2
            else:
                assign = 3

        idx = 0
        while c_o != NULL and idx < count:

            res_idx = 0
            for attr_idx in range(0, nattr):
                v = sos_value_init(&v_, c_o, res_attr[attr_idx])
                type_id = res_type[attr_idx]

                if res_acc[attr_idx].op == INLINE_ARRAY:
                    el_count = res_dim[attr_idx]
                else:
                    el_count = 1

                for el_idx in range(0, el_count):
                    if assign == 2:
                        result[idx][res_idx] = res_acc[attr_idx].acc_fn(v, el_idx, &res_acc[attr_idx])
                    elif assign == 3:
                        result[res_idx][idx] = res_acc[attr_idx].acc_fn(v, el_idx, &res_acc[attr_idx])
                    else:
                        result[idx] = res_acc[attr_idx].acc_fn(v, el_idx, &res_acc[attr_idx])
                    res_idx += 1
                sos_value_put(v)
            sos_obj_put(c_o)

            c_o = sos_filter_next(self.c_filt)
            idx += 1
            if idx >= count:
                sos_obj_put(c_o)
                break

        if idx < count:
            c_o = sos_filter_end(self.c_filt)
            sos_obj_put(c_o)

        free(res_attr)
        free(res_type)
        free(res_acc)
        free(res_dim)
        return (idx, result)

    def as_timeseries(self, size_t count, shape=None, order='attribute', cont=False,
                      timestamp='timestamp', interval_ms=None):
        """Return filter data as a Numpy array

        The keyword parameter 'shape' is used to specify the object
        attributes that will comprise the returned array. Each element
        of the array should be either a string or a dictionary object
        as follows:

            {
                "name" : <str attr-name>,
                "op" : <str>,
                "idx" : <int>
            }

        where name is the name of the object attribute and "op", and
        "idx" are used to control how array attributes are handled.

             [ self.attr.name() ]

        If the len(shape) is 1, the array will be simplified to a singly
        dimensioned array of the 'attr' attribute of each object.

        If the number of objects defined by the Filter is less than 'count',
        the array will be padded with zeros.

        The return value is a tuple containing the number of elements written
        to the array and the array itself. For example,

             schema = db.schema_by_name('Sample')
             tstamp = schema.attr_by_name('timestamp')
             f = Filter(tstamp)
             count, array = f.as_timeseries(1024, shape=['timestamp', 'current_freemem'],
                                            order='attribute')

        will return the number of elements actually written to 'array' of
        size 1024 elements. The array is an Numpy.ndarray as follows:

             [ [ 1425362400.0, 1425362460.0, ... ], [ 62453912.0, 6553912.0, ... ] ]

        which is an 'attribute' ordering of attribute values. This ordering is more
        natural for numerical analysis as each array contains data of the same type.

        For applications such as graphing, it is often preferred to have the
        attribute values grouped by index. Set the 'order' keyword argument to
        'index', and the array data will be ordered as follows:

             [ [ 1425362400.0, 62453912.0 ], [ 1425362460.0, 6553912.0 ], ... ]

        Columns in an ndarray must all be the same order, therefore
        when mixing scalars with arrays in the output, additional
        fields in the shape column argument are used to specify how
        the array should be handled;

        - If the 'op' attribute equals the string 'inline', or the shape
          argument is only a string containing the name of the array attribute,
          the output array contains additional columns for each array element in
          the attribute value. Assume that:

               shape = [ 'timestamp', 'a_array', 'b_array' ]

          and that a_array has a length of 2, b_array has a length of 4, and
          order = 'index', then the output array would contain the following:

          [ [ t0,
              a_array[0], a_array[1],
              b_array[0], b_array[1], b_array[2], b_array[3] ],
            ...
          ]

        - If the shape array element dictionary keyword "op" equals 'sum',
          'avg', 'max', or 'min', the output is a single scalar value that is
          either the sum, average, maximum, or minimum of the array elements for
          that column. For example:

              shape = [ 'timestamp', { 'name' : 'power', 'op' : 'avg' } ]

          would result in column 2 containing a single value that is the average
          of the values in the power array.

        - If the shape array element dictionary keyword "op" equals 'pick', the
          output is a single scalar value that is chosen from the array based on
          the value of the element dictionary keyword "idx" in the shape array
          element dictionary. For example:

              shape = [ 'timestamp', { 'name' : 'power', 'op' : 'pick', 'idx' : 5 } ]

          would result in column 2 containing a single value that corresponds to
          the 6-th element of the power array, i.e. power[5].

        Positional Parameters:
        -- The maximum number of rows in the output array. If this parameter is zero,
           the output size will be the size necessary to hold all matching
           data.

        Keyword Parameters:
        shape       -- Shape is an array/tuple specifies the contents of each column
                       in the output.
        order       -- One of 'attribute' (default) or 'index' as described
                       above
        cont        -- If true, the filter will continue where it left off,
                       i.e. processing will not begin at the first matching
                       key.
        interval_ms -- The number of milliseconds represented by each sample.
                       The default is None.
        timestamp   -- The name of the attribute containing a
                       SOS_TYPE_TYPESTAMP. The default is 'timestamp'
        array       -- If specified, one of INLINE_ARRAY, MAX_ARRAY, MIN_ARRAY, AVG_ARRAY,
                       or SUM_ARRAY
        array_idx   -- If array=='pick', used to specify which element of the
                       array is returned in the result.
        Returns:
        -- A two element tuple containing the number of samples written to
           the array and the array itself as follows: (count, array)

        """
        cdef sos_obj_t c_o
        cdef sos_value_s v_, t_
        cdef sos_value_t v, t
        cdef int el_idx
        cdef int el_count
        cdef int idx
        cdef int last_idx
        cdef int attr_idx
        cdef int res_idx
        cdef int val_idx
        cdef int atype
        cdef int nattr
        cdef int assign
        cdef Schema schema = self.attr.schema()
        cdef Attr attr
        cdef sos_attr_t c_attr, t_attr
        cdef sos_attr_t *res_attr
        cdef int *res_dim
        cdef int dim
        cdef int *res_type
        cdef shape_opt res_acc
        cdef int type_id
        cdef double obj_time, prev_time
        cdef double bin_width, bin_time, bin_value, bin_samples
        cdef double *tmp_res
        cdef double temp_a, temp_b

        if shape == None:
            shape = [ self.attr.name() ]
        nattr = len(shape)

        res_attr = <sos_attr_t *>malloc(sizeof(sos_attr_t) * nattr)
        if res_attr == NULL:
            raise MemoryError("Insufficient memory to allocate dimension array")
        res_type = <int *>malloc(sizeof(uint64_t) * nattr)
        if res_type == NULL:
            free(res_attr)
            raise MemoryError("Insufficient memory to allocate type array")
        res_acc = <shape_opt>malloc(sizeof(shape_opt_s) * nattr)
        if res_acc == NULL:
            free(res_attr)
            free(res_type)
            raise MemoryError("Insufficient memory to allocate type array")
        res_dim = <int *>malloc(sizeof(uint64_t) * nattr)
        if res_dim == NULL:
            free(res_attr)
            free(res_type)
            free(res_acc)
            raise MemoryError("Insufficient memory to allocate type array")

        t_attr = sos_schema_attr_by_name(schema.c_schema, timestamp.encode())
        if t_attr == NULL:
            raise ValueError("The timestamp attribute was not found in the schema. "
                             "Consider specifying the timestamp keyword parameter")
        if sos_attr_type(t_attr) != SOS_TYPE_TIMESTAMP:
            raise ValueError("The timestamp attribute {0} is not a SOS_TYPE_TIMESTAMP".format(timestamp))

        try:
            idx = 0
            for opt in shape:
                if type(opt) == str:
                    aname = opt
                elif type(opt) == dict:
                    aname = opt['name']
                else:
                    raise ValueError("shape elements must be either <str> or <dict.")
                attr = schema.attr_by_name(aname)
                res_attr[idx] = attr.c_attr
                res_type[idx] = sos_attr_type(attr.c_attr)
                # set access defaults
                res_acc[idx].idx = 0
                res_acc[idx].len = 0
                res_acc[idx].op = INLINE_ARRAY
                if type(opt) == dict:
                    # override defaults
                    res_acc[idx].op = array_op_to_offset(opt['op'])
                    if 'idx' in opt:
                        res_acc[idx].idx = <int>opt['idx']
                    if 'len' in opt:
                        res_acc[idx].len = <int>opt['len']
                res_acc[idx].acc_fn = get_accessor_for_type(res_type[idx], res_acc[idx].op)
                idx += 1
        except Exception as e:
                free(res_attr)
                free(res_type)
                free(res_acc)
                free(res_dim)
                raise ValueError("Error '{0}' processing the shape keyword parameter".format(str(e)))

        if cont:
            c_o = sos_filter_next(self.c_filt)
        else:
            c_o = sos_filter_begin(self.c_filt)

        # determine the result depth in the 'attribute' dimension
        dim = 0
        if c_o != NULL:
            t = sos_value_init(&t_, c_o, t_attr)
            obj_time = <double>t.data.prim.timestamp_.tv.tv_usec * 1.0e6 \
                       + <double>t.data.prim.timestamp_.tv.tv_usec
            sos_value_put(t)
            self.start_us = obj_time
            for attr_idx in range(0, nattr):
                # expand the attribute dimension if there are arrays
                if sos_attr_is_array(res_attr[attr_idx]):
                    v = sos_value_init(&v_, c_o, res_attr[attr_idx])
                    res_dim[attr_idx] = sos_array_count(v)
                    sos_value_put(v)
                else:
                    res_dim[attr_idx] = 1
                if res_acc[attr_idx].op == INLINE_ARRAY:
                    dim += res_dim[attr_idx]
                else:
                    dim += 1

        tmp_res = <double *>malloc(dim *sizeof(double))
        if tmp_res == NULL:
            raise MemoryError("Insufficient memory to allocate temporary result array")

        if interval_ms is not None and self.start_us != 0 and self.end_us != 0:
            bin_width = interval_ms * 1.0e3
            bin_count = self.end_us - self.start_us
            bin_count = bin_count / bin_width
            if count == 0:
                count = int(bin_count)
        else:
            bin_width = 0.0

        if nattr > 1:
            if order == 'index':
                ndshape = [ count, dim ]
            elif order == 'attribute':
                ndshape = [ dim, count ]
            else:
                raise ValueError("The 'order' keyword parameter must be one " \
                                 "of 'index' or 'attribute'")
        else:
            # if dim == nattr, there are no arrays in the result
            if dim > nattr:
                ndshape = [ count, dim ]
            else:
                ndshape = [ count ]

        result = np.zeros(ndshape, dtype=np.float64, order='C')

        bin_samples = 0.0
        bin_dur = 0.0
        bin_time = 0.0
        prev_time = obj_time
        self.start_us = obj_time # This make idx_0 == start_time

        if nattr == 1 and dim == nattr:
            assign = 0
        else:
            if order == 'index':
                assign = 2
            else:
                assign = 3

        idx = 0
        last_idx = 0
        for res_idx in range(0, dim):
            tmp_res[res_idx] = 0.0

        if bin_width > 0.0:
            idx = int((obj_time - self.start_us) / bin_width)

        while c_o != NULL and idx < count and last_idx < count:

            res_idx = 0
            for attr_idx in range(0, nattr):
                v = sos_value_init(&v_, c_o, res_attr[attr_idx])
                type_id = res_type[attr_idx]

                if res_acc[attr_idx].op == INLINE_ARRAY:
                    el_count = res_dim[attr_idx]
                else:
                    el_count = 1

                for el_idx in range(0, el_count):
                    temp_a = tmp_res[res_idx]
                    temp_b = res_acc[attr_idx].acc_fn(v, el_idx, &res_acc[attr_idx])
                    temp_a = ((temp_a * bin_samples) + temp_b) / (bin_samples + 1.0)
                    tmp_res[res_idx] = temp_a
                    res_idx += 1

                sos_value_put(v)
            sos_obj_put(c_o)

            # Get the next sample
            prev_time = obj_time
            c_o = sos_filter_next(self.c_filt)
            if c_o != NULL:
                t = sos_value_init(&t_, c_o, t_attr)
                obj_time = <double>t.data.prim.timestamp_.tv.tv_usec * 1.0e6 \
                           + <double>t.data.prim.timestamp_.tv.tv_usec
                sos_value_put(t)

            # compute the next bin index
            if bin_width > 0.0:
                idx = int((obj_time - self.start_us) / bin_width)
            else:
                idx += 1

            # If the next object is not in the same bin, save this result
            if idx != last_idx:
                if last_idx == 0 and idx - last_idx > 2:
                    # The code supports down-sampling, but not up-sampling.
                    bin_width = 0.0
                    idx = last_idx + 1
                if assign == 2:
                    while last_idx < idx and last_idx < count:
                        for res_idx in range(0, dim):
                            result[last_idx][res_idx] = tmp_res[res_idx]
                        last_idx += 1
                elif assign == 3:
                    while last_idx < idx and last_idx < count:
                        for res_idx in range(0, dim):
                            result[res_idx][last_idx] = tmp_res[res_idx]
                        last_idx += 1
                else:
                    while last_idx < idx and last_idx < count:
                        for res_idx in range(0, dim):
                            result[last_idx] = tmp_res[res_idx]
                        last_idx += 1
                for res_idx in range(0, dim):
                    tmp_res[res_idx] = 0.0
                bin_samples = 0.0
            else:
                bin_samples += 1.0

            if idx >= count:
                sos_obj_put(c_o)
                break

        # if idx < count:
        #     c_o = sos_filter_end(self.c_filt)
        #     sos_obj_put(c_o)

        free(tmp_res)
        free(res_attr)
        free(res_type)
        free(res_acc)
        free(res_dim)
        return (last_idx, result)

    def release(self):
        if self.c_obj:
            sos_obj_put(self.c_obj)
            self.c_obj = NULL
        if self.c_filt:
            sos_filter_free(self.c_filt)
            self.c_filt = NULL

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        self.release()

cdef class ColSpec:

    LEFT = 1
    RIGHT = 2
    CENTER = 3

    cdef name                   # The text name for the column
    cdef query                  # The Query
    cdef int cursor_idx         # The index in the cursor for the Object
                                # specifying the filter
    cdef attr                   # The attribute in the Object
    cdef int col_width          # How wide to make the column
    cdef cvt_fn                 # Function to convert the attribute value
                                # to some other value type for output
    cdef default_fn             # Function to provide the default value if
                                # there is no object for this column at the
                                # cursor index
    cdef fill
    cdef align

    cdef data                   # A place where an application can store its data

    def __init__(self, name,
                 cvt_fn=None, default_fn=None, attr_type=None,
                 col_width=None, align=ColSpec.RIGHT, fill=' '):
        self.name = name
        self.cvt_fn = cvt_fn
        self.default_fn = default_fn
        if col_width:
            self.col_width = col_width
        else:
            self.col_width = 0
        self.align = align
        self.fill = fill

    @property
    def col_name(self):
        return self.name

    @property
    def schema_name(self):
        return self.attr.schema().name()

    @property
    def attr_name(self):
        return self.attr.name()

    @property
    def attr_type(self):
        return self.attr.type()

    @property
    def attr_id(self):
        return self.attr.attr_id()

    @property
    def attr_idx(self):
        return self.cursor_idx

    @property
    def is_array(self):
        return self.attr.is_array()

    @property
    def width(self):
        return self.col_width

    def update(self, query, cursor_idx, attr):
        # self.query = query
        self.cursor_idx = cursor_idx
        self.attr = attr
        if self.col_width == 0:
            self.col_width = query.get_col_width()

    def __repr__(self):
        r = "name      : {0}\n".format(self.name)
        r += "query     : {0}\n".format(self.query)
        r += "attr      : {0}\n".format(self.attr)
        r += "cvt_fn    : {0}\n".format(self.cvt_fn)
        r += "cursor_idx: {0}\n".format(self.cursor_idx)
        r += "col_width : {0}\n".format(self.col_width)
        r += "align     : {0}\n".format(self.align)
        r += "fill      : {0}".format(self.fill)
        return r

    def convert(self, value):
        """Return the value with any data conversion defined for the column"""
        if self.cvt_fn:
            return self.cvt_fn(value)
        return value

    def format(self, value):
        """Return a column formatted string for the input value"""
        if type(value) == bytearray:
            v = repr(value)
        else:
            v = str(value)
        if self.align == ColSpec.RIGHT:
            return v.rjust(self.col_width, self.fill)
        elif self.align == ColSpec.LEFT:
            return v.ljust(self.col_width, self.fill)
        else:
            return v.center(self.col_width, self.fill)

cdef class Index(object):
    cdef sos_index_t c_index
    cdef sos_index_stat_s c_stats

    def __init__(self, Attr attr=None, name=None):
        if attr:
            self.c_index = sos_attr_index(attr.c_attr)
        else:
            self.c_index = NULL

    cdef assign(self, sos_index_t idx):
        self.c_index = idx
        return self

    def insert(self, Key key, Object obj):
        """Inserts an object in the index

        Positional Parameters:

        - The Key for the object
        - The Object to associate with the Key

        Returns:
        0  - Success
        !0 - An errno indicating the reason for failure.
        """
        cdef int rc
        rc = sos_index_insert(self.c_index, key.c_key, obj.c_obj)
        return rc

    def remove(self, Key key, Object obj):
        """Remove a key from the index

        Removes a Key from the Index. An Object is specified to allow
        the code to discriminate between duplicate keys.

        Positional Parameters:

        - The Key to remove
        - The Object associated with the Key

        Returns:
        0  - Success
        !0 - An errno indicating the reason for failure, e.g. ENOENT.

        """
        cdef int rc
        rc = sos_index_remove(self.c_index, key.c_key, obj.c_obj)
        return rc

    def find(self, Key key):
        """Positions the index at the first matching key

        Return the object that matches the specified key.
        If no match was found, None is returned.

        Positional Arguments:
        -- The Key to match
        Returns:
        -- The Object at the index position
        """
        cdef sos_obj_t c_obj = sos_index_find(self.c_index, key.c_key)
        if c_obj != NULL:
            o = Object()
            return o.assign(c_obj)
        return None

    def find_inf(self, Key key):
        """Positions the index at the infinum of the specified key

        Return the object at the key that is the infinum (greatest
        lower bound) of the specified key. If no match was found, None
        is returned.

        Positional Arguments:
        -- The Key to match
        Returns:
        -- The Object at the index position or None if the infinum was not found

        """
        cdef sos_obj_t c_obj = sos_index_find_inf(self.c_index, key.c_key)
        if c_obj != NULL:
            o = Object()
            return o.assign(c_obj)
        return None

    def find_sup(self, Key key):
        """Positions the index at the supremum of the specified key

        Return the object at the key that is the supremum (least
        upper bound) of the specified key. If no match was found, None
        is returned.

        Positional Arguments:
        -- The Key to match
        Returns:
        -- The Object at the index position or None if the supremum was not found
        """
        cdef sos_obj_t c_obj = sos_index_find_sup(self.c_index, key.c_key)
        if c_obj != NULL:
            o = Object()
            return o.assign(c_obj)
        return None

    def name(self):
        """Return the name of the index"""
        return sos_index_name(self.c_index).decode('utf-8')

    def stats(self):
        """Return a dictionary of index statistics as follows:
            cardinality - Number of index entries
            duplicates  - Number of duplicate keys
            size        - The storage size consumed by the index in bytes
        """
        cdef int rc = sos_index_stat(self.c_index, &self.c_stats)
        return self.c_stats

    def show(self):
        """Print the contents of the index to stdout"""
        sos_index_print(self.c_index, NULL);

################################
# Object getter functions
################################
cdef object get_DOUBLE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('float64'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.double_[i]
    return res

cdef object get_LONG_DOUBLE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('float128'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.long_double_[i]
    return res

cdef object get_FLOAT_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('float32'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.float_[i]
    return res

cdef object get_UINT64_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('uint64'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.uint64_[i]
    return res

cdef object get_UINT32_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('uint32'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.uint32_[i]
    return res

cdef object get_UINT16_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('uint16'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.uint16_[i]
    return res

cdef object get_BYTE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('uint8'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.byte_[i]
    return res

cdef object get_CHAR_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.array.data.char_[:c_data.array.count].decode()

cdef object get_INT64_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('int64'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.int64_[i]
    return res

cdef object get_INT32_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('int32'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.int32_[i]
    return res

cdef object get_INT16_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    res = np.ndarray([ c_data.array.count ], dtype=np.dtype('int16'), order="C")
    for i in range(0, c_data.array.count):
        res[i] = c_data.array.data.int16_[i]
    return res

cdef object get_TIMESTAMP(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return (c_data.prim.timestamp_.tv.tv_sec, c_data.prim.timestamp_.tv.tv_usec)

cdef object get_DOUBLE(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.double_

cdef object get_LONG_DOUBLE(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.long_double_

cdef object get_FLOAT(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.float_

cdef object get_UINT64(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.uint64_

cdef object get_UINT32(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.uint32_

cdef object get_UINT16(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.uint16_

cdef object get_INT64(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.int64_

cdef object get_INT32(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.int32_

cdef object get_INT16(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.prim.int16_

cdef object get_STRUCT(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return bytearray(c_data.struc.char_[:sos_attr_size(c_attr)])

cdef object get_JOIN(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    cdef sos_value_data_t c_attr_data
    cdef sos_attr_t c_join_attr
    cdef sos_array_t join_ids
    cdef sos_obj_t arr_obj
    cdef int i

    if c_obj == NULL:
        raise ValueError("The c_obj parameter cannot be NULL")

    join_ids = sos_attr_join_list(c_attr);
    value = ""
    for i in range(0, join_ids.count):
        c_join_attr = sos_schema_attr_by_id(sos_attr_schema(c_attr), join_ids.data.uint32_[i])
        c_attr_data = sos_obj_attr_data(c_obj, c_join_attr, &arr_obj)
        if i != 0:
            value += ":" + str(type_getters[<int>sos_attr_type(c_join_attr)](c_obj, c_attr_data, c_join_attr))
        else:
            value += str(type_getters[<int>sos_attr_type(c_join_attr)](c_obj, c_attr_data, c_join_attr))
        if arr_obj != NULL:
            sos_obj_put(arr_obj)
    return value

cdef object get_ERROR(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    raise ValueError("Get is not supported on this attribute type.")

ctypedef object (*type_getter_fn_t)(sos_obj_t, sos_value_data_t, sos_attr_t)
cdef type_getter_fn_t type_getters[SOS_TYPE_LAST+1]
type_getters[<int>SOS_TYPE_INT16] = get_INT16
type_getters[<int>SOS_TYPE_INT32] = get_INT32
type_getters[<int>SOS_TYPE_INT64] = get_INT64
type_getters[<int>SOS_TYPE_UINT16] = get_UINT16
type_getters[<int>SOS_TYPE_UINT32] = get_UINT32
type_getters[<int>SOS_TYPE_UINT64] = get_UINT64
type_getters[<int>SOS_TYPE_FLOAT] = get_FLOAT
type_getters[<int>SOS_TYPE_DOUBLE] = get_DOUBLE
type_getters[<int>SOS_TYPE_LONG_DOUBLE] = get_LONG_DOUBLE
type_getters[<int>SOS_TYPE_TIMESTAMP] = get_TIMESTAMP
type_getters[<int>SOS_TYPE_OBJ] = get_ERROR
type_getters[<int>SOS_TYPE_STRUCT] = get_STRUCT
type_getters[<int>SOS_TYPE_JOIN] = get_JOIN
type_getters[<int>SOS_TYPE_BYTE_ARRAY] = get_BYTE_ARRAY
type_getters[<int>SOS_TYPE_CHAR_ARRAY] = get_CHAR_ARRAY
type_getters[<int>SOS_TYPE_INT16_ARRAY] = get_INT16_ARRAY
type_getters[<int>SOS_TYPE_INT32_ARRAY] = get_INT32_ARRAY
type_getters[<int>SOS_TYPE_INT64_ARRAY] = get_INT64_ARRAY
type_getters[<int>SOS_TYPE_UINT16_ARRAY] = get_UINT16_ARRAY
type_getters[<int>SOS_TYPE_UINT32_ARRAY] = get_UINT32_ARRAY
type_getters[<int>SOS_TYPE_UINT64_ARRAY] = get_UINT64_ARRAY
type_getters[<int>SOS_TYPE_FLOAT_ARRAY] = get_FLOAT_ARRAY
type_getters[<int>SOS_TYPE_DOUBLE_ARRAY] = get_DOUBLE_ARRAY
type_getters[<int>SOS_TYPE_LONG_DOUBLE_ARRAY] = get_LONG_DOUBLE_ARRAY
type_getters[<int>SOS_TYPE_OBJ_ARRAY] = get_ERROR

cdef check_len(sos_value_data_t c_data, int sz):
    if c_data.array.count < sz:
        raise ValueError("The array can only accomodate "
                         "{0} members, {1} were provided".\
                         format(c_data.array.count, sz))

################################
# Object attribute setter functions
################################
cdef set_LONG_DOUBLE_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.long_double_[i] = val[i]

cdef set_DOUBLE_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.double_[i] = val[i]

cdef set_FLOAT_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.float_[i] = val[i]

cdef set_UINT64_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.uint64_[i] = val[i]

cdef set_UINT32_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.uint32_[i] = val[i]

cdef set_UINT16_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.uint16_[i] = val[i]

cdef set_BYTE_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef uint8_t *s
    cdef int sz = len(val)
    check_len(c_data, sz)
    t = type(val)
    if t != bytearray:
        if t == str:
            val = val.encode()
        elif t == list:
            val = bytearray(val)
        else:
            raise ValueError(f"BYTE_ARRAY accepts only "
                             "str, list, and bytearray, not {t}")
    s = val
    memcpy(c_data.array.data.byte_, s, sz);

cdef set_INT64_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.int64_[i] = <int64_t>val[i]

cdef set_INT32_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.int32_[i] = val[i]

cdef set_INT16_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int i, sz = len(val)
    check_len(c_data, sz)
    for i in range(sz):
        c_data.array.data.int16_[i] = val[i]

cdef set_CHAR_ARRAY(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef char *s
    cdef int sz = len(val)
    check_len(c_data, sz)
    t = type(val)
    if t != bytearray:
        if t == str:
            val = val.encode()
        elif t == list:
            val = bytearray(val)
        else:
            raise ValueError(f"CHAR_ARRAY accepts only "
                             "str, list, and bytearray, not {t}")
    s = val
    memcpy(c_data.array.data.char_, s, sz);

cdef set_TIMESTAMP(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef int secs
    cdef int usecs
    cdef typ = type(val)
    if typ == tuple:
        try:
            secs = <int>val[0]
            usecs = <int>val[1]
        except:
            raise ValueError("Error assigning {0} to timestamp."\
                              "A timestamp must be a tuple of "\
                              "( int(secs), int(usecs) )".format(val))
    elif typ == float or typ == np.float64 or typ == np.float32:
        try:
            secs = int(val)
            usecs = int((val - float(secs)) * 1.e6)
        except:
            raise ValueError("Error assigning {0} to timestamp. "\
                             "The time value of a floating point "\
                             "number must be non-negative".format(val))
    elif typ == np.datetime64:
        ts = val.astype('int')
        secs = ts / 1000000L
        usecs = ts % 1000000L
    elif typ == dt.datetime:
        ts = (val - dt.datetime(1970,1,1)).total_seconds()
        secs = int(ts)
        usecs = int((ts - secs) * 1.e6)
    elif typ == int:
        secs = val
        usecs = 0
    else:
        raise ValueError("Timestamp of type {0}, must be float, tuple or int".format(type(val)))
    c_data.prim.timestamp_.tv.tv_sec = secs
    c_data.prim.timestamp_.tv.tv_usec = usecs

cdef set_LONG_DOUBLE(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.long_double_ = <long double>val

cdef set_DOUBLE(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.double_ = <double>val

cdef set_FLOAT(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.float_ = <float>val

cdef set_UINT64(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.uint64_ = <uint64_t>val

cdef set_UINT32(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.uint32_ = <uint32_t>val

cdef set_UINT16(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.uint16_ = <uint16_t>val

cdef set_INT64(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.int64_ = <int64_t>val

cdef set_INT32(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.int32_ = <int32_t>val

cdef set_INT16(sos_attr_t c_attr, sos_value_data_t c_data, val):
    c_data.prim.int16_ = <int16_t>val

cdef set_STRUCT(sos_attr_t c_attr, sos_value_data_t c_data, val):
    cdef const unsigned char *s
    cdef int count
    cdef int l, i
    if c_attr == NULL:
        raise ValueError("Cannot set a STRUCT attribute value without an sos_attr_t value")
    count = sos_attr_size(c_attr)
    l = min(len(val), count)
    if l < count:
        memset(&c_data.struc.byte_[l], 0, count - l)
    if isinstance(val, np.ndarray):
        b = bytearray(val.tobytes())
        for i in range(0, l):
            c_data.struc.byte_[i] = b[i]
    else:
        s = val
        memcpy(c_data.struc.byte_, s, l)

cdef set_ERROR(sos_attr_t c_attr, sos_value_data_t c_data, val):
    raise ValueError("Set is not supported on this attribute type")

ctypedef object (*type_setter_fn_t)(sos_attr_t c_attr, sos_value_data_t c_data, val)
cdef type_setter_fn_t type_setters[SOS_TYPE_LAST+1]
type_setters[<int>SOS_TYPE_INT16] = set_INT16
type_setters[<int>SOS_TYPE_INT32] = set_INT32
type_setters[<int>SOS_TYPE_INT64] = set_INT64
type_setters[<int>SOS_TYPE_UINT16] = set_UINT16
type_setters[<int>SOS_TYPE_UINT32] = set_UINT32
type_setters[<int>SOS_TYPE_UINT64] = set_UINT64
type_setters[<int>SOS_TYPE_FLOAT] = set_FLOAT
type_setters[<int>SOS_TYPE_DOUBLE] = set_DOUBLE
type_setters[<int>SOS_TYPE_LONG_DOUBLE] = set_LONG_DOUBLE
type_setters[<int>SOS_TYPE_TIMESTAMP] = set_TIMESTAMP
type_setters[<int>SOS_TYPE_OBJ] = set_ERROR
type_setters[<int>SOS_TYPE_JOIN] = set_ERROR
type_setters[<int>SOS_TYPE_STRUCT] = set_STRUCT
type_setters[<int>SOS_TYPE_BYTE_ARRAY] = set_BYTE_ARRAY
type_setters[<int>SOS_TYPE_CHAR_ARRAY] = set_CHAR_ARRAY
type_setters[<int>SOS_TYPE_INT16_ARRAY] = set_INT16_ARRAY
type_setters[<int>SOS_TYPE_INT32_ARRAY] = set_INT32_ARRAY
type_setters[<int>SOS_TYPE_INT64_ARRAY] = set_INT64_ARRAY
type_setters[<int>SOS_TYPE_UINT16_ARRAY] = set_UINT16_ARRAY
type_setters[<int>SOS_TYPE_UINT32_ARRAY] = set_UINT32_ARRAY
type_setters[<int>SOS_TYPE_UINT64_ARRAY] = set_UINT64_ARRAY
type_setters[<int>SOS_TYPE_FLOAT_ARRAY] = set_FLOAT_ARRAY
type_setters[<int>SOS_TYPE_DOUBLE_ARRAY] = set_DOUBLE_ARRAY
type_setters[<int>SOS_TYPE_LONG_DOUBLE_ARRAY] = set_LONG_DOUBLE_ARRAY
type_setters[<int>SOS_TYPE_OBJ_ARRAY] = set_ERROR

cdef set_key_LONG_DOUBLE_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(float(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(long double)
    for i in range(count):
        c_key.long_double_[i] = <long double>val[i]

cdef set_key_DOUBLE_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(float(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(double)
    for i in range(count):
        c_key.double_[i] = <double>val[i]

cdef set_key_FLOAT_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(float(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(float)
    for i in range(count):
        c_key.float_[i] = <float>val[i]

cdef set_key_UINT64_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(int(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(uint64_t)
    for i in range(count):
        c_key.uint64_[i] = <uint64_t>val[i]

cdef set_key_UINT32_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(int(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(uint32_t)
    for i in range(count):
        c_key.uint32_[i] = <uint32_t>val[i]

cdef set_key_UINT16_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(int(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(uint16_t)
    for i in range(count):
        c_key.uint16_[i] = <uint16_t>val[i]

cdef set_key_BYTE_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    if type(val) == list:
        c_key.len = len(val)
        for i in range(c_key.len):
            c_key.byte_[i] = <char>val[i]
        return
    if type(val) == bytearray:
        c_key.len = len(val)
        for i in range(c_key.len):
            c_key.byte_[i] = <char>val[i]
        return
    raise ValueError("The value for a BYTE_ARRAY key must be a list "
                     "or a bytearray()")

cdef set_key_INT64_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(int(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(int64_t)
    for i in range(count):
        c_key.int64_[i] = <int64_t>val[i]

cdef set_key_INT32_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(int(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(int32_t)
    for i in range(count):
        c_key.int32_[i] = <int32_t>val[i]

cdef set_key_INT16_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    cdef int count
    if type(val) == str:
        val = list(int(v) for v in val.split(','))
    count = len(val)
    c_key.len = count * sizeof(int16_t)
    for i in range(count):
        c_key.int16_[i] = <int16_t>val[i]

cdef set_key_CHAR_ARRAY(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef int i
    c_key.len = len(val)
    ba = bytearray(val, encoding='utf-8')
    for i in range(c_key.len):
        c_key.value[i] = <char>ba[i]

cdef set_key_TIMESTAMP(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == int:
        c_key.tv_.tv_sec = val
        c_key.tv_.tv_usec = 0
    elif type(val) == tuple or type(val) == list:
        c_key.tv_.tv_sec = val[0]
        c_key.tv_.tv_usec = val[1]
    elif type(val) == float:
        c_key.tv_.tv_sec = int(val)
        c_key.tv_.tv_usec = int((val - int(val)) * 1.0e6)
    else:
        raise ValueError("The time value is a tuple, list, float or unix timestamp")

cdef set_key_LONG_DOUBLE(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = float(val)
    c_key.long_double_[0] = <long double>val

cdef set_key_DOUBLE(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = float(val)
    c_key.len = 8
    c_key.double_[0] = <double>val

cdef set_key_FLOAT(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = float(val)
    c_key.len = 4
    c_key.float_[0] = <float>val

cdef set_key_UINT64(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = int(val)
    c_key.len = 8
    c_key.uint64_[0] = <uint64_t>val

cdef set_key_UINT32(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = int(val)
    c_key.len = 4
    c_key.uint32_[0] = <uint32_t>val

cdef set_key_UINT16(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = int(val)
    c_key.len = 2
    c_key.uint16_[0] = <uint16_t>val

cdef set_key_INT64(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = int(val)
    c_key.len = 8
    c_key.int64_[0] = <int64_t>val

cdef set_key_INT32(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = int(val)
    c_key.len = 4
    c_key.int32_[0] = <int32_t>val

cdef set_key_INT16(sos_attr_t c_attr, ods_key_value_t c_key, val):
    if type(val) == str:
        val = int(val)
    c_key.len = 2
    c_key.int16_[0] = <int16_t>val

cdef set_key_STRUCT(sos_attr_t c_attr, ods_key_value_t c_key, val):
    cdef char *s
    cdef int count
    cdef int l = len(val)
    if c_attr == NULL:
        raise ValueError("Attr is required to set STRUCT key")
    val = val.encode()
    s = val
    count = sos_attr_size(c_attr)
    c_key.len = count
    if l < count:
        memset(c_key.value, 0, count)
        count = l
    memcpy(c_key.value, s, count)

cdef set_key_ERROR(sos_attr_t c_attr, ods_key_value_t c_key, val):
    raise ValueError("Set is not supported on this attribute type")

ctypedef object (*key_setter_fn_t)(sos_attr_t c_attr, ods_key_value_t c_key, val)
cdef key_setter_fn_t key_setters[SOS_TYPE_LAST+1]
key_setters[<int>SOS_TYPE_INT16] = set_key_INT16
key_setters[<int>SOS_TYPE_INT32] = set_key_INT32
key_setters[<int>SOS_TYPE_INT64] = set_key_INT64
key_setters[<int>SOS_TYPE_UINT16] = set_key_UINT16
key_setters[<int>SOS_TYPE_UINT32] = set_key_UINT32
key_setters[<int>SOS_TYPE_UINT64] = set_key_UINT64
key_setters[<int>SOS_TYPE_FLOAT] = set_key_FLOAT
key_setters[<int>SOS_TYPE_DOUBLE] = set_key_DOUBLE
key_setters[<int>SOS_TYPE_LONG_DOUBLE] = set_key_LONG_DOUBLE
key_setters[<int>SOS_TYPE_TIMESTAMP] = set_key_TIMESTAMP
key_setters[<int>SOS_TYPE_OBJ] = set_key_ERROR
key_setters[<int>SOS_TYPE_JOIN] = set_key_ERROR
key_setters[<int>SOS_TYPE_STRUCT] = set_key_STRUCT
key_setters[<int>SOS_TYPE_BYTE_ARRAY] = set_key_BYTE_ARRAY
key_setters[<int>SOS_TYPE_CHAR_ARRAY] = set_key_CHAR_ARRAY
key_setters[<int>SOS_TYPE_INT16_ARRAY] = set_key_INT16_ARRAY
key_setters[<int>SOS_TYPE_INT32_ARRAY] = set_key_INT32_ARRAY
key_setters[<int>SOS_TYPE_INT64_ARRAY] = set_key_INT64_ARRAY
key_setters[<int>SOS_TYPE_UINT16_ARRAY] = set_key_UINT16_ARRAY
key_setters[<int>SOS_TYPE_UINT32_ARRAY] = set_key_UINT32_ARRAY
key_setters[<int>SOS_TYPE_UINT64_ARRAY] = set_key_UINT64_ARRAY
key_setters[<int>SOS_TYPE_FLOAT_ARRAY] = set_key_FLOAT_ARRAY
key_setters[<int>SOS_TYPE_DOUBLE_ARRAY] = set_key_DOUBLE_ARRAY
key_setters[<int>SOS_TYPE_LONG_DOUBLE_ARRAY] = set_key_LONG_DOUBLE_ARRAY
key_setters[<int>SOS_TYPE_OBJ_ARRAY] = set_key_ERROR

cdef int size_INT16(arg):
    return sizeof(int16_t)

cdef int size_UINT16(arg):
    return sizeof(uint16_t)

cdef int size_INT32(arg):
    return sizeof(int32_t)

cdef int size_UINT32(arg):
    return sizeof(uint32_t)

cdef int size_INT64(arg):
    return sizeof(int64_t)

cdef int size_UINT64(arg):
    return sizeof(uint64_t)

cdef int size_FLOAT(arg):
    return sizeof(float)

cdef int size_DOUBLE(arg):
    return sizeof(double)

cdef int size_LONG_DOUBLE(arg):
    return sizeof(long double)

cdef int size_TIMESTAMP(arg):
    return sizeof(ods_timeval_s)

cdef int size_ERROR(arg):
    raise ValueError("The type has no key size.")

cdef int size_STRUCT(arg):
    return len(arg)

cdef int size_JOIN(arg):
    return len(arg)

cdef int size_BYTE_ARRAY(arg):
    return len(arg)

cdef int size_CHAR_ARRAY(arg):
    return len(arg)

cdef int size_INT16_ARRAY(arg):
    return sizeof(int16_t) * len(arg)

cdef int size_UINT16_ARRAY(arg):
    return sizeof(uint16_t) * len(arg)

cdef int size_INT32_ARRAY(arg):
    return sizeof(int32_t) * len(arg)

cdef int size_UINT32_ARRAY(arg):
    return sizeof(uint32_t) * len(arg)

cdef int size_INT64_ARRAY(arg):
    return sizeof(int64_t) * len(arg)

cdef int size_UINT64_ARRAY(arg):
    return sizeof(uint64_t) * len(arg)

cdef int size_FLOAT_ARRAY(arg):
    return sizeof(float) * len(arg)

cdef int size_DOUBLE_ARRAY(arg):
    return sizeof(double) * len(arg)

cdef int size_LONG_DOUBLE_ARRAY(arg):
    return sizeof(long double) * len(arg)

ctypedef int (*type_size_fn_t)(arg)
cdef type_size_fn_t type_sizes[SOS_TYPE_LAST+1]
type_sizes[<int>SOS_TYPE_INT16] = size_INT16
type_sizes[<int>SOS_TYPE_INT32] = size_INT32
type_sizes[<int>SOS_TYPE_INT64] = size_INT64
type_sizes[<int>SOS_TYPE_UINT16] = size_UINT16
type_sizes[<int>SOS_TYPE_UINT32] = size_UINT32
type_sizes[<int>SOS_TYPE_UINT64] = size_UINT64
type_sizes[<int>SOS_TYPE_FLOAT] = size_FLOAT
type_sizes[<int>SOS_TYPE_DOUBLE] = size_DOUBLE
type_sizes[<int>SOS_TYPE_LONG_DOUBLE] = size_LONG_DOUBLE
type_sizes[<int>SOS_TYPE_TIMESTAMP] = size_TIMESTAMP
type_sizes[<int>SOS_TYPE_OBJ] = size_ERROR
type_sizes[<int>SOS_TYPE_JOIN] = size_JOIN
type_sizes[<int>SOS_TYPE_STRUCT] = size_STRUCT
type_sizes[<int>SOS_TYPE_BYTE_ARRAY] = size_BYTE_ARRAY
type_sizes[<int>SOS_TYPE_CHAR_ARRAY] = size_CHAR_ARRAY
type_sizes[<int>SOS_TYPE_INT16_ARRAY] = size_INT16_ARRAY
type_sizes[<int>SOS_TYPE_INT32_ARRAY] = size_INT32_ARRAY
type_sizes[<int>SOS_TYPE_INT64_ARRAY] = size_INT64_ARRAY
type_sizes[<int>SOS_TYPE_UINT16_ARRAY] = size_UINT16_ARRAY
type_sizes[<int>SOS_TYPE_UINT32_ARRAY] = size_UINT32_ARRAY
type_sizes[<int>SOS_TYPE_UINT64_ARRAY] = size_UINT64_ARRAY
type_sizes[<int>SOS_TYPE_FLOAT_ARRAY] = size_FLOAT_ARRAY
type_sizes[<int>SOS_TYPE_DOUBLE_ARRAY] = size_DOUBLE_ARRAY
type_sizes[<int>SOS_TYPE_LONG_DOUBLE_ARRAY] = size_LONG_DOUBLE_ARRAY
type_sizes[<int>SOS_TYPE_OBJ_ARRAY] = size_ERROR

cdef class Value(object):
    cdef sos_value_s c_v_
    cdef sos_value_t c_v
    cdef sos_attr_t c_attr
    cdef sos_obj_t c_obj
    cdef int c_str_sz
    cdef char *c_str
    cdef type_setter_fn_t set_fn
    cdef type_getter_fn_t get_fn

    def __init__(self, Attr attr, Object obj=None):
        cdef int typ
        if obj is not None:
            self.c_obj = sos_obj_get(obj.c_obj)
        else:
            self.c_obj = NULL
        self.c_attr = attr.c_attr
        self.c_v = sos_value_init(&self.c_v_, self.c_obj, self.c_attr)
        if self.c_v == NULL:
            raise ValueError("The value could not be initialized from {0}".format(attr.name()))
        typ = sos_attr_type(self.c_attr)
        self.set_fn = type_setters[typ]
        self.get_fn = type_getters[typ]

    @property
    def value(self):
        return self.get_fn(self.c_obj, self.c_v.data, self.c_attr)

    @value.setter
    def value(self, v):
        cdef int sz
        if 0 == sos_attr_is_array(self.c_attr):
            self.set_fn(self.c_attr, self.c_v.data, v)
        else:
            sz = len(v)
            self.c_v = sos_array_new(&self.c_v_, self.c_attr, self.c_obj, sz)
            if self.c_v == NULL:
                raise MemoryError()
            self.set_fn(self.c_attr, <sos_value_data_t>sos_array(self.c_v), v)

    cdef assign(self, sos_obj_t c_obj):
        cdef int typ
        if self.c_obj:
            sos_obj_put(self.c_obj)
        self.c_obj = c_obj
        self.c_v = sos_value_init(&self.c_v_, self.c_obj, self.c_attr)
        if self.c_v == NULL:
            raise ValueError("The c_obj could not be assigned to the value")
        typ = sos_attr_type(self.c_attr)
        self.set_fn = type_setters[typ]
        self.get_fn = type_getters[typ]
        return self

    @property
    def obj(self):
        return None

    def set_obj(self, Object o):
        self.assign(o.c_obj)

    cdef _set_obj_(self, Object o):
        self.assign(o.c_obj)

    @obj.setter
    def obj(self, o):
        self._set_obj_(o)

    def name(self):
        """Return the value's attribute name"""
        return sos_attr_name(self.c_attr).decode('utf-8')

    def strlen(self):
        """
        Return the length of the string if the value
        were formatted as a string
        """
        return sos_value_strlen(self.c_v)

    def from_str(self, string):
        """Set the value from the string"""
        return sos_value_from_str(self.c_v, string.encode(), NULL)

    def to_str(self):
        """Return the value as a formatted string"""
        cdef int rc
        cdef int sz = sos_value_strlen(self.c_v)
        if self.c_str == NULL:
            self.c_str_sz = sz
            self.c_str = <char *>malloc(self.c_str_sz + 1)
            if self.c_str == NULL:
                raise MemoryError("Insufficient memory to allocate {0} bytes.".format(sz))
        else:
            if self.c_str_sz < sz:
                self.c_str_sz = sz
                free(self.c_str)
                self.c_str = <char *>malloc(self.c_str_sz + 1)
                if self.c_str == NULL:
                    raise MemoryError("Insufficient memory to allocate {0} bytes.".format(sz))
        return sos_value_to_str(self.c_v, self.c_str, sz).decode('utf-8')

    def __str__(self):
        return self.to_str()

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        if self.c_str:
            free(self.c_str)
        if self.c_v:
            sos_value_put(self.c_v)
            self.c_v = NULL
        if self.c_obj:
            sos_obj_put(self.c_obj)
            self.c_obj = NULL

cdef class Object(object):
    """
    The Object encapsulates the SOS container object in memory.
    To support python-like usage, it implements an internal
    dictionary of values indexed by the attribute name, and an
    internal list of values indexed by attribute id. From Python,
    O['<name>'] will return the value of that attribute as a native
    Python object. Similarly, O[<id>] will return the same Python
    object where the id == sos_attr_id() of the attribute with
    the name <name>.
    """
    cdef sos_obj_t c_obj
    cdef sos_schema_t c_schema
    cdef id_list
    cdef name_dict

    def __init__(self):
        self.c_obj = NULL
        self.c_schema = NULL

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        self.release()

    def release(self):
        if self.c_obj:
            sos_obj_put(self.c_obj)
            self.c_obj = NULL

    cdef assign(self, sos_obj_t obj):
        if obj == NULL:
            raise ValueError("obj argument cannot be NULL")
        self.c_obj = obj
        self.c_schema = sos_obj_schema(obj)
        return self

    cdef get_py_value(self, sos_obj_t c_obj, sos_attr_t c_attr, sos_value_data_t c_data):
        cdef int t = sos_attr_type(c_attr)
        return <object>type_getters[<int>t](c_obj, c_data, c_attr)

    cdef set_py_array_value(self, sos_attr_t c_attr, val):
        cdef sos_value_s v_
        cdef sos_value_s *v
        cdef int t = sos_attr_type(c_attr)
        v = sos_array_new(&v_, c_attr, self.c_obj, len(val))
        if v == NULL:
            raise MemoryError()
        <object>type_setters[<int>t](c_attr, v.data, val)
        sos_value_put(v)

    cdef set_py_value(self, sos_attr_t c_attr, val):
        cdef sos_value_data_t c_data
        cdef int t = sos_attr_type(c_attr)
        c_data = sos_obj_attr_data(self.c_obj, c_attr, NULL)
        <object>type_setters[<int>t](c_attr, c_data, val)

    def __getitem__(self, idx):
        cdef sos_obj_t arr_obj
        cdef sos_attr_t c_attr
        cdef sos_value_data_t c_data
        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")

        if slice == type(idx):
            # Slice item retrieval
            ret = []
            _start = idx.start if idx.start else 0
            _stop = idx.stop if idx.stop \
                    else sos_schema_attr_count(self.c_schema)
            _step = idx.step if idx.step else 1
            for _i in range(_start, _stop, _step):
                c_attr = sos_schema_attr_by_id(sos_obj_schema(self.c_obj), _i)
                if c_attr == NULL:
                    raise ValueError("Object has no attribute with id '{0}'".format(idx))
                arr_obj = NULL
                c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
                ret.append(self.get_py_value(self.c_obj, c_attr, c_data))
                if arr_obj != NULL:
                    sos_obj_put(arr_obj);
            return ret
        if int == type(idx):
            c_attr = sos_schema_attr_by_id(sos_obj_schema(self.c_obj), idx)
        else:
            c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), idx.encode())
        if c_attr == NULL:
            raise StopIteration("Object has no attribute with id '{0}'".format(idx))
        arr_obj = NULL
        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        res = self.get_py_value(self.c_obj, c_attr, c_data)
        if arr_obj != NULL:
            sos_obj_put(arr_obj)
        return res

    def __getattr__(self, name):
        cdef sos_obj_t arr_obj
        cdef sos_attr_t c_attr
        cdef sos_value_data_t c_data
        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")
        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name.encode())
        if c_attr == NULL:
            raise ValueError("Object has no attribute with name '{0}'".format(name))
        arr_obj = NULL
        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        res = self.get_py_value(self.c_obj, c_attr, c_data)
        if arr_obj != NULL:
            sos_obj_put(arr_obj);
        return res

    def __setitem__(self, idx, val):
        cdef int iidx
        cdef sos_attr_t c_attr
        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")
        if type(idx) == slice:
            # slice assignment -- obj[a:b] = seq
            _start = idx.start if idx.start else 0
            _stop = idx.stop if idx.stop \
                    else sos_schema_attr_count(self.c_schema)
            _step = idx.step if idx.step else 1
            for (_i, _v) in zip(range(_start, _stop, _step), val):
                c_attr = sos_schema_attr_by_id(self.c_schema, _i)
                if c_attr == NULL:
                    raise ValueError("Object has no attribute with id '{0}'".format(_i))
                if 0 == sos_attr_is_array(c_attr):
                    self.set_py_value(c_attr, _v)
                else:
                    self.set_py_array_value(c_attr, _v)
            return

        # single index assignment
        elif type(idx) == str:
            c_attr = sos_schema_attr_by_name(self.c_schema, idx.encode())
        # assume it's a number
        else:
            iidx = int(idx)
            c_attr = sos_schema_attr_by_id(self.c_schema, iidx)
            if c_attr == NULL:
                raise ValueError("Object has no attribute with id '{0}'".format(idx))
        if 0 == sos_attr_is_array(c_attr):
            self.set_py_value(c_attr, val)
        else:
            self.set_py_array_value(c_attr, val)

    def __setattr__(self, name, val):
        cdef sos_attr_t c_attr
        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")
        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name.encode())
        if c_attr == NULL:
            raise ObjAttrError(name)
        if 0 == sos_attr_is_array(c_attr):
            self.set_py_value(c_attr, val)
        else:
            self.set_py_array_value(c_attr, val)

    def set_array_size(self, name, size):
        """
        For array attributes, this will initialize the array to the
        specified size. Other attribute types will generate a
        ValueError()
        """
        cdef sos_attr_t c_attr
        cdef sos_type_t c_type
        cdef sos_value_s v_
        cdef sos_value_s *v

        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")
        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name.encode())
        if c_attr == NULL:
            raise ObjAttrError(name)
        c_type = sos_attr_type(c_attr)
        if c_type < SOS_TYPE_ARRAY:
            raise TypeError("This method only works on array types")
        v = sos_array_new(&v_, c_attr, self.c_obj, size)
        if v == NULL:
            raise MemoryError()
        sos_value_put(v)

    def commit(self):
        return sos_obj_commit(self.c_obj)

    def index_add(self):
        """
        Add the object to all schema indices
        """
        if self.c_obj == NULL:
            self.abort("There is no container object associated with this Object")
        rc = sos_obj_commit(self.c_obj)
        if rc:
            return rc
        return sos_obj_index(self.c_obj)

    def index_del(self):
        """
        Remove the object from all schema indices
        """
        if self.c_obj == NULL:
            self.abort("There is no container object associated with this Object")
        return sos_obj_remove(self.c_obj)

    def delete(self):
        """
        Remove the object from the container
        """
        sos_obj_delete(self.c_obj)
        sos_obj_put(self.c_obj)
        self.c_obj = NULL

    def type_of(self):
        if self.c_obj == NULL:
            self.abort("There is no container object associated with this Object")
        return sos_schema_name(sos_obj_schema(self.c_obj))

    def get_schema(self):
        s = Schema()
        s.c_schema = self.c_schema
        return s

    def as_ndarray(self, name, eltype = np.uint64):
        """
        Return the object data or an attributes data as an ndarray of the
        specified type

        Positional Parameters:
        -- The name of the object attribute

        Keyword parameters:
        eltype - The type of each element in the ndarray
        """
        cdef int t
        cdef sos_obj_t arr_obj
        cdef sos_attr_t c_attr
        cdef sos_value_data_t c_data
        cdef np.npy_int size
        cdef np.npy_intp shape[1]

        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")

        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name.encode())
        if c_attr == NULL:
            raise ObjAttrError(name)
        t = sos_attr_type(c_attr)
        if t < SOS_TYPE_ARRAY:
            size = sos_attr_size(c_attr)
        else:
            raise TypeError("Use the attribute accessor directly for arrays")

        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        # convert size in bytes to array count
        size = size // np.dtype(eltype).itemsize
        shape[0] = size
        res = np.PyArray_SimpleNewFromData(1, shape, np.dtype(eltype).num,
                                           c_data.array.data.byte_)
        if arr_obj != NULL:
            sos_obj_put(arr_obj)
        return res

class ObjAttrError(NameError):
    def __init__(self, attr):
        NameError.__init__(self,
                           "Object has not attribute with the name '{0}'" \
                           .format(attr))

ctypedef void (*nda_setter_fn_t)(np.ndarray nda, int idx, sos_value_t v)
ctypedef void (*nda_resample_fn_t)(np.ndarray nda, int idx, sos_value_t v,
                                   double bin_samples, double bin_width)
cdef struct nda_setter_opt_s:
    sos_attr_t attr             # attribute in schema
    int idx                     # index of this object in objects[]
    nda_setter_fn_t setter_fn
    nda_resample_fn_t resample_fn
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

cdef nda_setter_fn_t *nda_setters = [
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

cdef void int16_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                             double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    nda[idx] = ((nda[idx] * bc) + v.data.prim.int16_) / (bc + 1)

cdef void int32_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                             double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    nda[idx] = ((nda[idx] * bc) + v.data.prim.int32_) / (bc + 1)

cdef void int64_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                             double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    nda[idx] = ((nda[idx] * bc) + v.data.prim.int16_) / (bc + 1)

cdef void uint16_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                              double bin_samples, double bin_width):
    nda[idx] = ((nda[idx] * bin_samples) + <double>v.data.prim.uint16_) / (bin_samples + 1.0)

cdef void uint32_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                              double bin_samples, double bin_width):
    nda[idx] = ((nda[idx] * bin_samples) + <double>v.data.prim.uint32_) / (bin_samples + 1.0)

cdef void uint64_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                              double bin_samples, double bin_width):
    nda[idx] = ((nda[idx] * bin_samples) + <double>v.data.prim.uint64_) / (bin_samples + 1.0)

cdef void float_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                             double bin_samples, double bin_width):
    nda[idx] = ((nda[idx] * bin_samples) + v.data.prim.float_) / (bin_samples + 1.0)

cdef void double_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                              double bin_samples, double bin_width):
    nda[idx] = ((nda[idx] * bin_samples) + v.data.prim.double_) / (bin_samples + 1.0)

cdef void timestamp_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                 double bin_samples, double bin_width):
    cdef uint64_t t
    cdef uint64_t bw
    if bin_samples == 0.0:
        bw = int(bin_width)
        t = (v.data.prim.timestamp_.tv.tv_sec * 1000000L) + \
            v.data.prim.timestamp_.tv.tv_usec
        nda[idx] = t - (t % bw)

cdef void uint8_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                   double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bc) + v.data.array.data.byte_[i]) / (bc + 1)

cdef void int8_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                  double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bc) + v.data.array.data.char_[i]) / (bc + 1)

cdef void int16_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                   double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bc) + v.data.array.data.int16_[i]) / (bc + 1)

cdef void int32_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                   double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bc) + v.data.array.data.int32_[i]) / (bc + 1)

cdef void int64_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                   double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bc) + v.data.array.data.int64_[i]) / (bc + 1)

cdef void uint16_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                    double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bc) + v.data.array.data.uint16_[i]) / (bc + 1)

cdef void uint32_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                    double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bc) + v.data.array.data.uint32_[i]) / (bc + 1)

cdef void uint64_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                    double bin_samples, double bin_width):
    cdef int bc = int(bin_samples)
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bc) + v.data.array.data.uint64_[i]) / (bc + 1)

cdef void float_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                    double bin_samples, double bin_width):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bin_samples) + v.data.array.data.float_[i]) / (bin_samples + 1.0)

cdef void double_array_nda_resample(np.ndarray nda, int idx, sos_value_t v,
                                    double bin_samples, double bin_width):
    cdef int i
    ndb = nda[idx]
    for i in range(0, v.data.array.count):
        ndb[i] = ((ndb[i] * bin_samples) + v.data.array.data.double_[i]) / (bin_samples + 1)


cdef nda_resample_fn_t *nda_resamplers = [
    int16_nda_resample,
    int32_nda_resample,
    int64_nda_resample,
    uint16_nda_resample,
    uint32_nda_resample,
    uint64_nda_resample,
    float_nda_resample,
    double_nda_resample,
    NULL,                       # long double
    timestamp_nda_resample,
    NULL,                       # obj
    NULL,                       # struct
    NULL,                       # join
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,     # twenty zeros to get to 32, i.e. SOS_TYPE_BYTE_ARRAY
    uint8_array_nda_resample,     # byte_array
    int8_array_nda_resample,      # char_array
    int16_array_nda_resample,
    int32_array_nda_resample,
    int64_array_nda_resample,
    uint16_array_nda_resample,
    uint32_array_nda_resample,
    uint64_array_nda_resample,
    float_array_nda_resample,
    double_array_nda_resample,
    NULL,                       # long double
    NULL,                       # obj array
]

cdef class QueryInputer:
    DEFAULT_ARRAY_LIMIT = 256
    cdef int start
    cdef int row_limit
    cdef int col_count
    cdef int row_count
    cdef sos_obj_t *objects

    def __init__(self, Query q, int limit, int start=0):
        self.row_limit = limit
        self.col_count = len(q.filters)
        self.objects = <sos_obj_t *>calloc(self.row_limit,
                                           sizeof(sos_obj_t) * self.col_count)
        if self.objects == NULL:
            raise MemoryError("Memory allocation failure")
        self.row_count = 0

    @property
    def capacity(self):
        """Return the row capacity of the result"""
        return self.row_limit

    @property
    def count(self):
        """Return the number of rows in the result"""
        return self.row_count

    def __len__(self):
        """Return the number of rows in the result"""
        return self.row_limit

    def __dealloc__(self):
        cdef int row_no
        cdef int filt_no
        cdef int row_idx
        cdef int obj_idx
        if self.objects == NULL:
            return
        for row_no in range(0, self.row_limit):
            row_idx = row_no * self.col_count
            for col_no in range(0, self.col_count):
                obj_idx = row_idx + col_no
                if self.objects[obj_idx] != NULL:
                    sos_obj_put(self.objects[obj_idx])
                    self.objects[obj_idx] = NULL
        free(self.objects)
        self.objects = NULL

    def __getitem__(self, idx):
        cdef int idx_ = (idx[0] * self.col_count) + idx[1]
        o = Object()
        o.assign(self.objects[idx_])
        return o

    def __setitem__(self, idx, value):
        cdef int idx_ = (idx[0] * self.col_count) + idx[1]
        self.objects[idx_] = <sos_obj_t>value

    def concat(self, QueryInputer result):
        """Concatenate query results

        Appends the QueryResult 'result' to the end of this result.

        A MemoryError exception is raised if there are insufficient
        resources to expand this result buffer.

        Postional Parameters:
        result -- The query result to append

        """
        cdef int newRows = self.row_count + result.row_count
        cdef int newBytes, oldBytes
        oldBytes = self.row_count * self.col_count * sizeof(sos_obj_t)
        newBytes = newRows * self.col_count * sizeof(sos_obj_t)
        if newRows > self.row_limit:
            self.objects = <sos_obj_t *>realloc(self.objects, newBytes)
            if self.objects == NULL:
                raise MemoryError("Insufficent resources for {0} bytes".format(newBytes))
        memcpy(&self.objects[oldBytes], result.objects, newBytes - oldBytes)
        self.row_count += result.row_count

    def input(self, Query query, reset=True):
        """Reads query results into memory

        Read as many as self.limit records from the container. If
        there is no more matching data or the input buffer is
        exhausted, the function returns False. Otherwise, the function
        returns True (i.e. more data remaining).

        Returns:
        True  -- There is more data available, but buffer space is exhausted
        False -- There is no buffer space available or there is no more matching data
        """
        cdef int filt_count = len(query.filters)
        cdef int idx, start, row_no, filt_no
        cdef int reset_ = int(reset)
        cdef sos_obj_t c_obj
        cdef Filter f

        if filt_count == 0:
            raise ValueError("The select method must be called before query")

        if reset_:
            start = self.start
        else:
            start = self.row_count

        for filt_no in range(start, filt_count):
            f = query.filters[filt_no]
            if reset_:
                if not query.desc:
                    c_obj = sos_filter_begin(f.c_filt)
                else:
                    c_obj = sos_filter_end(f.c_filt)
            else:
                if not query.desc:
                    c_obj = sos_filter_next(f.c_filt)
                else:
                    c_obj = sos_filter_prev(f.c_filt)
            if c_obj == NULL:
                return False
            else:
                self.objects[filt_no] = c_obj
        self.row_count = 1

        for row_no in range(start+1, self.row_limit):
            for filt_no in range(0, filt_count):
                f = query.filters[filt_no]
                c_obj = sos_filter_next(f.c_filt)
                if c_obj == NULL:
                    return False
                idx = (row_no * filt_count) + filt_no
                self.objects[idx] = c_obj
            self.row_count += 1

        if c_obj:
            return False
        return True

    def to_timeseries(self, Query query, timestamp='timestamp', interval_ms=None,
                      max_array=DEFAULT_ARRAY_LIMIT,
                      max_string=DEFAULT_ARRAY_LIMIT):
        """Return the QueryResult data as a DataSet"""
        cdef sos_obj_t c_o
        cdef sos_value_s v_, t_
        cdef sos_value_t v, t
        cdef int idx
        cdef int attr_idx
        cdef int res_idx
        cdef int atype
        cdef int nattr
        cdef Schema schema
        cdef Attr attr
        cdef sos_attr_t c_attr, t_attr
        cdef sos_attr_t *res_attr
        cdef int *res_type
        cdef nda_setter_opt res_acc
        cdef int type_id
        cdef double obj_time
        cdef double bin_width, bin_time, bin_value, bin_samples
        cdef typ_str
        cdef ColSpec col

        nattr = len(query.columns)

        res_attr = <sos_attr_t *>malloc(sizeof(sos_attr_t) * nattr)
        if res_attr == NULL:
            raise MemoryError("Insufficient memory to allocate dimension array")
        res_type = <int *>malloc(sizeof(uint64_t) * nattr)
        if res_type == NULL:
            free(res_attr)
            raise MemoryError("Insufficient memory to allocate type array")
        res_acc = <nda_setter_opt>malloc(sizeof(nda_setter_opt_s) * nattr)
        if res_acc == NULL:
            free(res_attr)
            free(res_type)
            raise MemoryError("Insufficient memory to allocate type array")

        schema = query.filters[0].get_attr().schema()
        t_attr = sos_schema_attr_by_name(schema.c_schema, timestamp.encode())
        if t_attr == NULL:
            raise ValueError("The timestamp attribute was not found in the schema. "
                             "Consider specifying the timestamp keyword parameter")
        if sos_attr_type(t_attr) != SOS_TYPE_TIMESTAMP:
            raise ValueError("The timestamp attribute {0} "
                             "is not a SOS_TYPE_TIMESTAMP".format(timestamp))

        result = []
        try:
            idx = 0
            for col in query.columns:

                attr = col.attr
                res_attr[idx] = <sos_attr_t>attr.c_attr
                res_type[idx] = attr.type()

                # set access defaults
                res_acc[idx].idx = col.cursor_idx
                res_acc[idx].setter_fn = nda_setters[res_type[idx]]
                res_acc[idx].resample_fn = nda_resamplers[res_type[idx]]

                atyp = col.attr_type
                if atyp == SOS_TYPE_TIMESTAMP:
                    typ_str = 'datetime64[us]'
                elif atyp == SOS_TYPE_STRUCT:
                    typ_str = 'uint8'
                elif atyp == SOS_TYPE_UINT64:
                    typ_str = 'double'
                elif atyp == SOS_TYPE_UINT32:
                    typ_str = 'double'
                elif atyp == SOS_TYPE_INT64:
                    typ_str = 'double'
                elif atyp == SOS_TYPE_INT32:
                    typ_str = 'double'
                else:
                    typ_str = sos_type_strs[atyp].lower()
                    typ_str = typ_str.replace('_array', '')

                if atyp >= TYPE_IS_ARRAY:
                    if atyp == SOS_TYPE_STRING:
                        data = np.zeros([ self.row_limit ],
                                        dtype=np.dtype('U{0}'.format(max_string)))
                    else:
                        data = np.zeros([ self.row_limit, int(max_array) ],
                                        dtype=np.dtype(typ_str))
                elif atyp == SOS_TYPE_STRUCT:
                    data = np.zeros([ self.row_limit, sos_attr_size(attr.c_attr) ],
                                    dtype=np.dtype(np.uint8))
                else:
                    data = np.zeros([ self.row_limit ], dtype=np.dtype(typ_str))
                result.append(data)
                idx += 1
        except Exception as e:
                free(res_attr)
                free(res_type)
                free(res_acc)
                raise ValueError("Error '{0}' processing the "
                                 "shape keyword parameter".format(str(e)))

        c_o = self.objects[0]
        t = sos_value_init(&t_, c_o, t_attr)
        obj_time = (<double>t.data.prim.timestamp_.tv.tv_sec * 1.0e6) + \
                   <double>t.data.prim.timestamp_.tv.tv_usec
        sos_value_put(t)

        if interval_ms is not None:
            bin_width = interval_ms * 1.0e3
        else:
            bin_width = 0.0

        res_idx = 0
        obj_idx = 0

        if bin_width == 0.0:
            for row_idx in range(0, self.row_limit):
                obj_idx = row_idx * self.col_count
                for attr_idx in range(0, nattr):
                    c_o = self.objects[obj_idx + res_acc[attr_idx].idx]
                    v = sos_value_init(&v_, c_o, res_attr[attr_idx])
                    res_acc[attr_idx].setter_fn(result[attr_idx], res_idx, v)
                    sos_value_put(v)
                res_idx += 1
        else:
            bin_start = obj_time - (obj_time % bin_width)
            bin_end = bin_start + bin_width
            bin_samples = 0.0
            for row_idx in range(0, self.row_limit):

                for attr_idx in range(0, nattr):
                    c_o = self.objects[obj_idx + res_acc[attr_idx].idx]
                    v = sos_value_init(&v_, c_o, res_attr[attr_idx])
                    res_acc[attr_idx].resample_fn(result[attr_idx], res_idx, v,
                                                  bin_samples, bin_width)
                    sos_value_put(v)

                obj_idx += self.col_count
                c_o = self.objects[obj_idx]
                t = sos_value_init(&t_, c_o, t_attr)
                obj_time = (<double>t.data.prim.timestamp_.tv.tv_sec * 1.0e6) + \
                           <double>t.data.prim.timestamp_.tv.tv_usec
                sos_value_put(t)

                if obj_time >= bin_end:
                    bin_start = bin_end
                    bin_end = bin_start + bin_width
                    bin_samples = 0.0
                    res_idx += 1
                else:
                    bin_samples += 1.0

        free(res_attr)
        free(res_type)
        free(res_acc)
        res = DataSet()
        for attr_idx in range(0, nattr):
            res.append_array(res_idx, query.columns[attr_idx].col_name,
                             result[attr_idx])
        return res

    def to_dataset(self, Query query, max_array=DEFAULT_ARRAY_LIMIT, max_string=DEFAULT_ARRAY_LIMIT):
        """Return the Query data as a DataSet"""
        cdef sos_obj_t c_o
        cdef sos_value_s v_
        cdef sos_value_t v
        cdef int idx
        cdef int attr_idx
        cdef int res_idx
        cdef int obj_idx
        cdef int row_idx
        cdef int nattr
        cdef Attr attr
        cdef int *res_type
        cdef nda_setter_opt res_acc
        cdef typ_str
        cdef ColSpec col

        nattr = len(query.columns)
        if nattr == 0 or self.row_count == 0:
            return None

        res_acc = <nda_setter_opt>malloc(sizeof(nda_setter_opt_s) * nattr)
        if res_acc == NULL:
            raise MemoryError("Insufficient memory")

        idx = 0
        result = []

        for col in query.columns:

            attr = col.attr

            res_acc[idx].attr = <sos_attr_t>attr.c_attr
            res_acc[idx].idx = col.cursor_idx
            res_acc[idx].setter_fn = nda_setters[attr.type()]
            res_acc[idx].resample_fn = nda_resamplers[attr.type()]

            atyp = col.attr_type
            if atyp == SOS_TYPE_TIMESTAMP:
                typ_str = 'datetime64[us]'
            elif atyp == SOS_TYPE_STRUCT:
                typ_str = 'uint8'
            elif atyp == SOS_TYPE_UINT64:
                typ_str = 'double'
            elif atyp == SOS_TYPE_UINT32:
                typ_str = 'double'
            elif atyp == SOS_TYPE_INT64:
                typ_str = 'double'
            elif atyp == SOS_TYPE_INT32:
                typ_str = 'double'
            else:
                typ_str = sos_type_strs[atyp].lower()
                typ_str = typ_str.replace('_array', '')

            if atyp >= TYPE_IS_ARRAY:
                if atyp == SOS_TYPE_STRING:
                    data = np.zeros([ self.row_count ],
                                    dtype=np.dtype('U{0}'.format(max_string)))
                else:
                    data = np.zeros([ self.row_count, max_array ],
                                    dtype=np.dtype(typ_str))
            elif atyp == SOS_TYPE_STRUCT:
                data = np.zeros([ self.row_limit, sos_attr_size(attr.c_attr) ],
                                dtype=np.dtype(np.uint8))
            else:
                data = np.zeros([ self.row_count ], dtype=np.dtype(typ_str))
            result.append(data)
            idx += 1

        res_idx = 0
        for row_idx in range(0, self.row_count):
            obj_idx = row_idx * self.col_count
            for attr_idx in range(0, nattr):
                c_o = self.objects[obj_idx + res_acc[attr_idx].idx]
                v = sos_value_init(&v_, c_o, res_acc[attr_idx].attr)
                res_acc[attr_idx].setter_fn(result[attr_idx], res_idx, v)
                sos_value_put(v)
            res_idx += 1
        self.row_count = 0

        res = DataSet()
        for attr_idx in range(0, nattr):
            res.append_array(res_idx,
                             sos_attr_name(res_acc[attr_idx].attr).decode(),
                             result[attr_idx])
        res.set_series_size(res_idx)
        free(res_acc)
        return res

    def to_dataframe(self, Query query, index=None,
                     max_array=DEFAULT_ARRAY_LIMIT, max_string=DEFAULT_ARRAY_LIMIT):
        """Return the Query data as a DataFrame"""
        cdef sos_obj_t c_o
        cdef sos_value_s v_
        cdef sos_value_t v
        cdef int idx
        cdef int attr_idx
        cdef int res_idx
        cdef int obj_idx
        cdef int row_idx
        cdef int nattr
        cdef Attr attr
        cdef int *res_type
        cdef nda_setter_opt res_acc
        cdef typ_str
        cdef ColSpec col

        nattr = len(query.columns)
        if nattr == 0 or self.row_count == 0:
            return None

        res_acc = <nda_setter_opt>malloc(sizeof(nda_setter_opt_s) * nattr)
        if res_acc == NULL:
            raise MemoryError("Insufficient memory")

        idx = 0
        result = []

        for col in query.columns:

            attr = col.attr

            res_acc[idx].attr = <sos_attr_t>attr.c_attr
            res_acc[idx].idx = col.cursor_idx
            res_acc[idx].setter_fn = nda_setters[attr.type()]
            res_acc[idx].resample_fn = nda_resamplers[attr.type()]

            atyp = col.attr_type
            if atyp == SOS_TYPE_TIMESTAMP:
                typ_str = 'datetime64[us]'
            elif atyp == SOS_TYPE_STRUCT:
                typ_str = 'uint8'
            elif atyp == SOS_TYPE_UINT64:
                typ_str = 'double'
            elif atyp == SOS_TYPE_UINT32:
                typ_str = 'double'
            elif atyp == SOS_TYPE_INT64:
                typ_str = 'double'
            elif atyp == SOS_TYPE_INT32:
                typ_str = 'double'
            else:
                typ_str = sos_type_strs[atyp].lower()
                typ_str = typ_str.replace('_array', '')


            if atyp >= TYPE_IS_ARRAY:
                if atyp == SOS_TYPE_STRING:
                    data = np.zeros([ self.row_count ],
                                    dtype=np.dtype('U{0}'.format(max_string)))
                else:
                    data = np.zeros([ self.row_count, max_array ],
                                    dtype=np.dtype(typ_str))
            elif atyp == SOS_TYPE_STRUCT:
                data = np.zeros([ self.row_limit, sos_attr_size(attr.c_attr) ],
                                dtype=np.dtype(np.uint8))
            else:
                data = np.zeros([ self.row_count ], dtype=np.dtype(typ_str))
            result.append(data)
            idx += 1

        res_idx = 0
        for row_idx in range(0, self.row_count):
            obj_idx = row_idx * self.col_count
            for attr_idx in range(0, nattr):
                c_o = self.objects[obj_idx + res_acc[attr_idx].idx]
                v = sos_value_init(&v_, c_o, res_acc[attr_idx].attr)
                res_acc[attr_idx].setter_fn(result[attr_idx], res_idx, v)
                sos_value_put(v)
            res_idx += 1
        self.row_count = 0

        pdres = {}
        df_idx = None
        for attr_idx in range(0, nattr):
            col_name = sos_attr_name(res_acc[attr_idx].attr).decode()
            pdres[col_name] = result[attr_idx]
            if index == col_name:
                df_idx = pd.DatetimeIndex(result[attr_idx])
        if index and df_idx is None:
            raise ValueError("The index column {0} is not present in the result".format(index))
        if df_idx is not None:
            res = pd.DataFrame(pdres, index=df_idx)
        else:
            res = pd.DataFrame(pdres)
        free(res_acc)
        return res

cdef class Query:
    QUERY_RESULT_LIMIT = 4096
    cdef Container cont     # The container
    cdef filter_idx             # dictionary mapping schema to filter index
    cdef filters                # array of filters
    cdef schema                 # array of schema
    cdef columns                # array of columns
    cdef int col_width          # default width of an output column
    cdef primary                # primary attribute
    cdef group_fn               # function that decides how objects are grouped
    cdef unique                 # Boolean indicating if queries are unique
    cdef desc	                # Boolean defining results order
    cdef inputer                # Maintains query results

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
        self.primary = None
        self.cont = container
        self.filter_idx = {}
        self.filters = []
        self.columns = []
        self.schema = []
        self.col_width = 15
        self.desc = False

    cdef __find_schema_with_attr(self, name):
        if len(self.schema) > 0:
            for s in self.schema:
                if name in s:
                    return s
        else:
            for s in self.cont.schema_iter():
                if name in s:
                    return s
        return None

    cdef __decode_attr_name(self, name):
        try:
            i = name.find('[')
            if i >= 0:
                sname = name[:i]
                aname = name[i+1:-1]
                schema = self.cont.schema_by_name(sname)
            else:
                schema = self.__find_schema_with_attr(name)
                aname = name
        except Exception as e:
            return (None, None)
        if schema:
            return (schema, schema[aname])
        raise ValueError("The attribute '{0}' is not present in any schema".
                         format(aname))

    @property
    def index_name(self):
        return self.primary

    @property
    def descending(self):
        return self.desc

    def set_col_width(self, width):
        self.col_width = width

    def get_col_width(self):
        return self.col_width

    def col_by_name(self, name):
        for col in self.columns:
            if name == col.col_name:
                return col
        return None

    def get_schemas(self):
        """Return all schema names in the container"""
        result = []
        for schema in self.cont.schema_iter():
            result.append(schema.name())
        return result

    def get_indices(self):
        """Return all indexed attributes in the container"""
        result = []
        for schema in self.cont.schema_iter():
            for attr in schema.attr_iter():
                if attr.is_indexed():
                    result.append(schema.name() + '[' + attr.name() + ']')
        return result

    def query(self, inputer=None, reset=True, wait=None):
        """Calls the inputer class' input function

        If the input() function returns True, the query continues and
        the input() function is called again with the next record. If
        the input() function returns False and wait is None, the query
        is discontinued and the call returns with the number of times
        input() was called.

        If wait is None, the query completes when the data is
        exhausted, or the row limit has been reached. Otherwise, wait
        must be a tuple containing a wait_fn and a wait_arg. The
        wait_fn(self, row, row_count, wait_arg) is called if input()
        returns False before the limit is reached. If wait_fn returns
        False, the query returns with the data already obtained, if
        wait_fn return True, the query loop continues.

        If query() is called again with reset=False, the query will
        continue with the next record following the last one delivered
        to input().

        The record argument to the input() function is a list of
        column values as defined in the column argument to select()

        Positional Parameters:

        -- An instance of an Inputer class that defines an
           Inputer.input() function and a Inputer.to_dataset(self)
           function. The input() fuction is called to read from the
           DataSource. The to_dataset() function is called to return
           the data as a DataSet. See the Inputer() class for more
           information.

        Keyword Parameters:
        reset -- Start at the beginning of the query (default is True)
        wait  -- A tuple containing a wait_fn and a wait_arg
                 (default is None)

        Returns:

            The number of times the Inputer's input function was
            called and returned True.

        Example:

            class Inputer(object):
                def __init__(self, query, row_limit=16, file=sys.stdout):
                    self.row_count = 0
                    self.row_limit = row_limit
                    self.query = query
                    self.file = file

                def input(self, row):
                    if self.row_count >= self.row_limit:
                        return False
                    for col in self.query.get_columns():
                        print(col, end=' ', file=self.file)
                    print("", file=self.file)
                    self.row_count += 1
                    return True

        """
        cdef int reset_
        if reset:
            reset_ = 1
        else:
            reset_ = 0
        if inputer is None:
            inputer = QueryInputer(self, self.QUERY_RESULT_LIMIT)
        self.inputer = inputer
        while True:
            if inputer.input(self, reset=reset_) == False:
                # The input limit has been reached
                break
            else:
                reset_ = 0
                if wait:
                    if wait[0](self, inputer, wait[1]) == False:
                        # The waiter says we have enough
                        break
                else:
                    break

        return inputer.count

    def _from_(self, schema):
        if type(schema) != tuple and type(schema) != list:
            raise ValueError("The from_ argument must be a list or tuple")
        self.schema = []
        for name in schema:
            sch = self.cont.schema_by_name(name)
            if sch is None:
                raise ValueError("The schema name '{0}' does not exist.".format(name))
            self.schema.append(sch)

    def _order_by(self, attr_name, from_):
        self.primary = None
        if from_ is None:
            schema_list = [ s for s in self.cont.schema_iter() ]
        else:
            schema_list = [ self.cont.schema_by_name(n) for n in from_ ]
        for schema in schema_list:
            # Check for exact match
            for attr in schema:
                if attr.is_indexed():
                    if attr_name == attr.name():
                        self.primary = attr_name
                        return

        for schema in schema_list:
            # If there is no exact match, check the prefix against join keys
            match = []
            for attr in schema:
                if attr.is_indexed() and attr.join_list() is not None:
                    if attr.name().startswith(attr_name):
                        match.append(attr.name())

        if len(match) == 0:
            raise ValueError("There is no index that "
                             "could be matched with {0}".format(attr_name))

        best = match[0]
        maxlen = len(best)

        # The best match is the one with the longest common prefix
        for m in match[1:]:
            l = len(m)
            if l > maxlen:
                best = m
                maxlen = l

        self.primary = best

    cdef _add_colspec(self, ColSpec colspec):
        schema, attr = self.__decode_attr_name(colspec.name)
        if not schema:
            raise ValueError("The attribute {0} was not found "
                             "in any schema.".format(colspec.name))
        if schema.name() not in self.filter_idx:
            # There is no filter yet for this schema. If there is
            # already a primary key, use it, otherwise this atttribute
            # becomes the primary key.
            if self.primary is None:
                self.primary = colspec.name
            try:
                primary_attr = schema[self.primary]
                f = Filter(primary_attr)
                if self.unique:
                    f.unique()
            except:
                raise ValueError("The schema {0} does not have the primary "
                                 "key attribute {1}".format(schema.name(),
                                                            self.primary))
            idx = len(self.filters)
            self.filter_idx[schema.name()] = idx
            self.filters.append(f)
        else:
            idx = self.filter_idx[schema.name()]
        colspec.update(self, idx, attr)
        self.columns.append(colspec)

    def release(self):
        self.primary = None
        self.schema = []
        for f in self.filters:
            f.release()
        self.filter_idx = {}
        self.filters = []
        self.columns = []

    def select(self, columns, order_by=None, desc=False,
               where=None, from_=None, unique=False):
        """Set the attribute list returned in the result

        Positional Parmeters:

        -- An array of column-specifications. A column-specification is
           either an attribute name (string), or a ColSpec object. The
           ColSpec object allows for the specification of how the data
           will be converted and formatted on output.

           Examples:

           The simplest format is an array of names:

             [ 'meminfo[timestamp]',
               'vmstat[timestamp]',
               'meminfo[job_id]',
               'meminfo[MemFree]',
               'vmstat[nr_free_pages]'
             ]

           A ColSpec object can be used to control data conversion and formatting:

              def fmt_timestamp(ts):
                  return str(dt.datetime.fromtimestamp(ts[0]))

              [ ColSpec('meminfo[timestamp]', cvt_fn=fmt_timestamp,
                   col_width=24, align=ColSpec.LEFT),
                ColSpec('vmstat[timestamp]', cvt_fn=fmt_timestamp,
                   col_width=24, align=ColSpec.LEFT),
                ColSpec('meminfo[job_id]', cvt_fn=int, col_width=12),
                'meminfo[MemFree]',
                'vmstat[nr_free_pages]',
              ]

        Keyword Parameters:

        from_     -- A list of schema names
        where     -- A list of conditions to filter the data
        order_by  -- The attribute to use as the primary index
        desc      -- Boolean to indicate if results should be in reverse order
        unique    -- Return only a single result for each matching key

        FROM_

          The 'from_' keyword is a list of schema names to search
          for attribute names.

          Example:

            from_ = [ 'meminfo', 'jobinfo' ]

        WHERE

          The 'where' keyword is a list of filter condition tuples.
          Each condition must be applicable to every schema. A
          condition tuple contains three elements as follows:

          ( attribute_name, condition, value )

          Example:

            where = [( 'timestamp', COND_GE, 1510597567.001617 )]

        ORDER_BY

          Specifies the name of the primary key. This attribute must be
          indexed and present in all schema referenced in the columns
          list.

          If the order_by keyword argument is omitted, the first column in the column
          list is used. If this attribute is not indexed, an exception
          will be raised.

          Example:

            order_by = 'job_comp_time'

        DESC

          Specifies if results should be returned in reverse order.
          The default is False.

          Example:

            desc = True

        UNIQUE

          Set the unique keyword to True to return only the 1st result
          if there are duplicate values for a key.

          Example:

            unique = True

        """
        if type(columns) != list and type(columns) != tuple:
            raise ValueError("The columns argument must be a list or tuple.")

        self.release()

        self.unique = unique
        self.desc = desc

        if order_by:
            # Must be before the Filter(s) are created
            self._order_by(order_by, from_)

        if from_:
            self._from_(from_)

        for col in columns:
            if type(col) == ColSpec:
                spec = copy.copy(col)
                self._add_colspec(spec)
            else:
                if '*' in col:
                    if '*' == col:
                        if len(self.columns) > 1:
                            raise ValueError("Ambiguous wildcard in column specification, "
                                             "use schema-name.* to identify column source")
                        if from_ is None:
                            raise ValueError("from_ is required with wildcard column names")
                        name = from_[0]
                    else:
                        name = col.split('[')[0]
                    schema = self.cont.schema_by_name(name)
                    if schema is None:
                        raise ValueError("Schema {0} was not found.".format(name))
                    for attr in schema.attr_iter():
                        if attr.type() == SOS_TYPE_JOIN:
                            continue
                        if len(from_) > 1:
                            spec = ColSpec(name + '[' + attr.name() + ']',
                                           col_width=self.get_col_width())
                        else:
                            spec = ColSpec(attr.name(), col_width=self.get_col_width())
                        self._add_colspec(spec)
                else:
                    spec = ColSpec(col, col_width=self.get_col_width())
                    self._add_colspec(spec)

        if where:
            # Must be after the Filter(s) are created
            self._where(where)

    def get_columns(self):
        """Return list of columns-specification (ColSpec)"""
        return self.columns

    def get_filters(self):
        """Return list of Filter objects"""
        return self.filters

    def default_group_fn(self, o1, o2):
        pass

    def set_group_fn(self, group_fn):
        """Sets the function that will be called when collection objects into a single row
        """
        self.group_fn = group_fn

    def _where(self, clause):
        if type(clause) != list and type(clause) != tuple:
            self.release()
            raise ValueError("The where arguement is a list of conditions.")
        for c in clause:
            if type(c) != list and type(c) != tuple:
                self.release()
                raise ValueError("Each condition is a list/tuple; "
                                 "[<attr-name>, <condition>, <value>]")
            for f in self.filters:
                f.add_condition(f.get_attr().schema()[c[0]], c[1], c[2])

    def to_timeseries(self, timestamp='timestamp', interval_ms=None,
                      max_array=QueryInputer.DEFAULT_ARRAY_LIMIT,
                      max_string=QueryInputer.DEFAULT_ARRAY_LIMIT):
        if self.inputer:
            return self.inputer.to_timeseries(self, timestamp, interval_ms, max_array, max_string)
        return None

    def to_dataset(self, max_array=QueryInputer.DEFAULT_ARRAY_LIMIT,
                   max_string=QueryInputer.DEFAULT_ARRAY_LIMIT):
        if self.inputer:
            return self.inputer.to_dataset(self, max_array, max_string)
        return None

    def to_dataframe(self, index=None,
                     max_array=QueryInputer.DEFAULT_ARRAY_LIMIT,
                     max_string=QueryInputer.DEFAULT_ARRAY_LIMIT):
        if self.inputer:
            return self.inputer.to_dataframe(self, index, max_array, max_string)
        return None

    def __dealloc__(self):
        for col in self.columns:
            del col
        self.filters = None
        self.inputer = None

    cdef object make_row(self, cursor):
        row = []
        for col in self.columns:
            value = col.convert(cursor[col.attr_idx][col.attr_id])
            row.append(value)
        return row

    def begin(self):
        """Position the cursor at the first object"""
        cursor = []
        for f in self.filters:
            o = f.begin()
            if o:
                cursor.append(o)
            else:
                return None
        return self.make_row(cursor)

    def end(self):
        cursor = []
        for f in self.filters:
            o = f.end()
            if o:
                cursor.append(o)
            else:
                return None
        return self.make_row(cursor)

    def __next__(self):
        cursor = []
        for f in self.filters:
            o = next(f)
            if o:
                cursor.append(o)
            else:
                return None
        return self.make_row(cursor)

    def prev(self):
        cursor = []
        for f in self.filters:
            o = f.prev()
            if o:
                cursor.append(o)
            else:
                return None
        return self.make_row(cursor)

cdef class SqlQuery:
    DEFAULT_ARRAY_LIMIT = 256
    cdef DsosContainer cont
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

    def __init__(self, DsosContainer cont, int row_limit, int max_array=256):
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
        cdef sos_attr_t c_attr
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
            self.c_res_acc[c_col_no].setter_fn = nda_setters[atyp]
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
