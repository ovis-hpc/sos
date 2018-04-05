from __future__ import print_function
from cpython cimport PyObject, Py_INCREF
from libc.stdint cimport *
from libc.stdlib cimport malloc, free
import numpy as np
import struct
cimport numpy as np
cimport Sos

#
# Python C-API
#
cdef extern from "Python.h":
    object PyString_FromStringAndSize(char *s, Py_ssize_t len)
    object PyByteArray_FromStringAndSize(char *s, Py_ssize_t len)

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

cdef class SchemaIter(SosObject):
    cdef sos_schema_t c_next_schema
    def __init__(self, Container cont):
        self.c_next_schema = sos_schema_first(cont.c_cont)

    def __iter__(self):
        return self

    def __next__(self):
        if self.c_next_schema == NULL:
            raise StopIteration
        s = Schema()
        s.assign(self.c_next_schema)
        self.c_next_schema = sos_schema_next(self.c_next_schema)
        return s

cdef class PartIter(SosObject):
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

    def __init__(self, path=None, o_perm=SOS_PERM_RW):
        SosObject.__init__(self)
        self.c_cont = NULL
        if path:
            self.open(path, o_perm=o_perm)

    def open(self, path, o_perm=SOS_PERM_RW):
        if self.c_cont != NULL:
            self.abort(EBUSY)
        self.c_cont = sos_container_open(path, o_perm)
        if self.c_cont == NULL:
            raise self.abort(errno)

    def create(self, path, o_mode=0660):
        cdef int rc
        if self.c_cont != NULL:
            self.abort(EBUSY)
        rc = sos_container_new(path, o_mode)
        if rc != 0:
            self.abort(rc)

    def delete(self):
        cdef int rc
        if self.c_cont == NULL:
            self.abort(EINVAL)
        rc = sos_container_delete(self.c_cont)
        if rc != 0:
            self.abort(rc)

    def close(self, commit=SOS_COMMIT_ASYNC):
        if self.c_cont == NULL:
            self.abort(EINVAL)
        sos_container_close(self.c_cont, commit)
        self.c_cont = NULL

    def commit(self, commit=SOS_COMMIT_ASYNC):
        cdef int rc
        rc = sos_container_commit(self.c_cont, commit)
        if rc != 0:
            self.abort(rc)

    def part_create(self, name, path=None):
        cdef int rc
        if self.c_cont == NULL:
            raise ValueError("The container is not open.")
        if path:
            rc = sos_part_create(self.c_cont, name, path)
        else:
            rc = sos_part_create(self.c_cont, name, NULL)
        if rc != 0:
            self.abort(rc)

    def part_by_name(self, name):
        cdef sos_part_t c_part = sos_part_find(self.c_cont, name)
        if c_part != NULL:
            p = Partition()
            p.assign(c_part)
            return p
        return None

    def part_iter(self):
        return PartIter(self)

    def index_iter(self):
        return IndexIter(self)

    def schema_by_name(self, name):
        cdef sos_schema_t c_schema = sos_schema_by_name(self.c_cont, name)
        if c_schema != NULL:
            s = Schema()
            s.assign(c_schema)
            return s
        return None

    def schema_by_id(self, id_):
        cdef sos_schema_t c_schema = sos_schema_by_id(self.c_cont, id_)
        if c_schema != NULL:
            s = Schema()
            s.assign(c_schema)
            return s
        return None

    def schema_iter(self):
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
    # cdef sos_part_stat_s c_stat

    def __init__(self):
        self.c_part = NULL

    cdef assign(self, sos_part_t c_part):
        self.c_part = c_part
        return self

    def part_id(self):
        """Returns the Partition id"""
        return sos_part_id(self.c_part)

    def name(self):
        """Returns the partition name"""
        return sos_part_name(self.c_part)

    def path(self):
        """Returns the partition path"""
        return sos_part_path(self.c_part)

    def state(self):
        """Returns the partition state"""
        return PartState(sos_part_state(self.c_part))

    def stat(self):
        """Returns the partition PartStat (size, access time, etc...) information"""
        return PartStat(self)

    def delete(self):
        """Delete the paritition"""
        cdef int rc = sos_part_delete(self.c_part)
        if rc != 0:
            self.abort(rc)
        self.c_part = NULL

    def move(self, new_path):
        """Move the paritition to a different location"""
        cdef int rc = sos_part_move(self.c_part, new_path)
        if rc != 0:
            self.abort(rc)

    def state_set(self, new_state):
        """Set the partition state"""
        cdef int rc
        cdef sos_part_state_t state
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

    def export(self, Container dst_cont, reindex=False):
        """Export the contents of this partition to another container"""
        return sos_part_export(self.c_part, dst_cont.c_cont, reindex)

    def index(self):
        """Index the contents of this partition"""
        return sos_part_index(self.c_part)

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        if self.c_part:
            sos_part_put(self.c_part)
            self.c_part = NULL

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
    "FIRST" : SOS_TYPE_FIRST,
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

    cdef assign(self, sos_schema_t c_schema):
        if c_schema == NULL:
            raise ValueError("schema argument cannot be NULL")
        self.c_schema = c_schema
        return self

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

    def from_template(self, name, template):
        """Create a schema from a template specification

        The template parameter is a python array of attribute
        definitions. An attribute definition is a Python dictionary
        object as follows:

        {
            "name" : <attribute-name-str>,
            "type" : <type-str>,
            "index" : <index-dict>
        }

        For example:

        [
            { "name" : "timestamp", "type" : "timestamp", "index" : {} },
            { "name" : "component_id", "type" : "uint64" },
            { "name" : "flits", "type" : "double" },
            { "name" : "stalls", "type" : "double" },
            { "name" : "comp_time", "type" : "join",
              "join_attrs" : [ "component_id", "timestamp" ],
              "index" : {} }
        ]

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

        self.c_schema = sos_schema_new(name)
        if self.c_schema == NULL:
            self.abort(ENOMEM)
        for attr in template:
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
                join_attrs = attr['join_attrs']
                join_count = len(join_attrs)
                join_args = <char **>malloc(join_count * 8)
                rc = 0
                for attr_name in join_attrs:
                    join_args[rc] = <char *>attr_name
                    rc += 1
                rc = sos_schema_attr_add(self.c_schema, attr['name'],
                                         t, <size_t>join_count, join_args)
            elif t == SOS_TYPE_STRUCT:
                if 'size' not in attr:
                    raise ValueError("The type {0} must have a 'size'.".format(n))
                sz = attr['size']
                rc = sos_schema_attr_add(self.c_schema, attr['name'], t, <size_t>sz)
            else:
                rc = sos_schema_attr_add(self.c_schema, attr['name'], t, 0)

            if rc != 0:
                raise ValueError("The attribute named {0} resulted in error {1}". \
                                 format(attr['name'], rc))

            if 'index' in attr:
                rc = sos_schema_index_add(self.c_schema, attr['name'])
                if rc != 0:
                    self.abort(rc)

                idx_type = "BXTREE"
                # The index modifiers are optional
                idx = attr['index']
                if 'type' in idx:
                    idx_type = idx['type']
                if 'key' in idx:
                    idx_key = idx['key']
                if 'args' in idx:
                    idx_args = idx['args']
                rc = sos_schema_index_modify(self.c_schema,
                                             attr['name'],
                                             idx_type,
                                             idx_key,
                                             <const char *>idx_args)
                if rc != 0:
                    self.abort(rc)

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

    def schema_id(self):
        """Returns the unique schema id"""
        return sos_schema_id(self.c_schema)

    def name(self):
        """Returns the name of the schema"""
        return sos_schema_name(self.c_schema)

    def alloc(self):
        """Allocate a new object of this type in the container"""
        cdef sos_obj_t c_obj = sos_obj_new(self.c_schema)
        if c_obj == NULL:
            self.abort()
        o = Object()
        return o.assign(c_obj)

    def __getitem__(self, attr_id):
        if type(attr_id) == int:
            return Attr(self, attr_id=attr_id)
        elif type(attr_id) == str:
            return Attr(self, attr_name=attr_id)
        raise ValueError("The index must be a string or an integer.")

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
        elif self.sos_type == SOS_TYPE_BYTE_ARRAY:
            return PyByteArray_FromStringAndSize(v.array.char_, v.array.count)
        elif self.sos_type == SOS_TYPE_CHAR_ARRAY:
            return PyString_FromStringAndSize(v.array.char_, v.array.count)
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
        cdef int i, j, count, typ
        cdef sos_comp_key_spec_t specs
        cdef sos_key_t c_key

        count = len(args) / 2
        if count * 2 != len(args):
            raise ValueError("The argument list must consist of a pairs of type, value")

        specs = <sos_comp_key_spec_t>malloc(count * sizeof(sos_comp_key_spec))
        if specs == NULL:
            raise MemoryError("Could not allocate the component key spec list.")

        i = 0
        j = 0
        while i < len(args):
            typ = <int>args[i]
            if typ < SOS_TYPE_LAST:
                specs[j].type = typ
            else:
                raise ValueError("Invalid value type {0} specifed".format(typ))
            type_setters[typ](&specs[j].data, args[i+1])
            i += 2
            j += 1

        i = sos_comp_key_set(self.c_key, count, specs);
        if i != 0:
            raise ValueError("Error encoding the composite key")
        free(specs)
        return self

    def split(self):
        """Split a join key into it's component parts"""
        cdef int rc, i, j
        cdef int typ
        cdef size_t count
        cdef sos_comp_key_spec_t specs
        cdef sos_key_t c_key

        rc = sos_comp_key_get(self.c_key, &count, NULL);
        if rc != 0:
            raise ValueError("Error {0} decoding key.".format(rc))

        specs = <sos_comp_key_spec_t>malloc(count * sizeof(sos_comp_key_spec))
        if specs == NULL:
            raise MemoryError("Could not allocate the component key spec list.")

        rc = sos_comp_key_get(self.c_key, &count, specs)
        if rc:
            raise ValueError("Error {0} decoding key after allocation.".format(rc))

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
            elif typ == SOS_TYPE_BYTE_ARRAY:
                res.append(PyByteArray_FromStringAndSize(specs[i].data.array.data.char_,
                                                         specs[i].data.array.count))
            elif typ == SOS_TYPE_CHAR_ARRAY:
                res.append(PyString_FromStringAndSize(specs[i].data.array.data.char_,
                                                      specs[i].data.array.count))
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
                raise ValueError("Invalid type {0} found in key.".format(typ))
        free(specs)
        return res

    cdef assign(self, sos_key_t c_key):
        if c_key == NULL:
            raise ValueError("key argument cannot be NULL")
        if self.c_key:
            sos_key_put(self.c_key)
        self.c_key = c_key
        return self

    def get_attr(self):
        return self.attr

    def set_value(self, value):
        """Set the value of a key.

        Set the value of a key.

        Positional Parameters:
        value - The value to assign to the key

        Keyword Parameters:
        key_type - The sos_type_t of the value
        """
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        type_setters[<int>self.sos_type](data, value)

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
    The AttrIter implements begin(), end(), prev(), and next() to
    iterate through objects in the index. Each of these methods
    returns True if there is an object at the current iterator
    position or False otherwise. For example:

    ```python
    b = it.begin()
    while b:
        o = it.item()
        # do something with the object
        b = it.next()
    ```
    There are also methods that take a Key as an argument to position
    the iterator at an object with the specified key. See find(),
    find_sup(), and find_inf() for documentation on these methods.
    """
    cdef Attr attr
    cdef sos_iter_t c_iter

    def __init__(self, Attr attr):
        """Instantiate an AttrIter object

        Positional Arguments:
        attr	The Attr with the Index on which the iterator is being
                created.
        """
        self.c_iter = sos_attr_iter_new(attr.c_attr)
        self.attr = attr
        if self.c_iter == NULL:
            raise ValueError("The {0} attribute is not indexed".format(self.attr.name()))

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
            k = Key(attr=self.attr)
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

    def get_pos(self):
        """Returns the currrent iterator position as a string

        The returned string represents the current iterator
        position. The string can be passed to the set_pos() method to
        set an iterator on the same index to this position.

        An intent of this string is that it can be exchanged over the
        network to enable paging of objects at the browser

        """
        cdef const char *c_str
        cdef sos_pos_t c_pos
        cdef int rc = sos_iter_pos_get(self.c_iter, &c_pos)
        if rc != 0:
            return None
        c_str = sos_pos_to_str(c_pos)
        return c_str

    def set_pos(self, pos_str):
        """Sets the currrent position from a string

        Positional Parameters:
        -- String representation of the iterator position
        """
        cdef sos_pos_t c_pos
        cdef int rc = sos_pos_from_str(&c_pos, pos_str)
        if rc == 0:
            return sos_iter_pos_set(self.c_iter, c_pos)
        return rc

    def put_pos(self, pos_str):
        """Puts (deletes) the specified position

        Positional Parameters:
        -- String representation of the iterator position
        """
        cdef sos_pos_t c_pos
        cdef int rc = sos_pos_from_str(&c_pos, pos_str)
        if rc == 0:
            return sos_iter_pos_put(self.c_iter, c_pos)
        return rc

    def __del__(self):
        if self.c_iter != NULL:
            sos_iter_free(self.c_iter)
            self.c_iter = NULL

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
TYPE_BYTE_ARRAY = SOS_TYPE_BYTE_ARRAY
TYPE_CHAR_ARRAY = SOS_TYPE_CHAR_ARRAY
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
PERM_RO = SOS_PERM_RO

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
            self.c_attr = sos_schema_attr_by_name(schema.c_schema, attr_name)
        if self.c_attr == NULL:
            if attr_id:
                name = attr_id
            elif attr_name:
                name = attr_name
            else:
                name = "unspecified"
            raise SchemaAttrError(name, schema.name())

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
        return s.assign(sos_attr_schema(self.c_attr))

    def attr_id(self):
        """Returns the attribute id"""
        return sos_attr_id(self.c_attr)

    def is_array(self):
        """Return True if the attribute is an array"""
        return (0 != sos_attr_is_array(self.c_attr))

    def name(self):
        """Returns the attribute name"""
        return sos_attr_name(self.c_attr)

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
        a = OAArray()
        return a.set_data(NULL, array.data.byte_, array.count, np.NPY_UINT32)

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
            join_list = []
            specs_len = len(args)
            specs = <sos_comp_key_spec_t>malloc(specs_len * sizeof(sos_comp_key_spec))
            if specs == NULL:
                raise MemoryError("Could not allocate the component key spec list.")
            for i in range(attrs.count):
                attr = sos_schema_attr_by_id(self.c_schema, attrs.data.uint32_[i])
                typ = sos_attr_type(attr)
                join_list.append(typ)
                arg = args[i]
                join_list.append(arg)
                specs[i].type = typ
                type_setters[typ](&specs[i].data, arg)
            size = sos_comp_key_size(specs_len, specs)
            free(specs)
            key = Key(size=size, sos_type=SOS_TYPE_JOIN)
            key.join(*join_list)
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
        cdef int t

        if not self.is_indexed():
            return None

        c_obj = sos_index_find_max(sos_attr_index(self.c_attr))
        if c_obj == NULL:
            return None
        c_data = sos_obj_attr_data(c_obj, self.c_attr, &c_arr_obj)

        t = sos_attr_type(self.c_attr)
        v = <object>type_getters[<int>t](c_obj, c_data, self.c_attr)

        sos_obj_put(c_obj)
        if c_arr_obj != NULL:
            sos_obj_put(c_arr_obj)
        return v

    def min(self):
        """Return the minimum value of this attribute in the container"""
        cdef sos_obj_t c_obj
        cdef sos_obj_t c_arr_obj
        cdef sos_value_data_t c_data
        cdef int t

        if not self.is_indexed():
            return None

        c_obj = sos_index_find_min(sos_attr_index(self.c_attr))
        if c_obj == NULL:
            return None
        c_data = sos_obj_attr_data(c_obj, self.c_attr, &c_arr_obj)

        t = sos_attr_type(self.c_attr)
        v = <object>type_getters[<int>t](c_obj, c_data, self.c_attr)

        sos_obj_put(c_obj)
        if c_arr_obj != NULL:
            sos_obj_put(c_arr_obj)
        return v

    def __str__(self):
        cdef sos_index_t c_idx
        s = '{{ "name" : "{0}", "type" : "{1}", "size" : {2}'.format(
            sos_attr_name(self.c_attr), sos_type_strs[sos_attr_type(self.c_attr)],
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
    NULL, NULL, NULL, NULL,     # twenty zeroes to get to 32, i.e. SOS_TYPE_BYTE_ARRAY
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

    The Filter implements begin(), end(), prev(), and next() to
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
        o = it.next()
    ```
    """
    cdef Attr attr
    cdef sos_iter_t c_iter
    cdef sos_filter_t c_filt
    cdef sos_obj_t c_obj
    cdef double start_us
    cdef double end_us

    def __init__(self, Attr attr):
        """Positional Parameters:
        -- The primary filter attribute
        """
        self.attr = attr
        self.c_iter = sos_attr_iter_new(attr.c_attr)
        if self.c_iter == NULL:
            raise ValueError("The attribute {0} must be indexed.".format(attr.name()))

        self.c_filt = sos_filter_new(self.c_iter)
        self.start_us = 0.0
        self.end_us = 0.0

    def get_attr(self):
        """Return the iterator attribute for this filter"""
        return self.attr

    def attr_by_name(self, name):
        """Return the attribute with this name"""
        return self.attr.schema[name]

    def add_condition(self, Attr cond_attr, cond, value_str):
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
        objects less than a particular value, then
        SOS_COND_GT/SOS_COND_GE with the desired start value.

        Positional parameters:
        -- The attribute whose value is being compared
        -- The condition:
           SOS_COND_LE    less-or-equal
           SOS_COND_LT    less-than
           SOS_COND_EQ    equal
           SOS_COND_NE    not-equal
           SOS_COND_GE    greater-or-equal
           SOS_COND_GT    greater-than
        -- A string representation of the value

        """
        cdef int rc
        cdef sos_value_t cond_v

        if type(value_str) != str:
            value_str = str(value_str)

        # strip embedded '"' from value if present
        value_str = value_str.replace('"', '')

        cond_v = sos_value_new()
        if not cond_v:
            raise ValueError("The attribute value for {0} could not be created.".format(cond_attr.name()))

        cond_v = sos_value_init(cond_v, NULL, cond_attr.c_attr)
        if sos_attr_type(cond_attr.c_attr) == SOS_TYPE_STRUCT:
            ba = bytearray(value_str)
            for rc in range(0, len(ba)):
                cond_v.data.prim.struc_[rc] = ba[rc]
        else:
            rc = sos_value_from_str(cond_v, value_str, NULL)
            if rc != 0:
                raise ValueError("The value {0} is invalid for the {1} attribute."
                                 .format(value_str, cond_attr.name()))

        if sos_attr_type(cond_attr.c_attr) == SOS_TYPE_TIMESTAMP:
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
        sos_filter_flags_set(self.c_filt, SOS_ITER_F_UNIQUE)

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

    def next(self):
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

    def skip(self, count):
        cdef sos_obj_t c_obj = sos_filter_skip(self.c_filt, count)
        if c_obj == NULL:
            return None
        o = Object()
        return o.assign(c_obj)

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

    def get_pos(self):
        """Returns the currrent filter position as a string

        The intent is that this string can be exchanged over the network
        to enable paging of filter records at the browser
        """
        cdef const char *c_str
        cdef sos_pos_t c_pos
        cdef int rc = sos_filter_pos_get(self.c_filt, &c_pos)
        if rc != 0:
            return None
        c_str = sos_pos_to_str(c_pos)
        return c_str

    def set_pos(self, pos_str):
        """Sets the currrent filter position from a string

        The string parameter is converted to a sos_pos_t and used to set the
        current filter position. The string was returned from a
        previous call to self.pos()

        Positional Parameters:
        -- String representation of the iterator position
        """
        cdef sos_pos_t c_pos
        cdef int rc = sos_pos_from_str(&c_pos, pos_str)
        if rc == 0:
            return sos_filter_pos_set(self.c_filt, c_pos)
        return rc

    def put_pos(self, pos_str):
        """Puts the specified position string

        Positional Parameters:
        -- String representation of the iterator position
        """
        cdef sos_pos_t c_pos
        cdef int rc = sos_pos_from_str(&c_pos, pos_str)
        if rc == 0:
            return sos_filter_pos_put(self.c_filt, c_pos)
        return rc

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
        the array will be padded with zeroes.

        The return value is a tuple containing the number of elements written
        to the array and the array itself. For example,

             schema = db.schema_by_name('Sample')
             tstamp = schema.attr_by_name('timestamp')
             f = Filter(tstamp)
             count, array = f.as_ndarray(1024, shape=['timestamp', 'current_freemem'],
                                         order=['attribute'])

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
        the array will be padded with zeroes.

        The return value is a tuple containing the number of elements written
        to the array and the array itself. For example,

             schema = db.schema_by_name('Sample')
             tstamp = schema.attr_by_name('timestamp')
             f = Filter(tstamp)
             count, array = f.as_ndarray(1024, shape=['timestamp', 'current_freemem'],
                                         order=['attribute'])

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

        t_attr = sos_schema_attr_by_name(schema.c_schema, timestamp)
        if t_attr == NULL:
            raise ValueError("The timestamp attribute was not found in the schema. " +\
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

        free(tmp_res)
        free(res_attr)
        free(res_type)
        free(res_acc)
        free(res_dim)
        return (last_idx, result)

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        if self.c_obj:
            sos_obj_put(self.c_obj)
            self.c_obj = NULL
        if self.c_iter:
            sos_iter_free(self.c_iter)
            self.c_iter = NULL
        if self.c_filt:
            sos_filter_free(self.c_filt)
            self.c_filt = NULL

cdef class Query(object):
    cdef Container cont         # The container
    cdef filter_idx             # dictionary mapping schema to filter index
    cdef filters                # array of filters
    cdef schema                 # array of schema
    cdef columns                # array of columns
    cdef int col_width          # default width of an output column
    cdef primary                # primary attribute
    cdef cursor                 # array of objects at the current index position
    cdef fill_fn                # function that returns values for missing objects
    cdef group_fn               # function that decides how objects are grouped

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
        container -- The Sos.Container to query
        """
        self.primary = 'timestamp'
        self.cont = container
        self.filter_idx = {}
        self.filters = []
        self.columns = []
        self.cursor = []
        self.schema = []
        self.col_width = 15

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
            i = name.find('.')
            if i >= 0:
                sname = name[:i]
                aname = name[i+1:]
                schema = self.cont.schema_by_name(sname)
            else:
                schema = self.__find_schema_with_attr(name)
                aname = name
        except:
            return (None, None)
        if schema:
            return (schema, schema[aname])
        raise ValueError('No schema contains the attribute "{0}"'.format(aname))

    def set_col_width(self, width):
        self.col_width = width

    def show_all_schema(self):
        """Return all schema names in the container"""
        result = []
        for schema in self.cont.schema_iter():
            result.append(schema.name())
        return result

    def show_all_indices(self):
        """Return all indexed attributes in the container"""
        result = []
        for schema in self.cont.schema_iter():
            for attr in schema.attr_iter():
                if attr.is_indexed():
                    result.append(schema.name() + '.' + attr.name())
        return result

    def show_schema(self, name):
        """Return a string describing the schema"""
        return str(self.cont.schema_by_name(name))

    def show_results(self, row_limit=0, col_width=0, fmt='table'):
        if col_width == 0L:
            col_width = self.col_width
        if format == 'table':
            return self.show_table_results(row_limit, col_width)
        print("{0} is an unsupported format".format(fmt))

    def show_table_results(self, row_limit, col_width):
        for col in self.columns:
            print("{0:{width}}".format(col[1].schema().name(), width=col_width), end=' ')
        print("")
        for col in self.columns:
            print("{0:{width}}".format(col[1].name(), width=col_width), end=' ')
        print("")
        for col in self.columns:
            print("{0:{width}}".format('-'.ljust(col_width, '-'), width=col_width), end=' ')
        print("")
        count = 0
        row = self.begin()
        while row and (row_limit == 0 or count < row_limit):
            for col in range(0, len(self.columns)):
                print("{0:{width}}".format(row[col], width=col_width), end=' ')
            print("")
            row = self.next()
            count += 1
        for col in self.columns:
            print("{0:{width}}".format('-'.ljust(col_width, '-'), width=col_width), end=' ')
        print("\n{0} record(s)".format(count))

    def show_json_results(self, row_limit, col_width):
        for col in self.columns:
            print("{0:{width}}".format(col[1].schema().name(), width=col_width), end=' ')
        print("")
        for col in self.columns:
            print("{0:{width}}".format(col[1].name(), width=col_width), end=' ')
        print("")
        for col in self.columns:
            print("{0:{width}}".format('-'.ljust(col_width, '-'), width=col_width), end=' ')
        print("")
        count = 0
        row = self.begin()
        while row and (row_limit == 0 or count < row_limit):
            for col in range(0, len(self.columns)):
                print("{0:{width}}".format(row[col], width=col_width), end=' ')
            print("")
            row = self.next()
            count += 1
        for col in self.columns:
            print("{0:{width}}".format('-'.ljust(col_width, '-'), width=col_width), end=' ')
        print("\n{0} record(s)".format(count))

    def from_(self, schema):
        """Specify the schema(s) to query

        If from_() is not called, the default is all schema present in
        the container; otherwise, it must be called before select().

        Positional Parameters:
        schema - An array of schema names

        Example:
        query.from_(['meminfo', 'vmstat'])

        """
        self.schema = []
        for name in schema:
            self.schema.append(self.cont.schema_by_name(name))

    def order_by(self, name):
        """Specify the attribute name to use for ordering the result

        Must be called before select().

        Positional Parameters:
        name - This name of the attribute used to order the data.
               This attribute must be present and indexed in all schema.
        """
        self.primary = name

    def select(self, attr_list):
        """Set the attribute list returned in the result

        The attr_list is an array of attribute names (logically
        similar to column names in SQL.)  The attribute names should
        be adorned with the containing schema name if the name is
        ambiguous, i.e. multiple schema may contain the same attribute
        name. Good examples in the LDMS use case are the component_id
        and job_id attributes.

        The syntax of an attribute name is as follows:

        name      := [[:alpha:]_]+ [[:alnum:]_]
        attribute := '*'
                  |  name
                  |  name '.' name
                  |  name '.' '*'
                  |  '*' '.' '*'
                  ;

        The special character '*' (asterisk) represents a wild
        card. If '*' is alone, it represents all attribute names in
        all schema. If a schema name precedes it, e.g. meminfo.*, it
        is all attribute names in the schema named 'meminfo'. The
        strings '*.*' and '*' are synonyms.

        The set of schema that are part of the query is inferred from
        the select clause.

        Positional Parameters:
        attr_list - An array of attribute names

        """
        self.filter_idx = {}
        self.filters = []
        self.columns = []
        for name in attr_list:
            schema, attr = self.__decode_attr_name(name)
            if not schema:
                raise ValueError("The attribute {0} was not found in any schema.".format(name))
            if schema.name() not in self.filter_idx:
                ts = schema[self.primary]
                f = Sos.Filter(schema[self.primary])
                idx = len(self.filters)
                self.filter_idx[schema.name()] = idx
                self.filters.append(f)
            else:
                idx = self.filter_idx[schema.name()]
            self.columns.append((idx, attr))

    def get_columns(self):
        return self.columns

    def get_filters(self):
        return self.filters

    def set_fill_fn(self, fill_fn):
        """Sets the function that will be called when a value cannot be filled
        from the current object
        """
        self.fill_fn = fill_fn

    def default_group_fn(self, o1, o2):
        pass

    def set_group_fn(self, group_fn):
        """Sets the function that will be called when collection objects into a single row
        """
        self.group_fn = group_fn

    def set_(self, fill_fn):
        """Sets the function that will be called when a value cannot be filled
        from the current object
        """
        self.fill_fn = fill_fn

    def where(self, clause):
        """Specify the filter conditions

        'clause' is an array of filter condition tuples.  Each
        condition must be applicable to every Filter/schema. A
        condition tuple contains three elements as follows:

        ( attribute_name, condition, value )

        For example:

        ( 'timestamp', COND_GE, 1510597567.001617 )

        A valid call then is:

        query.where(clause = [ ( 'timestamp', COND_GE, 1510597567.001617 ) ] )

        Keyword Parameters:
        clause -- An array of filter conditions as described above
        timespec -- A two element tuple of start and end timestamps
        """
        for c in clause:
            for f in self.filters:
                f.add_condition(f.get_attr().schema()[c[0]], c[1], c[2])

    cdef object make_row(self):
        row = []
        for col in self.columns:
            obj = self.cursor[col[0]]
            attr_id = col[1].attr_id()
            row.append(obj[attr_id])
        return row

    def begin(self):
        self.cursor = []
        for f in self.filters:
            self.cursor.append(f.begin())
        return self.make_row()

    def end(self):
        self.cursor = (f.end() for f in self.filters)
        return self.cursor

    def next(self):
        self.cursor = []
        for f in self.filters:
            self.cursor.append(f.next())
        return self.make_row()

    def prev(self):
        self.cursor = (f.prev() for f in self.filters)
        return self.cursor

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
        return sos_index_name(self.c_index)

    def stats(self):
        """Return a dictionary of index statistics as follows:
            cardinality - Number of index entries
            duplicates  - Number of duplicate keys
            size        - The storage size consumed by the index in bytes
        """
        cdef int rc = sos_index_stat(self.c_index, &self.c_stats)
        return self.c_stats

    def show(self):
        sos_index_print(self.c_index, NULL);

################################
# Object getter functions
################################
cdef class OAArray:
    cdef sos_obj_t c_obj
    cdef void *c_ptr
    cdef int c_size
    cdef int c_type

    def __init__(self):
        self.c_obj = NULL
        self.c_ptr = NULL
        self.c_size = 0

    cdef set_data(self, sos_obj_t obj, void *data_ptr, int size, int el_type):
        cdef np.ndarray ndarray
        self.c_obj = obj
        if self.c_obj != NULL:
            sos_obj_get(obj)
        self.c_ptr = data_ptr
        self.c_size = size
        self.c_type = el_type
        ndarray = np.array(self, copy=False)
        Py_INCREF(self)
        ndarray.base = <PyObject *>self
        return ndarray

    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.c_size
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               self.c_type, self.c_ptr)
        return ndarray

    def __del__(self):
        self.__dealloc__()

    def __dealloc__(self):
        if self.c_obj:
            sos_obj_put(self.c_obj)
            self.c_obj = NULL

cdef object get_DOUBLE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_FLOAT64)

cdef object get_LONG_DOUBLE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_LONGDOUBLE)

cdef object get_FLOAT_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_FLOAT32)

cdef object get_UINT64_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT64)

cdef object get_UINT32_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT32)

cdef object get_UINT16_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT16)

cdef object get_BYTE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT8)

cdef object get_CHAR_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return c_data.array.data.char_

cdef object get_INT64_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT64)

cdef object get_INT32_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT32)

cdef object get_INT16_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT16)

cdef object get_TIMESTAMP(sos_obj_t c_obj, sos_value_data_t c_data, sos_attr_t c_attr):
    return <double>c_data.prim.timestamp_.tv.tv_sec \
        + (<double>c_data.prim.timestamp_.tv.tv_usec / 1000000.0)

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
    n = sos_attr_size(c_attr)
    return <object>c_data.struc.char_[:n]

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

################################
# Object setter functions
################################
cdef set_LONG_DOUBLE_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    for i in range(sz):
        c_data.array.data.long_double_[i] = val[i]

cdef set_DOUBLE_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.double_[i] = val[i]

cdef set_FLOAT_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.float_[i] = val[i]

cdef set_UINT64_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.uint64_[i] = val[i]

cdef set_UINT32_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.uint32_[i] = val[i]

cdef set_UINT16_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    for i in range(sz):
        c_data.array.data.uint16_[i] = val[i]

cdef set_BYTE_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.byte_[i] = <uint8_t>val[i]

cdef set_INT64_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.int64_[i] = <int64_t>val[i]

cdef set_INT32_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.int32_[i] = val[i]

cdef set_INT16_ARRAY(sos_value_data_t c_data, val):
    cdef int i, sz
    sz = len(val)
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.int16_[i] = val[i]

cdef set_CHAR_ARRAY(sos_value_data_t c_data, val):
    cdef char *s
    cdef int i, sz
    sz = len(val)
    s = val
    c_data.array.count = sz
    for i in range(sz):
        c_data.array.data.char_[i] = s[i]

cdef set_TIMESTAMP(sos_value_data_t c_data, val):
    cdef int secs
    cdef int usecs
    try:
        secs = val
        usecs = (val - secs) * 1000000
        c_data.prim.timestamp_.tv.tv_sec = secs
        c_data.prim.timestamp_.tv.tv_usec = usecs
    except Exception as e:
        raise ValueError("The time value is a floating point number representing "
                         "seconds since the epoch")

cdef set_LONG_DOUBLE(sos_value_data_t c_data, val):
    c_data.prim.long_double_ = <long double>val

cdef set_DOUBLE(sos_value_data_t c_data, val):
    c_data.prim.double_ = <double>val

cdef set_FLOAT(sos_value_data_t c_data, val):
    c_data.prim.float_ = <float>val

cdef set_UINT64(sos_value_data_t c_data, val):
    c_data.prim.uint64_ = <uint64_t>val

cdef set_UINT32(sos_value_data_t c_data, val):
    c_data.prim.uint32_ = <uint32_t>val

cdef set_UINT16(sos_value_data_t c_data, val):
    c_data.prim.uint16_ = <uint16_t>val

cdef set_INT64(sos_value_data_t c_data, val):
    c_data.prim.int64_ = <int64_t>val

cdef set_INT32(sos_value_data_t c_data, val):
    c_data.prim.int32_ = <int32_t>val

cdef set_INT16(sos_value_data_t c_data, val):
    c_data.prim.int16_ = <int16_t>val

cdef set_STRUCT(sos_value_data_t c_data, val):
    cdef char *s = val
    memcpy(&c_data.prim.byte_, s, len(val))

cdef set_ERROR(sos_value_data_t c_data, val):
    raise ValueError("Set is not supported on this attribute type")

ctypedef object (*type_setter_fn_t)(sos_value_data_t c_data, val)
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
            self.set_fn(self.c_v.data, v)
        else:
            sz = len(v)
            self.c_v = sos_array_new(&self.c_v_, self.c_attr, self.c_obj, sz)
            self.set_fn(self.c_v.data, v)

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
        return sos_attr_name(self.c_attr)

    def strlen(self):
        """
        Return the length of the string if the value
        were formatted as a string
        """
        return sos_value_strlen(self.c_v)

    def to_key(self):
        """Return a Key() object initialized from the associated Attr"""
        return 
    def from_str(self, string):
        """Set the value from the string"""
        return sos_value_from_str(self.c_v, string, NULL)

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
        return sos_value_to_str(self.c_v, self.c_str, sz)

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
    O.<name> will return the value of that attribute as a native
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
        <object>type_setters[<int>t](v.data, val)
        sos_value_put(v)

    cdef set_py_value(self, sos_attr_t c_attr, val):
        cdef sos_value_data_t c_data
        cdef int t = sos_attr_type(c_attr)
        c_data = sos_obj_attr_data(self.c_obj, c_attr, NULL)
        <object>type_setters[<int>t](c_data, val)

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
            c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), idx)
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
        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name)
        if c_attr == NULL:
            raise ValueError("Object has no attribute with name '{0}'".format(name))
        arr_obj = NULL
        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        res = self.get_py_value(self.c_obj, c_attr, c_data)
        if arr_obj != NULL:
            sos_obj_put(arr_obj);
        return res

    def __setitem__(self, idx, val):
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
        if type(idx) == int:
            c_attr = sos_schema_attr_by_id(self.c_schema, idx)
        elif type(idx) == str:
            c_attr = sos_schema_attr_by_name(self.c_schema, idx)
        else:
            raise ValueError("Object has no attribute with id '{0}'".format(idx))
        if 0 == sos_attr_is_array(c_attr):
            self.set_py_value(c_attr, val)
        else:
            self.set_py_array_value(c_attr, val)

    def __setattr__(self, name, val):
        cdef sos_attr_t c_attr
        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")
        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name)
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
        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name)
        if c_attr == NULL:
            raise ObjAttrError(name)
        c_type = sos_attr_type(c_attr)
        if c_type < SOS_TYPE_ARRAY:
            raise TypeError("This method only works on array types")
        v = sos_array_new(&v_, c_attr, self.c_obj, size)
        if v == NULL:
            raise MemoryError()
        sos_value_put(v)

    def index_add(self):
        """
        Add the object to all schema indices
        """
        if self.c_obj == NULL:
            self.abort("There is no container object associated with this Object")
        sos_obj_index(self.c_obj)

    def index_del(self):
        """
        Remove the object from all schema indices
        """
        if self.c_obj == NULL:
            self.abort("There is no container object associated with this Object")
        sos_obj_remove(self.c_obj)

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
        s.assign(self.c_schema)
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

        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name)
        if c_attr == NULL:
            raise ObjAttrError(name)
        t = sos_attr_type(c_attr)
        if t < SOS_TYPE_ARRAY:
            size = sos_attr_size(c_attr)
        else:
            raise TypeError("Use the attribute accessor directly for arrays")

        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        # convert size in bytes to array count
        size = size / np.dtype(eltype).itemsize
        shape[0] = size
        res = np.PyArray_SimpleNewFromData(1, shape, np.dtype(eltype).num,
                                           c_data.array.data.byte_)
        if arr_obj != NULL:
            sos_obj_put(arr_obj)
        return res

class SchemaAttrError(NameError):
    def __init__(self, attr, schema):
        NameError.__init__(self,
                           "Attribute name '{0}' is not " \
                           "present in schema '{1}'".format(attr, schema))

class ObjAttrError(NameError):
    def __init__(self, attr, schema):
        NameError.__init__(self,
                           "Object has not attribute with the name '{0}'" \
                           .format(attr, schema))
