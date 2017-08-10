from cpython cimport PyObject, Py_INCREF
from libc.stdint cimport *
from libc.stdlib cimport malloc, free
import numpy as np
import struct
cimport numpy as np
cimport Sos

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
        cdef sos_schema_t c_schema = sos_schema_by_name(self.c_cont, id_)
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

    def __dealloc__(self):
        if self.c_part:
            sos_part_put(self.c_part)

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
            { "name" : "timestamp", "type" : "timestamp", "index" = {} },
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
    cdef Attr attr

    def __init__(self, size=None, attr=None):
        if not size and not attr:
            raise ValueError("Either size or attr must be specified")
        if attr:
            self.attr = attr
            if not size:
                size = attr.size()
        else:
            self.attr = None
        self.c_key = sos_key_new(size)
        self.c_size = size

    def __len__(self):
        return self.c_size

    def __str__(self):
        cdef const char *s
        if self.attr:
            s = sos_attr_key_to_str(self.attr.c_attr, self.c_key)
        else:
            s = <char *>sos_key_value(self.c_key)
        return s

    cdef assign(self, sos_key_t c_key):
        if c_key == NULL:
            raise ValueError("key argument cannot be NULL")
        if self.c_key:
            sos_key_put(self.c_key)
        self.c_key = c_key
        return self

    def get_attr(self):
        return self.attr

    def set_value(self, py_val):
        """Set the value of a key.
        For primitive types (e.g. int, float), this method conveniently set the
        internal key value data according to types. For `SOS_TYPE_STRUCT`, the
        supplied value `py_val` must be prepared with `struct.pack()`.
        """
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        cdef char *c_val
        if not self.attr:
            raise TypeError("Key is not bound to any attribute.")
        typ = self.attr.type()
        type_setters[<int>typ](NULL, self.attr.c_attr, data, py_val)

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
        cdef int rc = sos_iter_begin(self.c_iter);
        if rc == 0:
            return True
        return False

    def end(self):
        """Position the iterator at the last object in the index"""
        cdef int rc = sos_iter_end(self.c_iter);
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

    def schema(self):
        """Returns the schema for which this attribute is a member"""
        s = Schema()
        return s.assign(sos_attr_schema(self.c_attr))

    def attr_id(self):
        """Returns the attribute id"""
        return sos_attr_id(self.c_attr)

    def name(self):
        """Returns the attribute name"""
        return sos_attr_name(self.c_attr)

    def type(self):
        """Returns the attribute type"""
        return sos_attr_type(self.c_attr)

    def type_name(self):
        """Returns the type name of this attribute"""
        return sos_type_strs[sos_attr_type(self.c_attr)]

    def indexed(self):
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

    def find(self, Key key):
        cdef sos_index_t c_index = sos_attr_index(self.c_attr)
        cdef sos_obj_t c_obj = sos_index_find(c_index, key.c_key)
        if c_obj == NULL:
            return None
        o = Object()
        o.assign(c_obj)
        return o

    def max(self):
        """Return the maximum Key value of this attribute in the container"""
        cdef sos_iter_t c_iter
        cdef int c_rc
        cdef sos_key_t c_key

        if not self.indexed:
            return 0

        c_iter = sos_attr_iter_new(self.c_attr)
        c_rc = sos_iter_end(c_iter)
        if c_rc:
            return None

        c_key = sos_iter_key(c_iter)
        key = Key(attr=self)
        sos_key_set(key.c_key, sos_key_value(c_key), sos_key_len(c_key))
        sos_key_put(c_key)
        sos_iter_free(c_iter)
        return key

    def min(self):
        """Return the minimum Key value of this attribute in the container"""
        cdef sos_iter_t c_iter
        cdef int c_rc
        cdef sos_key_t c_key

        if not self.indexed:
            return 0

        c_iter = sos_attr_iter_new(self.c_attr)
        c_rc = sos_iter_begin(c_iter)
        if c_rc:
            return None

        c_key = sos_iter_key(c_iter)
        key = Key(attr=self)
        sos_key_set(key.c_key, sos_key_value(c_key), sos_key_len(c_key))
        sos_key_put(c_key)
        sos_iter_free(c_iter)
        return key

COND_LT = SOS_COND_LT
COND_LE = SOS_COND_LE
COND_EQ = SOS_COND_EQ
COND_GE = SOS_COND_GE
COND_GT = SOS_COND_GT
COND_NE = SOS_COND_NE

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

    def __init__(self, Attr attr):
        """Positional Parameters:
        -- The primary filter attribute
        """
        self.attr = attr
        self.c_iter = sos_attr_iter_new(attr.c_attr)
        if self.c_iter == NULL:
            raise ValueError("The attribute {0} must be indexed.".format(attr.name()))

        self.c_filt = sos_filter_new(self.c_iter)

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
        rc = sos_value_from_str(cond_v, value_str, NULL)
        if rc != 0:
            raise ValueError("The value {0} is invalid for the {1} attribute."
                             .format(value_str, cond_attr.name()))

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

        The keyword parameter 'shape' is used to specify the object attributes
        that will comprise the returned array. Each element of the array should
        be the name of an object attribute in the same schema that the positional
        'attr' argument is a member of. The default value of 'shape' is:

             [ self.attr.name() ]

        If the len(shape) is 1, the array will be simplified to a singly
        dimensioned array of the 'attr' attribute of the each object.

        If the number of objects defined by the Filter is less than 'count',
        the array will be padded with zeroes.

        The return value is a tuple containing the number of elements written
        to the array and the array itself. For example,

             schema = db.schema_by_name('Sample')
             tstamp = schema.attr_by_name('timestamp')
             f = Filter(tstamp)
             count, array = f.as_ndarray(1024, shape=['timestamp', 'current_freemem'], order=['attribute'])

        will return the number of elements actually written to 'array' of
        size 1024 elements. The array is an Numpy.ndarray as follows:

             [ [ 1425362400.0, 1425362460.0, ... ], [ 62453912.0, 6553912.0, ... ] ]

        which is an 'attribute' ordering of attribute values. This ordering is more
        natural for numerical analysis as each array contains data of the same type.

        For applications such as graphing, it is often preferred to have the
        attribute values grouped by index. Set the 'order' keyword argument to
        'index', and the array data will be ordered as follows:

             [ [ 1425362400.0, 62453912.0 ], [ 1425362460.0, 6553912.0 ], ... ]

        Positional Parameters:
        -- A size_t count that specifies the maximum number of 'entries' in
           the array, where an entry may have n-dimensions.

        Keyword Parameters:
        shape  -- A tuple that specifies the attribute name of each
                  column in the output array.
        order  -- One of 'attribute' (default) or 'index' as described above
        cont   -- If true, the filter will continue where it left off, i.e.
                  processing will not begin at the first matching key
        """
        cdef sos_obj_t c_o
        cdef sos_value_s v_
        cdef sos_value_t v
        cdef int idx
        cdef int el_idx
        cdef int atype
        cdef int nattr
        cdef Schema schema = self.attr.schema()
        cdef Attr attr
        cdef sos_attr_t c_attr
        cdef sos_attr_t *res_attr
        cdef int *res_type

        if shape == None:
            shape = [ self.attr.name() ]

        nattr = len(shape)
        if nattr > 1:
            if order == 'index':
                ndshape = [ count, nattr ]
            elif order == 'attribute':
                ndshape = [ nattr, count ]
            else:
                raise ValueError("The 'order' keyword parameter must be one of 'index' or 'attribute'")
        else:
            ndshape = [ count ]

        res_attr = <sos_attr_t *>malloc(sizeof(sos_attr_t) * nattr)
        if res_attr == NULL:
            raise MemoryError("Insufficient memroy to allocate dimension array")
        res_type = <int *>malloc(sizeof(uint64_t) * nattr)
        if res_type == NULL:
            free(res_attr)
            raise MemoryError("Insufficient memory to allocate type array")

        result = np.zeros(ndshape, dtype=np.float64, order='C')

        idx = 0
        for aname in shape:
            try:
                attr = schema.attr_by_name(aname)
                res_attr[idx] = attr.c_attr
                res_type[idx] = sos_attr_type(attr.c_attr)
                idx += 1
            except:
                free(res_attr)
                free(res_type)
                raise ValueError("The attribute {0} does not exist in the schema {1}".format(aname, schema.name()))

        if cont:
            c_o = sos_filter_next(self.c_filt)
        else:
            c_o = sos_filter_begin(self.c_filt)
        idx = 0
        if nattr == 1:
            c_attr = res_attr[0]
            atype = sos_attr_type(c_attr)
            while c_o != NULL:
                v = sos_value_init(&v_, c_o, c_attr)
                result[idx] = <object>type_getters[atype](c_o, v.data)
                sos_value_put(v)
                sos_obj_put(c_o)
                idx = idx + 1
                if idx >= count:
                    break
                c_o = sos_filter_next(self.c_filt)
        else:
            if order == 'index':
                while c_o != NULL:
                    for el_idx in range(0, nattr):
                        v = sos_value_init(&v_, c_o, res_attr[el_idx])
                        result[idx][el_idx] = <object>type_getters[res_type[el_idx]](c_o, v.data)
                        sos_value_put(v)
                        sos_obj_put(c_o)
                    idx = idx + 1
                    if idx >= count:
                        break
                    c_o = sos_filter_next(self.c_filt)
            elif order == 'attribute':
                while c_o != NULL:
                    for el_idx in range(0, nattr):
                        v = sos_value_init(&v_, c_o, res_attr[el_idx])
                        result[el_idx][idx] = <object>type_getters[res_type[el_idx]](c_o, v.data)
                        sos_value_put(v)
                        sos_obj_put(c_o)
                    idx = idx + 1
                    if idx >= count:
                        break
                    c_o = sos_filter_next(self.c_filt)
        free(res_attr)
        free(res_type)
        return (idx, result)

    def __dealloc__(self):
        if self.c_obj:
            sos_obj_put(self.c_obj)
        if self.c_iter:
            sos_iter_free(self.c_iter)
        if self.c_filt:
            sos_filter_free(self.c_filt)

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

        Return the object at the key that matches the specified key.
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

    def __dealloc__(self):
        if self.c_obj:
            sos_obj_put(self.c_obj)
            self.c_obj = NULL

cdef object get_DOUBLE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_FLOAT64)

cdef object get_FLOAT_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_FLOAT32)

cdef object get_UINT64_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT64)

cdef object get_UINT32_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT32)

cdef object get_UINT16_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT16)

cdef object get_BYTE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT8)

cdef object get_CHAR_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT8)

cdef object get_INT64_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT64)

cdef object get_INT32_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT32)

cdef object get_INT16_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = OAArray()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT16)

cdef object get_TIMESTAMP(sos_obj_t c_obj, sos_value_data_t c_data):
    return <double>c_data.prim.timestamp_.fine.secs \
        + (<double>c_data.prim.timestamp_.fine.usecs / 1000000.0)

cdef object get_DOUBLE(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.double_

cdef object get_FLOAT(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.float_

cdef object get_UINT64(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.uint64_

cdef object get_UINT32(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.uint32_

cdef object get_UINT16(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.uint16_

cdef object get_INT64(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.int64_

cdef object get_INT32(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.int32_

cdef object get_INT16(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.int16_

cdef object get_STRUCT(sos_obj_t c_obj, sos_value_data_t c_data):
    return c_data.prim.uint64_

cdef object get_ERROR(sos_obj_t c_obj, sos_value_data_t c_data):
    raise ValueError("Get is not supported on this attribute type.")

ctypedef object (*type_getter_fn_t)(sos_obj_t, sos_value_data_t)
cdef type_getter_fn_t type_getters[SOS_TYPE_LAST+1]
type_getters[<int>SOS_TYPE_INT16] = get_INT16
type_getters[<int>SOS_TYPE_INT32] = get_INT32
type_getters[<int>SOS_TYPE_INT64] = get_INT64
type_getters[<int>SOS_TYPE_UINT16] = get_UINT16
type_getters[<int>SOS_TYPE_UINT32] = get_UINT32
type_getters[<int>SOS_TYPE_UINT64] = get_UINT64
type_getters[<int>SOS_TYPE_FLOAT] = get_FLOAT
type_getters[<int>SOS_TYPE_DOUBLE] = get_DOUBLE
type_getters[<int>SOS_TYPE_TIMESTAMP] = get_TIMESTAMP
type_getters[<int>SOS_TYPE_OBJ] = get_ERROR
type_getters[<int>SOS_TYPE_STRUCT] = get_STRUCT
type_getters[<int>SOS_TYPE_JOIN] = get_STRUCT
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
type_getters[<int>SOS_TYPE_OBJ_ARRAY] = get_ERROR

################################
# Object setter functions
################################
cdef object set_DOUBLE_ARRAY(sos_obj_t c_obj,
                             sos_attr_t c_attr,
                             sos_value_data_t c_data,
                             val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.double_[i] = val[i]
    sos_value_put(v)

cdef object set_FLOAT_ARRAY(sos_obj_t c_obj,
                            sos_attr_t c_attr,
                            sos_value_data_t c_data,
                            val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.float_[i] = val[i]
    sos_value_put(v)

cdef object set_UINT64_ARRAY(sos_obj_t c_obj,
                             sos_attr_t c_attr,
                             sos_value_data_t c_data,
                             val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.uint64_[i] = val[i]
    sos_value_put(v)

cdef object set_UINT32_ARRAY(sos_obj_t c_obj,
                             sos_attr_t c_attr,
                             sos_value_data_t c_data,
                             val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.uint64_[i] = val[i]
    sos_value_put(v)

cdef object set_UINT16_ARRAY(sos_obj_t c_obj,
                             sos_attr_t c_attr,
                             sos_value_data_t c_data,
                             val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.uint64_[i] = val[i]
    sos_value_put(v)

cdef object set_BYTE_ARRAY(sos_obj_t c_obj,
                           sos_attr_t c_attr,
                           sos_value_data_t c_data,
                           val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.uint64_[i] = val[i]
    sos_value_put(v)

cdef object set_INT64_ARRAY(sos_obj_t c_obj,
                            sos_attr_t c_attr,
                            sos_value_data_t c_data,
                            val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.uint64_[i] = val[i]
    sos_value_put(v)

cdef object set_INT32_ARRAY(sos_obj_t c_obj,
                            sos_attr_t c_attr,
                            sos_value_data_t c_data,
                            val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.uint64_[i] = val[i]
    sos_value_put(v)

cdef object set_INT16_ARRAY(sos_obj_t c_obj,
                            sos_attr_t c_attr,
                            sos_value_data_t c_data,
                            val):
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    for i in range(sz):
        v.data.array.data.uint64_[i] = val[i]
    sos_value_put(v)

cdef object set_CHAR_ARRAY(sos_obj_t c_obj,
                            sos_attr_t c_attr,
                            sos_value_data_t c_data,
                            val):
    cdef char *s
    cdef int i, sz
    cdef sos_value_s v_
    cdef sos_value_s *v
    sz = len(val)
    v = sos_array_new(&v_, c_attr, c_obj, sz)
    s = val
    for i in range(sz):
        v.data.array.data.char_[i] = s[i]
    sos_value_put(v)

cdef object set_TIMESTAMP(sos_obj_t c_obj,
                          sos_attr_t c_attr,
                          sos_value_data_t c_data,
                          val):
    cdef int secs
    cdef int usecs
    secs = <int>val
    usecs = (val - secs) * 1000000
    c_data.prim.timestamp_.fine.secs = secs
    c_data.prim.timestamp_.fine.usecs = usecs

cdef object set_DOUBLE(sos_obj_t c_obj,
                       sos_attr_t c_attr,
                       sos_value_data_t c_data,
                       val):
    c_data.prim.double_ = <double>val

cdef object set_FLOAT(sos_obj_t c_obj,
                      sos_attr_t c_attr,
                      sos_value_data_t c_data,
                      val):
    c_data.prim.float_ = <float>val

cdef object set_UINT64(sos_obj_t c_obj,
                       sos_attr_t c_attr,
                       sos_value_data_t c_data,
                       val):
    c_data.prim.uint64_ = <uint64_t>val

cdef object set_UINT32(sos_obj_t c_obj,
                          sos_attr_t c_attr,
                          sos_value_data_t c_data,
                          val):
    c_data.prim.uint32_ = <uint32_t>val

cdef object set_UINT16(sos_obj_t c_obj,
                       sos_attr_t c_attr,
                       sos_value_data_t c_data,
                       val):
    c_data.prim.uint16_ = <uint16_t>val

cdef object set_INT64(sos_obj_t c_obj,
                      sos_attr_t c_attr,
                      sos_value_data_t c_data,
                      val):
    c_data.prim.int64_ = <int64_t>val

cdef object set_INT32(sos_obj_t c_obj,
                      sos_attr_t c_attr,
                      sos_value_data_t c_data,
                      val):
    c_data.prim.int32_ = <int32_t>val

cdef object set_INT16(sos_obj_t c_obj,
                      sos_attr_t c_attr,
                      sos_value_data_t c_data,
                      val):
    c_data.prim.int16_ = <int16_t>val

cdef object set_STRUCT(sos_obj_t c_obj,
                       sos_attr_t c_attr,
                       sos_value_data_t c_data,
                       val):
    cdef char *s = val
    memcpy(&c_data.prim.byte_, s, sos_attr_size(c_attr))

cdef object set_ERROR(sos_obj_t c_obj,
                      sos_attr_t c_attr,
                      sos_value_data_t c_data,
                      val):
    raise ValueError("Set is not supported on this attribute type")

ctypedef object (*type_setter_fn_t)(sos_obj_t, sos_attr_t, sos_value_data_t, val)
cdef type_setter_fn_t type_setters[SOS_TYPE_LAST+1]
type_setters[<int>SOS_TYPE_INT16] = set_INT16
type_setters[<int>SOS_TYPE_INT32] = set_INT32
type_setters[<int>SOS_TYPE_INT64] = set_INT64
type_setters[<int>SOS_TYPE_UINT16] = set_UINT16
type_setters[<int>SOS_TYPE_UINT32] = set_UINT32
type_setters[<int>SOS_TYPE_UINT64] = set_UINT64
type_setters[<int>SOS_TYPE_FLOAT] = set_FLOAT
type_setters[<int>SOS_TYPE_DOUBLE] = set_DOUBLE
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
type_setters[<int>SOS_TYPE_OBJ_ARRAY] = set_ERROR

cdef class Value(object):
    cdef sos_value_s c_v_
    cdef sos_value_t c_v
    cdef sos_attr_t c_attr
    cdef sos_obj_t c_obj

    def __init__(self, Attr attr, Object obj=None):
        if obj is not None:
            self.c_obj = sos_obj_get(obj.c_obj)
        else:
            self.c_obj = NULL
        self.c_attr = attr.c_attr
        self.c_v = sos_value_init(&self.c_v_, self.c_obj, self.c_attr)
        if self.c_v == NULL:
            raise ValueError()

    @property
    def value(self):
        return type_getters[<int>sos_attr_type(self.c_attr)](self.c_obj,
                                                             self.c_v.data)

    @value.setter
    def value(self, v):
        type_setters[<int>sos_attr_type(self.c_attr)](self.c_obj,
                                                      self.c_attr,
                                                      self.c_v.data,
                                                      v)

    cdef assign(self, sos_obj_t c_obj):
        if self.c_obj:
            sos_obj_put(self.c_obj)
        self.c_obj = c_obj
        self.c_v = sos_value_init(&self.c_v_, self.c_obj, self.c_attr)
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
        return sos_attr_name(self.c_attr)

    def __dealloc__(self):
        if self.c_v:
            sos_value_put(self.c_v)
        if self.c_obj:
            sos_obj_put(self.c_obj)

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
        if t == SOS_TYPE_STRUCT:
            n = sos_attr_size(c_attr)
            return <object>c_data.struc.char_[:n]
        else:
            return <object>type_getters[<int>t](c_obj, c_data)

    cdef set_py_value(self, sos_attr_t c_attr, sos_value_data_t c_data, val):
        cdef int t = sos_attr_type(c_attr)
        return <object>type_setters[<int>t](self.c_obj, c_attr, c_data, val)

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
                ret.append(self.get_py_value(arr_obj, c_attr, c_data))
            return ret
        if int == type(idx):
            c_attr = sos_schema_attr_by_id(sos_obj_schema(self.c_obj), idx)
        else:
            c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), idx)
        if c_attr == NULL:
            raise ValueError("Object has no attribute with id '{0}'".format(idx))
        arr_obj = NULL
        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        return self.get_py_value(arr_obj, c_attr, c_data)

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
        return self.get_py_value(arr_obj, c_attr, c_data)

    def __setitem__(self, idx, val):
        cdef sos_obj_t arr_obj
        cdef sos_attr_t c_attr
        cdef sos_value_data_t c_data
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
                c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
                self.set_py_value(c_attr, c_data, _v)
            return
        # single index assignment
        c_attr = sos_schema_attr_by_id(self.c_schema, idx)
        if c_attr == NULL:
            raise ValueError("Object has no attribute with id '{0}'".format(idx))
        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        self.set_py_value(c_attr, c_data, val)

    def __setattr__(self, name, val):
        cdef sos_obj_t arr_obj
        cdef sos_attr_t c_attr
        cdef sos_value_data_t c_data
        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")
        c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), name)
        if c_attr == NULL:
            raise ObjAttrError(name)
        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        self.set_py_value(c_attr, c_data, val)

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
        return np.PyArray_SimpleNewFromData(1, shape, np.dtype(eltype).num,
                                            c_data.array.data.byte_)
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
