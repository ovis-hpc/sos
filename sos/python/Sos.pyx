from cpython cimport PyObject, Py_INCREF
from libc.stdint cimport *
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
    EPROTO : "Protocol error",
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

cdef class Container(SosObject):
    cdef sos_t c_cont
    cdef sos_schema_t c_next_schema

    def __init__(self, path=None, o_perm=SOS_PERM_RO):
        SosObject.__init__(self)
        self.c_cont = NULL
        if path:
            self.open(path, o_perm=o_perm)

    def open(self, path, o_perm=SOS_PERM_RW):
        if self.c_cont != NULL:
            self.abort(EBUSY)
        self.c_cont = sos_container_open(path, o_perm)
        if self.c_cont == NULL:
            self.abort()

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

    def __iter__(self):
        self.c_next_schema = sos_schema_first(self.c_cont)
        return self

    def __next__(self):
        if self.c_next_schema == NULL:
            raise StopIteration
        s = Schema()
        s.assign(self.c_next_schema)
        self.c_next_schema = sos_schema_next(self.c_next_schema)
        return s

cdef class Partition(SosObject):
    cdef sos_part_t c_part
    def __init__(self):
        self.c_part = NULL

    cdef assign(self, sos_part_t c_part):
        self.c_part = c_part

    def delete(self):
        cdef int rc = sos_part_delete(self.c_part)
        if rc != 0:
            self.abort(rc)

    def move(self, new_path):
        cdef int rc = sos_part_move(self.c_part, new_path)
        if rc != 0:
            self.abort(rc)

    def state_set(self, new_state):
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

    def __iter__(self):
        self.c_next_attr = sos_schema_attr_first(self.c_schema)
        return self

    def __next__(self):
        if self.c_next_attr == NULL:
            raise StopIteration
        a = self.attr_by_id(sos_attr_id(self.c_next_attr))
        self.c_next_attr = sos_schema_attr_next(self.c_next_attr)
        return a

    def from_template(self, name, schema_def):
        cdef int rc
        cdef const char *idx_type = NULL
        cdef const char *idx_key = NULL
        cdef const char *idx_args = NULL
        self.c_schema = sos_schema_new(name)
        if self.c_schema == NULL:
            self.abort(ENOMEM)
        for attr in schema_def:
            if 'name' not in attr:
                raise ValueError("The 'name' is missing from the attribute")

            if 'type' not in attr:
                raise ValueError("The 'type' is missing from the attribute")

            n = attr['type'].upper()
            if n in sos_attr_types:
                t = sos_attr_types[n]
            else:
                raise ValueError("Invalid attribute type {0}.".format(n))

            sz = 0
            if t == SOS_TYPE_STRUCT:
                if 'size' not in attr:
                    raise ValueError("The type {0} must have a 'size'.".format(n))
                sz = attr['size']

            rc = sos_schema_attr_add(self.c_schema, attr['name'], t, <size_t>sz)
            if rc != 0:
                raise ValueError("The attribute named {0} resulted in error {1}".format(attr['name'], rc))

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
        cdef int rc
        rc = sos_schema_add(cont.c_cont, self.c_schema)
        if rc != 0:
            self.abort(rc)

    def attr_by_name(self, name):
        return Attr(self, attr_name=name)

    def attr_by_id(self, attr_id):
        return Attr(self, attr_id=attr_id)

    def AttrCount(self):
        return sos_schema_attr_count(self.c_schema)

    def schema_id(self):
        return sos_schema_id(self.c_schema)

    def name(self):
        return sos_schema_name(self.c_schema)

    def alloc(self):
        """
        Allocate a new object of this type in the container
        """
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

cdef class Key(SosObject):
    cdef sos_key_t c_key
    cdef size_t c_size

    def __init__(self, size):
        SosObject.__init__(self)
        self.c_key = sos_key_new(size)
        self.c_size = size

    def __len__(self):
        return self.c_size

    def __str__(self):
        s = sos_key_value(self.c_key)
        return s

    def __del__(self):
        self.release()

    cdef assign(self, sos_key_t c_key):
        if c_key == NULL:
            raise ValueError("key argument cannot be NULL")
        if self.c_key:
            sos_key_put(self.c_key)
        self.c_key = c_key

    def release(self):
        if self.c_key != NULL:
            sos_key_put(self.c_key)
            self.c_key = NULL

    def set_value(self, string):
        """Set the value of a key.
        This method *only* works with a string that has been prepared with
        struct.pack. Using a normal python string will result in PyString
        header garbage in your key.
        """
        cdef char *s = string
        memcpy(sos_key_value(self.c_key), s, self.c_size)

    def uint64(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.uint64_

    cdef uint32_t uint32(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.uint32_

    cdef uint16_t uint16(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.uint16_

    cdef uint8_t byte(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.byte_

    cdef int64(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.int64_

    cdef int32(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.int32_

    cdef int16(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.int16_

    cdef float32(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.float_

    cdef float64(self):
        cdef sos_value_data_t data = <sos_value_data_t>sos_key_value(self.c_key)
        return data.prim.double_

cdef class AttrIter(SosObject):

    cdef Attr attr
    cdef sos_iter_t c_iter
    cdef Key end_key

    def __init__(self, Attr attr,
                 Key start_key=None, Key end_key=None,
                 Key filt_key=None):
        self.c_iter = sos_attr_iter_new(attr.c_attr)
        if start_key is None:
            sos_iter_begin(self.c_iter)
        else:
            sos_iter_sup(self.c_iter, start_key.c_key)
        self.end_key = end_key
        self.attr = attr

    def __iter__(self):
        return self

    def __next__(self):
        cdef sos_obj_t c_obj
        c_obj = sos_iter_obj(self.c_iter)
        if c_obj == NULL:
            raise StopIteration()
        sos_iter_next(self.c_iter)
        o = Object()
        o.assign(c_obj)
        return o

    def find(self, Key key):
        cdef int rc = sos_iter_find(self.c_iter, key.c_key)
        if rc == 0:
            return True
        return False

    def find_sup(self, Key key):
        cdef int rc = sos_iter_sup(self.c_iter, key.c_key)
        if rc == 0:
            return True
        return False

    def find_inf(self, Key key):
        cdef int rc = sos_iter_inf(self.c_iter, key.c_key)
        if rc == 0:
            return True
        return False

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
TYPE_BYTE_ARRAY = SOS_TYPE_BYTE_ARRAY
TYPE_CHAR_ARRAY = SOS_TYPE_CHAR_ARRAY
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

    def attr_id(self):
        return sos_attr_id(self.c_attr)

    def name(self):
        return sos_attr_name(self.c_attr)

    def type(self):
        return sos_attr_type(self.c_attr)

    def type_name(self):
        return sos_type_strs[sos_attr_type(self.c_attr)]

    def indexed(self):
        if sos_attr_index(self.c_attr) != NULL:
            return True
        return False

    def size(self):
        return sos_attr_size(self.c_attr)

    def index(self):
        return Index(self)

    def set_start(self, Key key):
        self.start_key = key

    def set_end(self, Key key):
        self.end_key = key

    def __iter__(self):
        return AttrIter(self)

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

        c_iter = sos_attr_iter_new(self.c_attr)
        c_rc = sos_iter_end(c_iter)
        if c_rc:
            return None

        c_key = sos_iter_key(c_iter)
        key = Key(sos_key_size(c_key))
        sos_key_set(key.c_key, sos_key_value(c_key), sos_key_len(c_key))
        sos_key_put(c_key)
        sos_iter_free(c_iter)
        return key

    def min(self):
        """Return the minimum Key value of this attribute in the container"""
        cdef sos_iter_t c_iter
        cdef int c_rc
        cdef sos_key_t c_key

        c_iter = sos_attr_iter_new(self.c_attr)
        c_rc = sos_iter_end(c_iter)
        if c_rc:
            return None

        c_key = sos_iter_key(c_iter)
        key = Key(sos_key_size(c_key))
        sos_key_set(key.c_key, sos_key_value(c_key), sos_key_len(c_key))
        sos_key_put(c_key)
        sos_iter_free(c_iter)
        return key

FILT_COND_LT = SOS_COND_LT
FILT_COND_LE = SOS_COND_LE
FILT_COND_EQ = SOS_COND_EQ
FILT_COND_GE = SOS_COND_GE
FILT_COND_GT = SOS_COND_GT
FILT_COND_NE = SOS_COND_NE

cdef class Filter(SosObject):

    cdef Attr attr
    cdef sos_iter_t c_iter
    cdef sos_filter_t c_filt
    cdef sos_obj_t c_obj

    def __init__(self, Attr attr):
        self.attr = attr
        self.c_iter = sos_attr_iter_new(attr.c_attr)
        if self.c_iter == NULL:
            raise ValueError("The attribute {0} must be indexed.".format(attr.name()))

        self.c_filt = sos_filter_new(self.c_iter)

    def where(self, sos_cond_t cond, Value value):
        cdef int rc
        rc = sos_filter_cond_add(self.c_filt,
                                 <sos_attr_t>value.c_attr,
                                 cond,
                                 <sos_value_t>value.c_v)
        if rc:
            raise ValueError("Unsupported condition added " \
                             "with error {0}".format(rc))

    def first(self):
        cdef sos_obj_t o_0 = sos_filter_begin(self.c_filt)
        if o_0 == NULL:
            return None
        O_0 = Object()
        O_0.assign(o_0)
        return O_0

    def last(self):
        cdef sos_obj_t o_1 = sos_filter_end(self.c_filt)
        if o_1 == NULL:
            raise SystemError("If first is found, last should not be NULL")
        O_1 = Object()
        O_1.assign(o_1)
        return O_1

    def count(self):
        cdef size_t count = 0
        cdef sos_obj_t c_o = sos_filter_begin(self.c_filt)
        if c_o == NULL:
            return count
        sos_obj_put(c_o)
        while c_o != NULL:
            c_o = sos_filter_next(self.c_filt)
            sos_obj_put(c_o)
            count = count + 1
        return count

    def as_array(self, count):
        cdef size_t sample_count = count
        cdef sos_obj_t c_o
        cdef Value v = Value(self.attr)
        cdef int idx = 0

        shape = []
        shape.append(<np.npy_intp>sample_count)
        result = np.zeros(shape, dtype=np.float64, order='C')

        c_o = sos_filter_begin(self.c_filt)
        if c_o == NULL:
            return result

        sos_obj_put(c_o)
        idx = 0
        while c_o != NULL:
            v.assign(c_o)
            result[idx] = v.value
            c_o = sos_filter_next(self.c_filt)
            sos_obj_put(c_o)
            idx = idx + 1

        return result

    def __iter__(self):
        self.c_obj = sos_filter_begin(self.c_filt)
        return self

    def __next__(self):
        cdef sos_obj_t c_obj = self.c_obj
        if c_obj == NULL:
            raise StopIteration
        o = Object()
        o.assign(c_obj)
        self.c_obj = sos_filter_next(self.c_filt)
        return o

    def __dealloc__(self):
        if self.c_obj:
            sos_obj_put(self.c_obj)
        if self.c_iter:
            sos_iter_free(self.c_iter)
        if self.c_filt:
            sos_filter_free(self.c_filt)

cdef class Index(SosObject):
    cdef sos_index_t c_index
    def __init__(self, Attr attr=None, name=None):
        SosObject.__init__(self)
        if attr:
            self.c_index = sos_attr_index(attr.c_attr)
        else:
            self.c_index = NULL

    def find(self, Key key):
        cdef sos_obj_t c_obj = sos_index_find(self.c_index, key.c_key)
        if c_obj != NULL:
            o = Object()
            return o.assign(c_obj)
        return None

    def find_inf(self, Key key):
        cdef sos_obj_t c_obj = sos_index_find_inf(self.c_index, key.c_key)
        if c_obj != NULL:
            o = Object()
            return o.assign(c_obj)
        return None

    def find_sup(self, Key key):
        cdef sos_obj_t c_obj = sos_index_find_sup(self.c_index, key.c_key)
        if c_obj != NULL:
            o = Object()
            return o.assign(c_obj)
        return None

################################
# Object getter functions
################################
cdef class Array:
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
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_FLOAT64)

cdef object get_FLOAT_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_FLOAT32)

cdef object get_UINT64_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT64)

cdef object get_UINT32_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT32)

cdef object get_UINT16_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT16)

cdef object get_BYTE_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_UINT8)

cdef object get_CHAR_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT8)

cdef object get_INT64_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT64)

cdef object get_INT32_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
    return array.set_data(c_obj, c_data.array.data.char_, c_data.array.count, np.NPY_INT32)

cdef object get_INT16_ARRAY(sos_obj_t c_obj, sos_value_data_t c_data):
    array = Array()
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
# type_getters[<int>SOS_TYPE_OBJ] = get_OBJ
type_getters[<int>SOS_TYPE_STRUCT] = get_STRUCT
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
# type_getters[<int>SOS_TYPE_OBJ_ARRAY] = get_OBJ_ARRAY

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
# type_setters[<int>SOS_TYPE_OBJ] = set_OBJ
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
# type_setters[<int>SOS_TYPE_OBJ_ARRAY] = set_OBJ_ARRAY

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

cdef class Object(SosObject):
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

        if int == type(idx):
            c_attr = sos_schema_attr_by_id(sos_obj_schema(self.c_obj), idx)
        else:
            c_attr = sos_schema_attr_by_name(sos_obj_schema(self.c_obj), idx)
        if c_attr == NULL:
            raise ValueError("Object has no attribute with id {0}".format(idx))
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
            raise ValueError("Object has no attribute with name {0}".format(name))
        arr_obj = NULL
        c_data = sos_obj_attr_data(self.c_obj, c_attr, &arr_obj)
        return self.get_py_value(arr_obj, c_attr, c_data)

    def __setitem__(self, idx, val):
        cdef sos_obj_t arr_obj
        cdef sos_attr_t c_attr
        cdef sos_value_data_t c_data
        if self.c_obj == NULL:
            raise ValueError("No SOS object assigned to Object")
        c_attr = sos_schema_attr_by_id(sos_obj_schema(self.c_obj), idx)
        if c_attr == NULL:
            raise ValueError("Object has no attribute with id {0}".format(idx))
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

    def as_array(self, name, eltype = np.uint64):
        """
        Return the object data or an attributes data as an ndarray of the
        specified type
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
                           "Attribute name {0} is not " \
                           "present in schema {1}".format(attr, schema))

class ObjAttrError(NameError):
    def __init__(self, attr, schema):
        NameError.__init__(self,
                           "Object has not attribute with the name {0}" \
			   .format(attr, schema))

