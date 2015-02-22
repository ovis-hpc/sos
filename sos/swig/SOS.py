#!/usr/bin/python
import time
import sys
import sos
import os

class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Key:
    def __init__(self, attr, size=0):
        self.attr = attr
        self.key = sos.attr_key_get(attr.attr, size)

    def set(self, key_str):
        sos.attr_key_set(self.attr.attr, self.key, str(key_str))

    def __lt__(self, other):
        rc = sos.sos_key_cmp(self.attr.attr, self.key, other.key)
        if rc < 0:
            return True
        return False

    def __le__(self, other):
        rc = sos.sos_key_cmp(self.attr.attr, self.key, other.key)
        if rc < 0 or rc == 0:
            return True
        return False

    def __eq__(self, other):
        rc = sos.sos_key_cmp(self.attr.attr, self.key, other.key)
        return rc == 0

    def __ge__(self, other):
        rc = sos.sos_key_cmp(self.attr.attr, self.key, other.key)
        if rc == 0 or rc > 0:
            return True
        return False

    def __gt__(self, other):
        rc = sos.sos_key_cmp(self.attr.attr, self.key, other.key)
        if rc > 0:
            return True
        return False

    def __ne__(self, other):
        rc = sos.sos_key_cmp(self.attr.attr, self.key, other.key)
        return rc != 0

    def __str__(self):
        s = sos.sos_key_to_str(self.attr.attr, self.key)
        return s

    def __del__(self):
        sos.sos_key_put(self.key)

class Timestamp:
    def __init__(self, secs, nsecs):
        self.secs = secs
        self.nsecs = nsecs
    def seconds(self):
        return self.secs
    def nanoseconds(self):
        return self.nsecs
    def __str__(self):
        t = time.localtime(self.secs)
        return time.strftime('%Y/%m/%d %H:%M:%S', t)
    def __float__(self):
        return float(self.secs) + float(self.nsecs) / 1.0e9
    def __int__(self):
        return float(self.secs)

class Object:

    json_fmt = 1
    csv_fmt = 2
    table_fmt = 3
    def_fmt = json_fmt

    def __init__(self, obj, schema, str_fmt=None):
        if not str_fmt:
            self.str_fmt = self.def_fmt
        else:
            self.str_fmt = str_fmt
        self.obj = obj
        self.schema = schema
        self.values = {}
        self.col_widths = {}
        for attr_name, attr in schema.attrs.iteritems():
            self.col_widths[attr_name] = attr.col_width
            v = sos.sos_value(obj, attr.attr)
            t = attr.sos_type
            an_ = attr_name+'_'
            if t == sos.SOS_TYPE_INT32:
                self.__dict__[an_] = v.data.prim.int32_
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_INT64:
                self.__dict__[an_] = v.data.prim.int64_
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_UINT32:
                self.__dict__[an_] = v.data.prim.uint32_
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_UINT64:
                self.__dict__[an_] = v.data.prim.uint64_
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_FLOAT:
                self.__dict__[an_] = v.data.prim.float_
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_DOUBLE:
                self.__dict__[an_] = v.data.prim.double_
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_LONG_DOUBLE:
                self.__dict__[an_] = v.data.prim.long_double_
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_TIMESTAMP:
                self.__dict__[an_] = Timestamp(v.data.prim.timestamp_.fine.secs,
                                               v.data.prim.timestamp_.fine.nsecs)
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_BYTE_ARRAY:
                self.__dict__[an_] = sos.value_as_str(v)
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            elif t == sos.SOS_TYPE_OBJ:
                self.__dict__[an_] = sos.value_as_str(v)
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            else:
                self.__dict__[an_] = ""
                self.__dict__[attr_name] = lambda an=an_ : self.__dict__[an]
            self.values[attr_name] = self.__dict__[an_]
            sos.sos_value_put(v)

    def key(self, attr_name):
        attr = self.schema.attr(attr_name)
        key = Key(attr)
        key.set(str(self.values[attr_name]))
        return key

    def __del__(self):
        sos.sos_obj_put(self.obj)

    def json_header(self):
        return ''
    def json_str(self):
        s = '{{"{0}":{{'.format(self.schema.name())
        comma = False
        for attr_name, value in self.values.iteritems():
            if comma:
                s += ','
            s += '"{0}":"{1}"'.format(
                attr_name, value)
            comma = True
        s += "}}"
        return s

    def csv_header(self):
        comma = False
        s = '#'
        for attr_name, value in self.values.iteritems():
            if comma:
                s += ','
            s += attr_name
            comma = True
        return s
    def csv_str(self):
        comma = False
        s = ''
        for attr_name, value in self.values.iteritems():
            if comma:
                s += ','
            s += str(value)
            comma = True
        return s

    def table_header(self):
        s = ''
        l = ''
        for attr_name, value in self.values.iteritems():
            w = self.col_widths[attr_name]
            s += "{0!s:{1}} ".format(attr_name, w)
            l += '-'*w + ' '
        return s + '\n' + l + '\n'
    def table_str(self):
        s = ''
        for attr_name, value in self.values.iteritems():
            s += "{0!s:{1}} ".format(value, self.col_widths[attr_name])
        return s

    def header_str(self):
        if self.str_fmt == self.json_fmt:
            return self.json_header()
        elif self.str_fmt == self.csv_fmt:
            return self.csv_header()
        elif self.table_fmt == self.table_fmt:
            return self.table_header()
    def __str__(self):
        if self.str_fmt == self.json_fmt:
            return self.json_str()
        elif self.str_fmt == self.csv_fmt:
            return self.csv_str()
        elif self.table_fmt == self.table_fmt:
            return self.table_str()

class Condition:
    def __init__(self, attr, value, comparator):
        self.attr = attr
        self.value = value
        self.comparator = comparator

    def __str__(self):
        return "{0} {1} {2}".format(self.attr.name(), self.comparator, self.value)

class Iterator:
    def __init__(self, iter_, attr):
        self.iter_ = iter_
        self.attr = attr
        self.pos_ = sos.sos_pos()

    def __del__(self):
        pass

    def begin(self):
        rc = sos.sos_iter_begin(self.iter_)
        if rc == 0:
            obj = sos.sos_iter_obj(self.iter_)
            if obj:
                return Object(obj, self.attr.schema)
        else:
            return None

    def end(self):
        rc = sos.sos_iter_end(self.iter_)
        if rc == 0:
            obj = sos.sos_iter_obj(self.iter_)
            if obj:
                return Object(obj, self.attr.schema)
        return None

    def next(self):
        rc = sos.sos_iter_next(self.iter_)
        if rc == 0:
            obj = sos.sos_iter_obj(self.iter_)
            if obj:
                return Object(obj, self.attr.schema)
        return None

    def prev(self):
        rc = sos.sos_iter_prev(self.iter_)
        if rc == 0:
            obj = sos.sos_iter_obj(self.iter_)
            if obj:
                return Object(obj, self.attr.schema)
        return None

    def find(self, key):
        rc = sos.sos_iter_find(self.iter_, key.key)
        if rc == 0:
            obj = sos.sos_iter_obj(self.iter_)
            if obj:
                return Object(obj, self.attr.schema)
        return None

    def sup(self, key):
        rc = sos.sos_iter_sup(self.iter_, key.key)
        if rc == 0:
            obj = sos.sos_iter_obj(self.iter_)
            if obj:
                return Object(obj, self.attr.schema)
        return None

    def inf(self, key):
        rc = sos.sos_iter_inf(self.iter_, key.key)
        if rc == 0:
            obj = sos.sos_iter_obj(self.iter_)
            if obj:
                return Object(obj, self.attr.schema)
        return None

    def pos(self):
        rc = sos.sos_iter_pos(self.iter_, self.pos_)
        return (rc, sos.pos_to_str(self.pos_))

    def set(self, pos):
        rc = sos.pos_from_str(self.pos_, pos)
        if rc == 0:
            return sos.sos_iter_set(self.iter_, self.pos_)
        return 22;

    def cardinality(self):
        return sos.sos_iter_card(self.iter_)

    def duplicates(self):
        return sos.sos_iter_dups(self.iter_)

class Filter:
    comparators = {
        'lt' : sos.SOS_COND_LT,
        'le' : sos.SOS_COND_LE,
        'eq' : sos.SOS_COND_EQ,
        'ge' : sos.SOS_COND_GE,
        'gt' : sos.SOS_COND_GT,
        'ne' : sos.SOS_COND_NE
    }
    def __init__(self, iter_):
        self.iter_ = iter_
        self.filt = sos.sos_filter_new(iter_.iter_)
        self.pos_ = sos.sos_pos()

    def cardinality(self):
        return sos.sos_iter_card(self.iter_.iter_)

    def duplicates(self):
        return sos.sos_iter_dups(self.iter_.iter_)

    def add(self, attr, cmp_str, value_str):
        # strip embedded '"' from value if present
        value_str = value_str.replace('"', '')
        cond_v = sos.sos_value_new()
        if not cond_v:
            raise Error("The attribute value for {0} "
                        "could not be created.".format(attr.name()))
        cond_v = sos.sos_value_init(cond_v, None, attr.attr);
        rc = sos.sos_value_from_str(cond_v, str(value_str), None)
        if rc != 0:
            raise Error("The value {0} is invalid for the {1} attribute."
                        .format(value_str, attr.name()))

        if not cmp_str.lower() in Filter.comparators:
            raise Error("The comparison {0} is invalid.".format(comp_str))

        rc = sos.sos_filter_cond_add(self.filt, attr.attr,
                                     Filter.comparators[cmp_str.lower()], cond_v)
        if rc != 0:
            raise Error("Invalid filter condition, error {0}".format(rc))

    def unique(self):
        sos.sos_filter_flags_set(self.filt, sos.SOS_ITER_F_UNIQUE)

    def begin(self):
        obj = sos.sos_filter_begin(self.filt)
        if obj:
            return Object(obj, self.iter_.attr.schema)
        return None

    def end(self):
        obj = sos.sos_filter_end(self.filt)
        if obj:
            return Object(obj, self.iter_.attr.schema)
        return None

    def next(self):
        obj = sos.sos_filter_next(self.filt)
        if obj:
            return Object(obj, self.iter_.attr.schema)
        return None

    def prev(self):
        obj = sos.sos_filter_prev(self.filt)
        if obj:
            return Object(obj, self.iter_.attr.schema)
        return None

    def obj(self):
        obj = sos.sos_filter_obj(self.filt)
        if obj:
            return Object(obj, self.iter_.attr.schema)

    def pos(self):
        rc = sos.sos_filter_pos(self.filt, self.pos_)
        return (rc, sos.pos_to_str(self.pos_))

    def set(self, pos):
        rc = sos.pos_from_str(self.pos_, pos)
        if rc == 0:
            return sos.sos_filter_set(self.filt, self.pos_)
        return 22;

col_widths = {
    sos.SOS_TYPE_INT32 : 10,
    sos.SOS_TYPE_INT64 : 18,
    sos.SOS_TYPE_UINT32 : 10,
    sos.SOS_TYPE_UINT64 : 18,
    sos.SOS_TYPE_FLOAT : 12,
    sos.SOS_TYPE_DOUBLE : 24,
    sos.SOS_TYPE_LONG_DOUBLE : 48,
    sos.SOS_TYPE_TIMESTAMP : 18,
    sos.SOS_TYPE_OBJ : 8,
    sos.SOS_TYPE_BYTE_ARRAY : -1,
    sos.SOS_TYPE_INT32_ARRAY : 8,
    sos.SOS_TYPE_INT64_ARRAY : 8,
    sos.SOS_TYPE_UINT32_ARRAY : 8,
    sos.SOS_TYPE_UINT64_ARRAY : 8,
    sos.SOS_TYPE_FLOAT_ARRAY : 8,
    sos.SOS_TYPE_DOUBLE_ARRAY : 8,
    sos.SOS_TYPE_LONG_DOUBLE_ARRAY : 8,
    sos.SOS_TYPE_OBJ_ARRAY : 8,
}

sos_type_name = {
    sos.SOS_TYPE_INT32 : "INT32",
    sos.SOS_TYPE_INT64 : "INT64",
    sos.SOS_TYPE_UINT32 : "UINT32",
    sos.SOS_TYPE_UINT64 : "UINT64",
    sos.SOS_TYPE_FLOAT : "FLOAT",
    sos.SOS_TYPE_DOUBLE : "DOUBLE",
    sos.SOS_TYPE_LONG_DOUBLE : "LONG_DOUBLE",
    sos.SOS_TYPE_TIMESTAMP : "TIMESTAMP",
    sos.SOS_TYPE_OBJ : "OBJ",
    sos.SOS_TYPE_BYTE_ARRAY : "BYTE_ARRAY",
    sos.SOS_TYPE_INT32_ARRAY : "INT32_ARRAY",
    sos.SOS_TYPE_INT64_ARRAY : "INT64_ARRAY",
    sos.SOS_TYPE_UINT32_ARRAY : "UINT32_ARRAY",
    sos.SOS_TYPE_UINT64_ARRAY : "UINT64_ARRAY",
    sos.SOS_TYPE_FLOAT_ARRAY : "FLOAT_ARRAY",
    sos.SOS_TYPE_DOUBLE_ARRAY : "DOUBLE_ARRAY",
    sos.SOS_TYPE_LONG_DOUBLE_ARRAY : "LONG_DOUBLE_ARRAY",
    sos.SOS_TYPE_OBJ_ARRAY : "OBJ_ARRAY",
}

class Attr:
    def __init__(self, attr, schema):
        self.attr = attr
        self.sos_type = sos.sos_attr_type(attr)
        self.schema = schema
        self.has_index = sos.sos_attr_index(self.attr)
        if self.has_index:
            self.iter_ = Iterator(sos.sos_iter_new(self.attr), self)
        else:
            self.iter_ = None
        self.col_width = col_widths[self.sos_type]
        if self.col_width < 0:
            self.col_width = sos.sos_attr_size(self.attr)

    def name(self):
        return sos.sos_attr_name(self.attr)

    def size(self):
        return sos.sos_attr_size(self.attr)

    def iterator(self):
        return self.iter_

    def indexed(self):
        if self.has_index:
            return True
        return False

    def set_col_width(self, w):
        self.col_width = w

    def type_str(self):
        return sos_type_name[self.sos_type]

    def attr_id(self):
        return sos.sos_attr_id(self.attr)

    def value(self, obj):
        return sos.sos_value(obj.obj, self.attr)

class Schema:
    def __init__(self, schema):
        self.schema = schema
        self.name_ = sos.sos_schema_name(schema)
        self.attrs = {}
        for attr_id in range(0, sos.sos_schema_attr_count(schema)):
            attr = sos.sos_schema_attr_by_id(schema, attr_id)
            self.attrs[sos.sos_attr_name(attr)] = Attr(attr, self)

    def name(self):
        return self.name_

    def attr(self, name):
        return self.attrs[name]

    def attr_count(self):
        return sos.sos_schema_attr_count(self.schema)

    def __del__(self):
        sos.sos_schema_put(self.schema)

class Container:
    RW = sos.SOS_PERM_RW
    RO = sos.SOS_PERM_RO
    def __init__(self, path, mode=RW):
        self.schemas = {}
        try:
            self.path = path
            self.container = sos.container_open(path, mode)
            if self.container == None:
                raise Error('The container "{0}" could not be opened.'.format(path))
            self.schemas = {}
            schema = sos.sos_schema_first(self.container)
            while schema != None:
                self.schemas[sos.sos_schema_name(schema)] = Schema(schema)
                schema = sos.sos_schema_next(schema)
        except Exception as e:
            raise Error('Exception: ' + str(e))

    def name(self):
        return self.path

    def schema(self, name):
        try:
            return self.schemas[name]
        except:
            return None

    def info(self):
        sos.container_info(self.container)

def dump_schema_objects(schema):
    # Find the first attribute with an index and use it to iterate
    # over the objects in the container
    for attr_name, attr in schema.attrs.iteritems():
        if attr.has_index:
            iter = attr.iter_
            break

    if not iter:
        print("No attributes with indices in this schema")
        return

    Object.def_fmt = Object.csv_fmt
    o = iter.begin()
    print(o.header_str())
    while o is not None:
        print(o)
        o = iter.next()

    Object.def_fmt = Object.table_fmt
    o = iter.begin()
    print(o.header_str())
    while o is not None:
        print(o)
        o = iter.next()

    Object.def_fmt = Object.json_fmt
    o = iter.begin()
    print(o.header_str())
    while o is not None:
        print(o)
        o = iter.next()

if __name__ == "__main__":
    c = Container("/SOS/bluewaters")
    for schema_name, schema in c.schemas.iteritems():
        dump_schema_objects(schema)
