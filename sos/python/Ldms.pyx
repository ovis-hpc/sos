from __future__ import print_function
import cython
# if you forget this, your module cannot be imported by other cythong modules
cimport cython
# cimport Sos
# cimport Array

from libc.stdint cimport *
from libc.stdlib cimport *
from libc.string cimport *
cimport numpy as np
from numpy cimport npy_intp

import sys
import numpy as np
import struct

cimport Sos
import Sos
# from sos import Array
import Array

cdef class Vector(object):
    cdef cont
    cdef schema
    cdef time_name
    cdef comp_name

    def __init__(self, container, schema,
                 time_name="timestamp", comp_name="component_id"):
        self.cont = container
        self.schema = container.schema_by_name(schema)
        if not self.schema:
            raise NameError("The schema named '{0}' does not exist.".format(schema))
        self.time_name = time_name
        self.comp_name = comp_name

    def range(self, comp_id=None, time_name=None, comp_name=None):
        """Return the time window of a sample for a component

        Returns a ( start, end ) tuple of the Unix timestamps for the
        first and last samples available for the specified metric and
        component. If comp_id is not specified, the first and last
        sample times in the container will be returned.

        Positional arguments:
        -- The attribute name for the metric

        Keyword Arguments:
        time_name -- The name of the Timestamp attribute.
        comp_name -- The name of the Component Id attribute
        comp_id   -- The component Id.
        """
        if not time_name:
            time_name = self.time_name
        if not comp_name:
            comp_name = self.comp_name
        attr = self.schema.attr_by_name(time_name)
        if attr is None:
            raise Sos.SchemaAttrError(time_name, self.schema.name())

        filt = Sos.Filter(attr)
        if comp_id is not None:
            attr = self.schema.attr_by_name(comp_name)
            if attr is None:
                raise Sos.SchemaAttrError(comp_name, self.schema.name())
            v = Sos.Value(attr)
            v.value = comp_id
            filt.where(Sos.FILT_COND_EQ, v)

        first = filt.first()
        if first is None:
            return None

        last = filt.last()
        if last is None:
            raise SystemError("Internal error")

        return ( first[time_name], last[time_name] )

    def query(self, metric_name, comp_id,
              start_time=None, end_time=None, index_name=None):
        """Query the store for a metric vector

        This function queries the SOS container and returns a 1
        dimensional Numpy array. If comp_id is specified, only
        metrics for the specified components will be considered.
        If comp_is None, samples will be accumulated across
        components.

        It is assumed that an attribute named 'timestamp' contains the
        Unix timestamp. The name can be overridden with the time_attr
        keyword parameter.

        It is assumed that an attributed named 'component_id' contains
        the Component Id. This name can be overridden with the
        comp_attr keyword parameter.

        Positional arguments:
        -- The attribute name for the metric
        -- The Unix timestamp start time
        -- The Unix timestamp end time
        -- The component Id

        Keyword Arguments:
        time_name -- The name of the Timestamp attribute.
        comp_name -- The name of the Component Id attribute
        index_name -- The name of the attribute to use for the Index
        """
        cdef double start
        cdef double end
        cdef size_t rec_no

        metric_attr = self.schema.attr_by_name(metric_name)
        if not metric_attr:
            raise Sos.SchemaAttrError(metric_name, self.schema.name())
        if not index_name:
            index_name = self.time_name

        time_attr = self.schema.attr_by_name(self.time_name)
        if not time_attr:
            raise Sos.SchemaAttrError(self.time_name, self.schema.name())
        comp_attr = self.schema.attr_by_name(self.comp_name)
        if not comp_attr:
            raise Sos.SchemaAttrError(self.comp_name, self.schema.name())

        if index_name:
            index_attr = self.schema.attr_by_name(index_name)
            if not index_attr:
                raise Sos.SchemaAttrError(index_name, self.schema.name())
        else:
            index_attr = time_attr

        if start_time:
            start = start_time
        else:
            start = 0

        if end_time:
            end = end_time
        else:
            end = 0

        x = Array.Array()
        y = Array.Array()

        ct_attr = self.schema.attr_by_name(index_name)
        key = Sos.Key(ct_attr.size())
        kv = struct.pack(">QQ", comp_id, start)
        key.set_value(kv)
        it = Sos.AttrIter(ct_attr, start_key=key)

        rec_no = 0
        time = Sos.Value(time_attr)
        metric = Sos.Value(metric_attr)
        comp = Sos.Value(comp_attr)

        for sample in it:
            time.set_obj(sample)
            if end != 0 and time.value > end:
                break
            comp.set_obj(sample)
            if comp.value != comp_id:
                print("Next component is {0}".format(comp.value))
                break
            metric.set_obj(sample)
            x.append(time.value)
            y.append(metric.value)
            rec_no += 1
        # return (rec_no, x.as_ndarray(), y.as_ndarray())
        return (rec_no, x, y)

    def __dealloc__(self):
        pass
