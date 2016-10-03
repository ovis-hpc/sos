from __future__ import print_function
from cpython cimport PyObject, Py_INCREF
from libc.stdint cimport *
import sys
import numpy as np
import struct
from numpy cimport npy_intp
cimport numpy as np
import Sos
cimport Sos

np.import_array()

cdef class Vector(object):
    cdef cont
    cdef schema
    cdef time_name
    cdef comp_name

    def __init__(self, container, schema,
                 time_name="timestamp", comp_name="component_id"):
        self.cont = container
        self.schema = container.schema_by_name(schema)
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

    def query(self, metric_name, start_time, end_time, comp_id=None, samples=None,
              time_name=None, comp_name=None):
        """Query the store for a metric vector

        This function queries the SOS container and returns a 1
        dimensional Numpy array. If comp_id is specified, only
        metrics for the specified components will be considered.
        If comp_is None, samples will be accumulated across
        components.

        If samples is specified, the timeseries will be clamped to the
        specified number of buckets and values will be summed inside each
        bucket.

        It is assumed that an attribute named 'timestamp' contains the
        Unix timestamp. The name can be overridden with the time_name
        keyword parameter.

        It is assumed that an attributed named 'component_id' contains
        the Component Id. This name can be overridden with the
        comp_name keyword parameter.

        Positional arguments:
        -- The attribute name for the metric
        -- The Unix timestamp start time
        -- The Unix timestamp end time
        -- The component Id
        -- The maximum number of samples in the resulting array

        Keyword Arguments:
        time_name -- The name of the Timestamp attribute.
        comp_name -- The name of the Component Id attribute
        """

        cdef size_t sample_count
        cdef double bin_width
        cdef np.npy_intp shape[1]
        cdef size_t bkt
        cdef double start
        cdef double end
        cdef double sample_time

        if not time_name:
            time_name = self.time_name
        if not comp_name:
            comp_name = self.comp_name

        #
        # Create the filter with a primary index on 'timestamp'
        #
        attr = self.schema.attr_by_name(time_name)
        if attr is None:
            raise Sos.SchemaAttrError(time_name, self.schema.name())

        filt = Sos.Filter(attr)

        # Add the start time condition
        v = Sos.Value(attr)
        v.value = <int>start_time
        filt.where(Sos.FILT_COND_GE, v)

        # Add the end time condition. NB: this has to be a new attr to
        # avoid overwriting the start time condition value
        v = Sos.Value(attr)
        v.value = <int>end_time
        filt.where(Sos.FILT_COND_LE, v)

        # Add the component if specified
        if comp_id is not None:
            attr = self.schema.attr_by_name(comp_name)
            if attr is None:
                raise Sos.SchemaAttrError(comp_name, self.schema.name())
            v = Sos.Value(attr)
            v.value = comp_id
            filt.where(Sos.FILT_COND_EQ, v)

        # Get the number of records the filter matches
        print("Counting samples...", end='')
        sys.stdout.flush()
        if samples is None:
            sample_count = filt.count()
        else:
            sample_count = samples
        print("{0}".format(sample_count))

        # Divide the time span into equidistant buckets based
        # on the sample count
        start = start_time
        end = end_time
        bin_width = <double>(end - start + 1) / <double>sample_count;

        shape[0] = <np.npy_intp>sample_count
        result = np.zeros(shape, dtype=np.float64, order='C')
        print("Computing result...", end='')
        sys.stdout.flush()
        for o in filt:
            sample_time = <double>o[time_name]
            bkt = <long>((sample_time - start) / bin_width)
            result[bkt] += <double>o[metric_name]
        print("complete")
        result = filt.as_array(sample_count)
        return result

    def __dealloc__(self):
        pass
