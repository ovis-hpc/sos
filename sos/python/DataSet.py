#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import datetime as dt
import textwrap
import copy
import time
import os
import sys

class DataSetIter(object):
    def __init__(self, dataSet, reverse=False):
        self.dataSet = dataSet
        self.array = dataSet.array_with_series_idx[0]
        self.reverse = reverse
        self.series_size = dataSet.series_size
        if reverse:
            self.row_no = self.series_size - 1
        else:
            self.row_no = 0

    def __iter__(self):
        return self

    def next(self):
        if self.reverse:
            if self.row_no < 0:
                raise StopIteration
            res = self.array[self.row_no]
            self.row_no -= 1
        else:
            if self.row_no == self.series_size:
                raise StopIteration
            res = self.array[self.row_no]
            self.row_no += 1
        return res


class DataSet(object):
    """A class for accessing datasource results

    The DataSet class wraps Numpy arrays that are returned by a
    DataSource. The class provides a convenient ways to access,
    combine and manipulate datasource results by name.

    For example:

        ds.config(path = '~/meminfo.csv')
        ds.select( [ 'timestamp', 'jobinfo.job_name',
                     'MemFree', 'MemAvailable' ],
                   from_ [ 'meminfo', 'jobinfo' ])
        result = ds.get_results()

    Returns the queried data as a DataSet. The following returns the
    'timestamp' data as a numpy array:

        tstamps = result['timestamp']

    The tstamps variable is a Numpy array that contains all of the
    timestamp data.

    Similarly,

        job_names = result['jobinfo.job_name']

    The job_names variable is a numpy array of strings containing the
    'jobinfo.job_name' series.

    Indexing is always [series][item]; the first diminesion is the
    series name (string) or number (int), and the second dimension
    is the datum index (int). Therefore data_set['timestamp'][0] is the
    first datum in the 'timestamp' series.

    """

    def __init__(self):
        """Create an instance of a DataSet"""
        self.arrays = []
        self.series_size = 0
        self.series_names = []
        self.array_with_series_name = {}
        self.array_with_series_idx = []

    def copy(self, my_ser, my_offset, src_set, src_ser, src_offset, count):
        dst = self.array_with_series_name[my_ser]
        src = src_set.array_with_series_name[src_ser]
        dst[my_offset:my_offset+count] = src[src_offset:src_offset+count]

    def series_idx(self, name):
        return self.series_names.index(name)

    def rename(self, oldname, newname):
        """Rename a series

        Changes the name of the data series. The index is not change.

        Positional Parameters
        -- The series name
        -- The desired new series name

        """
        if type(oldname) == int:
            oldname = self.series_names[oldname]
        if newname in self.series_names:
            raise ValueError("The series name {0} already exists.".format(newname))
        if oldname not in self.series_names:
            raise ValueError("The series name {0} does not exist.".format(oldname))
        idx = self.series_names.index(oldname)
        self.series_names[idx] = newname
        array = self.array_with_series_name[oldname]
        del self.array_with_series_name[oldname]
        self.array_with_series_name[newname] = array

    def set_series_size(self, series_size):
        """Sets the size of the series present in the set

        This function is used to set how much data is present in
        each. Use the len() function to get the capacity (max-size) of
        a series.
        """
        self.series_size = series_size

    def get_series_size(self):
        """Gets the size of the series

        This function is used to query how much data is actually
        present in a series. Use len() to get the capacity of a
        series.
        """
        return self.series_size

    def get_series_count(self):
        return len(self.series_names)

    def concat(self, aset):
        """Concatenate a set to the DataSet

        Add the new set's data to the end of this set

        Positional Parameters:
        -- The DataSet to get the data from

        """
        series_list = aset.series

        # Create the new DataSet
        newds = DataSet()
        row_count = self.series_size + aset.series_size
        for ser in series_list:
            nda = self.array_with_series_name[ser]
            shape = nda.shape
            if len(shape) > 1:
                # arrays
                array = np.zeros([ row_count ] + shape[1:], dtype=nda.dtype)
            else:
                array = np.zeros([ row_count ], dtype=nda.dtype)
            newds.append_array(row_count, ser, array)
        newds.set_series_size(row_count)

        # Copy the data from self
        for ser in series_list:
            newds.copy(ser, 0, self, ser, 0, self.series_size)

        # copy the data from aset
        for ser in series_list:
            newds.copy(ser, self.series_size, aset, ser, 0, aset.series_size)

        return newds

    def append_array(self, series_size, series_name, array):
        """Append a numpy array to a DataSet

        Add a series to the DataSet.

        Positional Parameters:
        -- The size of the series being added
        -- The series name
        -- The Numpy array containing the series data
        """
        if series_name in self.series_names:
            raise ValueError("A series named '{0}' already exists "
                             "in the set.".format(series_name))
        if type(array) == DataSet:
            raise ValueError("The array parameter must be a numpy array")

        self.series_names.append(series_name)
        self.array_with_series_idx.append(array)
        self.array_with_series_name[series_name] = array
        self.set_series_size(series_size)
        return self

    def append_series(self, aset, series_list=None):
        """Append series from one DataSet to another

        If the 'series_list' keyword argument is not specified, all
        series in the set are added; otherwise, only the series in the
        series_list are added.

        Positional Parameters:
        -- The source DataSet

        Keyword Parameters:
        series_list -- A list of series names from the source DataSet
                       to append

        """
        if aset.get_series_size() == 0:
            raise ValueError("Empty sets cannot be appended.")

        if series_list is None:
            series_list = aset.series_names

        for ser in series_list:
            if ser in self.series_names:
                raise ValueError("The series {0} is already present.".format(ser))

        for ser in series_list:
            self.append_array(aset.series_size, ser,
                              aset.array_with_series_name[ser])
        return self

    def tolist(self):
        """Return a JSon encode-able representaiton of this dataset.

        Numpy datatypes are not directly encodable in JSon. This
        function returns a Python list of lists containing Python
        types that will work with json.dumps()

        np.intXX types are int(x)
        np.floatXX types become float(x)
        np.array types become str(x)
        """
        aSet = []
        for row_no in range(0, self.series_size):
            aRow = []
            for col in range(0, self.series_count):
                v = self[col, row_no]
                typ = type(v)
                if typ == np.ndarray or typ == np.string_:
                    v = str(v)
                elif typ == np.float32 or typ == np.float64:
                    v = float(v)
                elif typ == np.int64 or typ == np.uint64:
                    v = int(v)
                elif typ == np.int32 or typ == np.uint32:
                    v = int(v)
                elif typ == np.int16 or typ == np.uint16:
                    v = int(v)
                elif typ == np.datetime64:
                    # convert to milliseconds from microseconds
                    v = v.astype('int') / 1000
                aRow.append(v)
            aSet.append(aRow)
        return aSet

    def __getitem__(self, idx):
        if type(idx) == str or type(idx) == int:
            idx = [ idx ]
        ds = DataSet()
        for i in idx:
            if type(i) == str:
                ds.append_array(self.series_size, i,
                                self.array_with_series_name[i])
            elif type(i) == int:
                ds.append_array(self.series_size, self.series_names[i],
                                self.array_with_series_idx[i])
        return ds

    def __setitem__(self, idx, value):
        if type(idx) == str:
            nda = self.array_with_series_name[idx]
            nda[:] = value
        elif type(idx) == int:
            nda = self.array_with_series_idx[idx]
            nda[:] = value
        elif type(idx) == tuple or type(idx) == slice:
            if type(idx[0]) == str:
                nda = self.array_with_series_name[idx[0]]
                nda[idx[1:]] = value
            else:
                nda = self.array_with_series_idx[idx[0]]
                nda[idx[1:]] = value
        else:
            raise ValueError("Unsupported index type {0}".format(type(idx)))

    def __len__(self):
        return len(self.array_with_series_idx[0])

    def show(self, series=None, limit=None, width=16):
        wrap = []
        if series is None:
            series_names = self.series_names
        else:
            series_names = series
        for ser in series_names:
            name = ser.replace('.', '. ')
            wrap.append(textwrap.wrap(name, width))
        max_len = 0
        for col in wrap:
            if len(col) > max_len:
                max_len = len(col)
        for r in range(0, max_len):
            for col in wrap:
                skip = max_len - len(col)
                if r < skip:
                    label = ""
                else:
                    label = col[r - skip]
                print("{0:>{width}}".format(label, width=width), end=' ')
            print('')
        for ser in series_names:
            print("{0:{width}}".format('-'.ljust(width, '-'), width=width), end=' ')
        print('')
        count = 0
        if limit is None:
            limit = self.get_series_size()
        else:
            limit = min(limit, self.get_series_size())
        for i in range(0, limit):
            for ser in series_names:
                v = self.array(ser)[i]
                if type(v) == np.timedelta64 or type(v) == np.datetime64:
                    print("{0:{width}}".format(str(v), width=width), end=' ')
                elif type(v) == float:
                    print("{0:{width}f}".format(v, width=width), end=' ')
                else:
                    print("{0:{width}}".format(v, width=width), end=' ')
            print('')
            count += 1
        for ser in series_names:
            print("{0:{width}}".format('-'.ljust(width, '-'), width=width), end=' ')
        print('\n{0} results'.format(count))

    def new(self, series_size, series_list, shapes=None, types=None):
        """Factory function for creating a new DataSet

        Positional Parametes:
        -- Series size
        -- Series names

        Keyword Parameters:
        shapes -- Dictionary(by series name) of array shapes
        types  -- Dictionary of array element types

        Returns:
        A new DataSet with series of the specified size
        """
        for ser in series_list:
            if shapes and ser in shapes:
                shape = shapes[ser]
            else:
                shape = [ series_size ]
            if types and ser in types:
                typ = types[ser]
            else:
                typ = np.dtype('float64')
            self.append_array(series_size, ser, np.zeros(shape, dtype=typ))
        self.set_series_size(series_size)
        return self

    @property
    def series_count(self):
        """Return the number of series in the DataSet"""
        return len(self.series_names)

    @property
    def series(self):
        """Return a copy of the series names

        This method returns a copy of the series names so that they
        can be manipulated by the caller without affecting the DataSet

        Use series_count() to obtain the number of series instead of
        len(series) to avoid unnecessarily copying data to obtain a
        series count.
        """
        return copy.copy(self.series_names)

    def array(self, series):
        """Returns the array for the series"""
        if type(series) == int:
            return self.array_with_series_idx[series][0:self.series_size]
        return self.array_with_series_name[series][0:self.series_size]

    def toset(self, series):
        """Returns a DataSet with a single series"""
        return self[series]

    def __iter__(self):
        return DataSetIter(self)

    def __reversed__(self):
        return DataSetIter(self, reverse=True)

    def __sub__(self, rhs):
        """Implements lhs - rhs"""
        result = DataSet()
        if type(rhs) == DataSet:
            for idx in range(0, len(self.array_with_series_idx)):
                ser_name = '(' + self.series_names[idx] + '-' + rhs.series_names[idx] + ')'
                ary = self.array(idx) - rhs.array(idx)
                result.append_array(self.series_size, ser_name, ary)
        else:
            for idx in range(0, len(self.array_with_series_idx)):
                ser_name = '(' + self.series_names[idx] + '-' + str(rhs) + ')'
                ary = self.array(idx) - rhs
                result.append_array(self.series_size, ser_name, ary)
        return result

    def __add__(self, rhs):
        """Implements lhs + rhs"""
        result = DataSet()
        if type(rhs) == DataSet:
            for idx in range(0, len(self.array_with_series_idx)):
                ser_name = '(' + self.series_names[idx] + '+' + rhs.series_names[idx] + ')'
                ary = self.array(idx) + rhs.array(idx)
                result.append_array(self.series_size, ser_name, ary)
        else:
            for idx in range(0, len(self.array_with_series_idx)):
                ser_name = '(' + self.series_names[idx] + '+' + str(rhs) + ')'
                ary = self.array(idx) + rhs
                result.append_array(self.series_size, ser_name, ary)
        return result

    def __mul__(self, rhs):
        """Implements lhs * rhs"""
        result = DataSet()
        if type(rhs) == DataSet:
            for idx in range(0, len(self.array_with_series_idx)):
                ser_name = '(' + self.series_names[idx] + '*' + rhs.series_names[idx] + ')'
                ary = self.array(idx) * rhs.array(idx)
                result.append_array(self.series_size, ser_name, ary)
        else:
            for idx in range(0, len(self.array_with_series_idx)):
                ser_name = '(' + self.series_names[idx] + '*' + str(rhs) + ')'
                ary = self.array(idx) * rhs
                result.append_array(self.series_size, ser_name, ary)
        return result

    def __div__(self, rhs):
        """Implements lhs / rhs"""
        result = DataSet()
        if type(rhs) == DataSet:
            for idx in range(0, len(self.array_with_series_idx)):
                ser_name = '(' + self.series_names[idx] + '/' + rhs.series_names[idx] + ')'
                with np.errstate(all='ignore'):
                    ary = self.array(idx) / rhs.array(idx)
                ary[np.isnan(ary)] = 0
                result.append_array(self.series_size, ser_name, ary)
        else:
            for idx in range(0, len(self.array_with_series_idx)):
                ser_name = '(' + self.series_names[idx] + '/' + str(rhs) + ')'
                ary = self.array(idx) / rhs
                result.append_array(self.series_size, ser_name, ary)
        return result

    def __lshift__(self, val):
        ds = DataSet()
        for i in range(0, len(self.series_names)):
            ds.append_array(self.series_size, self.series_names[i], self.array_with_series_idx[i])
        if type(val) == DataSet:
            for i in range(0, len(val.series_names)):
                ds.append_array(val.series_size, val.series_names[i], val.array_with_series_idx[i])
        elif type(val) == tuple:
            # ( count, series-name, np.dtype, initial-value )
            nda = np.ndarray(val[0], dtype=val[2])
            nda[:] = val[3]
            ds.append_array(val[0], val[1], nda)
        return ds

    def __rshift__(self, rename):
        if type(rename) == str:
            self.rename(0, rename)
        else:
            self.rename(rename[0], rename[1])
        return self

    def min(self, series_list=None):
        """Compute min for series across rows
        """
        if series_list is None:
            series_list = self.series_names
        res = DataSet()
        for ser in series_list:
            m = np.min(self.array(ser))
            nda = np.ndarray([ 1 ], dtype=m.dtype)
            nda[0] = m
            res.append_array(1, ser, nda)
        return res

    def max(self, series_list=None):
        """Compute min for series across rows
        """
        if series_list is None:
            series_list = self.series_names
        res = DataSet()
        for ser in series_list:
            m = np.max(self.array(ser))
            nda = np.ndarray([ 1 ], dtype=m.dtype)
            nda[0] = m
            res.append_array(1, ser, nda)
        return res

