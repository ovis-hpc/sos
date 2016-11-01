from __future__ import print_function
import cython
# if you forget this, your module cannot be imported by other cythong modules
cimport cython
from cpython cimport PyObject, Py_INCREF
from libc.stdint cimport *
from libc.stdlib cimport *
from libc.string cimport *
import sys
import numpy as np
from numpy cimport npy_intp
cimport numpy as np

#
# Initialize the numpy array support. Hurray can be converted,
# i.e. flattened into numpy arrays for use with the Numpy
# numerical analysis functions.
#
np.import_array()

cdef class Array:
    """Array is a growable array intended for the analsysis of SOS Object Data

    Numpy arrays are not growable, i.e. they are fixed size and/or require
    copying of the data if they are extended. This presents a
    performance challenge for data that is being queried from a SOS
    database for the purpose of numerical analysis using SciPy or
    other Numpy toolkits. The Array solves this problem by
    implementing a growable array that can be converted to a zero-copy
    numpy array after the data has been read from the database.

    Keyword Arguments:
    el_type -- The numpy type of the elements of the array. The default
               is np.double
    """
    cdef int pg_sz              # The grain size for aligning blocks
    cdef int blk_sz             # The size of each block in blks
    cdef void **blks            # Array of ptrs to blk_sz blocks
    cdef int blk_cnt            # Pages in the blks array
    cdef int el_cap             # Capacity of the array
    cdef int el_cnt             # Elements that have been appended to the array
    cdef el_type                # The NumPy type for the array
    cdef int el_sz              # The size of each element
    cdef int el_pb              # Number of elements per blk

    cdef public double ___getitem___(self, int pg, int idx):
        cdef double *p = <double*>self.blks[pg]
        return p[idx]

    cdef public ___setitem___(self, int pg, int idx, v):
        cdef double *p = <double*>self.blks[pg]
        p[idx] = <double>v

    def __init__(self, el_type=np.double):
        self.pg_sz = 4096
        self.blk_sz = self.pg_sz
        self.blk_cnt = 16
        self.blks = <void **>calloc(self.blk_cnt, 8)
        self.el_type = np.dtype(el_type)
        self.el_sz = self.el_type.itemsize
        self.el_pb = self.blk_sz / self.el_sz
        self.el_cap = 0
        self.el_cnt = 0

    def __str__(self):
        s = "Hurray@{0}[".format(self.el_cnt)
        for i in range(0, self.el_cnt):
            if i > 0:
                s += ","
            s += "{0}".format(self[i])
            if i > 3:
                break
        if i < self.el_cnt - 1:
            s += ",...,{0}".format(self[self.el_cnt-1])
        s += "]"
        return s

    def __len__(self):
        return self.el_cnt

    def block_size(self):
        """Return the size of each block"""
        return self.blk_sz

    def block_count(self):
        """Return the current block count"""
        return self.blk_cnt

    def capacity(self):
        """Return the number of elements the array could handle without expansion"""
        return self.el_cap

    def as_ndarray(self):
        """Return a Numpy ndarray

        Array are implemented as arrays of blocks of memory containing
        the array elements. Although this makes them efficient for
        filling with data from SOS collections that are of an unknown
        size, a Numpy array must be contiguous in memory. To convert
        to a Numpy array, the Hurray is first flattened into a single
        block of sufficient size to contain all elements in the array
        and then a Numpy nd_array is constructed that points to this
        block. The data is not copied. An Array so converted can still
        be appended to, however, because numpy array are not
        extendable, the subsequent size exapansions will not be seen
        be the ndarray returned here.
        """
        cdef np.ndarray ndarray
        cdef char *new_blk
        cdef size_t blk_size
        cdef size_t blk_used    # number of bytes used in each blk
        cdef uint64_t off

        if self.blk_cnt > 1:
            # flatten the array so that all the elements fit in a single block
            blk_size = self.el_sz * self.el_cnt
            blk_size = (blk_size + self.pg_sz - 1) & ~(self.pg_sz - 1) # align to page
            new_blk = <char *>malloc(blk_size)
            if new_blk == NULL:
                raise MemoryError("Insufficient memory to flatten array")
            blk_used = self.el_pb * self.el_sz

            # copy all the current block data to the new large block, and
            # free the old blocks
            off = 0
            for blk in range(0, self.blk_cnt):
                if self.blks[blk] == NULL:
                    break
                memcpy(&new_blk[off], self.blks[blk], blk_used)
                off += blk_used
                free(self.blks[blk])
                self.blks[blk] = NULL
            self.blks[0] = new_blk
            self.blk_cnt = 1
            self.blk_sz = blk_size
            self.el_pb = blk_size / self.el_sz
            self.el_cap = self.el_pb

        ndarray = np.array(self, copy=False)
        Py_INCREF(self)
        ndarray.base = <PyObject *>self
        return ndarray

    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.el_cnt
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               self.el_type.num,
                                               self.blks[0])
        return ndarray

    @cython.cdivision(True)
    def append(self, v):
        """Append an element to the Array"""
        cdef int pg
        cdef int idx
        cdef int i = self.el_cnt

        pg = i / self.el_pb
        idx = i % self.el_pb

        if pg >= self.blk_cnt:
            # Allocate a batch of new page slots to hold the additonal pages
            self.blks = <void **>realloc(self.blks,
                                        (self.blk_cnt + 16) * 8)
            if self.blks == NULL:
                raise MemoryError
            i = self.blk_cnt
            self.blk_cnt += 16
            for i in range(pg, self.blk_cnt):
                self.blks[i] = NULL

        # If the target page is empty, allocate a new page
        if self.blks[pg] == NULL:
            self.blks[pg] = malloc(4096)
            if self.blks[pg] == NULL:
                raise MemoryError
            self.el_cap += self.el_pb

        self.el_cnt += 1
        self.___setitem___(pg, idx, v)

    def __getitem__(self, i):
        cdef int pg
        cdef int idx
        cdef int i_ = i

        if i_ >= self.el_cnt:
            raise IndexError

        pg = i_ / self.el_pb
        idx = i_ % self.el_pb
        return self.___getitem___(pg, idx)

    def __setitem__(self, i, v):
        cdef int pg
        cdef int idx
        cdef int c_i = <int>i

        if c_i >= self.el_cnt:
            raise IndexError

        pg = c_i / self.el_pb
        idx = c_i % self.el_pb

        self.___setitem___(pg, idx, v)

    def __dealloc__(self):
        for pg in range(0, self.blk_cnt):
            if self.blks[pg]:
                free(self.blks[pg])
            else:
                break
        free(self.blks)
