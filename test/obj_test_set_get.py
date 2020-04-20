#!/usr/bin/env python
from past.builtins import execfile
from builtins import next
from builtins import range
from builtins import object
import unittest
import shutil
import logging
import os
import random
import numpy as np
from sosdb import Sos
from sosunittest import SosTestCase

class Debug(object): pass

logger = logging.getLogger(__name__)

class ObjTestSetGet(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("obj_set_get_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('obj_set_get', [
            { "name" : "int16", "type" : "int16" },
            { "name" : "int32", "type" : "int32" },
            { "name" : "int64", "type" : "int64" },
            { "name" : "uint16", "type" : "uint16" },
            { "name" : "uint32", "type" : "uint32" },
            { "name" : "uint64", "type" : "uint64" },
            { "name" : "float", "type" : "float" },
            { "name" : "double", "type" : "double" },
            { "name" : "long_double", "type" : "long_double" },
            { "name" : "timestamp", "type" : "timestamp" },
            { "name" : "struct", "type" : "struct", "size" : 24 },
            { "name" : "byte_array", "type" : "byte_array" },
            { "name" : "char_array", "type" : "char_array" },
            { "name" : "int16_array", "type" : "int16_array" },
            { "name" : "int32_array", "type" : "int32_array" },
            { "name" : "int64_array", "type" : "int64_array" },
            { "name" : "uint16_array", "type" : "uint16_array" },
            { "name" : "uint32_array", "type" : "uint32_array" },
            { "name" : "uint64_array", "type" : "uint64_array" },
            { "name" : "float_array", "type" : "float_array" },
            { "name" : "double_array", "type" : "double_array" },
            { "name" : "long_double_array", "type" : "long_double_array" }
        ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def chk_set_attr(self, obj, idx, name, v0, v1):
        # check set by name
        obj[name] = v0
        self.assertEqual(obj[name], v0)
        self.assertEqual(obj[idx], v0)
        # check set by idx
        obj[idx] = v1
        self.assertEqual(obj[name], v1)
        self.assertEqual(obj[idx], v1)

    def test_set_int16(self):
        self.chk_set_attr(self.schema.alloc(), 0, 'int16', -1234, -4567)

    def test_set_int32(self):
        self.chk_set_attr(self.schema.alloc(), 1, 'int32', -1234567, -456789)

    def test_set_int64(self):
        self.chk_set_attr(self.schema.alloc(), 2, 'int64', -98765, -4321)

    def test_set_uint16(self):
        self.chk_set_attr(self.schema.alloc(), 3, 'uint16', 1234, 4567)

    def test_set_uint32(self):
        self.chk_set_attr(self.schema.alloc(), 4, 'uint32', 1234567, 456789)

    def test_set_uint64(self):
        self.chk_set_attr(self.schema.alloc(), 5, 'uint64', 98765, 4321)

    def test_set_float(self):
        # only 7 significant digits
        self.chk_set_attr(self.schema.alloc(), 6, 'float', 12345.0, 45678.0)

    def test_set_double(self):
        # 16 significant digits
        self.chk_set_attr(self.schema.alloc(), 7, 'double', 1234567.431, 456789.123)

    def test_set_long_double(self):
        # 32? significant digits
        self.chk_set_attr(self.schema.alloc(), 8, 'long_double', 1234567.431, 456789.123)

    def chk_set_array_attr(self, obj, idx, name, v0, v1):
        # check set by name
        obj[name] = v0
        self.assertEqual(len(obj[name]), len(v0))
        for i in range(len(obj[name])):
            self.assertEqual(obj[name][i], v0[i])
            self.assertEqual(obj[idx][i], v0[i])
        # check set by idx
        obj[idx] = v1
        self.assertEqual(len(obj[idx]), len(v1))
        for i in range(len(obj[idx])):
            self.assertEqual(obj[name][i], v1[i])
            self.assertEqual(obj[idx][i], v1[i])

    def test_set_byte_array(self):
        self.chk_set_array_attr(self.schema.alloc(), 11, 'byte_array',
                                [ 0xA0, 0xB0 ],
                                [ 0xA0, 0xB0, 0xC0, 0xD0 ])

    def test_set_char_array(self):
        test_str = 'this is not a test'
        # check set by name
        obj = self.schema.alloc()
        obj['char_array'] = test_str
        self.assertEqual(len(obj['char_array']), len(test_str))
        self.assertEqual(obj['char_array'], test_str)
        self.assertEqual(obj[12], test_str)
        test_str = 'this is _still_ not a test'
        obj[12] = test_str
        self.assertEqual(len(obj['char_array']), len(test_str))
        self.assertEqual(obj['char_array'], test_str)
        self.assertEqual(obj[12], test_str)

    def test_set_int16_array(self):
        self.chk_set_array_attr(self.schema.alloc(), 13, 'int16_array',
                                [-1234], [-4567, -7654])

    def test_set_int32_array(self):
        self.chk_set_array_attr(self.schema.alloc(), 14, 'int32_array',
                                [-1234567], [-456789, -987654])

    def test_set_int64_array(self):
        self.chk_set_array_attr(self.schema.alloc(), 15, 'int64_array',
                                [-98765], [-4321, -1234])

    def test_set_uint16_array(self):
        self.chk_set_array_attr(self.schema.alloc(), 16, 'uint16_array',
                                [1234], [4567, 7554])

    def test_set_uint32_array(self):
        self.chk_set_array_attr(self.schema.alloc(), 17, 'uint32_array',
                                [1234567], [456789, 987654])

    def test_set_uint64_array(self):
        self.chk_set_array_attr(self.schema.alloc(), 18, 'uint64_array',
                                [98765], [4321, 1234])

    def test_set_float_array(self):
        # Single precision floating point #'s have to be represented
        # with a float or the compare won't match. Force this cast
        # (Python is all double) by using a numpy float32 array.
        aa = np.ndarray(shape=[1], dtype=np.float32)
        aa[0] = 12345.0
        ab = np.ndarray(shape=[2], dtype=np.float32)
        ab[0] = 1234532.034
        ab[1] = 431235.831
        self.chk_set_array_attr(self.schema.alloc(), 19, 'float_array', aa, ab)

    def test_set_double_array(self):
        # 16 significant digits
        self.chk_set_array_attr(self.schema.alloc(), 20, 'double_array',
                                [1234567.431], [456789.123, 321.987654])

    def test_set_long_double_array(self):
        # 32? significant digits
        self.chk_set_array_attr(self.schema.alloc(), 21, 'long_double_array',
                                [1234567.431],
                                [456789.123, 321.987654])

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
