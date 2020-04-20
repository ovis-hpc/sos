#!/usr/bin/env python
from past.builtins import execfile
from builtins import str
from builtins import object
import unittest
import shutil
import logging
import os
import random
from sosdb import Sos
from sosunittest import SosTestCase

class Debug(object): pass

logger = logging.getLogger(__name__)

class ValueTestFromStr(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("val_from_str_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('value_from_str', [
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

    def test_int16_strlen(self):
        a = self.schema['int16']
        v = Sos.Value(a)
        v.value = -1234
        self.assertEqual(v.strlen(), 5)

    def test_int16_from_str(self):
        a = self.schema['int16']
        v = Sos.Value(a)
        # octal
        v.from_str('-01234')
        self.assertEqual(v.value, -668)
        # decimal
        v.from_str('-1234')
        self.assertEqual(v.value, -1234)
        # hex
        v.from_str('-0x1234')
        self.assertEqual(v.value, -4660)

    def test_uint16_strlen(self):
        a = self.schema['uint16']
        v = Sos.Value(a)
        v.value = 1234
        self.assertEqual(v.strlen(), 4)

    def test_uint16_from_str(self):
        a = self.schema['uint16']
        v = Sos.Value(a)
        # octal
        v.from_str('01234')
        self.assertEqual(v.value, 668)
        # decimal
        v.from_str('1234')
        self.assertEqual(v.value, 1234)
        # hex
        v.from_str('0x1234')
        self.assertEqual(v.value, 4660)

    def test_int32_strlen(self):
        a = self.schema['int32']
        v = Sos.Value(a)
        v.value = -12345678
        self.assertEqual(v.strlen(), 9)

    def test_int32_from_str(self):
        a = self.schema['int32']
        v = Sos.Value(a)
        # octal
        v.from_str('-012345678')
        self.assertEqual(v.value, -342391)
        # decimal
        v.from_str('-12345678')
        self.assertEqual(v.value, -12345678)
        # hex
        v.from_str('-0x12345678')
        self.assertEqual(v.value, -305419896)

    def test_uint32_strlen(self):
        a = self.schema['uint32']
        v = Sos.Value(a)
        v.value = 12345678
        self.assertEqual(v.strlen(), 8)

    def test_uint32_from_str(self):
        a = self.schema['uint32']
        v = Sos.Value(a)
        # octal
        v.from_str('012345678')
        self.assertEqual(v.value, 342391)
        # decimal
        v.from_str('12345678')
        self.assertEqual(v.value, 12345678)
        # hex
        v.from_str('0x12345678')
        self.assertEqual(v.value, 305419896)

    def test_int64_strlen(self):
        a = self.schema['int64']
        v = Sos.Value(a)
        v.value = -123456789
        self.assertEqual(v.strlen(), 10)

    def test_int64_from_str(self):
        a = self.schema['int64']
        v = Sos.Value(a)
        # octal
        v.from_str('-0123456789')
        self.assertEqual(v.value, -342391)
        # decimal
        v.from_str('-123456789')
        self.assertEqual(v.value, -123456789)
        # hex
        v.from_str('-0x123456789')
        self.assertEqual(v.value, -4886718345)

    def test_float_strlen(self):
        a = self.schema['float']
        v = Sos.Value(a)
        v.value = 1234.5
        self.assertGreaterEqual(v.strlen(), 6)

    def test_float_from_str(self):
        a = self.schema['float']
        v = Sos.Value(a)
        v.from_str('1234.50000')
        self.assertEqual(v.value, 1234.5)
        v.from_str('1.2345e3')
        self.assertEqual(v.value, 1234.5)
        v.from_str('1234.4e-3')
        self.assertEqual(v.value, 1.2344000339508057)

    def test_float_to_str(self):
        a = self.schema['float']
        v = Sos.Value(a)
        v.value = 1234.5
        s = v.to_str()
        self.assertEqual(s, '1234.50000')

    def test_double_strlen(self):
        a = self.schema['double']
        v = Sos.Value(a)
        v.value = 1234.5
        self.assertGreaterEqual(v.strlen(), 6)

    def test_double_from_str(self):
        a = self.schema['double']
        v = Sos.Value(a)
        v.from_str('1234.50000')
        self.assertEqual(v.value, 1234.5)
        v.from_str('1.2345e3')
        self.assertEqual(v.value, 1234.5)
        v.from_str('1234.4e-3')
        self.assertEqual(v.value, 1.2344)

    def test_double_to_str(self):
        a = self.schema['double']
        v = Sos.Value(a)
        v.value = 1234.5
        s = v.to_str()
        self.assertEqual(s, '1234.50000')

    def test_timestamp_strlen(self):
        a = self.schema['timestamp']
        v = Sos.Value(a)
        v.value = ( 1511885835, 12345 )
        self.assertGreaterEqual(v.strlen(), 6)

    def test_timestamp_from_str(self):
        a = self.schema['timestamp']
        v = Sos.Value(a)

        v.from_str('1511885835.012345')
        self.assertEqual(v.value, ( 1511885835, 12345 ))

        v.from_str('1511885835.12345')
        self.assertEqual(v.value, ( 1511885835, 12345 ))

        v.from_str('1511885835.123450')
        self.assertEqual(v.value, ( 1511885835, 123450 ))

    def test_timestamp_to_str(self):
        a = self.schema['timestamp']
        v = Sos.Value(a)
        v.value = ( 1511885835, 123450 )
        s = v.to_str()
        self.assertEqual(s, '1511885835.123450')

        v.value = ( 1511885835, 12345 )
        s = v.to_str()
        self.assertEqual(s, '1511885835.012345')

    # Array strlen
    def chk_array_strlen(self, a, av):
        v = Sos.Value(a)
        v.value = av
        s = str(v)
        self.assertGreaterEqual(v.strlen(), len(s))

    def test_char_array_strlen(self):
        self.chk_array_strlen(self.schema['char_array'], "this is a test")

    def test_byte_array_strlen(self):
        self.chk_array_strlen(self.schema['byte_array'], [ 0xA0, 0xB0, 0xC0, 0xFF ])

    def test_int16_array_strlen(self):
        self.chk_array_strlen(self.schema['int16_array'], [ -1, 2, -3, 4 ])

    def test_int32_array_strlen(self):
        self.chk_array_strlen(self.schema['int32_array'], [ -1, 2, -3, 4 ])

    def test_int64_array_strlen(self):
        self.chk_array_strlen(self.schema['int64_array'], [ -1, 2, -3, 4 ])

    def test_uint16_array_strlen(self):
        self.chk_array_strlen(self.schema['uint16_array'], [ 1, 2, 3, 4 ])

    def test_uint32_array_strlen(self):
        self.chk_array_strlen(self.schema['uint32_array'], [ 1, 2, 3, 4 ])

    def test_uint64_array_strlen(self):
        self.chk_array_strlen(self.schema['uint64_array'], [ 1, 2, 3, 4 ])

    # Array to_str
    def chk_array_to_str(self, a, av, ev):
        v = Sos.Value(a)
        v.value = av
        s = str(v)
        self.assertEqual(s, ev)

    def test_char_array_to_str(self):
        self.chk_array_to_str(self.schema['char_array'], 'this is a test', 'this is a test')

    def test_byte_array_to_str(self):
        self.chk_array_to_str(self.schema['byte_array'],
                              [ 0xA0, 0xB0, 0xC0, 0xFF ],
                              'A0:B0:C0:FF')

    def test_int16_array_to_str(self):
        self.chk_array_to_str(self.schema['int16_array'], [ -1, 2, -3, 4 ], '-1,2,-3,4')

    def test_int32_array_to_str(self):
        self.chk_array_to_str(self.schema['int32_array'], [ -1, 2, -3, 4 ], '-1,2,-3,4')

    def test_int64_array_to_str(self):
        self.chk_array_to_str(self.schema['int64_array'], [ -1, 2, -3, 4 ], '-1,2,-3,4')

    def test_float_array_to_str(self):
        self.chk_array_to_str(self.schema['float_array'],
                              [ -1.1234, 2.2345, -3.4567, 4.5678 ],
                              '-1.123400,2.234500,-3.456700,4.567800')

    def test_double_array_to_str(self):
        self.chk_array_to_str(self.schema['double_array'],
                              [ -1.1234, 2.2345, -3.4567, 4.5678 ],
                              '-1.123400,2.234500,-3.456700,4.567800')

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
