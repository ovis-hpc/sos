#!/usr/bin/env python
from past.builtins import execfile
from builtins import next
from builtins import str
from builtins import range
from builtins import object
import unittest
import shutil
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase, Dprint
import random
import numpy as np
import numpy.random as nprnd
import datetime as dt
class Debug(object): pass

logger = logging.getLogger(__name__)
data = []

class KeyTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("key_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('key_test',
                                 [
                                 { "name" : "int16", "type" : "int16", "index" : {} },
                                 { "name" : "int32", "type" : "int32", "index" : {} },
                                 { "name" : "int64", "type" : "int64", "index" : {} },
                                 { "name" : "uint16", "type" : "uint16", "index" : {} },
                                 { "name" : "uint32", "type" : "uint32", "index" : {} },
                                 { "name" : "uint64", "type" : "uint64", "index" : {} },
                                 { "name" : "float", "type" : "float", "index" : {} },
                                 { "name" : "double", "type" : "double", "index" : {} },
                                 { "name" : "timestamp", "type" : "timestamp", "index" : {} },
                                 { "name" : "struct", "type" : "struct", "size" : 16, "index" : {} },
                                 { "name" : "string", "type" : "char_array", "index" : {} },
                                 { "name" : "byte_array", "type" : "byte_array", "index" : {} },
                                 { "name" : "int16_array", "type" : "int16_array", "index" : {} },
                                 { "name" : "int32_array", "type" : "int32_array", "index" : {} },
                                 { "name" : "int64_array", "type" : "int64_array", "index" : {} },
                                 { "name" : "uint16_array", "type" : "uint16_array", "index" : {} },
                                 { "name" : "uint32_array", "type" : "uint32_array", "index" : {} },
                                 { "name" : "uint64_array", "type" : "uint64_array", "index" : {} },
                                 { "name" : "float_array", "type" : "float_array", "index" : {} },
                                 { "name" : "double_array", "type" : "double_array", "index" : {} },
                                 ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def __generate_data(self):
        for i in range(0, 1024):
            t = dt.datetime.now()
            data.append((
                -i, -i, -i,
                i, i, i,
                random.random(), random.random(),
                (t.second, t.microsecond),
                bytearray(str(-i), encoding='utf-8'),
                "{0}".format(i), bytearray("{0}".format(i), encoding='utf-8'),
                [ -i, -i, -i ], [ -i, -i, -i ], [ -i, -i, -i ],
                [ i, i, i ], [ i, i, i ], [ i, i, i ],
                [ random.random(), random.random(), random.random() ],
                [ random.random(), random.random(), random.random() ]
            ))

    def test_00_add_data(self):
        global data
        self.__generate_data()
        for t in data:
            obj = self.schema.alloc()
            obj[:] = t
            rc = obj.index_add()
            self.assertEqual( rc, 0 )

    def _test_int_key_type(self, attr_name, value):
        attr = self.schema[attr_name]
        idx = attr.index()

        # generate key from value
        key = attr.key(value)
        o = idx.find(key)
        if o is not None:
            Dprint(o[:])
        self.assertIsNotNone( o )
        self.assertEqual( o[attr_name], value )

    def _test_int_key_str(self, attr_name, value):
        attr = self.schema[attr_name]
        idx = attr.index()

        # generate key from string
        key = attr.key(str(value))
        o = idx.find(key)
        self.assertIsNotNone( o )
        self.assertEqual( o[attr_name], value )

        # cause exception by giving key a garbage integer value
        try:
            key = attr.key('five-hundred')
        except Exception as e:
            self.assertEqual( type(e), ValueError )
            return
        self.assertTrue( False )

    def _test_array_key_type(self, attr_name, value):
        attr = self.schema[attr_name]
        idx = attr.index()

        # generate key from value
        key = attr.key(value)
        o = idx.find(key)
        self.assertTrue( o is not None )
        Dprint(o[:])
        a = o[attr_name]
        b = value
        self.assertEqual( len(a), len(b) )
        for i in range(len(a)):
            self.assertEqual( a[i], b[i] )

    def _test_array_key_str(self, attr_name, value):
        attr = self.schema[attr_name]
        idx = attr.index()

        # generate key from value
        key = attr.key(value)
        o = idx.find(key)
        self.assertTrue( o is not None )
        a = o[attr_name]
        b = value.split(',')
        self.assertEqual( len(a), len(b) )
        for i in range(len(a)):
            self.assertEqual( a[i], int(b[i]) )

    def test_01_int16(self):
        self._test_int_key_type('int16', -500)
        self._test_int_key_str('int16', -500)

    def test_02_int32(self):
        self._test_int_key_type('int32', -500)
        self._test_int_key_str('int32', -500)

    def test_03_int64(self):
        self._test_int_key_type('int64', -500)
        self._test_int_key_str('int64', -500)

    def test_04_uint16(self):
        self._test_int_key_type('uint16', 500)
        self._test_int_key_str('uint16', 500)

    def test_05_uint32(self):
        self._test_int_key_type('uint32', 500)
        self._test_int_key_str('uint32', 500)

    def test_06_uint64(self):
        self._test_int_key_type('uint64', 500)
        self._test_int_key_str('uint64', 500)

    def test_07_float(self):
        return                  # can't handle loss of precision
        d = data[500]
        self._test_int_key_type('float', d[6])

    def test_08_double(self):
        d = data[500]
        self._test_int_key_type('double', d[7])

    def test_09_timestamp(self):
        d = data[501]
        self._test_int_key_type('timestamp', d[8])

    def test_10_struct(self):
        attr = self.schema['struct']
        idx = attr.index()

        # generate key from value
        value = '-503'
        key = attr.key(value)
        o = idx.find(key)
        if o is not None:
            Dprint(o[:])
        self.assertIsNotNone( o )
        self.assertEqual( o['struct'][:len(value)],
                          bytearray(value, encoding='utf-8') )

    def test_11_char_array(self):
        self._test_int_key_type('string', '502')

    def test_12_byte_array(self):
        self._test_array_key_type('byte_array', bytearray('503', encoding='utf-8'))
        self._test_array_key_type('byte_array', [53, 48, 51])

    def test_13_int16_array(self):
        self._test_array_key_type('int16_array', [-504, -504, -504])
        self._test_array_key_str('int16_array', '  -504  ,  -504, -504')

    def test_14_uint16_array(self):
        self._test_array_key_type('uint16_array', [504, 504, 504])
        self._test_array_key_str('uint16_array', '  504  ,  504, 504')

    def test_15_int32_array(self):
        self._test_array_key_type('int32_array', [-504, -504, -504])
        self._test_array_key_str('int32_array', '  -504  ,  -504, -504')

    def test_16_uint32_array(self):
        self._test_array_key_type('uint32_array', [504, 504, 504])
        self._test_array_key_str('uint32_array', '  504  ,  504, 504')


if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
