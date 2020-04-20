#!/usr/bin/env python
from __future__ import print_function
from past.builtins import execfile
from builtins import next
from builtins import range
from builtins import object
import unittest
import logging
import os
import random
import numpy
from sosdb import Sos
from sosunittest import SosTestCase, Dprint
import datetime as dt

class Debug(object): pass

logger = logging.getLogger(__name__)
data = []

class FilterTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb('filter_test_cont')
        cls.schema = Sos.Schema()
        cls.schema.from_template('filter_test',
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

                                 { "name" : "join_i16_i32_i64", "type" : "join",
                                   "join_attrs" : [ "int16", "int32", "int64" ], "index" : {} },

                                 { "name" : "join_u16_u32_u64", "type" : "join",
                                   "join_attrs" : [ "uint16", "uint32", "uint64" ], "index" : {} },

                                 { "name" : "join_i16_u32_i64", "type" : "join",
                                   "join_attrs" : [ "int16", "uint32", "int64" ], "index" : {} },

                                 { "name" : "join_i16_i32_u64", "type" : "join",
                                   "join_attrs" : [ "int16", "int32", "uint64" ], "index" : {} },

                                 { "name" : "join_i64_double", "type" : "join",
                                   "join_attrs" : [ "int64", "double" ], "index" : {} },

                                 { "name" : "join_u64_timestamp", "type" : "join",
                                   "join_attrs" : [ "uint64", "timestamp" ], "index" : {} },
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

    def __test_next_prev(self, attr_name, min_v, max_v):
        attr = self.schema[attr_name]
        f = Sos.Filter(attr)
        f.add_condition(attr, Sos.COND_GE, min_v)
        f.add_condition(attr, Sos.COND_LE, max_v)
        o = f.begin()
        next_count = 0
        while o:
            next_count += 1
            v = o[attr_name]
            if type(v) == numpy.ndarray:
                v = v.tolist()
            Dprint("{0} >= {1}".format(v, min_v))
            Dprint("{0} <= {1}".format(v, max_v))
            self.assertTrue( v >= min_v )
            self.assertTrue( v <= max_v )
            o = next(f)
        self.assertTrue( next_count > 0 )

        # iterate backwards, the count should be the same
        o = f.end()
        prev_count = 0
        while o:
            prev_count += 1
            v = o[attr_name]
            if type(v) == numpy.ndarray:
                v = v.tolist()
            Dprint("{0} >= {1}".format(v, min_v))
            Dprint("{0} <= {1}".format(v, max_v))
            self.assertTrue( v >= min_v )
            self.assertTrue( v <= max_v )
            o = f.prev()
        self.assertTrue( prev_count > 0 )

        self.assertEqual(next_count, prev_count)

    def __join_test_next_prev(self, join_attr_name, attr_name, min_v, max_v):
        join_attr = self.schema[join_attr_name]
        attr = self.schema[attr_name]
        f = Sos.Filter(join_attr)
        f.add_condition(attr, Sos.COND_GE, min_v)
        f.add_condition(attr, Sos.COND_LE, max_v)
        o = f.begin()
        next_count = 0
        while o:
            Dprint(o[:])
            next_count += 1
            self.assertTrue( o[attr_name] >= min_v )
            self.assertTrue( o[attr_name] <= max_v )
            o = next(f)

        # iterate backwards, the count should be the same
        o = f.end()
        prev_count = 0
        while o:
            Dprint(o[:])
            prev_count += 1
            self.assertTrue( o[attr_name] >= min_v )
            self.assertTrue( o[attr_name] <= max_v )
            o = f.prev()

        self.assertEqual(next_count, prev_count)

    def test_01_next_prev_int16(self):
        self.__test_next_prev("int16", -600, -500)

    def test_02_u_next_prev_uint16(self):
        self.__test_next_prev("uint16", 500, 600)

    def test_03_next_prev_int32(self):
        self.__test_next_prev("int32", -600, -500)

    def test_04_next_prev_uint32(self):
        self.__test_next_prev("uint32", 500, 600)

    def test_05_next_prev_int64(self):
        self.__test_next_prev("int64", -600, -500)

    def test_06_next_prev_uint64(self):
        self.__test_next_prev("uint64", 500, 600)

    def test_07_next_prev_float(self):
        self.__test_next_prev("float", 0, 1)

    def test_08_next_prev_double(self):
        self.__test_next_prev("double", 0, 1)

    def test_09_next_prev_int16_array(self):
        self.__test_next_prev("int16_array", [-600, -600, -600], [-500, -500, -500])

    def test_10_next_prev_int32_array(self):
        self.__test_next_prev("int32_array", [-600, -600, -600], [-500, -500, -500])

    def test_11_next_prev_int64_array(self):
        self.__test_next_prev("int64_array", [-600, -600, -600], [-500, -500, -500])

    def test_12_next_prev_uint16_array(self):
        self.__test_next_prev("uint16_array", [500, 500, 500], [600, 600, 600])

    def test_13_next_prev_uint32_array(self):
        self.__test_next_prev("uint32_array", [500, 500, 500], [600, 600, 600])

    def test_14_next_prev_uint64_array(self):
        self.__test_next_prev("uint64_array", [500, 500, 500], [600, 600, 600])

    def test_15_next_prev_float_array(self):
        self.__test_next_prev("float_array", [0, 0, 0], [1, 1, 1])

    def test_16_next_prev_double_array(self):
        self.__test_next_prev("double_array", [0, 0, 0], [1, 1, 1])


    def test_200_next_prev_join_i16_i32_i64(self):
        self.__join_test_next_prev("join_i16_i32_i64", "int16", -600, -500)

    def test_201_next_prev_join_i16_i32_i64(self):
        self.__join_test_next_prev("join_i16_i32_i64", "int32", -600, -500)

    def test_202_next_prev_join_i16_i32_i64(self):
        self.__join_test_next_prev("join_i16_i32_i64", "int64", -600, -500)


    def test_300_next_prev_join_u16_u32_u64(self):
        self.__join_test_next_prev("join_u16_u32_u64", "uint16", 500, 600)

    def test_301_next_prev_join_u16_u32_u64(self):
        self.__join_test_next_prev("join_u16_u32_u64", "uint32", 500, 600)

    def test_302_next_prev_join_u16_u32_u64(self):
        self.__join_test_next_prev("join_u16_u32_u64", "uint64", 500, 600)


    def test_400_next_prev_join_i16_u32_i64(self):
        self.__join_test_next_prev("join_i16_u32_i64", "int16", -600, -500)

    def test_401_next_prev_join_i16_u32_i64(self):
        self.__join_test_next_prev("join_i16_u32_i64", "uint32", 500, 600)

    def test_402_next_prev_join_i16_u32_i64(self):
        self.__join_test_next_prev("join_i16_u32_i64", "int64", -600, -500)


    def test_500_next_prev_join_i16_i32_u64(self):
        self.__join_test_next_prev("join_i16_i32_u64", "int16", -600, -500)

    def test_501_next_prev_join_i16_i32_u64(self):
        self.__join_test_next_prev("join_i16_i32_u64", "int32", -600, -500)

    def test_502_next_prev_join_i16_i32_u64(self):
        self.__join_test_next_prev("join_i16_i32_u64", "uint64", 500, 600)


    def test_600_next_prev_join_i64_double(self):
        self.__join_test_next_prev("join_i64_double", "int64", -600, -500)

    def test_601_next_prev_join_i64_double(self):
        self.__join_test_next_prev("join_i64_double", "double", 0, 1)


    def test_700_next_prev_join_u64_timestamp(self):
        self.__join_test_next_prev("join_u64_timestamp", "timestamp", data[100][8], data[200][8])

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
