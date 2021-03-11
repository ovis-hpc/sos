#!/usr/bin/env python3
from past.builtins import execfile
from builtins import object
import unittest
import shutil
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase
import random
class Debug(object): pass

logger = logging.getLogger(__name__)

data = [
    ( 0, -100, 0, -1000, 0.0, "a this is a test" ),
    ( 100, -3, 100, -3, 1.e6, "b this is a test" ),
    ( 200, -2, 200, -2, 2.e6, "c this is a test" ),
    ( 300, -1, 300, -1, 3.e6, "d this is a test" ),
]

class IndexTestMinMax(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("min_max_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_min_max',
                             [
                                 { "name" : "uint32", "type" : "uint32", "index" : {} },
                                 { "name" : "int32", "type" : "int32", "index" : {} },
                                 { "name" : "uint64", "type" : "uint64", "index" : {} },
                                 { "name" : "int64", "type" : "int64", "index" : {} },
                                 { "name" : "double", "type" : "double", "index" : {} },
                                 { "name" : "string", "type" : "string", "index" : {} },
                             ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_00_add_obj(self):
        # Twot objects go in the 'default' partition
        obj = self.schema.alloc()
        obj[:] = data[0]
        obj.index_add()

        obj = self.schema.alloc()
        obj[:] = data[1]
        obj.index_add()

        # Create a new partition and make it primary
        self.db.part_create("ROOT")
        root = self.db.part_by_name("ROOT")
        root.state_set("PRIMARY")

        # These go in the new partition
        obj = self.schema.alloc()
        obj[:] = data[2]
        obj.index_add()

        obj = self.schema.alloc()
        obj[:] = data[3]
        obj.index_add()

    def test_01_min_uint32(self):
        a = self.schema.attr_by_name("uint32")
        v = a.min()
        self.assertEqual(v, data[0][0])

    def test_02_min_int32(self):
        a = self.schema.attr_by_name("int32")
        v = a.min()
        self.assertEqual(v, data[0][1])

    def test_03_min_uint64(self):
        a = self.schema.attr_by_name("uint64")
        v = a.min()
        self.assertEqual(v, data[0][2])

    def test_04_min_int64(self):
        a = self.schema.attr_by_name("int64")
        v = a.min()
        self.assertEqual(v, data[0][3])

    def test_05_min_double(self):
        a = self.schema.attr_by_name("double")
        v = a.min()
        self.assertEqual(v, data[0][4])

    def test_06_min_string(self):
        a = self.schema.attr_by_name("string")
        v = a.min()
        self.assertEqual(v, data[0][5])

    def test_07_max_uint32(self):
        a = self.schema.attr_by_name("uint32")
        v = a.max()
        self.assertEqual(v, data[3][0])

    def test_08_max_int32(self):
        a = self.schema.attr_by_name("int32")
        v = a.max()
        self.assertEqual(v, data[3][1])

    def test_09_max_uint64(self):
        a = self.schema.attr_by_name("uint64")
        v = a.max()
        self.assertEqual(v, data[3][2])

    def test_10_max_int64(self):
        a = self.schema.attr_by_name("int64")
        v = a.max()
        self.assertEqual(v, data[3][3])

    def test_11_max_double(self):
        a = self.schema.attr_by_name("double")
        v = a.max()
        self.assertEqual(v, data[3][4])

    def test_12_min_string(self):
        a = self.schema.attr_by_name("string")
        v = a.max()
        self.assertEqual(v, data[3][5])

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
