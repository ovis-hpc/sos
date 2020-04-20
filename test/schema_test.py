#!/usr/bin/env python
from past.builtins import execfile
from builtins import str
from builtins import range
import unittest
import shutil
import logging
import os
import random
from sosdb import Sos
from sosunittest import SosTestCase

logger = logging.getLogger(__name__)

attr_list = [
    { "name" : "0", "type" : "uint64", "index" : {} },
    { "name" : "1", "type" : "int16" },
    { "name" : "2", "type" : "int32" },
    { "name" : "3", "type" : "int64" },
    { "name" : "4", "type" : "uint16" },
    { "name" : "5", "type" : "uint32" },
    { "name" : "6", "type" : "uint64" },
    { "name" : "7", "type" : "float" },
    { "name" : "8", "type" : "double" },
    { "name" : "9", "type" : "long_double" },
    { "name" : "10", "type" : "timestamp" },
    { "name" : "11", "type" : "struct", "size" : 24 },
    { "name" : "12", "type" : "byte_array" },
    { "name" : "13", "type" : "char_array" },
    { "name" : "14", "type" : "int16_array" },
    { "name" : "15", "type" : "int32_array" },
    { "name" : "16", "type" : "int64_array" },
    { "name" : "17", "type" : "uint16_array" },
    { "name" : "18", "type" : "uint32_array" },
    { "name" : "19", "type" : "uint64_array" },
    { "name" : "20", "type" : "float_array" },
    { "name" : "21", "type" : "double_array" },
    { "name" : "22", "type" : "long_double_array" }
]

class SchemaTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("schema_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('schema_attr_test', attr_list)
        cls.schema.add(cls.db)
        cls.types = [
            Sos.TYPE_UINT64,
            Sos.TYPE_INT16,
            Sos.TYPE_INT32,
            Sos.TYPE_INT64,
            Sos.TYPE_UINT16,
            Sos.TYPE_UINT32,
            Sos.TYPE_UINT64,
            Sos.TYPE_FLOAT,
            Sos.TYPE_DOUBLE,
            Sos.TYPE_LONG_DOUBLE,
            Sos.TYPE_TIMESTAMP,
            Sos.TYPE_STRUCT,
            Sos.TYPE_BYTE_ARRAY,
            Sos.TYPE_CHAR_ARRAY,
            Sos.TYPE_INT16_ARRAY,
            Sos.TYPE_INT32_ARRAY,
            Sos.TYPE_INT64_ARRAY,
            Sos.TYPE_UINT16_ARRAY,
            Sos.TYPE_UINT32_ARRAY,
            Sos.TYPE_UINT64_ARRAY,
            Sos.TYPE_FLOAT_ARRAY,
            Sos.TYPE_DOUBLE_ARRAY,
            Sos.TYPE_LONG_DOUBLE_ARRAY
        ]


    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_00_test_types(self):
        for i in range(0, self.schema.attr_count()):
            a = self.schema[i]
            self.assertEqual(a.type(), self.types[i])

    def test_01_test___getitem__by_id(self):
        for i in range(0, self.schema.attr_count()):
            a = self.schema[i]
            self.assertEqual(str(i), a.name())

    def test_02_test___getitem__by_name(self):
        for i in range(0, self.schema.attr_count()):
            a = self.schema[i]
            b = self.schema[a.name()]
            self.assertEqual(a.attr_id(), b.attr_id())

    def test_03_test_is_array(self):
        for i in range(12, self.schema.attr_count()):
            a = self.schema[i]
            self.assertTrue(a.is_array())

    def test_04_test_is_indexed(self):
        a = self.schema[0]
        self.assertTrue(a.is_indexed())

        for i in range(1, self.schema.attr_count()):
            a = self.schema[i]
            self.assertFalse(a.is_indexed())

    def test_05_many_schema(self):
        for i in range(0, 10 * 1024):
            schema = Sos.Schema()
            schema.from_template('schema_{0}'.format(i), attr_list)
            schema.add(self.db)
        self.assertTrue(1 == 1)

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
