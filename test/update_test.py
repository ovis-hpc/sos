#!/usr/bin/env python
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

class UpdateTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("update_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('update_test', [
            { "name" : "int16", "type" : "int16" },
            { "name" : "int32", "type" : "int32" },
            { "name" : "int64", "type" : "int64" },
            { "name" : "uint16", "type" : "uint16" },
            { "name" : "uint32", "type" : "uint32" },
            { "name" : "uint64", "type" : "uint64" },
            { "name" : "float", "type" : "float" },
            { "name" : "double", "type" : "double" },
            { "name" : "key", "type" : "uint64", "index" : {} }
        ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_01_add_data(self):
        for n in range(0, 1024):
            v = [ n for i in range(0, 9) ]
            o = self.schema.alloc()
            o[:] = v
            o.index_add()

    def test_02_update(self):
        key_attr = self.schema['key']
        for k in range(0, 1024):
            key = key_attr.key(k)
            o = key_attr.find(key)
            self.assertIsNotNone(o)
            v = o[:]
            v = [ i + 1 for i in v[:-1]]
            o[:] = v

    def test_03_new_value(self):
        # verify that the objects have the new value
        key_attr = self.schema['key']
        for k in range(0, 1024):
            key = key_attr.key(k)
            o = key_attr.find(key)
            v = o[:]
            for kv in v[:-1]:
                self.assertEqual( kv, k + 1 )

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
