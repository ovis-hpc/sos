#!/usr/bin/env python
import unittest
import shutil
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase
import random
import time
class Debug(object): pass

logger = logging.getLogger(__name__)

dup_time = time.time()
data = [
    ( dup_time, 99, ),
    ( dup_time, 99, ),
    ( dup_time, 99, ),
    ( time.time(), 0, ),
    ( time.time(), 1, ),
    ( time.time(), 2, ),
    ( time.time(), 3, ),
    ( time.time(), 4 ),
]

class DeleteTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("delete_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_delete',
                             [ { "name" : "ts", "type" : "timestamp" },
                               { "name" : "id", "type" : "uint64" },
                               { "name" : "key", "type" : "join",
                                 "join_attrs" : [ "ts", "id" ],
                                 "index" : {}}
                           ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_00_add_obj(self):
        for seq in data:
            obj = self.schema.alloc()
            obj[:] = seq
            obj.index_add()

    def test_01_del(self):
        k_attr = self.schema['key']
        idx = k_attr.index()
        for d in data:
            k = k_attr.key(d[0], d[1])
            o = idx.find(k)
            self.assertTrue(o is not None)
            o.index_del()
            o.delete()
            o = idx.find(k)
            if d[0] != dup_time:
                self.assertTrue(o is None)

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
