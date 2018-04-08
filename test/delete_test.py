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
data = []

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
        # cls.tearDownDb()
        pass

    def test_00_add_obj(self):
        for cnt in range(0, 50):
            t = time.time()
            i = random.random()
            for dups in range(1, 4):
                obj = self.schema.alloc()
                obj[:] = ( t, i )
                data.append( ( t, i ) )
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
            # self.assertTrue(o is None)

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
