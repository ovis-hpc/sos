#!/usr/bin/env python3
from builtins import range
from builtins import object
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
data1 = []
data2 = []

class DeleteTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        global data1
        global data2
        data1 = []
        data2 = []
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
        pass

    def __time_to_tuple(self, t):
        return ( int(t), int(int((t - int(t)) * 1.0e6 )) )

    def test_00_add_obj(self):
        global data1
        for cnt in range(0, 25):
            t = self.__time_to_tuple(time.time())
            i = random.random()
            for dups in range(1, 4):
                obj = self.schema.alloc()
                obj[:] = ( t, i )
                data1.append( ( t, i ) )
                obj.index_add()

    def test_01_del(self):
        global data1
        k_attr = self.schema['key']
        idx = k_attr.index()
        # delete the objects
        for i in range(0, len(data1)):
            d = data1[i]
            k = k_attr.key(d[0], d[1])
            o = idx.find(k)
            self.assertTrue(o is not None)
            rc = o.index_del()
            self.assertEqual( rc, 0 )
            o.delete()

    def test_02_add_obj(self):
        global data2
        for cnt in range(0, 50):
            t = self.__time_to_tuple(time.time())
            i = random.random()
            for dups in range(1, 4):
                obj = self.schema.alloc()
                obj[:] = ( t, i )
                data2.append( ( t, i ) )
                obj.index_add()

    def test_03_del(self):
        global data2
        k_attr = self.schema['key']
        idx = k_attr.index()
        for i in range(0, len(data2) // 2):
            d = data2[i]
            k = k_attr.key(d[0], d[1])
            o = idx.find(k)
            self.assertTrue(o is not None)
            o.index_del()
            o.delete()
            o = idx.find(k)
            # self.assertTrue(o is None)

class LsosDeleteTest(DeleteTest):
    @classmethod
    def backend(cls):
        return Sos.BE_LSOS

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
