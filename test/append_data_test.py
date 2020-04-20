#!/usr/bin/env python
from __future__ import print_function
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
key = 1000

class AppendDataTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb('append_data_test_cont')
        cls.schema = Sos.Schema()
        cls.schema.from_template('append_data_test',
                                 [
                                 { "name" : "int32", "type" : "int32", "index" : {} },
                                 { "name" : "string", "type" : "char_array" }
                               ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def __generate_data(self, data, key):
        for i in range(0, 16):
            # three duplicates of each key
            data.append(
                ( key, "{:0>4}".format(key) )
            )
            data.append(
                ( key, "{:0>4}".format(key) )
            )
            data.append(
                ( key, "{:0>4}".format(key) )
            )
            key += 1
        return key

    def __add_data(self, key, data):
        key = self.__generate_data(data, key)
        for t in data:
            obj = self.schema.alloc()
            obj[:] = t
            rc = obj.index_add()
            self.assertEqual( rc, 0 )
        return ( key, data )

    def __test_next(self, attr_name):
        global key
        global data
        key, data = self.__add_data(key, data)
        attr = self.schema[attr_name]
        f = Sos.Filter(attr)
        # Iterate to the end
        count = 0
        o = f.begin()
        while o:
            d = data[count]
            v = o[:]
            self.assertEqual(d[0], v[0])
            self.assertEqual(d[1], v[1])
            count += 1
            o = next(f)
        self.assertEqual(count, len(data))

        # Add more data
        new_data = []
        key, new_data = self.__add_data( key, new_data )

        # next(f) should return the new data
        o = next(f)
        self.assertIsNotNone( o )
        count = 0
        while o:
            d = new_data[count]
            v = o[:]
            self.assertEqual(d[0], v[0])
            self.assertEqual(d[1], v[1])
            count += 1
            o = next(f)
        # we should not see any object twice
        self.assertEqual( count, len(new_data) )
        # Make the global data match the container
        data = data + new_data

    def __test_prev(self, attr_name):
        global key
        global data
        attr = self.schema[attr_name]
        f = Sos.Filter(attr)
        # Iterate to the start
        count = len(data)
        o = f.end()
        while o:
            d = data[count-1]
            v = o[:]
            self.assertEqual(d[0], v[0])
            self.assertEqual(d[1], v[1])
            count -= 1
            o = f.prev()
        self.assertEqual( count, 0 )

        # Add more data
        key = 500   # Put the key before the 1st key of the last test
        new_data = []
        key, new_data = self.__add_data( key, new_data )
        count = len(new_data)

        # f.prev should return the new data
        o = f.prev()
        self.assertIsNotNone( o )
        while o:
            d = new_data[count-1]
            v = o[:]
            self.assertEqual(d[0], v[0])
            self.assertEqual(d[1], v[1])
            count -= 1
            o = f.prev()
        # we should not see any object twice
        self.assertEqual( count, 0 )

    def test_01_next(self):
        self.__test_next("int32")

    def test_02_prev(self):
        self.__test_prev("int32")

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
