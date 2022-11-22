#!/usr/bin/env python3
from builtins import next
from builtins import range
import unittest
import shutil
import logging
import os
import random
from sosdb import Sos
from sosunittest import SosTestCase
from pandas import Timestamp

logger = logging.getLogger(__name__)

data = [
    [ b'this is a byte_array', "this is a char_array",
      1, 1, 1, 1, 1, 1, 1.1230000257492065, 3.1415, Timestamp(1668727891493526, unit='us')
    ],
    [ b'this is a test', "this is a test",  
      2, 2, 2, 2, 2, 2, 2.123000144958496, 4.1415, Timestamp(1668727892493526, unit='us')
    ],
    [ b'this is not a test', "this is not a test",
      3, 3, 3, 3, 3, 3, 3.123000144958496, 5.1415, Timestamp(1668727893493526, unit='us')
    ],
]

class DsosAstTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.session()
        cls.cont = cls.session.open("dsos_ast_test_cont", o_perm=Sos.PERM_RW|Sos.PERM_CREAT)
        cls.schema = Sos.Schema()
        cls.schema.from_template('dsos_ast_test', [
            { "name" : "byte_array", "type" : "byte_array" },
            { "name" : "char_array", "type" : "char_array", "index": {} },
            { "name" : "int16", "type" : "int16", "index" : {} },
            { "name" : "int32", "type" : "int32", "index" : {} },
            { "name" : "int64", "type" : "int64", "index" : {} },
            { "name" : "uint16", "type" : "uint16", "index" : {} },
            { "name" : "uint32", "type" : "uint32", "index" : {} },
            { "name" : "uint64", "type" : "uint64", "index" : {} },
            { "name" : "float", "type" : "float", "index" : {} },
            { "name" : "double", "type" : "double", "index" : {} },
            { "name" : "timestamp", "type" : "timestamp", "index" : {} },
        ])
        cls.dschema = cls.cont.schema_add(cls.schema)

    @classmethod
    def tearDownClass(cls):
        # cls.tearDownDb()
        pass

    def test_00_add_data(self):
        self.cont.transaction_begin()
        for d in data:
            o = self.dschema.malloc()
            for e in d:
                o[:] = d[:]
            self.dschema.obj_create(o)
        self.cont.transaction_end()

    def __test_find_data(self, key_no, key_name):
        q = self.cont.query()
        for d in data:
            k = d[key_no]
            if type(k) == str:
                q.select(f'select {key_name} from dsos_ast_test where {key_name} == "{k}"')
            elif type(k) == Timestamp:
                s = int(int(k.to_datetime64())/1000000000)
                us = k.microsecond
                q.select(f'select {key_name} from dsos_ast_test where {key_name} == "{s}.{us}"')
            else:
                q.select(f'select {key_name} from dsos_ast_test where {key_name} == {k}')
            df = q.next()
            self.assertIsNotNone(df, None)
            self.assertEqual(len(df), 1)
            self.assertEqual(df[key_name][0], k)

    def __test_wrong_key_type(self, key_no):
        q = self.cont.query()
        for d in data:
            k = d[1]
            try:
                q.select(f'select char_array from dsos_ast_test where char_array == 1')
            except Exception as e:
                # We are expecting an error here due to the invalid constant comparison
                return
        self.assertFalse(True)

    def test_01_get_data(self):
        self.__test_find_data(1, 'char_array')

    def test_02_get_data(self):
        self.__test_find_data(2, 'int16')

    def test_03_get_data(self):
        self.__test_find_data(3, 'int32')

    def test_04_get_data(self):
        self.__test_find_data(4, 'int64')

    def test_05_get_data(self):
        self.__test_find_data(5, 'uint16')

    def test_06_get_data(self):
        self.__test_find_data(6, 'uint32')

    def test_07_get_data(self):
        self.__test_find_data(7, 'uint64')

    def test_08_get_data(self):
        self.__test_find_data(8, 'float')

    def test_09_get_data(self):
        self.__test_find_data(9, 'double')

    def test_10_get_data(self):
        q = self.cont.query()
        q.select(f'select timestamp from dsos_ast_test order_by timestamp')
        df = q.next()
        self.__test_find_data(10, 'timestamp')

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
