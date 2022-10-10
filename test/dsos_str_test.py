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

logger = logging.getLogger(__name__)

data = [
    { "byte_array" : b'this is a byte_array', "char_array" : "this is a char_array" },
    { "byte_array" : b'this is a test', "char_array" : "this is a test" },
    { "byte_array" : b'this is not a test', "char_array" : "this is not a test" }
]

class DsosStrTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.session()
        cls.cont = cls.session.open("dsos_str_test_cont", o_perm=Sos.PERM_RW|Sos.PERM_CREAT)
        cls.schema = Sos.Schema()
        cls.schema.from_template('dsos_str_test', [
            { "name" : "byte_array", "type" : "byte_array" },
            { "name" : "char_array", "type" : "char_array", "index": {} }
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
                o[e] = d[e]
            self.dschema.obj_create(o)
        self.cont.transaction_end()

    def test_01_get_data(self):
        q = self.cont.query()
        for d in data:
            k = d['char_array']
            q.select(f'select char_array from dsos_str_test where char_array == "{k}"')
            df = q.next()
            self.assertIsNotNone(df, None)
            self.assertEqual(len(df), 1)
            self.assertEqual(df['char_array'][0], k)

    def test_02_wrong_key_type(self):
        q = self.cont.query()
        for d in data:
            k = d['char_array']
            try:
                q.select(f'select char_array from dsos_str_test where char_array == 1')
            except Exception as e:
                # We are expecting an error here due to the invalid constant comparison
                return
        self.assertFalse(True)

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
