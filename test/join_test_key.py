#!/usr/bin/env python
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

def random_16():
    return int(random.random() * 32767)

def random_32():
    return int(random.random() * (1 << 31))

def random_64():
    return int(random.random() * (1 << 63))

def random_float():
    return float(random.random())

class JoinTestKey(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("join_key_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_u32',
                             [ { "name" : "a_1", "type" : "uint32" },
                               { "name" : "a_2", "type" : "uint32" },
                               { "name" : "a_3", "type" : "uint32" },
                               { "name" : "a_join", "type" : "join",
                                 "join_attrs" : [ "a_1", "a_2", "a_3" ],
                                 "index" : {}}
                           ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_00_add_obj(self):
        data = [
            ( 0, 0, 0 ),
            ( 0, 1, 1 ),
            ( 0, 1, 2 ),
            ( 0, 2, 1 ),
            ( 100, 1, 1 ),
        ]
        for seq in data:
            obj = self.schema.alloc()
            obj[:] = seq
            obj.index_add()

    def test_01_order(self):
        a = self.schema.attr_by_name("a_join")
        k = a.key(1, 0, 0)
        idx = a.index()
        o = idx.find_sup(k)

    def compare_join_split(self, values):
        k = Sos.Key(size=256)
        k.join(*values)
        result = k.split()
        self.assertEqual(values, result)

    def test_16_str_16(self):
        self.compare_join_split([ Sos.TYPE_UINT16, random_16(),
                                  Sos.TYPE_CHAR_ARRAY, 'this is a your day',
                                  Sos.TYPE_UINT16, random_16()
                              ])

    def test_16_str_16_str(self):
        self.compare_join_split([ Sos.TYPE_UINT16, random_16(),
                                  Sos.TYPE_CHAR_ARRAY, 'this is a your day',
                                  Sos.TYPE_UINT16, random_16(),
                                  Sos.TYPE_CHAR_ARRAY, 'or is it?'
                              ])

    def test_32_str_64_str(self):
        self.compare_join_split([ Sos.TYPE_UINT32, random_32(),
                                  Sos.TYPE_CHAR_ARRAY, 'this is a your day',
                                  Sos.TYPE_UINT64, random_64(),
                                  Sos.TYPE_CHAR_ARRAY, 'or is it?'
                              ])

    def test_32_str_64_str_16(self):
        self.compare_join_split([ Sos.TYPE_UINT32, random_32(),
                                  Sos.TYPE_CHAR_ARRAY, 'this is a your day',
                                  Sos.TYPE_UINT64, random_64(),
                                  Sos.TYPE_CHAR_ARRAY, 'or is it?',
                                  Sos.TYPE_UINT16, random_16()
                              ])

    def test_16_32_64_double(self):
        self.compare_join_split([ Sos.TYPE_UINT16, random_16(),
                                  Sos.TYPE_UINT32, random_32(),
                                  Sos.TYPE_UINT64, random_64(),
                                  Sos.TYPE_DOUBLE, random_float()
                              ])

    def test_16_64_32_double(self):
        self.compare_join_split([ Sos.TYPE_UINT16, random_16(),
                                  Sos.TYPE_UINT64, random_64(),
                                  Sos.TYPE_UINT32, random_32(),
                                  Sos.TYPE_DOUBLE, random_float()
                              ])

    def test_16_double_32_64(self):
        self.compare_join_split([ Sos.TYPE_UINT16, random_16(),
                                  Sos.TYPE_DOUBLE, random_float(),
                                  Sos.TYPE_UINT32, random_32(),
                                  Sos.TYPE_UINT64, random_64(),
                              ])

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
