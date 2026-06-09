#!/usr/bin/env python3
from builtins import next
from builtins import range
from builtins import object
import unittest
import shutil
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase

class Debug(object): pass

logger = logging.getLogger(__name__)

class JoinTestStr(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("join_test_str_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_str',
                             [ { "name" : "a_1", "type" : "string", "index" : {} },
                               { "name" : "a_2", "type" : "uint64", "index" : {} }
                             ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        # cls.tearDownDb()
        pass

    def test_str_add_objects(self):
        data = []
        for i in range(1, 1024 * 1024):
            o = self.schema.alloc()
            o[:] = ( str(i), i )
            o.index_add()


class LsosJoinTestStr(JoinTestStr):
    @classmethod
    def backend(cls):
        return Sos.BE_LSOS

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
