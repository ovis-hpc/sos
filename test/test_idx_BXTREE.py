#!/usr/bin/env python

from test_idx_util import *

class TestBXTREE(TestIndexBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.STORE_PATH = "./bxt.store"
        cls.PART_NAME = "part"
        cls.SCHEMA_NAME = "schema"
        cls.IDX_TYPE = "BXTREE"
        cls.IDX_ARG = "ORDER=5 SIZE=3"
        super(TestBXTREE, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestBXTREE, cls).tearDownClass()


if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    unittest.main()
