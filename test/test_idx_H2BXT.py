#!/usr/bin/env python3

from test_idx_util import *

class TestBXTREE(TestIndexBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.STORE_PATH = "./bxt.store"
        cls.PART_NAME = "part"
        cls.SCHEMA_NAME = "schema"
        cls.IDX_TYPE = "H2BXT"
        cls.IDX_ARG = ""
        super(TestBXTREE, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestBXTREE, cls).tearDownClass()


if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    unittest.main()
