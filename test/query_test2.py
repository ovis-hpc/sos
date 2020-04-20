#!/usr/bin/env python
from __future__ import print_function
from past.builtins import execfile
from builtins import object
import unittest
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase, Dprint

class Debug(object): pass

logger = logging.getLogger(__name__)

class QueryTest2(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb('query_test2_cont')
        cls.schema = Sos.Schema()
        cls.schema.from_template('query_test2', [{ "name" : "int16", "type" : "int16", "index" : {} }])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    # Regression: This used to trip a refcount bug in ods_idx.c.
    def test_01(self):
        obj = self.schema.alloc()
        obj[:] = [1]
        obj.index_add()
        query = Sos.Query(self.db)
        attr_name = 'int16'
        query.select([attr_name],
                     from_ = ['query_test2'],
                     where = [(attr_name, Sos.COND_GE, 1), (attr_name, Sos.COND_LE, 2)],
                     order_by = attr_name)
        query.begin()

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
