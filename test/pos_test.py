#!/usr/bin/env python
from __future__ import print_function
from past.builtins import execfile
from builtins import next
from builtins import range
from builtins import object
import unittest
import shutil
import numpy as np
import logging
import time
import os
from sosdb import Sos
import random
import unittest
from sosunittest import SosTestCase
class Debug(object): pass

logger = logging.getLogger(__name__)

class FilterPosTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb('pos_test_cont')

        cls.schema = Sos.Schema()
        cls.schema_name = '3x_u64'
        cls.schema.from_template(cls.schema_name,
                                 [ { "name" : "job_id", "type" : "uint64" },
                                   { "name" : "timestamp", "type" : "timestamp" },
                                   { "name" : "component_id", "type" : "uint64" },
                                   { "name" : "job_time_comp", "type" : "join",
                                     "join_attrs" : [ "job_id", "timestamp", "component_id" ],
                                     "index" : {}}
                               ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def setUp(self):
        self.filt = Sos.Filter(self.schema['job_time_comp'])

    def tearDown(self):
        del self.filt

    def test_00_add_objs(self):
        for job in range(1, 6):
            start = int(time.time())
            for ts in range(start, start + 60 * int(random.random() * 60)):
                for comp in range(1, int(random.random() * 16) + 1):
                    o = self.schema.alloc()
                    o[:] = (job, ( ts, 0 ), comp)
                    o.index_add()

    def test_01_test_pos(self):
        pos_list = []
        o = self.filt.begin()
        while o:
            pos = self.filt.get_pos()
            pos_list.append( [ pos, o[:] ] )
            o = next(self.filt)

        for p in pos_list:
            rc = self.filt.set_pos( p[0] )
            o = self.filt.obj()
            self.assertEqual( rc, 0 )
            self.assertEqual( o[:], p[1] )

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
