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

class FilterCountTest(SosTestCase):
    """Test the Filter.count() method"""
    @classmethod
    def setUpClass(cls):
        cls.setUpDb('pos_test_cont')

        cls.schema = Sos.Schema()
        cls.schema_name = '3x_u64'
        cls.schema.from_template(cls.schema_name,
                                 [ { "name" : "job_id", "type" : "uint64" },
                                   { "name" : "timestamp", "type" : "timestamp" },
                                   { "name" : "component_id", "type" : "uint64" },
                                   { "name" : "job_time_cond", "type" : "join",
                                     "join_attrs" : [ "job_id", "timestamp", "component_id" ],
                                     "index" : {}}
                               ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def setUp(self):
        self.filt = Sos.Filter(self.schema['job_time_cond'])

    def tearDown(self):
        del self.filt

    def test_00_add_objs(self):
        for job in range(1, 6):
            start = int(time.time())
            for ts in range(start, start + 60 * int(random.random() * 60) + 2):
                for comp in range(1, int(random.random() * 16) + 2):
                    o = self.schema.alloc()
                    o[:] = (job, ( ts, 0 ), comp)
                    o.index_add()

    def test_01_test_count(self):
        # compute the record counts for each job
        counts = {}
        for job_id in range(1, 6):
            filt = Sos.Filter(self.schema['job_time_cond'])
            filt.add_condition(self.schema['job_id'], Sos.COND_EQ, job_id)
            count = 0
            o = filt.begin()
            while o:
                count += 1
                o = next(filt)
            counts[job_id] = count
            del filt

        # confirm that the count returned by filt.count() matches
        # the computed count
        for job_id in range(1, 6):
            filt = Sos.Filter(self.schema['job_time_cond'])
            filt.add_condition(self.schema['job_id'], Sos.COND_EQ, job_id)
            count = filt.count()
            self.assertEqual(count, counts[job_id])
            # print("filt.count() = {0}, computed count = {1}".format(count, counts[job_id]))

    def test_02_test_count(self):
        # compute the record counts for each job + comp
        counts = {}
        comps = {}
        for job_id in range(1, 6):
            filt = Sos.Filter(self.schema['job_time_cond'])
            filt.add_condition(self.schema['job_id'], Sos.COND_EQ, job_id)
            count = 0
            o = filt.begin()
            self.assertIsNotNone( o )
            comp_id = o['component_id']
            comps[job_id] = comp_id
            filt.add_condition(self.schema['component_id'], Sos.COND_EQ, comp_id)
            o = filt.begin()
            while o:
                count += 1
                o = next(filt)
            counts[job_id] = count
            del filt

        # confirm that the count returned by filt.count() matches
        # the computed count
        for job_id in range(1, 6):
            filt = Sos.Filter(self.schema['job_time_cond'])
            filt.add_condition(self.schema['job_id'], Sos.COND_EQ, job_id)
            filt.add_condition(self.schema['component_id'], Sos.COND_EQ, comps[job_id])
            count = filt.count()
            self.assertEqual(count, counts[job_id])
            # print("filt.count() = {0}, computed count = {1}".format(count, counts[job_id]))

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
