#!/usr/bin/env python
from past.builtins import execfile
from builtins import next
from builtins import range
from builtins import object
import unittest
import shutil
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase, Dprint, DprintEnable
import time
import random

class Debug(object): pass

logger = logging.getLogger(__name__)

class GuiFilter(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("metric_gui_join")

        t = time.time()
        cls.start = t + 60
        cls.end = t + 120

        cls.schema = Sos.Schema()
        cls.schema.from_template('metric_set',
                                 [
                                     { "name" : "timestamp", "type" : "timestamp", "index" : {} },
                                     { "name" : "component_id", "type" : "uint64", "index" : {} },
                                     { "name" : "job_id", "type" : "uint64", "index" : {} },
                                     { "name" : "app_id", "type" : "uint64" },
                                     { "name" : "metric0", "type" : "uint64" },
                                     { "name" : "comp_time", "type" : "join",
                                       "join_attrs" : [ "component_id", "timestamp" ],
                                       "index" : {} },
                                     { "name" : "job_comp_time", "type" : "join",
                                       "join_attrs" : [ "job_id", "component_id", "timestamp" ],
                                       "index" : {} },
                                     { "name" : "job_time_comp", "type" : "join",
                                       "join_attrs" : [ "job_id", "timestamp", "component_id" ],
                                       "index" : {} }
                                 ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()
        pass

    def test_001_metric_add_objects(self):
        for job_id in range(1, 8):
            for comp_id in range(1, 16):
                for secs in range(0, 360):
                    o = self.schema.alloc()
                    tstamp = self.start + secs
                    o[:] = ( tstamp, comp_id, job_id, 0, random.random() * 1000 )
                    o.index_add()

    def __query_test(self, index, job_id, comp_id, start, end, expected_count):
        join = self.schema.attr_by_name(index)
        time_a = self.schema['timestamp']
        f = Sos.Filter(join)
        if job_id is not None:
            job_a = self.schema['job_id']
            f.add_condition(job_a, Sos.COND_EQ, job_id)
        if comp_id is not None:
            comp_a = self.schema['component_id']
            f.add_condition(comp_a, Sos.COND_EQ, comp_id)
        f.add_condition(time_a, Sos.COND_GE, start)
        f.add_condition(time_a, Sos.COND_LT, end)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:4])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, expected_count)
        del f

    def test_002_job_comp_time(self):
        self.__query_test('job_comp_time', 4, 8, self.start, self.end, 60)

    def test_003_job_time_comp(self):
        self.__query_test('job_time_comp', 4, 8, self.start, self.end, 60)

    def test_004_comp_time(self):
        self.__query_test('comp_time', None, 8, self.start, self.end, 60 * 7)

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    # DprintEnable()
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
