#!/usr/bin/env python
from __future__ import print_function
import unittest
import shutil
import numpy as np
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase

class Debug(object): pass

logger = logging.getLogger(__name__)

class FilterTestJoin3xU64(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb('join_test_3x_u64_cont')
        cls.schema = Sos.Schema()
        cls.schema.from_template('3x_u64',
                                 [ { "name" : "a_1", "type" : "uint64" },
                                   { "name" : "a_2", "type" : "uint64" },
                                   { "name" : "a_3", "type" : "uint64" },
                                   { "name" : "join_key", "type" : "join",
                                     "join_attrs" : [ "a_1", "a_2", "a_3" ],
                                     "index" : {}}
                               ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def setUp(self):
        self.min_a_1 = 1000
        self.max_a_1 = 10000
        self.filt = Sos.Filter(self.schema.attr_by_name('join_key'))
        a_1 = self.schema.attr_by_name('a_1')
        self.filt.add_condition(a_1, Sos.COND_GE, self.min_a_1)
        self.filt.add_condition(a_1, Sos.COND_LE, self.max_a_1)

    def tearDown(self):
        del self.filt

    def compute_min_max_a_23(self):
        # compute known min and max for the a_2 and a_3
        a_2_vals = {}
        a_3_vals = {}
        o = self.filt.begin()
        while o:
            if o[1] in a_2_vals:
                a_2_vals[o[1]] += 1
            else:
                a_2_vals[o[1]] = 1
            if o[2] in a_3_vals:
                a_3_vals[o[2]] += 1
            else:
                a_3_vals[o[2]] = 1
            o = self.filt.next()

        c1 = a_2_vals.keys()[len(a_2_vals)/2]
        c2 = a_2_vals.keys()[-1]
        if c1 > c2:
            min_a_2 = c2
            max_a_2 = c1
        else:
            min_a_2 = c1
            max_a_2 = c2
        c1 = a_3_vals.keys()[len(a_3_vals)/2]
        c2 = a_3_vals.keys()[-1]
        if c1 > c2:
            min_a_3 = c2
            max_a_3 = c1
        else:
            min_a_3 = c1
            max_a_3 = c2
        return (min_a_2, max_a_2, min_a_3, max_a_3)

    def test_00_add_objects(self):
        for i in range(0, 128 * 1024):
            o = self.schema.alloc()
            o[:] = np.random.rand(3) * 1000000
            o.index_add()

    def test_01_next_prev_a_1(self):
        o = self.filt.begin()
        next_count = 0
        while o:
            self.assertTrue(o[0] >= self.min_a_1)
            self.assertTrue(o[0] <= self.max_a_1)
            o = self.filt.next()
            next_count += 1

        # iterate backwards, the count should be the same
        o = self.filt.end()
        prev_count = 0
        while o:
            self.assertTrue(o[0] >= self.min_a_1)
            self.assertTrue(o[0] <= self.max_a_1)
            o = self.filt.prev()
            prev_count += 1

        self.assertEqual(next_count, prev_count)

    def test_02_next_prev_a_23(self):
        (min_a_2, max_a_2, min_a_3, max_a_3) = self.compute_min_max_a_23()

        # Add conditions an a_2 and a_3
        a_2 = self.schema.attr_by_name('a_2')
        a_3 = self.schema.attr_by_name('a_3')
        self.filt.add_condition(a_2, Sos.COND_GE, min_a_2)
        self.filt.add_condition(a_2, Sos.COND_LE, max_a_2)
        self.filt.add_condition(a_3, Sos.COND_GE, min_a_3)
        self.filt.add_condition(a_3, Sos.COND_LE, max_a_3)

        o = self.filt.begin()
        next_count = 0
        while o:
            self.assertTrue(o[1] >= min_a_2)
            self.assertTrue(o[1] <= max_a_2)
            self.assertTrue(o[2] >= min_a_3)
            self.assertTrue(o[2] <= max_a_3)
            o = self.filt.next()
            next_count += 1

        o = self.filt.end()
        prev_count = 0
        while o:
            self.assertTrue(o[1] >= min_a_2)
            self.assertTrue(o[1] <= max_a_2)
            self.assertTrue(o[2] >= min_a_3)
            self.assertTrue(o[2] <= max_a_3)
            o = self.filt.prev()
            prev_count += 1

        # The forward and reverse counts should be the same
        self.assertEqual(next_count, prev_count)

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
