#!/usr/bin/env python
from past.builtins import execfile
from builtins import next
from builtins import object
import unittest
import shutil
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase

class Debug(object): pass

logger = logging.getLogger(__name__)

class JoinTestU64(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("join_test_u64_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_u64',
                             [ { "name" : "a_1", "type" : "uint64" },
                               { "name" : "a_2", "type" : "uint64" },
                               { "name" : "a_3", "type" : "uint64" },
                               { "name" : "a_join", "type" : "join",
                                 "join_attrs" : [ "a_1", "a_2", "a_3" ],
                                 "index" : {}}
                           ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_u64_add_objects(self):
        data = [ (1, 2, 3), (2, 3, 4), (3, 4, 5) ]
        objs = []
        for seq in data:
            o = self.schema.alloc()
            objs.append(o)
            o[:] = seq
            o.index_add()

        # Check expected order
        a_join = self.schema.attr_by_name('a_join')
        it = a_join.attr_iter()
        i = 0
        b = it.begin()
        count = 0
        while b:
            count += 1
            o = it.item()
            self.assertEqual(o.a_1, data[i][0])
            self.assertEqual(o.a_2, data[i][1])
            self.assertEqual(o.a_3, data[i][2])
            b = next(it)
            i += 1
        self.assertEqual(count, 3)

    def u64_fwd_col(self, cond, cmp_fn, idx, expect, counts):
        a_join = self.schema.attr_by_name('a_join')
        attr = self.schema.attr_by_id(idx)

        f = a_join.filter()
        f.add_condition(attr, cond, expect[0])
        o = f.begin()
        count = 0
        while o:
            count += 1
            self.assertTrue(cmp_fn(o[idx], expect[0]))
            o = next(f)
        self.assertEqual(count, counts[0])
        del f

        f = a_join.filter()
        f.add_condition(attr, cond, expect[1])
        o = f.begin()
        count = 0
        while o:
            count += 1
            self.assertTrue(cmp_fn(o[idx], expect[1]))
            o = next(f)
        self.assertEqual(count, counts[1])
        del f

        f = a_join.filter()
        f.add_condition(attr, cond, expect[2])
        o = f.begin()
        count = 0
        while o:
            count += 1
            self.assertTrue(cmp_fn(o[idx], expect[2]))
            o = next(f)
        self.assertEqual(count, counts[2])
        del f

        f = a_join.filter()
        f.add_condition(attr, cond, expect[3])
        o = f.begin()
        count = 0
        while o:
            count += 1
            self.assertTrue(cmp_fn(o[idx], expect[3]))
            o = next(f)
        self.assertEqual(count, counts[3])
        del f

    # SOS_COND_LT
    def test_u64_fwd_col_0_lt(self):
        self.u64_fwd_col(Sos.COND_LT, lambda a, b : a < b, 0, (1, 2, 3, 4), (0, 1, 2, 3))

    def test_u64_fwd_col_1_lt(self):
        self.u64_fwd_col(Sos.COND_LT, lambda a, b : a < b, 1, (2, 3, 4, 5), (0, 1, 2, 3))

    def test_u64_fwd_col_2_lt(self):
        self.u64_fwd_col(Sos.COND_LT, lambda a, b : a < b, 2, (3, 4, 5, 6), (0, 1, 2, 3))

    # SOS_COND_LE
    def test_u64_fwd_col_0_le(self):
        self.u64_fwd_col(Sos.COND_LE, lambda a, b : a <= b, 0, (1, 2, 3, 4), (1, 2, 3, 3))

    def test_u64_fwd_col_1_le(self):
        self.u64_fwd_col(Sos.COND_LE, lambda a, b : a <= b, 1, (2, 3, 4, 5), (1, 2, 3, 3))

    def test_u64_fwd_col_2_le(self):
        self.u64_fwd_col(Sos.COND_LE, lambda a, b : a <= b, 2, (3, 4, 5, 6), (1, 2, 3, 3))

    # SOS_COND_EQ
    def test_u64_fwd_col_0_eq(self):
        self.u64_fwd_col(Sos.COND_EQ, lambda a, b : a == b, 0, (1, 2, 3, 4), (1, 1, 1, 0))

    def test_u64_fwd_col_1_eq(self):
        self.u64_fwd_col(Sos.COND_EQ, lambda a, b : a == b, 1, (2, 3, 4, 5), (1, 1, 1, 0))

    def test_u64_fwd_col_2_eq(self):
        self.u64_fwd_col(Sos.COND_EQ, lambda a, b : a == b, 2, (3, 4, 5, 6), (1, 1, 1, 0))

    # SOS_COND_GE
    def test_u64_fwd_col_0_ge(self):
        self.u64_fwd_col(Sos.COND_GE, lambda a, b : a >= b, 0, (1, 2, 3, 4), (3, 2, 1, 0))

    def test_u64_fwd_col_1_ge(self):
        self.u64_fwd_col(Sos.COND_GE, lambda a, b : a >= b, 1, (2, 3, 4, 5), (3, 2, 1, 0))

    def test_u64_fwd_col_2_ge(self):
        self.u64_fwd_col(Sos.COND_GE, lambda a, b : a >= b, 2, (3, 4, 5, 6), (3, 2, 1, 0))

    # SOS_COND_GT
    def test_u64_fwd_col_0_gt(self):
        self.u64_fwd_col(Sos.COND_GT, lambda a, b : a > b, 0, (1, 2, 3, 4), (2, 1, 0, 0))

    def test_u64_fwd_col_1_gt(self):
        self.u64_fwd_col(Sos.COND_GT, lambda a, b : a > b, 1, (2, 3, 4, 5), (2, 1, 0, 0))

    def test_u64_fwd_col_2_gt(self):
        self.u64_fwd_col(Sos.COND_GT, lambda a, b : a > b, 2, (3, 4, 5, 6), (2, 1, 0, 0))

    # SOS_COND_NE
    def test_u64_fwd_col_0_ne(self):
        self.u64_fwd_col(Sos.COND_NE, lambda a, b : a != b, 0, (1, 2, 3, 4), (2, 2, 2, 3))

    def test_u64_fwd_col_1_ne(self):
        self.u64_fwd_col(Sos.COND_NE, lambda a, b : a != b, 1, (2, 3, 4, 5), (2, 2, 2, 3))

    def test_u64_fwd_col_2_ne(self):
        self.u64_fwd_col(Sos.COND_NE, lambda a, b : a != b, 2, (3, 4, 5, 6), (2, 2, 2, 3))

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
