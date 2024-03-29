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

col_0_arg = ( "aaaaa", "b", "cccccc", "dd", "eeeeeeee" )
col_1_arg = ( "bbbbb", "c", "dddddd", "ee", "ffffffff" )
col_2_arg = ( "ccccc", "d", "eeeeee", "ff", "gggggggg" )

class JoinTestStr(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("join_test_str_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_str',
                             [ { "name" : "a_1", "type" : "string" },
                               { "name" : "a_2", "type" : "string" },
                               { "name" : "a_3", "type" : "string" },
                               { "name" : "a_join", "type" : "join",
                                 "join_attrs" : [ "a_1", "a_2", "a_3" ],
                                 "index" : {}}
                           ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        # cls.tearDownDb()
        pass

    def test_str_add_objects(self):
        data = []
        for i in range(0, 3):
            data.append( (col_0_arg[i], col_1_arg[i], col_2_arg[i]) )
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

    def str_fwd_col(self, cond, cmp_fn, idx, expect, counts):
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
    def test_str_fwd_col_0_lt(self):
        self.str_fwd_col(Sos.COND_LT, lambda a, b : a < b, 0, col_0_arg, (0, 1, 2, 3))

    def test_str_fwd_col_1_lt(self):
        self.str_fwd_col(Sos.COND_LT, lambda a, b : a < b, 1, col_1_arg, (0, 1, 2, 3))

    def test_str_fwd_col_2_lt(self):
        self.str_fwd_col(Sos.COND_LT, lambda a, b : a < b, 2, col_2_arg, (0, 1, 2, 3))

    # SOS_COND_LE
    def test_str_fwd_col_0_le(self):
        self.str_fwd_col(Sos.COND_LE, lambda a, b : a <= b, 0, col_0_arg, (1, 2, 3, 3))

    def test_str_fwd_col_1_le(self):
        self.str_fwd_col(Sos.COND_LE, lambda a, b : a <= b, 1, col_1_arg, (1, 2, 3, 3))

    def test_str_fwd_col_2_le(self):
        self.str_fwd_col(Sos.COND_LE, lambda a, b : a <= b, 2, col_2_arg, (1, 2, 3, 3))

    # SOS_COND_EQ
    def test_str_fwd_col_0_eq(self):
        self.str_fwd_col(Sos.COND_EQ, lambda a, b : a == b, 0, col_0_arg, (1, 1, 1, 0))

    def test_str_fwd_col_1_eq(self):
        self.str_fwd_col(Sos.COND_EQ, lambda a, b : a == b, 1, col_1_arg, (1, 1, 1, 0))

    def test_str_fwd_col_2_eq(self):
        self.str_fwd_col(Sos.COND_EQ, lambda a, b : a == b, 2, col_2_arg, (1, 1, 1, 0))

    # SOS_COND_GE
    def test_str_fwd_col_0_ge(self):
        self.str_fwd_col(Sos.COND_GE, lambda a, b : a >= b, 0, col_0_arg, (3, 2, 1, 0))

    def test_str_fwd_col_1_ge(self):
        self.str_fwd_col(Sos.COND_GE, lambda a, b : a >= b, 1, col_1_arg, (3, 2, 1, 0))

    def test_str_fwd_col_2_ge(self):
        self.str_fwd_col(Sos.COND_GE, lambda a, b : a >= b, 2, col_2_arg, (3, 2, 1, 0))

    # SOS_COND_GT
    def test_str_fwd_col_0_gt(self):
        self.str_fwd_col(Sos.COND_GT, lambda a, b : a > b, 0, col_0_arg, (2, 1, 0, 0))

    def test_str_fwd_col_1_gt(self):
        self.str_fwd_col(Sos.COND_GT, lambda a, b : a > b, 1, col_1_arg, (2, 1, 0, 0))

    def test_str_fwd_col_2_gt(self):
        self.str_fwd_col(Sos.COND_GT, lambda a, b : a > b, 2, col_2_arg, (2, 1, 0, 0))

    # SOS_COND_NE
    def test_str_fwd_col_0_ne(self):
        self.str_fwd_col(Sos.COND_NE, lambda a, b : a != b, 0, col_0_arg, (2, 2, 2, 3))

    def test_str_fwd_col_1_ne(self):
        self.str_fwd_col(Sos.COND_NE, lambda a, b : a != b, 1, col_1_arg, (2, 2, 2, 3))

    def test_str_fwd_col_2_ne(self):
        self.str_fwd_col(Sos.COND_NE, lambda a, b : a != b, 2, col_2_arg, (2, 2, 2, 3))

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
