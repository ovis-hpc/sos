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

class Debug(object): pass

logger = logging.getLogger(__name__)

class FilterJoinCond(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("join_test_filter_seek")

        cls.uint64_schema = Sos.Schema()
        cls.uint64_schema.from_template('uint64_four_way_join',
                                 [ { "name" : "k0", "type" : "uint64" },
                                   { "name" : "k1", "type" : "uint64" },
                                   { "name" : "k2", "type" : "uint64" },
                                   { "name" : "k3", "type" : "uint64" },
                                   { "name" : "a_join", "type" : "join",
                                     "join_attrs" : [ "k0", "k1", "k2", "k3" ],
                                     "index" : {}}
                               ])
        cls.uint64_schema.add(cls.db)

        cls.int64_schema = Sos.Schema()
        cls.int64_schema.from_template('int64_four_way_join',
                                       [ { "name" : "k0", "type" : "int64" },
                                         { "name" : "k1", "type" : "int64" },
                                         { "name" : "k2", "type" : "int64" },
                                         { "name" : "k3", "type" : "int64" },
                                         { "name" : "a_join", "type" : "join",
                                           "join_attrs" : [ "k0", "k1", "k2", "k3" ],
                                           "index" : {}}
                               ])
        cls.int64_schema.add(cls.db)

        cls.uint32_schema = Sos.Schema()
        cls.uint32_schema.from_template('uint32_four_way_join',
                                 [ { "name" : "k0", "type" : "uint32" },
                                   { "name" : "k1", "type" : "uint32" },
                                   { "name" : "k2", "type" : "uint32" },
                                   { "name" : "k3", "type" : "uint32" },
                                   { "name" : "a_join", "type" : "join",
                                     "join_attrs" : [ "k0", "k1", "k2", "k3" ],
                                     "index" : {}}
                               ])
        cls.uint32_schema.add(cls.db)

        cls.int32_schema = Sos.Schema()
        cls.int32_schema.from_template('int32_four_way_join',
                                       [ { "name" : "k0", "type" : "int32" },
                                         { "name" : "k1", "type" : "int32" },
                                         { "name" : "k2", "type" : "int32" },
                                         { "name" : "k3", "type" : "int32" },
                                         { "name" : "a_join", "type" : "join",
                                           "join_attrs" : [ "k0", "k1", "k2", "k3" ],
                                           "index" : {}}
                               ])
        cls.int32_schema.add(cls.db)

        cls.uint16_schema = Sos.Schema()
        cls.uint16_schema.from_template('uint16_four_way_join',
                                 [ { "name" : "k0", "type" : "uint16" },
                                   { "name" : "k1", "type" : "uint16" },
                                   { "name" : "k2", "type" : "uint16" },
                                   { "name" : "k3", "type" : "uint16" },
                                   { "name" : "a_join", "type" : "join",
                                     "join_attrs" : [ "k0", "k1", "k2", "k3" ],
                                     "index" : {}}
                               ])
        cls.uint16_schema.add(cls.db)

        cls.int16_schema = Sos.Schema()
        cls.int16_schema.from_template('int16_four_way_join',
                                       [ { "name" : "k0", "type" : "int16" },
                                         { "name" : "k1", "type" : "int16" },
                                         { "name" : "k2", "type" : "int16" },
                                         { "name" : "k3", "type" : "int16" },
                                         { "name" : "a_join", "type" : "join",
                                           "join_attrs" : [ "k0", "k1", "k2", "k3" ],
                                           "index" : {}}
                               ])
        cls.int16_schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_uint64_add_objects(self):
        for a_1 in range(0, 16):
            for a_2 in range(0, 16):
                for a_3 in range(0, 16):
                    for a_4 in range(0, 16):
                        o = self.uint64_schema.alloc()
                        o[:] = ( a_1, a_2, a_3, a_4 )
                        o.index_add()


    def test_int64_add_objects(self):
        for a_1 in range(-16, 1):
            for a_2 in range(-16, 1):
                for a_3 in range(-16, 1):
                    for a_4 in range(-16, 1):
                        o = self.int64_schema.alloc()
                        o[:] = ( a_1, a_2, a_3, a_4 )
                        o.index_add()


    def test_uint32_add_objects(self):
        for a_1 in range(0, 16):
            for a_2 in range(0, 16):
                for a_3 in range(0, 16):
                    for a_4 in range(0, 16):
                        o = self.uint32_schema.alloc()
                        o[:] = ( a_1, a_2, a_3, a_4 )
                        o.index_add()


    def test_int32_add_objects(self):
        for a_1 in range(-16, 1):
            for a_2 in range(-16, 1):
                for a_3 in range(-16, 1):
                    for a_4 in range(-16, 1):
                        o = self.int32_schema.alloc()
                        o[:] = ( a_1, a_2, a_3, a_4 )
                        o.index_add()


    def test_uint16_add_objects(self):
        for a_1 in range(0, 16):
            for a_2 in range(0, 16):
                for a_3 in range(0, 16):
                    for a_4 in range(0, 16):
                        o = self.uint16_schema.alloc()
                        o[:] = ( a_1, a_2, a_3, a_4 )
                        o.index_add()


    def test_int16_add_objects(self):
        for a_1 in range(-16, 1):
            for a_2 in range(-16, 1):
                for a_3 in range(-16, 1):
                    for a_4 in range(-16, 1):
                        o = self.int16_schema.alloc()
                        o[:] = ( a_1, a_2, a_3, a_4 )
                        o.index_add()


    # UINT64 Tests
    def test_uint64_k0(self):
        a_join = self.uint64_schema.attr_by_name('a_join')
        k0 = self.uint64_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 16 * 16 * 16)
        del f

    def test_uint64_k0_k1(self):
        a_join = self.uint64_schema.attr_by_name('a_join')
        k0 = self.uint64_schema.attr_by_name('k0')
        k1 = self.uint64_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 16 * 16)
        del f

    def test_uint64_k0_k1_k2(self):
        a_join = self.uint64_schema.attr_by_name('a_join')
        k0 = self.uint64_schema.attr_by_name('k0')
        k1 = self.uint64_schema.attr_by_name('k1')
        k2 = self.uint64_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 16)
        del f

    def test_uint64_k0_k1_k2_k3(self):
        a_join = self.uint64_schema.attr_by_name('a_join')
        k0 = self.uint64_schema.attr_by_name('k0')
        k1 = self.uint64_schema.attr_by_name('k1')
        k2 = self.uint64_schema.attr_by_name('k2')
        k3 = self.uint64_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        f.add_condition(k3, Sos.COND_GE, 4)
        f.add_condition(k3, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    def test_uint64_k0_rev(self):
        a_join = self.uint64_schema.attr_by_name('a_join')
        k0 = self.uint64_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 16 * 16 * 16)
        del f

    def test_uint64_k0_k1_rev(self):
        a_join = self.uint64_schema.attr_by_name('a_join')
        k0 = self.uint64_schema.attr_by_name('k0')
        k1 = self.uint64_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 16 * 16)
        del f

    def test_uint64_k0_k1_k2_prev(self):
        a_join = self.uint64_schema.attr_by_name('a_join')
        k0 = self.uint64_schema.attr_by_name('k0')
        k1 = self.uint64_schema.attr_by_name('k1')
        k2 = self.uint64_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 16)
        del f

    def test_uint64_k0_k1_k2_k3_prev(self):
        a_join = self.uint64_schema.attr_by_name('a_join')
        k0 = self.uint64_schema.attr_by_name('k0')
        k1 = self.uint64_schema.attr_by_name('k1')
        k2 = self.uint64_schema.attr_by_name('k2')
        k3 = self.uint64_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        f.add_condition(k3, Sos.COND_GE, 4)
        f.add_condition(k3, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    # INT64 Tests
    def test_int64_k0(self):
        a_join = self.int64_schema.attr_by_name('a_join')
        k0 = self.int64_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 17 * 17 * 17)
        del f

    def test_int64_k0_k1(self):
        a_join = self.int64_schema.attr_by_name('a_join')
        k0 = self.int64_schema.attr_by_name('k0')
        k1 = self.int64_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 17 * 17)
        del f

    def test_int64_k0_k1_k2(self):
        a_join = self.int64_schema.attr_by_name('a_join')
        k0 = self.int64_schema.attr_by_name('k0')
        k1 = self.int64_schema.attr_by_name('k1')
        k2 = self.int64_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 17)
        del f

    def test_int64_k0_k1_k2_k3(self):
        a_join = self.int64_schema.attr_by_name('a_join')
        k0 = self.int64_schema.attr_by_name('k0')
        k1 = self.int64_schema.attr_by_name('k1')
        k2 = self.int64_schema.attr_by_name('k2')
        k3 = self.int64_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        f.add_condition(k3, Sos.COND_LE, -4)
        f.add_condition(k3, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    def test_int64_k0_rev(self):
        a_join = self.int64_schema.attr_by_name('a_join')
        k0 = self.int64_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 17 * 17 * 17)
        del f

    def test_int64_k0_k1_rev(self):
        a_join = self.int64_schema.attr_by_name('a_join')
        k0 = self.int64_schema.attr_by_name('k0')
        k1 = self.int64_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 17 * 17)
        del f

    def test_int64_k0_k1_k2_prev(self):
        a_join = self.int64_schema.attr_by_name('a_join')
        k0 = self.int64_schema.attr_by_name('k0')
        k1 = self.int64_schema.attr_by_name('k1')
        k2 = self.int64_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 17)
        del f

    def test_int64_k0_k1_k2_k3_prev(self):
        a_join = self.int64_schema.attr_by_name('a_join')
        k0 = self.int64_schema.attr_by_name('k0')
        k1 = self.int64_schema.attr_by_name('k1')
        k2 = self.int64_schema.attr_by_name('k2')
        k3 = self.int64_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        f.add_condition(k3, Sos.COND_LE, -4)
        f.add_condition(k3, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    # UINT32 Tests
    def test_uint32_k0(self):
        a_join = self.uint32_schema.attr_by_name('a_join')
        k0 = self.uint32_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 16 * 16 * 16)
        del f

    def test_uint32_k0_k1(self):
        a_join = self.uint32_schema.attr_by_name('a_join')
        k0 = self.uint32_schema.attr_by_name('k0')
        k1 = self.uint32_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 16 * 16)
        del f

    def test_uint32_k0_k1_k2(self):
        a_join = self.uint32_schema.attr_by_name('a_join')
        k0 = self.uint32_schema.attr_by_name('k0')
        k1 = self.uint32_schema.attr_by_name('k1')
        k2 = self.uint32_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 16)
        del f

    def test_uint32_k0_k1_k2_k3(self):
        a_join = self.uint32_schema.attr_by_name('a_join')
        k0 = self.uint32_schema.attr_by_name('k0')
        k1 = self.uint32_schema.attr_by_name('k1')
        k2 = self.uint32_schema.attr_by_name('k2')
        k3 = self.uint32_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        f.add_condition(k3, Sos.COND_GE, 4)
        f.add_condition(k3, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    def test_uint32_k0_rev(self):
        a_join = self.uint32_schema.attr_by_name('a_join')
        k0 = self.uint32_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 16 * 16 * 16)
        del f

    def test_uint32_k0_k1_rev(self):
        a_join = self.uint32_schema.attr_by_name('a_join')
        k0 = self.uint32_schema.attr_by_name('k0')
        k1 = self.uint32_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 16 * 16)
        del f

    def test_uint32_k0_k1_k2_prev(self):
        a_join = self.uint32_schema.attr_by_name('a_join')
        k0 = self.uint32_schema.attr_by_name('k0')
        k1 = self.uint32_schema.attr_by_name('k1')
        k2 = self.uint32_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 16)
        del f

    def test_uint32_k0_k1_k2_k3_prev(self):
        a_join = self.uint32_schema.attr_by_name('a_join')
        k0 = self.uint32_schema.attr_by_name('k0')
        k1 = self.uint32_schema.attr_by_name('k1')
        k2 = self.uint32_schema.attr_by_name('k2')
        k3 = self.uint32_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        f.add_condition(k3, Sos.COND_GE, 4)
        f.add_condition(k3, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    # INT32 Tests
    def test_int32_k0(self):
        a_join = self.int32_schema.attr_by_name('a_join')
        k0 = self.int32_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 17 * 17 * 17)
        del f

    def test_int32_k0_k1(self):
        a_join = self.int32_schema.attr_by_name('a_join')
        k0 = self.int32_schema.attr_by_name('k0')
        k1 = self.int32_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 17 * 17)
        del f

    def test_int32_k0_k1_k2(self):
        a_join = self.int32_schema.attr_by_name('a_join')
        k0 = self.int32_schema.attr_by_name('k0')
        k1 = self.int32_schema.attr_by_name('k1')
        k2 = self.int32_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 17)
        del f

    def test_int32_k0_k1_k2_k3(self):
        a_join = self.int32_schema.attr_by_name('a_join')
        k0 = self.int32_schema.attr_by_name('k0')
        k1 = self.int32_schema.attr_by_name('k1')
        k2 = self.int32_schema.attr_by_name('k2')
        k3 = self.int32_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        f.add_condition(k3, Sos.COND_LE, -4)
        f.add_condition(k3, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    def test_int32_k0_rev(self):
        a_join = self.int32_schema.attr_by_name('a_join')
        k0 = self.int32_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 17 * 17 * 17)
        del f

    def test_int32_k0_k1_rev(self):
        a_join = self.int32_schema.attr_by_name('a_join')
        k0 = self.int32_schema.attr_by_name('k0')
        k1 = self.int32_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 17 * 17)
        del f

    def test_int32_k0_k1_k2_prev(self):
        a_join = self.int32_schema.attr_by_name('a_join')
        k0 = self.int32_schema.attr_by_name('k0')
        k1 = self.int32_schema.attr_by_name('k1')
        k2 = self.int32_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 17)
        del f

    def test_int32_k0_k1_k2_k3_prev(self):
        a_join = self.int32_schema.attr_by_name('a_join')
        k0 = self.int32_schema.attr_by_name('k0')
        k1 = self.int32_schema.attr_by_name('k1')
        k2 = self.int32_schema.attr_by_name('k2')
        k3 = self.int32_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        f.add_condition(k3, Sos.COND_LE, -4)
        f.add_condition(k3, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    # UINT16 Tests
    def test_uint16_k0(self):
        a_join = self.uint16_schema.attr_by_name('a_join')
        k0 = self.uint16_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 16 * 16 * 16)
        del f

    def test_uint16_k0_k1(self):
        a_join = self.uint16_schema.attr_by_name('a_join')
        k0 = self.uint16_schema.attr_by_name('k0')
        k1 = self.uint16_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 16 * 16)
        del f

    def test_uint16_k0_k1_k2(self):
        a_join = self.uint16_schema.attr_by_name('a_join')
        k0 = self.uint16_schema.attr_by_name('k0')
        k1 = self.uint16_schema.attr_by_name('k1')
        k2 = self.uint16_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 16)
        del f

    def test_uint16_k0_k1_k2_k3(self):
        a_join = self.uint16_schema.attr_by_name('a_join')
        k0 = self.uint16_schema.attr_by_name('k0')
        k1 = self.uint16_schema.attr_by_name('k1')
        k2 = self.uint16_schema.attr_by_name('k2')
        k3 = self.uint16_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        f.add_condition(k3, Sos.COND_GE, 4)
        f.add_condition(k3, Sos.COND_LE, 12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    def test_uint16_k0_rev(self):
        a_join = self.uint16_schema.attr_by_name('a_join')
        k0 = self.uint16_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 16 * 16 * 16)
        del f

    def test_uint16_k0_k1_rev(self):
        a_join = self.uint16_schema.attr_by_name('a_join')
        k0 = self.uint16_schema.attr_by_name('k0')
        k1 = self.uint16_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 16 * 16)
        del f

    def test_uint16_k0_k1_k2_prev(self):
        a_join = self.uint16_schema.attr_by_name('a_join')
        k0 = self.uint16_schema.attr_by_name('k0')
        k1 = self.uint16_schema.attr_by_name('k1')
        k2 = self.uint16_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 16)
        del f

    def test_uint16_k0_k1_k2_k3_prev(self):
        a_join = self.uint16_schema.attr_by_name('a_join')
        k0 = self.uint16_schema.attr_by_name('k0')
        k1 = self.uint16_schema.attr_by_name('k1')
        k2 = self.uint16_schema.attr_by_name('k2')
        k3 = self.uint16_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_GE, 4)
        f.add_condition(k0, Sos.COND_LE, 12)
        f.add_condition(k1, Sos.COND_GE, 4)
        f.add_condition(k1, Sos.COND_LE, 12)
        f.add_condition(k2, Sos.COND_GE, 4)
        f.add_condition(k2, Sos.COND_LE, 12)
        f.add_condition(k3, Sos.COND_GE, 4)
        f.add_condition(k3, Sos.COND_LE, 12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    # INT16 Tests
    def test_int16_k0(self):
        a_join = self.int16_schema.attr_by_name('a_join')
        k0 = self.int16_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 17 * 17 * 17)
        del f

    def test_int16_k0_k1(self):
        a_join = self.int16_schema.attr_by_name('a_join')
        k0 = self.int16_schema.attr_by_name('k0')
        k1 = self.int16_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 17 * 17)
        del f

    def test_int16_k0_k1_k2(self):
        a_join = self.int16_schema.attr_by_name('a_join')
        k0 = self.int16_schema.attr_by_name('k0')
        k1 = self.int16_schema.attr_by_name('k1')
        k2 = self.int16_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 17)
        del f

    def test_int16_k0_k1_k2_k3(self):
        a_join = self.int16_schema.attr_by_name('a_join')
        k0 = self.int16_schema.attr_by_name('k0')
        k1 = self.int16_schema.attr_by_name('k1')
        k2 = self.int16_schema.attr_by_name('k2')
        k3 = self.int16_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        f.add_condition(k3, Sos.COND_LE, -4)
        f.add_condition(k3, Sos.COND_GE, -12)
        o = f.begin()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = next(f)
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f

    def test_int16_k0_rev(self):
        a_join = self.int16_schema.attr_by_name('a_join')
        k0 = self.int16_schema.attr_by_name('k0')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 1)
        self.assertEqual(count, 9 * 17 * 17 * 17)
        del f

    def test_int16_k0_k1_rev(self):
        a_join = self.int16_schema.attr_by_name('a_join')
        k0 = self.int16_schema.attr_by_name('k0')
        k1 = self.int16_schema.attr_by_name('k1')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 2)
        self.assertEqual(count, 9 * 9 * 17 * 17)
        del f

    def test_int16_k0_k1_k2_prev(self):
        a_join = self.int16_schema.attr_by_name('a_join')
        k0 = self.int16_schema.attr_by_name('k0')
        k1 = self.int16_schema.attr_by_name('k1')
        k2 = self.int16_schema.attr_by_name('k2')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 3)
        self.assertEqual(count, 9 * 9 * 9 * 17)
        del f

    def test_int16_k0_k1_k2_k3_prev(self):
        a_join = self.int16_schema.attr_by_name('a_join')
        k0 = self.int16_schema.attr_by_name('k0')
        k1 = self.int16_schema.attr_by_name('k1')
        k2 = self.int16_schema.attr_by_name('k2')
        k3 = self.int16_schema.attr_by_name('k3')
        f = Sos.Filter(a_join)
        f.add_condition(k0, Sos.COND_LE, -4)
        f.add_condition(k0, Sos.COND_GE, -12)
        f.add_condition(k1, Sos.COND_LE, -4)
        f.add_condition(k1, Sos.COND_GE, -12)
        f.add_condition(k2, Sos.COND_LE, -4)
        f.add_condition(k2, Sos.COND_GE, -12)
        f.add_condition(k3, Sos.COND_LE, -4)
        f.add_condition(k3, Sos.COND_GE, -12)
        o = f.end()
        count = 0
        while o:
            # Dprint(o[:])
            count += 1
            o = f.prev()
        Dprint("Misses {0}".format(f.miss_count()))
        Dprint("Count {0}".format(count))
        self.assertTrue(f.miss_count() <= 4)
        self.assertEqual(count, 9 * 9 * 9 * 9)
        del f


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
