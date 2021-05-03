#!/usr/bin/env python3
from past.builtins import execfile
from builtins import object
import unittest
import shutil
import logging
import os
from shutil import rmtree
from sosdb import Sos
from sosunittest import SosTestCase
import random
class Debug(object): pass

logger = logging.getLogger(__name__)

data = [
    range(1, 512),
    range(256, 768),
    range(512, 1024),
    range(768, 1280)
]

part_path = ""

class PartitionTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("part_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_partitions',
                             [ { "name" : "uint32", "type" : "uint32", "index" : {} } ]
                             )
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()
        pass

    def __test_min(self):
        a = self.schema.attr_by_name("uint32")
        v = a.min()
        self.assertEqual(v, data[0][0])

    def __test_max(self):
        a = self.schema.attr_by_name("uint32")
        v = a.max()
        self.assertEqual(v, data[3][511])

    def __test_iter_begin(self):
        valdata = []
        for r in data:
            valdata += r
        valdata.sort()
        a = self.schema.attr_by_name("uint32")
        ai = Sos.AttrIter(a)
        rc = ai.begin()
        idx = 0
        while rc:
            k = ai.key()
            self.assertEqual(int(k), valdata[idx])
            rc = ai.next()
            idx += 1
            k.release()
        ai.release()

    def __test_iter_end(self):
        valdata = []
        for r in data:
            valdata += r
        valdata.sort(reverse=True)
        a = self.schema.attr_by_name("uint32")
        ai = Sos.AttrIter(a)
        rc = ai.end()
        idx = 0
        while rc:
            k = ai.key()
            self.assertEqual(int(k), valdata[idx])
            rc = ai.prev()
            idx += 1
            k.release()
        ai.release()

    def __test_iter_sup(self):
        valdata = []
        for r in data:
            valdata += r
        valdata = list(dict.fromkeys(valdata))
        valdata.sort()
        a = self.schema.attr_by_name("uint32")
        key = Sos.Key(attr=a)
        key.set_value(valdata[512])
        ai = Sos.AttrIter(a, unique=True)
        rc = ai.find_sup(key)
        key.release()
        idx = 512
        while rc:
            k = ai.key()
            self.assertEqual(int(k), valdata[idx])
            rc = ai.next()
            idx += 1
            k.release()
        ai.release()

    def __test_iter_inf(self):
        valdata = []
        for r in data:
            valdata += r
        valdata = list(dict.fromkeys(valdata))
        valdata.sort()
        a = self.schema.attr_by_name("uint32")
        key = Sos.Key(attr=a)
        key.set_value(valdata[512])
        ai = Sos.AttrIter(a, unique=True)
        rc = ai.find_inf(key)
        key.release()
        idx = 512
        while rc:
            k = ai.key()
            self.assertEqual(int(k), valdata[idx])
            rc = ai.next()
            idx += 1
            k.release()
        ai.release()

    def test_00_add_obj(self):
        pnum = 1
        for r in data:
            self.db.part_create(str(pnum), "partition {0}".format(pnum))
            part = self.db.part_by_name(str(pnum))
            part.state_set("PRIMARY")
            for o in r:
                obj = self.schema.alloc()
                obj[0] = o
                obj.index_add()
                obj.release()
            pnum += 1
            part.release()

    def test_01_min_uint32(self):
        self.__test_min()

    def test_02_max_uint32(self):
        self.__test_max()

    def test_03_iter_begin(self):
        self.__test_iter_begin()

    def test_04_iter_end(self):
        self.__test_iter_end()

    def test_04_iter_sup(self):
        self.__test_iter_sup()

    def test_05_iter_inf(self):
        self.__test_iter_inf()

    def test_06_part_move(self):
        # Move a partition to another location and ensure that the
        # content is iterable
        #
        # partition 1 contains the keys 1..511
        # partition 2 contains the keys 256..767
        global part_path

        p1 = self.db.part_by_name("1")
        self.assertNotEqual(p1, None)
        p1.state_set("OFFLINE")
        part_path = p1.path() + "-moved"
        p1.move(part_path)
        a = self.schema.attr_by_name("uint32")

        # The partition is moved and OFFLINE, make certain the
        # keys are not present
        ai = Sos.AttrIter(a)
        key = Sos.Key(attr=a)
        for i in range(1, 256):
            key.set_value(i)
            rc = ai.find(key)
            self.assertEqual(rc, False)
        key.release()
        ai.release()

        # Move the partition back to ACTIVE and the keys in that partition
        # should now be found
        p1.state_set("ACTIVE")
        ai = Sos.AttrIter(a)
        key = Sos.Key(attr=a)
        for i in range(1, 256):
            key.set_value(i)
            rc = ai.find(key)
            self.assertEqual(rc, True)
        key.release()
        ai.release()
        p1.release()    # release the partition reference

    def test_07_part_detach(self):
        # Detach the partition
        p1 = self.db.part_by_name("1")
        p1.state_set("OFFLINE")
        p1.detach()
        p1 = self.db.part_by_name("1")
        self.assertEqual(p1, None)

    def test_09_part_attach(self):
        # Clone a new empty container and import a
        # partition into it
        global part_path
        c = self.db.clone(self.db.path() + "-cloned")
        p = Sos.Partition()
        p.open(self.db.path() + "/1-moved")
        p.attach(c, "1-moved")
        p.release()
        p1 = c.part_by_name("1-moved")
        p1.state_set("PRIMARY")
        p1.release()

        # Check that the objects in the partition are present
        schema = c.schema_by_name("test_partitions")
        a = schema.attr_by_name("uint32")
        ai = Sos.AttrIter(a)
        rc = ai.begin()
        idx = 1
        while rc:
            k = ai.key()
            self.assertEqual(int(k), idx)
            rc = ai.next()
            idx += 1
            k.release()
        ai.release()

        # Import a 2nd partition that is now in both containers
        c.part_attach("2-too", self.db.path() + "/2")
        p1 = c.part_by_name("2-too")
        self.assertNotEqual(p1, None)
        p1.release()

        ai = Sos.AttrIter(a)
        rc = ai.begin()
        idx = 1
        while rc:
            k = ai.key()
            # self.assertEqual(int(k), idx)
            rc = ai.next()
            idx += 1
            k.release()
        ai.release()

        c.close()
        shutil.rmtree(c.path(), ignore_errors=True)

class LsosPartitionTest(PartitionTest):
    @classmethod
    def backend(cls):
        return Sos.BE_LSOS

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
