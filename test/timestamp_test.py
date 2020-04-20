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
from sosunittest import SosTestCase, Dprint
import random
import numpy as np
import numpy.random as nprnd
import time
class Debug(object): pass

logger = logging.getLogger(__name__)
data_tuple = []
data_unix = []
data_float = []

_debug = True

class TimestampTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("timestamp_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('timestamp_test',
                                 [
                                 { "name" : "timestamp", "type" : "timestamp", "index" : {} },
                                 { "name" : "timestr", "type" : "char_array" }
                                 ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()
        pass

    def __generate_data_tuple(self):
        for i in range(0, 10):
            t = time.time()
            data_tuple.append(
                (t, (int(t), int((t - int(t)) * 1.e6)),
                 time.strftime( '%X %x %Z', time.localtime(t) )
             ))

    def __generate_data_unix(self):
        # add a few seconds to void colliding the the tuple test data
        t = time.time() + 1
        for i in range(0, 10):
            data_unix.append(
                (t, int(t),
                 time.strftime( '%X %x %Z', time.localtime(t) )
             ))
            t += 1

    def __generate_data_float(self):
        # add a few seconds to void colliding the the unix test data
        t = time.time() + 20
        for i in range(0, 10):
            data_unix.append(
                (t, t,
                 time.strftime( '%X %x %Z', time.localtime(t) )
             ))
            t += 1

    def test_00_generate_tuple(self):
        self.__generate_data_tuple()
        for d in data_tuple:
            o = self.schema.alloc()
            o[:] = ( d[1], d[2] )
            rc = o.index_add()
            self.assertEqual( rc, 0 )
            Dprint( o[:] )
            del o

    def test_01_find(self):
        ts_attr = self.schema['timestamp']
        ts_idx = ts_attr.index()
        for d in data_tuple:
            k = ts_attr.key(d[1])
            o = ts_idx.find(k)
            self.assertTrue( o is not None )

    def test_02_filter_fwd(self):
        ts_attr = self.schema['timestamp']
        for i in range(len(data_tuple)):
            f = Sos.Filter(ts_attr)
            f.add_condition(ts_attr, Sos.COND_GE, data_tuple[i][1])
            f.add_condition(ts_attr, Sos.COND_LE, data_tuple[len(data_tuple)-1][1])
            Dprint(data_tuple[i][1])
            o = f.begin()
            count = len(data_tuple) - i
            while o:
                count -= 1
                Dprint(o[:])
                o = next(f)
            self.assertEqual( count, 0 )
            Dprint("--------------------------------")

    def test_03_filter_rev(self):
        ts_attr = self.schema['timestamp']
        for i in range(len(data_tuple) -1, -1, -1):
            f = Sos.Filter(ts_attr)
            f.add_condition(ts_attr, Sos.COND_GE, data_tuple[0][1])
            f.add_condition(ts_attr, Sos.COND_LE, data_tuple[i][1])
            Dprint(data_tuple[i][1])
            o = f.end()
            count = i + 1
            while o:
                count -= 1
                Dprint(o[:])
                o = f.prev()
            self.assertEqual( count, 0 )
            Dprint("--------------------------------")

    def test_04_generate_unix(self):
        self.__generate_data_unix()
        for d in data_unix:
            o = self.schema.alloc()
            o[:] = ( d[1], d[2] )
            rc = o.index_add()
            self.assertEqual( rc, 0 )
            Dprint( o[:] )
            del o

    def test_05_find(self):
        ts_attr = self.schema['timestamp']
        ts_idx = ts_attr.index()
        for d in data_unix:
            k = ts_attr.key((d[1], 0))
            o = ts_idx.find(k)
            self.assertTrue( o is not None )

            k = ts_attr.key(d[1])
            o = ts_idx.find(k)
            self.assertTrue( o is not None )

            k = ts_attr.key(float(d[1]))
            o = ts_idx.find(k)
            self.assertTrue( o is not None )

    def test_06_filter_fwd(self):
        ts_attr = self.schema['timestamp']
        for i in range(len(data_unix)):
            f = Sos.Filter(ts_attr)
            f.add_condition(ts_attr, Sos.COND_GE, data_unix[i][1])
            Dprint(data_unix[i][1])
            o = f.begin()
            count = len(data_unix) - i
            while o:
                count -= 1
                Dprint(o[:])
                o = next(f)
            self.assertEqual( count, 0 )
            Dprint("--------------------------------")

    def test_07_filter_rev(self):
        ts_attr = self.schema['timestamp']
        for i in range(len(data_unix) -1, -1, -1):
            f = Sos.Filter(ts_attr)
            f.add_condition(ts_attr, Sos.COND_GE, data_unix[0][1])
            f.add_condition(ts_attr, Sos.COND_LE, data_unix[i][1])
            Dprint(data_unix[0][1], data_unix[i][1])
            o = f.end()
            count = i + 1
            while o:
                count -= 1
                Dprint(o[:])
                o = f.prev()
            self.assertEqual( count, 0 )
            Dprint("--------------------------------")

    def test_08_generate_float(self):
        self.__generate_data_float()
        for d in data_float:
            o = self.schema.alloc()
            o[:] = ( d[1], d[2] )
            rc = o.index_add()
            self.assertEqual( rc, 0 )
            Dprint( o[:] )
            del o

    def test_09_find(self):
        ts_attr = self.schema['timestamp']
        ts_idx = ts_attr.index()
        for d in data_float:
            k = ts_attr.key(d[1])
            o = ts_idx.find(k)
            self.assertTrue( o is not None )

    def test_10_filter_fwd(self):
        ts_attr = self.schema['timestamp']
        for i in range(len(data_float)):
            f = Sos.Filter(ts_attr)
            f.add_condition(ts_attr, Sos.COND_GE, data_float[i][1])
            Dprint(data_float[i][1])
            o = f.begin()
            count = len(data_float) - i
            while o:
                count -= 1
                Dprint(o[:])
                o = next(f)
            self.assertEqual( count, 0 )
            Dprint("--------------------------------")

    def test_11_filter_rev(self):
        ts_attr = self.schema['timestamp']
        for i in range(len(data_float) -1, -1, -1):
            f = Sos.Filter(ts_attr)
            f.add_condition(ts_attr, Sos.COND_GE, data_float[0][1])
            f.add_condition(ts_attr, Sos.COND_LE, data_float[i][1])
            Dprint(data_float[0][1], data_float[i][1])
            o = f.end()
            count = i + 1
            while o:
                count -= 1
                Dprint(o[:])
                o = f.prev()
            self.assertEqual( count, 0 )
            Dprint("--------------------------------")

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
