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
from sosunittest import SosTestCase
import random
import numpy as np
import numpy.random as nprnd
import time
class Debug(object): pass

logger = logging.getLogger(__name__)

data1 = []
data2 = []
first_ptn_id = 256
next_ptn_id = first_ptn_id

class PatternTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("pattern_test_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_pattern',
                             [ { "name" : "ts", "type" : "timestamp", "index" : {}},
                               { "name" : "ptn_id", "type" : "uint64",
                                 "index" : {}, },
                               { "name" : "ts_ptn_key", "type" : "join",
                                 "join_attrs" : [ "ts", "ptn_id" ],
                                 "index" : {}}
                           ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()
        pass

    def __time_to_tuple(self, t):
        return ( int(t), int(int((t - int(t)) * 1.0e6 )) )

    def __add_pattern(self):
        global next_ptn_id
        obj = self.schema.alloc()
        t = time.time()
        obj[:] = ( self.__time_to_tuple(t), next_ptn_id )
        obj.index_add()
        next_ptn_id += 1

    def __update_pattern(self, assert_found = True):
        # pick a random pattern
        ptn_id = random.random() * (next_ptn_id - first_ptn_id)
        ptn_id = int(ptn_id + first_ptn_id)

        # test that we can find it by ptn_id
        id_attr = self.schema['ptn_id']
        id_key = id_attr.key(ptn_id)
        ptn = id_attr.index().find(id_key)
        if assert_found is False and ptn is None:
            return
        self.assertTrue( ptn is not None )

        # test that we can find it by ts_ptn_key
        ts_ptn_attr = self.schema['ts_ptn_key']
        ts_ptn_key = ts_ptn_attr.key(ptn['ts'], ptn['ptn_id'])
        ts_ptn_idx = ts_ptn_attr.index()
        ptn = ts_ptn_idx.find(ts_ptn_key)
        self.assertTrue( ptn is not None )

        ts_attr = self.schema['ts']
        ts_idx = ts_attr.index()
        ts_key = ts_attr.key(ptn['ts'])
        ptn = ts_idx.find(ts_key)
        self.assertTrue( ptn is not None )

        # test that we can delete it
        rc = ts_ptn_idx.remove( ts_ptn_key, ptn )
        self.assertEqual( rc, 0 )

        rc = ts_idx.remove( ts_key, ptn )
        self.assertEqual( rc, 0 )

        # test that it's gone
        del_ptn = ts_ptn_idx.find(ts_ptn_key)
        self.assertEqual( del_ptn, None )

        del_ptn = ts_idx.find(ts_key)
        self.assertEqual( del_ptn, None )

        # update the pattern and add it back
        ptn['ts'] = self.__time_to_tuple(time.time())
        ts_ptn_key = ts_ptn_attr.key(ptn['ts'], ptn['ptn_id'])
        rc = ts_ptn_idx.insert( ts_ptn_key, ptn )
        self.assertEqual( rc, 0 )

        ts_key = ts_attr.key(ptn['ts'])
        rc = ts_idx.insert( ts_key, ptn )
        self.assertEqual( rc, 0 )

        # Check that it's back
        upd_ptn = ts_ptn_idx.find( ts_ptn_key )
        self.assertTrue( upd_ptn is not None )
        self.assertEqual( upd_ptn['ts'], ptn['ts'] )
        self.assertEqual( upd_ptn['ptn_id'], ptn['ptn_id'] )

        upd_ptn = ts_idx.find( ts_key )
        self.assertTrue( upd_ptn is not None )
        self.assertEqual( upd_ptn['ts'], ptn['ts'] )
        self.assertEqual( upd_ptn['ptn_id'], ptn['ptn_id'] )

        return upd_ptn

    def test_00_add_patterns(self):
        for count in range(0, 48 * 1000):
            self.__add_pattern()

    def test_01_update_pattern(self):
        duration = 10
        start = time.time()
        test_time = 0
        while test_time < duration:
            ptn = self.__update_pattern()
            test_time = time.time() - start

    def test_02_delete_patterns(self):
        # Generate a random sequency of pattern ids
        nda = np.arange(256, 1256, 1)
        nprnd.shuffle(nda)

        for i in range(0, len(nda)):
            ptn_id = int( nda[i] )

            # test that we can find it by ptn_id
            id_attr = self.schema['ptn_id']
            id_key = id_attr.key(ptn_id)
            ptn = id_attr.index().find(id_key)
            self.assertTrue( ptn is not None )
            ptn.index_del()
            ptn.delete()
            del ptn

    def test_03_update_add(self):
        duration = 10
        start = time.time()
        test_time = 0
        while test_time < duration:
            ptn = self.__update_pattern(assert_found = False)
            test_time = time.time() - start
            # Add another pattern to the mix
            self.__add_pattern()

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
