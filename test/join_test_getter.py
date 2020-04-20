#!/usr/bin/env python
from past.builtins import execfile
from builtins import next
from builtins import str
from builtins import object
import unittest
import shutil
import logging
import os
from sosdb import Sos
from sosunittest import SosTestCase

class Debug(object): pass

logger = logging.getLogger(__name__)

col_1_arg = ("A-two", "B-three", "C-four", "D-five")

class JoinTestGet(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("join_test_get_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_u32_str_u32',
                             [ { "name" : "a_1", "type" : "uint32" },
                               { "name" : "a_2", "type" : "string" },
                               { "name" : "a_3", "type" : "uint32" },
                               { "name" : "a_join", "type" : "join",
                                 "join_attrs" : [ "a_1", "a_2", "a_3" ],
                                 "index" : {}}
                           ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_add_objects(self):
        data = [ (1, "A-two", 3), (2, "B-three", 4), (3, "C-four", 5) ]
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

    def test_join_get(self):
        a_join = self.schema.attr_by_name('a_join')
        f = a_join.filter()
        o = f.begin()
        while o:
            self.assertEqual(o['a_join'],
                             str(o['a_1']) + ':' +
                             str(o['a_2']) + ':' +
                             str(o['a_3']))
            o = next(f)

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
