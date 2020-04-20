#!/usr/bin/env python
from past.builtins import execfile
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
import sys
class Debug(object): pass

logger = logging.getLogger(__name__)

verify = []
next_tkn_id = 256

class H2HTBLTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("h2htbl_cont")
        cls.schema = Sos.Schema()
        cls.schema.from_template('test_h2htbl',
                             [ { "name" : "tkn_id", "type" : "uint64",
                                 "index" : { "type" : "H2HTBL" } },
                               { "name" : "tkn_count", "type" : "uint64", },
                               { "name" : "tkn_text", "type" : "char_array",
                                 "index" : { "type" : "h2htbl" } },
                           ])
        cls.schema.add(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()
        pass

    def test_00_add_tokens(self):
        global next_tkn_id
        #f = open(os.path.dirname(sys.argv[0]) + "/eng-dictionary", "r")
        f = open("eng-dictionary", "r")
        for word in f:
            tkn_text = word.rstrip()
            if next_tkn_id % 300 == 0:
                verify.append(tkn_text)
            tkn = self.schema.alloc()
            tkn[:] = ( next_tkn_id, 1, tkn_text )
            rc = tkn.index_add()
            self.assertEqual(rc, 0)
            next_tkn_id += 1

    def test_01_verify_find(self):
        tkn_text_attr = self.schema['tkn_text']
        idx = tkn_text_attr.index()
        for tkn_text in verify:
            k = tkn_text_attr.key(tkn_text)
            tkn = idx.find(k)
            tkn['tkn_count'] = 2
            self.assertTrue( tkn is not None )
            self.assertEqual( tkn_text, tkn['tkn_text'] )

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
