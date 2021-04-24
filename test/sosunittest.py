#!/usr/bin/env python3
from __future__ import print_function
import unittest
import shutil
import logging
import os
from sosdb import Sos

_sos_test_debug = False

def Dprint(*args):
    global _sos_test_debug
    if _sos_test_debug:
        print(*args)

def DprintEnable():
    global _sos_test_debug
    _sos_test_debug = True

def DprintDisable():
    global _sos_test_debug
    _sos_test_debug = False

class SosTestCase(unittest.TestCase):
    @classmethod
    def setUpDb(cls, db_name):
        global _sos_test_debug
        cls.db = Sos.Container()
        cls.db_name = db_name
        db_path = os.getenv("SOS_TEST_DATA_DIR")
        if db_path:
            cls.path = db_path + "/" + cls.db_name
        else:
            cls.path = cls.db_name
        shutil.rmtree(cls.path, ignore_errors=True)
        cls.db.create(cls.path)
        cls.db.open(cls.path)
        if os.environ.get("SOS_TEST_DEBUG"):
            _sos_test_debug = True

    @classmethod
    def tearDownDb(cls):
        if cls.db:
            cls.db.close()
            del cls.db
            cls.db = None
        if os.environ.get("SOS_TEST_KEEP_DB") is None:
            shutil.rmtree(cls.path, ignore_errors=True)

