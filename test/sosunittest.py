#!/usr/bin/env python3
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
    def backend(cls):
        return Sos.BE_MMOS

    @classmethod
    def session(cls):
        """Set up a D/SOS Session"""
        cls.session = Sos.Session("test_session.cfg")

    @classmethod
    def setUpDb(cls, db_name):
        global _sos_test_debug
        cls.db = Sos.Container()
        if cls.backend() == Sos.BE_MMOS:
            db_name = 'MMOS_' + db_name
        else:
            db_name = 'LSOS_' + db_name
        cls.db_name = db_name
        db_path = os.getenv("SOS_TEST_DATA_DIR")
        if db_path:
            cls.path = db_path + "/" + cls.db_name
        else:
            cls.path = cls.db_name
        shutil.rmtree(cls.path, ignore_errors=True)
        be = os.getenv("SOS_TEST_BACKEND")
        if be:
            if be == 'LSOS':
                be = Sos.BE_LSOS
            elif be == 'MMOS':
                be = Sos.BE_MMOS
            else:
                raise ValueError("Invalid setting {0} for SOS_TEST_BACKEND environent variable".format(be))
        else:
            be = cls.backend()

        cls.db.open(cls.path, create=True, backend=be)
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

