#!/usr/bin/env python3
import unittest
import shutil
import logging
import os
import random
from sosdb import Sos
from sosunittest import SosTestCase

logger = logging.getLogger(__name__)

class VersionTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb("vers_test_cont")

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def test_00_version(self):
        vers = self.db.version()
        self.assertEqual(vers['major'], Sos.VERS_MAJOR)
        self.assertEqual(vers['minor'], Sos.VERS_MINOR)
        self.assertEqual(vers['fix'], Sos.VERS_FIX)
        self.assertEqual(vers['git_commit_id'], Sos.GIT_COMMIT_ID)

class LsosVersionTest(VersionTest):
    @classmethod
    def backend(cls):
        return Sos.BE_LSOS

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
