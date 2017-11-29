#!/usr/bin/env python
import unittest
import logging
import os
import sys

logger = logging.getLogger(__name__)

from schema_test import SchemaTest
from value_test_from_str import ValueTestFromStr
from obj_test_set_get import ObjTestSetGet
from join_test_u16 import JoinTestU16
from join_test_u32 import JoinTestU32
from join_test_u64 import JoinTestU64
from join_test_u32_str_str import JoinTestU32_Str_Str
from join_test_u32_str_u32 import JoinTestU32_Str_U32
from join_test_u32_str_str import JoinTestU32_Str_Str
from filter_test import FilterTestJoin3xU64
from pos_test import FilterPosTest
from filt_count_test import FilterCountTest

tests = [ SchemaTest,
          ObjTestSetGet,
          ValueTestFromStr,
          JoinTestU16,
          JoinTestU32,
          JoinTestU64,
          JoinTestU32_Str_Str,
          JoinTestU32_Str_U32,
          FilterTestJoin3xU64,
          FilterPosTest,
          FilterCountTest
          ]

if __name__ == "__main__":
    chkpath = os.getenv("TEST_DATA_DIR")
    if chkpath is None:
        print("Please set the TEST_DATA_DIR environment variable" \
              " to a valid path where test data can created.")
        sys.exit(1)

    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)

    sos_suite = unittest.TestSuite()
    for t in tests:
        sos_suite.addTest(unittest.makeSuite(t))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(sos_suite)
