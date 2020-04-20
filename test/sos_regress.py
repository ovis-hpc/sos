#!/usr/bin/env python3
from __future__ import print_function
from past.builtins import execfile
import unittest
import logging
import os
import sys

logger = logging.getLogger(__name__)

from schema_test import SchemaTest
from query_test import QueryTest
from query_test2 import QueryTest2
from value_test_from_str import ValueTestFromStr
from obj_test_set_get import ObjTestSetGet
from index_test_min_max import IndexTestMinMax
from join_test_u16 import JoinTestU16
from join_test_u32 import JoinTestU32
from join_test_u64 import JoinTestU64
from join_test_i16 import JoinTestI16
from join_test_i32 import JoinTestI32
from join_test_i64 import JoinTestI64
from join_test_u32_str_str import JoinTestU32_Str_Str
from join_test_u32_str_u32 import JoinTestU32_Str_U32
from join_test_u32_str_str import JoinTestU32_Str_Str
from join_test_key import JoinTestKey
from join_test_getter import JoinTestGet
from filter_test import FilterTest
from test_filt_join_cond import FilterJoinCond
from test_gui_filt import GuiFilter
from pos_test import FilterPosTest
from filt_count_test import FilterCountTest
from delete_test import DeleteTest
from pattern_test import PatternTest
from h2htbl_test import H2HTBLTest
from key_test import KeyTest
from timestamp_test import TimestampTest
from array_test import ArrayTest
from version_test import VersionTest
from append_data_test import AppendDataTest
from update_test import UpdateTest

tests = [ SchemaTest,
          ObjTestSetGet,
          ValueTestFromStr,
          IndexTestMinMax,
          JoinTestU16,
          JoinTestU32,
          JoinTestU64,
          JoinTestI16,
          JoinTestI32,
          JoinTestI64,
          JoinTestU32_Str_Str,
          JoinTestU32_Str_U32,
          JoinTestKey,
          JoinTestGet,
          FilterTest,
          FilterPosTest,
          FilterCountTest,
          FilterJoinCond,
          GuiFilter,
          DeleteTest,
          PatternTest,
          H2HTBLTest,
          KeyTest,
          TimestampTest,
          ArrayTest,
          VersionTest,
          QueryTest,
          QueryTest2,
          AppendDataTest,
          UpdateTest,
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
