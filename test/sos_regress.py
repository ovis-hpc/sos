#!/usr/bin/env python3
import unittest
import logging
import os
import sys

logger = logging.getLogger(__name__)

from schema_test import SchemaTest, LsosSchemaTest
from query_test import QueryTest, LsosQueryTest
from query_test2 import QueryTest2, LsosQueryTest2
from value_test_from_str import ValueTestFromStr, LsosValueTestFromStr
from obj_test_set_get import ObjTestSetGet, LsosObjTestSetGet
from index_test_min_max import IndexTestMinMax, LsosIndexTestMinMax
from join_test_u16 import JoinTestU16, LsosJoinTestU16
from join_test_u32 import JoinTestU32, LsosJoinTestU32
from join_test_u64 import JoinTestU64, LsosJoinTestU64
from join_test_i16 import JoinTestI16, LsosJoinTestI16
from join_test_i32 import JoinTestI32, LsosJoinTestI32
from join_test_i64 import JoinTestI64, LsosJoinTestI64
from join_test_u32_str_str import JoinTestU32_Str_Str, LsosJoinTestU32_Str_Str
from join_test_u32_str_u32 import JoinTestU32_Str_U32, LsosJoinTestU32_Str_U32
from join_test_u32_str_str import JoinTestU32_Str_Str, LsosJoinTestU32_Str_Str
from join_test_key import JoinTestKey, LsosJoinTestKey
from join_test_getter import JoinTestGet, LsosJoinTestGet
from filter_test import FilterTest, LsosFilterTest
from test_filt_join_cond import FilterJoinCond, LsosFilterJoinCond
from test_gui_filt import GuiFilter, LsosGuiFilter
from filt_count_test import FilterCountTest, LsosFilterCountTest
from delete_test import DeleteTest, LsosDeleteTest
from pattern_test import PatternTest, LsosPatternTest
from h2htbl_test import H2HTBLTest, LsosH2HTBLTest
from key_test import KeyTest, LsosKeyTest
from timestamp_test import TimestampTest, LsosTimestampTest
from array_test import ArrayTest, LsosArrayTest
from version_test import VersionTest, LsosVersionTest
from append_data_test import AppendDataTest, LsosAppendDataTest
from update_test import UpdateTest, LsosUpdateTest
from partition_test import PartitionTest, LsosPartitionTest

tests = [
          SchemaTest,
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
          PartitionTest,
          LsosSchemaTest,
          LsosObjTestSetGet,
          LsosValueTestFromStr,
          LsosIndexTestMinMax,
          LsosJoinTestU16,
          LsosJoinTestU32,
          LsosJoinTestU64,
          LsosJoinTestI16,
          LsosJoinTestI32,
          LsosJoinTestI64,
          LsosJoinTestU32_Str_Str,
          LsosJoinTestU32_Str_U32,
          LsosJoinTestKey,
          LsosJoinTestGet,
          LsosFilterTest,
          LsosFilterCountTest,
          LsosFilterJoinCond,
          LsosGuiFilter,
          LsosDeleteTest,
          LsosPatternTest,
          LsosH2HTBLTest,
          LsosKeyTest,
          LsosTimestampTest,
          LsosArrayTest,
          LsosVersionTest,
          LsosQueryTest,
          LsosQueryTest2,
          LsosAppendDataTest,
          LsosUpdateTest,
          LsosPartitionTest,
          ]


if __name__ == "__main__":
    from sosdb import Sos
    chkpath = os.getenv("SOS_TEST_DATA_DIR")
    if chkpath is None:
        print("Please set the SOS_TEST_DATA_DIR environment variable" \
              " to a valid path where test data can created.")
        sys.exit(1)

    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    sos_suite = unittest.TestSuite()
    print(len(tests))
    for t in tests:
        sos_suite.addTest(unittest.makeSuite(t))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(sos_suite)
