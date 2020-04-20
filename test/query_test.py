#!/usr/bin/env python
from __future__ import print_function
from past.builtins import execfile
from builtins import next
from builtins import range
from builtins import object
import unittest
import logging
import random
import numpy
import os
from sosdb import Sos
from sosunittest import SosTestCase, Dprint
import datetime as dt

class Debug(object): pass

logger = logging.getLogger(__name__)

class QueryTest(SosTestCase):
    @classmethod
    def setUpClass(cls):
        cls.setUpDb('query_test_cont')
        cls.schema1 = Sos.Schema()
        cls.schema1.from_template('query_test',
                                 [
                                 { "name" : "int16", "type" : "int16", "index" : {} },
                                 { "name" : "int32", "type" : "int32", "index" : {} },
                                 { "name" : "int64", "type" : "int64", "index" : {} },
                                 { "name" : "uint16", "type" : "uint16", "index" : {} },
                                 { "name" : "uint32", "type" : "uint32", "index" : {} },
                                 { "name" : "uint64", "type" : "uint64", "index" : {} },
                                 { "name" : "float", "type" : "float", "index" : {} },
                                 { "name" : "double", "type" : "double", "index" : {} },
                                 { "name" : "timestamp", "type" : "timestamp", "index" : {} },
                                 { "name" : "string", "type" : "char_array", "index" : {} },
                                 { "name" : "byte_array", "type" : "byte_array", "index" : {} },
                                 { "name" : "int16_array", "type" : "int16_array", "index" : {} },
                                 { "name" : "int32_array", "type" : "int32_array", "index" : {} },
                                 { "name" : "int64_array", "type" : "int64_array", "index" : {} },
                                 { "name" : "uint16_array", "type" : "uint16_array", "index" : {} },
                                 { "name" : "uint32_array", "type" : "uint32_array", "index" : {} },
                                 { "name" : "uint64_array", "type" : "uint64_array", "index" : {} },
                                 { "name" : "float_array", "type" : "float_array", "index" : {} },
                                 { "name" : "double_array", "type" : "double_array", "index" : {} },
                                 { "name" : "int32_no_idx", "type" : "int32" },
                                 { "name" : "int16_int32_", "type" : "join",
                                   "join_attrs" : [ "int16", "int32" ], "index" : {} },
                                 { "name" : "int16_int32_int64", "type" : "join",
                                   "join_attrs" : [ "int16", "int32", "int64" ], "index" : {} }
                               ])
        cls.schema1.add(cls.db)
        cls.schema2 = Sos.Schema()
        cls.schema2.from_template('query_test_2',
                                 [
                                 { "name" : "int32_2", "type" : "int32", "index" : {} },
                                 { "name" : "timestamp", "type" : "timestamp", "index" : {} },
                               ])
        cls.schema2.add(cls.db)
        cls.query = Sos.Query(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tearDownDb()

    def __generate_data(self, schema, min_v, max_v):
        for i in range(min_v, max_v):
            t = dt.datetime.now()
            obj = schema.alloc()
            obj[:] = (
                -i, -2*i, -3*i,
                i, 2*i, 3*i,
                random.random(), random.random(),
                (t.second, t.microsecond),
                "{0}".format(i), bytearray("{0}".format(i), encoding='utf-8'),
                [ -i, -i, -i ], [ -i, -i, -i ], [ -i, -i, -i ],
                [ i, i, i ], [ i, i, i ], [ i, i, i ],
                [ random.random(), random.random(), random.random() ],
                [ random.random(), random.random(), random.random() ]
            )
            rc = obj.index_add()
            self.assertEqual( rc, 0 )

    def checkit(self, attr_name, min_v, max_v):
        # iterate forward, checking that everything the Query iterator returns
        # is also found by traversing with a non-filtered attribute iter
        o  = self.query.begin()
        it = self.schema1.attr_by_name(attr_name).attr_iter()
        rc = it.begin()
        next_count = 0
        while o:
            next_count += 1
            v = o[0]
            if type(v) == numpy.ndarray:
                v = v.tolist()
            Dprint("{0} >= {1}".format(v, min_v))
            Dprint("{0} <= {1}".format(v, max_v))
            self.assertTrue( v >= min_v )
            self.assertTrue( v <= max_v )
            foundit = False
            while rc:
                o2 = it.item()
                v2 = o2[attr_name]
                if type(v2) == numpy.ndarray:
                    v2 = v2.tolist()
                Dprint("checking {0}".format(v2))
                if v2 < v:
                    rc = next(it)
                else:
                    self.assertTrue( v == v2 )
                    foundit = True
                    next(it)
                    break
            self.assertTrue( foundit )
            o = next(self.query)
        self.assertTrue( next_count > 0 )

        # iterate backwards, the count should be the same
        o  = self.query.end()
        rc = it.end()
        prev_count = 0
        while o:
            prev_count += 1
            v = o[0]
            if type(v) == numpy.ndarray:
                v = v.tolist()
            Dprint("{0} >= {1}".format(v, min_v))
            Dprint("{0} <= {1}".format(v, max_v))
            self.assertTrue( v >= min_v )
            self.assertTrue( v <= max_v )
            foundit = False
            while rc:
                o2 = it.item()
                v2 = o2[attr_name]
                if type(v2) == numpy.ndarray:
                    v2 = v2.tolist()
                Dprint("checking {0}".format(v2))
                rc = it.prev()
                if v2 > v:
                    continue
                else:
                    self.assertTrue( v == v2 )
                    foundit = True
                    break
            self.assertTrue( foundit )
            o = self.query.prev()

        self.assertEqual(next_count, prev_count)

    def __test_next_prev(self, attr_name, min_v, max_v):
        self.query.select([attr_name],
                          from_ = ['query_test'],
                          where = [(attr_name, Sos.COND_GE, min_v), (attr_name, Sos.COND_LE, max_v)],
                          order_by = attr_name)
        self.checkit(attr_name, min_v, max_v)

    def test_000_add_data(self):
        self.__generate_data(self.schema1, 0, 1024)

    def test_001_next_prev_int16(self):
        self.__test_next_prev("int16", -600, -500)

    def test_002_next_prev_uint16(self):
        self.__test_next_prev("uint16", 500, 600)

    def test_003_next_prev_int32(self):
        self.__test_next_prev("int32", -600, -500)

    def test_004_next_prev_uint32(self):
        self.__test_next_prev("uint32", 500, 600)

    def test_005_next_prev_int64(self):
        self.__test_next_prev("int64", -600, -500)

    def test_006_next_prev_uint64(self):
        self.__test_next_prev("uint64", 500, 600)

    def test_007_next_prev_float(self):
        self.__test_next_prev("float", 0, 1)

    def test_008_next_prev_double(self):
        self.__test_next_prev("double", 0, 1)

    def test_009_next_prev_int16_array(self):
        self.__test_next_prev("int16_array", [-600, -600, -600], [-500, -500, -500])

    def test_010_next_prev_int32_array(self):
        self.__test_next_prev("int32_array", [-600, -600, -600], [-500, -500, -500])

    def test_011_next_prev_int64_array(self):
        self.__test_next_prev("int64_array", [-600, -600, -600], [-500, -500, -500])

    def test_012_next_prev_uint16_array(self):
        self.__test_next_prev("uint16_array", [500, 500, 500], [600, 600, 600])

    def test_013_next_prev_uint32_array(self):
        self.__test_next_prev("uint32_array", [500, 500, 500], [600, 600, 600])

    def test_014_next_prev_uint64_array(self):
        self.__test_next_prev("uint64_array", [500, 500, 500], [600, 600, 600])

    def test_015_next_prev_float_array(self):
        self.__test_next_prev("float_array", [0, 0, 0], [1, 1, 1])

    def test_016_next_prev_double_array(self):
        self.__test_next_prev("double_array", [0, 0, 0], [1, 1, 1])

    def __count(self, query):
        count = 0
        o = query.begin()
        while o:
            count = count + 1
            o = next(query)
        return count

    def test_050_unique(self):
        self.query.select(['uint16'],
                          from_ = ['query_test'],
                          where = [('uint16', Sos.COND_GE, 1)],
                          order_by = 'uint16')
        self.assertTrue( self.__count(self.query) == 1023 )
        self.query.select(['uint16'],
                          from_ = ['query_test'],
                          where = [('uint16', Sos.COND_GE, 1)],
                          order_by = 'uint16',
                          unique = True)
        self.assertTrue( self.__count(self.query) == 1023 )
        # now add another 1024 objects
        self.__generate_data(self.schema1, 0, 1024)
        self.query.select(['uint16'],
                          from_ = ['query_test'],
                          where = [('uint16', Sos.COND_GE, 1)],
                          order_by = 'uint16')
        self.assertTrue( self.__count(self.query) == 2046 )
        self.query.select(['uint16'],
                          from_ = ['query_test'],
                          where = [('uint16', Sos.COND_GE, 1)],
                          order_by = 'uint16',
                          unique = True)
        self.assertTrue( self.__count(self.query) == 1023 )

    def test_100_order_by(self):
        # order_by attr_name where attr_name does not exist in any schema
        try:
            self.query.select(['int32'],
                              order_by = 'does_not_exist')
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_101_order_by(self):
        # order_by attr_name where attr_name exists but is not indexed
        try:
            self.query.select(['int32'],
                              order_by = 'int32_no_idx')
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_102_order_by(self):
        # test that the order_by attribute must exist in all schema referenced by the colspecs
        try:
            self.query.select(['int32','int32_2'],
                              order_by = 'int32')
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_103_order_by(self):
        # test that an omitted order_by doesn't use the previous one
        self.query.select(['int32'],
                          order_by = 'int32')
        self.query.select(['int32_2'],
                          from_ = ['query_test_2'])

    # test unqualified attribute-naming "attr_name" errors

    def test_104_order_by(self):
        # test longest join match
        self.query.select(['int32'],
                          order_by = 'int16_int32')
        self.assertEqual(self.query.index_name, "int16_int32_int64")

    # test unqualified attribute-naming "attr_name" errors

    def test_200_attr_name(self):
        # from_ clause present: attr_name does not exist in any schema in from_
        try:
            self.query.select(['does_not_exist'],
                              from_ = ['query_test'])
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_201_attr_name(self):
        # from_ clause absent: attr_name does not exist in any schema in the test container
        try:
            self.query.select(['does_not_exist'])
        except ValueError:
            pass
        else:
            self.assertTrue( False )
        # these exist and should not generate exceptions
        try:
            self.query.select(['int32_2'])
        except Exception as e:
            print("{0}".format(e))
        self.query.select(['int32'])

    # test qualified attribute-naming "schema_name.attr_name" errors

    def test_202_attr_name(self):
        # schema_name in colspec does not exist
        try:
            self.query.select(['does_not_exist.int32'])
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_203_attr_name(self):
        # attr_name in colspec does not exist in schema_name
        try:
            self.query.select(['query_test.does_not_exist'])
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    # test from_ errors

    def test_300_from(self):
        # from_ clause with non-existent schema
        try:
            self.query.select(['int32'],
                              from_ = [ 'does_not_exist' ])
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_301_from(self):
        # regression for a bug where an omitted from_ clauses used the last
        # one specified in a previous select()
        self.query.select(['int32'],
                          from_ = ['query_test'])
        # this should not raise an exception, because int32_2 is found by searching all schema
        self.query.select(['int32_2'])

    def test_302_from(self):
        # from_ clause with bad from_ argument type
        try:
            self.query.select(['int32'],
                              from_ = 'query_test')
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    # test colspec errors

    def test_400_colspec(self):
        # * without a from_ clause
        try:
            self.query.select(['*'])
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_401_colspec(self):
        # * along with a non-*
        try:
            self.query.select(['*', 'int32'])
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_402_colspec(self):
        # * along with another *
        try:
            self.query.select(['*', '*'])
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    def test_403_colspec(self):
        # schema_name.* where schema_name does not exist
        try:
            self.query.select(['does_not_exist.*'])
        except ValueError:
            pass
        else:
            self.assertTrue( False )

    # test where-clause errors

    def test_500_where(self):
        # malformed where condition (must be a 3-tuple)
        try:
            self.query.select(['int32'],
                              where = [('int32')])
        except:
            pass
        else:
            self.assertTrue( False )
        try:
            self.query.select(['int32'],
                              where = [('int32', Sos.COND_EQ)])
        except:
            pass
        else:
            self.assertTrue( False )
        try:
            self.query.select(['int32'],
                              where = [('int32', Sos.COND_EQ, 1), ('int64')])
        except:
            pass
        else:
            self.assertTrue( False )

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
