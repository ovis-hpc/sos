#!/usr/bin/env python
import unittest
import shutil
import logging
import os
import random
from sosdb import Sos

class Debug(object): pass

logger = logging.getLogger(__name__)

class ValFromStrTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.db = Sos.Container()
        self.path = "val_from_str_cont"
        shutil.rmtree(self.path, ignore_errors=True)
        self.db.create(self.path)
        self.db.open(self.path)
        self.db.part_create("ROOT")
        root = self.db.part_by_name("ROOT")
        root.state_set("PRIMARY")

    @classmethod
    def tearDownClass(self):
        self.db.close()
        del self.db
        # shutil.rmtree(self.path, ignore_errors=True)

    def test_add_objects(self):
        self.schema = Sos.Schema()
        self.schema.from_template('test_val_from_str', [
            { "name" : "INT16", "type" : "int16", "index" : {} },
            { "name" : "INT32", "type" : "int32" },
            { "name" : "INT64", "type" : "int64" },
            { "name" : "UINT16", "type" : "uint16" },
            { "name" : "UINT32", "type" : "uint32" },
            { "name" : "UINT64", "type" : "uint64" },
            { "name" : "FLOAT", "type" : "float" },
            { "name" : "DOUBLE", "type" : "double" },
            { "name" : "LONG_DOUBLE", "type" : "long_double" },
            { "name" : "TIMESTAMP", "type" : "timestamp" },
            { "name" : "STRUCT", "type" : "struct", "size" : 24 },
            { "name" : "BYTE_ARRAY", "type" : "byte_array" },
            { "name" : "CHAR_ARRAY", "type" : "char_array" },
            { "name" : "INT16_ARRAY", "type" : "int16_array" },
            { "name" : "INT32_ARRAY", "type" : "int32_array" },
            { "name" : "INT64_ARRAY", "type" : "int64_array" },
            { "name" : "UINT16_ARRAY", "type" : "uint16_array" },
            { "name" : "UINT32_ARRAY", "type" : "uint32_array" },
            { "name" : "UINT64_ARRAY", "type" : "uint64_array" },
            { "name" : "FLOAT_ARRAY", "type" : "float_array" },
            { "name" : "DOUBLE_ARRAY", "type" : "double_array" },
            { "name" : "LONG_DOUBLE_ARRAY", "type" : "long_double_array" }
        ])
        self.schema.add(self.db)

        for x in range(0, 100):
            o = self.schema.alloc()
            for a in self.schema:
                if a.attr_id() == 0:
                    o[0] = x
                    continue
                t = a.type()
                i = a.attr_id()
                if t == Sos.TYPE_CHAR_ARRAY or t == Sos.TYPE_STRUCT:
                    o[i] = str(random.random() * 10000)
                elif t == Sos.TYPE_BYTE_ARRAY:
                    o[i] = [ x for x in bytearray(str(int(random.random() * 10000))) ]
                elif t == Sos.TYPE_LONG_DOUBLE:
                    o[i] = 1234.1234
                elif a.is_array():
                    o[i] = [ random.random() * 10000 for r in range(0, 8) ]
                else:
                    o[i] = random.random() * 10000
            o.index_add()

if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    _pystart = os.environ.get("PYTHONSTARTUP")
    if _pystart:
        execfile(_pystart)
    unittest.TestLoader.testMethodPrefix = "test_"
    unittest.main()
