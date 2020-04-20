#!/usr/bin/env python
from builtins import str
from test_idx_util import *

class TestHTBL(TestIndexBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.STORE_PATH = "./ht.store"
        cls.PART_NAME = "part"
        cls.SCHEMA_NAME = "schema"
        cls.IDX_TYPE = "HTBL"
        cls.IDX_ARG = ""
        super(TestHTBL, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestHTBL, cls).tearDownClass()

    def test_iter(self):
        for attr in self.schema:
            itr = sos.AttrIter(attr)
            data = set()
            itr.begin()
            for obj in SosIterWrap(itr):
                t = obj2tuple(obj)
                t = ( t[0], str(t[1]) )
                data.add(t)
            self.assertEqual(data, set(self.input_data))

    def test_iter_rev(self):
        for attr in self.schema:
            itr = sos.AttrIter(attr)
            data = set()
            itr.end()
            for obj in SosIterWrap(itr, rev=True):
                t = obj2tuple(obj)
                t = ( t[0], str(t[1]) )
                data.add(t)
            self.assertEqual(data, set(self.input_data))

    def test_iter_fwd_rev(self):
        # This test case is not applicable to HTBL
        pass

    def test_iter_begin(self):
        # HTBL is not ordered ... so we cannot really know what the first
        # element will be. At the least, we can test for consistency.
        for attr in self.schema:
            itr = sos.AttrIter(attr)
            itr.begin()
            obj = itr.item()
            obj2 = itr.item()
            self.assertEqual(obj2tuple(obj),obj2tuple(obj2))

    def test_iter_last(self):
        # HTBL is not ordered ... so we cannot really know what the first
        # element will be. At the least, we can test for consistency.
        for attr in self.schema:
            itr = sos.AttrIter(attr)
            itr.end()
            obj = itr.item()
            obj2 = itr.item()
            self.assertEqual(obj2tuple(obj),obj2tuple(obj2))

    def test_iter_inf(self):
        # This test case is not applicable to HTBL
        pass

    def test_iter_inf_exact(self):
        # This test case is not applicable to HTBL
        pass

    def test_iter_sup(self):
        # This test case is not applicable to HTBL
        pass

    def test_iter_sup_exact(self):
        # This test case is not applicable to HTBL
        pass


if __name__ == "__main__":
    LOGFMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=LOGFMT)
    logger.setLevel(logging.INFO)
    unittest.main()

