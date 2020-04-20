from __future__ import division
from __future__ import print_function
from builtins import object

ROOT_DIR = "."

class TestContext(object):
    def __init__(self, desc=None, detail=False):
        self.fail_count = 0
        self.pass_count = 0
        self.verbose = detail
        self.desc = desc

    def summarize(self):
        if self.desc:
            print("{0}".format(self.desc.center(80, ' ')))
            sep = '-'.ljust(len(self.desc), '-')
            print("{0}\n".format(sep.center(80, ' ')))
        total = self.pass_count + self.fail_count
        print("TOTAL TESTS : {0:>4}".format(total))
        print("PASSED      : {0:>4} (%{1:0.2f})".format(self.pass_count, (self.pass_count / total) * 100))
        print("FAILED      : {0:>4} (%{1:0.2f})".format(self.fail_count, (self.fail_count / total) * 100))

    def test(self, assertion, cond):
        if cond:
            if self.verbose:
                print("PASSED : {0}".format(assertion))
            self.pass_count += 1
        else:
            if self.verbose:
                print("FAILED : {0}".format(assertion))
            self.fail_count += 1


