#!/usr/bin/python
from __future__ import print_function
from builtins import next
from sosdb import Sos
from shutil import rmtree
from os import chdir
from sostestlib import ROOT_DIR, TestContext
import struct
import errno

def print_obj(obj):
    print("key : {0}  order : {1}".format(obj['key'], obj['order']))

# Destroy previous test data if present
try:
    chdir(ROOT_DIR)
    rmtree('iterator', ignore_errors = True)
except Exception as e:
    print(e)
    pass

# Open the test container
db = Sos.Container()
db.create('iterator')
db.open('iterator')

# Add a partition
db.part_create("ROOT")
part = db.part_by_name("ROOT")
part.state_set("primary")

# Create the test schema
template = [
    { "name" : "key", "type" : "uint64", "index" : {} },
    { "name" : "order", "type" : "uint64" }
]
schema = Sos.Schema()
schema.from_template("inf_sup", template)

# Add the schema to the container
schema.add(db)

# Add data to test the inf/sup functions
data = [
    [ 1,  1 ],
    [ 1,  2 ],
    [ 1,  3 ],
    [ 2,  4 ],
    [ 2,  5 ],
    [ 2,  6 ],
    [ 4,  7 ],
    [ 4,  8 ],
    [ 4,  9 ],
    [ 5, 10 ],
    [ 5, 11 ],
    [ 5, 12 ]
]

for entry in data:
    obj = schema.alloc()
    obj[0] = entry[0]
    obj[1] = entry[1]
    obj.index_add()

tests = TestContext(desc="AttrIter Unit Tests", detail=True)

key_attr = schema.attr_by_name('key')
it = key_attr.attr_iter()
key = Sos.Key(attr=key_attr)

key.set_value(0)
res = it.find(key)
tests.test('if key does not exist, then find(key) returns False', (res == False))
o = it.item()
tests.test('if find(key) is False, then it.item() returns None', (o == None))
k = it.key()
tests.test('if find(key) is False, then it.key() returns None', (k == None))

key.set_value(0)
res = it.find_inf(key)
tests.test('if key < min, then find_inf(key) returns False', (res == False))
o = it.item()
tests.test('if find_inf(key) is False, then it.item() returns None', (o == None))
k = it.key()
tests.test('if find_inf(key) is False, then it.key() returns None', (k == None))

key.set_value(1)
it.find_inf(key)
inf = it.item()
it.find_sup(key)
sup = it.item()
tests.test('if key 1 exists, then order of inf(1) <= sup(1)',
          inf['order'] <= sup['order'])

key.set_value(3)
it.find_inf(key)
inf = it.item()
it.find_sup(key)
sup = it.item()

tests.test('infinum and supremum correctly identified: inf(3) == 2 and sup(3) == 4',
           (inf['key'] == 2 and sup['key'] == 4))
tests.test('infinum and supremem return first dup',
           (inf['order'] == 4 and sup['order'] == 7))

key.set_value(0)
it.find_inf(key)
inf = it.item()
it.find_sup(key)
sup = it.item()
minkey = int(key_attr.min())

tests.test('infinum where key < min is None', (inf == None))
tests.test('supremem where key < min == min', (sup['key'] == minkey))
tests.test('order of supremum is 1 when key < min', (sup['order'] == 1))

key.set_value(6)
it.find_inf(key)
inf = it.item()
it.find_sup(key)
sup = it.item()
maxkey = int(key_attr.max())

tests.test('infinum where key > max == max', (inf['key'] == maxkey))
tests.test('order of infinum is first dup (10) when key > max', (inf['order'] == 10))
tests.test('supremem where key > max is None', (sup == None))

rc = it.prop_set("inf_last_dup", True)
rc = it.prop_set("sup_last_dup", True)
key.set_value(3)
rc = it.find_inf(key)
inf = it.item()
tests.test('find_inf with inf_last_dup==True should return last duplicate',
           (inf['order'] == 6))
rc = it.find_sup(key)
sup = it.item()
tests.test('find_sup with sup_last_dup==True should return last duplicate',
           (sup['order'] == 9))

# Iterate unique forward
rc = it.prop_set("unique", True)
rc = it.begin()
last_key = 0
while rc:
    o = it.item()
    tests.test('Iterator returns only the unique keys', (last_key != o['key']))
    rc = next(it)

# Iterate unique backward
rc = it.end()
last_key = 0
while rc:
    o = it.item()
    tests.test('Iterator returns only the unique keys', (last_key != o['key']))
    rc = it.prev()

# Iterate forward
rc = it.prop_set("unique", False)
rc = it.begin()
exp_order = 1
while rc:
    o = it.item()
    tests.test('Iterator returns next object with order {0}'.format(exp_order),
               (exp_order == o['order']))
    exp_order += 1
    rc = next(it)

# Iterate backward
rc = it.end()
exp_order = 12
while rc:
    o = it.item()
    tests.test('Iterator returns prev object with order {0}'.format(exp_order),
               (exp_order == o['order']))
    exp_order -= 1
    rc = it.prev()

# get_pos() tests
key.set_value(2)
rc = it.find(key)
pos = it.get_pos()
o1 = it.item()
tests.test('when iterator is positioned at an entry, get_pos() returns a position string', pos is not None)

key.set_value(3)
rc = it.find(key)
nonepos = it.get_pos()
tests.test('when iterator is NOT positioned at an entry, get_pos() returns None', nonepos is None)

key.set_value(5)
rc = it.find(key)
rc = it.set_pos(pos)
tests.test('The position was found and set_pos() returns success', rc == 0)
o2 = it.item()
tests.test('set_pos() sets the iterator to correct position', o1['order'] == o2['order'])
rc = it.set_pos(pos)
tests.test('Iterator positions are single use, calling set_pos() a second time will return ENOENT',
           rc == errno.ENOENT)

tests.summarize()
