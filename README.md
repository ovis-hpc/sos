SOS - Scalable Object Store
===========================

SOS (pronuounced "sôs") -- Scalable Object Store -- is a high-performance,
indexed, object-oriented database designed to efficiently manage structured data
on persistent media.

A **schema** is a collection of named, typed attributes. An **object**
is an instance of attribute values conforming to the object **schema**.

A SOS **container** is a place where SOS users can insert, query, update
and iterate over collections of objects objects.

`sos_cmd` is the CLI program to:
* Create SOS containers.
* Add schema to the container.
* Add/import objects into the container.
* Query objects from the container.

Please see `sos_cmd -h` for more information about how to use `sos_cmd`.

For more details about SOS documentation, please see [SOS
doc](http://www.opengridcomputing.com/sos_doc/index.html), which also contains
SOS C API.


Compile Dependencies
--------------------

* For SOS python interface:
  * Cython >= .29 (Cython 3.0)
  * Python >= 3.6

Installation
------------

```sh
./autogen.sh # this will call autoreconf to generate `configure` script
mkdir build
cd build
../configure --prefix=/SOS/INSTALL/PATH [--disable-python] [--enable-debug]
# add 'PYTHON=/PYTHON/EXECUTABLE/PATH' if PYTHON environment variable not set
# add `--enable-debug` to turn on debugging logic inside the SOS libraries
# add `--disable-python` to disable the Python commands and interface libraries
make && make install
```
