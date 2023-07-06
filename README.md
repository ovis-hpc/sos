SOS - Scalable Object Store
===========================

SOS (pronuounced "sôs") -- Scalable Object Store -- is a high-performance,
indexed, object-oriented database designed to efficiently manage structured
data on persistent media.

SOS was created to solve performance and scalability problems found with
other time series databases such as InfluxDB, OpenTSDB, and Graphite.

SOS is strictly typed and uses schema to define the objects stored in the
database. The schema specifies the attributes that comprise the object
and which attributes are indexed.

SOS implements its own back-end storage model. This allows
SOS to support:

* Very high insert rates
* Superior query performance
* Flexible storage management


Configuration Options:
* --disable-python
  * The python commands for managing and querying SOS will not be build
* --enable-doc
  * Man pages will be generated for SOS commands and API
* --enable-html
  * HTML documenation will be generated for SOS commands and API

Compile Dependencies
--------------------

* If --enable-doc or --enable-html is specified
  * Doxygen

RPM Packages

* gcc
* make
* autotools
* libtool
* openssl-devel
* libjansson-devel
* bison
* flex
* uuid-devel
* libuuid-devel
* jansson-devel

pip3 Packagges

* Cython
* Pandas
* Numpy

Installation
------------

```sh
./autogen.sh # this will call autoreconf to generate `configure` script
mkdir build
cd build
../configure --prefix=/SOS/INSTALL/PATH [--disable-python] [--enable-debug] \
        [--enable-doc] [--enable-html]
# add 'PYTHON=/PYTHON/EXECUTABLE/PATH' if PYTHON environment variable not set
# add `--enable-debug` to turn on debugging logic inside the SOS libraries
# add `--disable-python` to disable the Python commands and interface libraries
make && make install
```
