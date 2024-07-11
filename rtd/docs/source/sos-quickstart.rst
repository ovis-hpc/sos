SOS Quick Start
###########################

Introduction
*****************
SOS (pronuounced "sôs"), standing for Scalable Object Store, is a high-performance, indexed, object-oriented database designed to efficiently manage structured data on persistent media.

SOS was created to solve performance and scalability problems found with other time series databases such as InfluxDB, OpenTSDB, and Graphite.

SOS is strictly typed and uses schema to define the objects stored in the database. The schema specifies the attributes that comprise the object and which attributes are indexed.

SOS implements its own back-end storage model. This allows SOS to support:

* Very high insert rates
* Superior query performance
* Flexible storage management

Configuration Options
**********************

* --disable-python
       * The python commands for managing and querying SOS will not be build
* --enable-doc
       * Man pages will be generated for SOS commands and API
* --enable-html
       * HTML documenation will be generated for SOS commands and API

Compile Dependencies
********************

* If --disable-python is not specified
        * Cython >= .29 (Cython 3.0)
        * Python >= 3.6

* If --enable-doc or --enable-html is specified
        * Doxygen

Installation
****************
The following will build SOS and numsos into the directory /home/XXX/BuildSos

cd into the top level sos checkout directory

.. code-block:: console

    ./autogen.sh # this will call autoreconf to generate `configure` script
    mkdir build
    cd build
    ../configure --prefix=/home/XXX/BuildSos [--disable-python] [--enable-debug] \
        [--enable-doc] [--enable-html]
    # add 'PYTHON=/PYTHON/EXECUTABLE/PATH' if PYTHON environment variable not set
    # add `--enable-debug` to turn on debugging logic inside the SOS libraries
    # add `--disable-python` to disable the Python commands and interface libraries
    make && make install


The build will result in /home/XXX/BuildSos/lib/python3.X/site-packages with sosdb and numsos modules. The sosdb module includes the DataSet class and also the Array and Sos modules, which are written in C for efficiency. The numsos module includes the DataSource, DataSink, Stack, and Transform classes.

Set the environment variables appropriately using: 

.. code-block:: console

  export PATH=/home/XXX/BuildSos/bin:$PATH
  export PYTHONPATH=/home/XXX/BuildSos/lib/python3.X/site-packages:$PYTHONPATH

Importing a CSV file and using the command line tools
*********************************

.. list-table:: CSV and Formatting Files

    * - File
      - Use With
      - Description
    * - meminfo_qs.schema.json
      - sos-schema --add
      - A schema definition file
    * - meminfo_qs.map.json
      - sos-import-csv --map 	
      - A file that tells the import tool which CSV columns go to which schema attributes
    * - meminfo_qs.csv
      - sos-import-csv --csv 	
      - 1000 lines of CSV meminfo data 

These files can be obtained from a clone of the wiki under the quickstart directory in the top level of the sos repo.

.. code-block:: console

    > more meminfo_qs.schema.json
    {
    "name" : "meminfo_qs",
    "uuid": "33333333-3333-3333-3333-333333333333",
    "attrs" : [
        { "name" : "timestamp", "type" : "uint64" : "char_array",  "index" : {}  },
        { "name" : "component_id",      "type" : "char_array",  "index" : {}  },
        { "name" : "job_id",    "type" : "char_array",  "index" : {}  },
        { "name" : "app_id",    "type" : "uint64" },
        { "name" : "MemTotal",  "type" : "uint64" },
        { "name" : "MemFree",   "type" : "uint64" },
        ...
        { "name" : "DirectMap2M",       "type" : "uint64" },
        { "name" : "DirectMap1G",       "type" : "uint64" },
        { "name" : "time_job_comp", "type" : "join", "join_attrs" : [ "timestamp", "job_id", "component_id"],
        "index" : {} },
        { "name" : "time_comp_job", "type" : "join", "join_attrs" : [ "timestamp", "component_id", "job_id"],
        "index" : {} },
        { "name" : "job_comp_time", "type" : "join", "join_attrs" : [ "job_id", "component_id", "timestamp" ],
           "index" : {} },
        { "name" : "job_time_comp", "type" : "join", "join_attrs" : [ "job_id", "timestamp", "component_id" ],
           "index" : {} },
        { "name" : "comp_time_job", "type" : "join", "join_attrs" : [ "component_id", "timestamp", "job_id"],
        "index" : {} },
        { "name" : "comp_job_time", "type" : "join", "join_attrs" : [ "component_id", "job_id", "timestamp" ],
           "index" : {} }
         ]
     }
     > more meminfo_qs.map.json
     [
        { "target" : "timestamp", "source" : { "column" : 0 } },
        { "target" : "component_id", "source" : { "column" : 1 } },
        { "target" : "job_id", "source" : { "column" : 2 } },
        { "target" : "app_id", "source" : { "column" :  3 } },
        { "target" : "MemTotal", "source" : { "column" : 4 } },
        { "target" : "MemFree", "source" : { "column" : 5 } },
        ...
        { "target" : "DirectMap2M", "source" : { "column" : 49 } },
        { "target" : "DirectMap1G", "source" : { "column" : 50 } }
     ] ]
     >  more meminfo_qs.csv
     1703108908.000677,2448900245962755385,17165443304811230558,0.0,131928928.0...
     1703108908.000705,3501119766665329829,17326355104910386333,0.0,131928928.0...

Creating a SOS container

1. Create a container if you don't already have one:

.. code-block:: console

 > sos-db --path /dir/my-container --create

Adding a schema to a container

2. Add the schema to the container:

.. code-block:: console

 > sos-schema --path /dir/my-container --add meminfo_qs.schema.json

Querying for schema information

3. Query the schema to see what's in it:

a. Using sos-schema:

.. code-block:: console

 > sos-schema --path /dir/my-container --query --verbose
 meminfo_qs
 Id   Type             Indexed      Name                            
 ---- ---------------- ------------ --------------------------------
   0 TIMESTAMP        True         timestamp
   1 UINT64           True         component_id
   2 UINT64           True         job_id
   3 UINT64                        app_id
   4 UINT64                        MemTotal
   5 UINT64                        MemFree
  ...
  49 UINT64                                DirectMap2M
  50 UINT64                                DirectMap1G
  51 JOIN                     True         time_job_comp [timestamp+job_id+component_id]
  52 JOIN                     True         time_comp_job [timestamp+component_id+job_id]
  53 JOIN                     True         job_comp_time [job_id+component_id+timestamp]
  54 JOIN                     True         job_time_comp [job_id+timestamp+component_id]
  55 JOIN                     True         comp_time_job [component_id+timestamp+job_id]
  56 JOIN                     True         comp_job_time [component_id+job_id+timestamp]

b. OR using sos_cmd:

.. code-block:: console

 > sos_cmd -C /dir/my-container -l
 schema :
    name      : meminfo_qs
    schema_sz : 16728
    gen       : 0
    obj_sz    : 142
    uuid      : 33333333-3333-3333-3333-333333333333
    -attribute : timestamp
        type          : TIMESTAMP
        idx           : 0
        indexed       : 1
        offset        : 16
    -attribute : component_id
        type          : CHAR_ARRAY
        idx           : 1
        indexed       : 1
        offset        : 24
    -attribute : job_id
        type          : CHAR_ARRAY
        idx           : 2
        indexed       : 1
        offset        : 32
    ...
    -attribute : DirectMap2M
        type          : UINT16
        idx           : 49
        indexed       : 0
        offset        : 138
    -attribute : DirectMap1G
        type          : UINT16
        idx           : 50
        indexed       : 0
        offset        : 140
    -attribute : time_job_comp
        type          : JOIN
        idx           : 51
        indexed       : 1
        offset        : 142
    -attribute : time_comp_job
        type          : JOIN
        idx           : 52
        indexed       : 1
        offset        : 142
    -attribute : job_comp_time
        type          : JOIN
        idx           : 53
        indexed       : 1
        offset        : 142
    -attribute : job_time_comp
        type          : JOIN
        idx           : 54
        indexed       : 1
        offset        : 142
    -attribute : comp_time_job
        type          : JOIN
        idx           : 55
        indexed       : 1
        offset        : 142
    -attribute : comp_job_time
        type          : JOIN
        idx           : 56
        indexed       : 1
        offset        : 142

Note that there is no data yet in the container (using sos_cmd):

.. code-block:: console

 > sos_cmd -C /dir/my-container -q -S meminfo_qs -X time_job_comp
 ...
 -------------------------------- ------------------  ... -------------------------------- 
 Records 0/0.

Importing CSV data into a container

4. Import the CSV data into the container:

.. code-block:: console

 > sos-import-csv --path /dir/my-container --schema meminfo_qs --map meminfo_qs.map.json --csv meminfo_qs.csv
 Importing from CSV file meminfo_qs.csv into /tmp/my-container using map meminfo_qs.map.json
 Created 1000 records


5. You can monitor the progress from another window like this:

.. code-block:: console

 > sos-monitor --path /dir/my-container --schema meminfo_qs

It will take less than a second for 1000 lines, but you can see progress during larger file loads.
Querying data in a container

6. Query for the data in a container:

 a. Query all the data, using comp_time as an index, which will determine the output order
.. code-block:: console

 > sos_cmd -C /dir/my-container -q -S meminfo_qs -X time_job_comp
 ...
 -------------------------------- ------------------ ... -------------------------------- 
 Records 1000/1000.

b. Query only for certain variables (also using an index):

.. code-block:: console

 > sos_cmd -C /tmp/my-container/ -q -S meminfo_qs -X time_job_comp -f table -V timestamp -V component_id -V Active 
 timestamp                        component_id       Active             
 timestamp                        component_id Active
 -------------------------------- ------------ ------------------
               1703188156.000797 5427           29557660
               1703188156.000846 36            4825132
               1703188156.000873 4830            1784496
               1703188156.001007 5572           27297788
 ...
               1703188161.001589 9710           24505304
 --------------------------------  ------------------
 Records 1000/1000.

c. Querying with a filter:

.. code-block:: console

 > sos_cmd -C /tmp/my-container/ -q -S meminfo_qs -X time_job_comp -f table -V timestamp -V component_id -V Active -F timestamp:gt:1703188160
 timestamp                        component_id       Active             
 -------------------------------- ------------------ ------------------ 
   ...
               1703188161.001580 282            1999556
               1703188161.001588 5651          111678236
               1703188161.001589 9710           24505304
 --------------------------------  ------------------
 Records 248/248.


d. Querying with multiple filters:

.. code-block:: console

 > sos_cmd -C /tmp/my-container/ -q -S meminfo_qs -X time_job_comp -f table -V timestamp -V component_id -V Active -F timestamp:gt:1703188160 -F component_id:gt:9000
 timestamp                        component_id       Active             
 -------------------------------- ------------------ ------------------ 
 ...
               1703188161.001453               9274           26774688
               1703188161.001530               9593            2218724
               1703188161.001558               9097           57602824
               1703188161.001589               9710           24505304
 -------------------------------- ------------------ ------------------
 Records 23/23.


