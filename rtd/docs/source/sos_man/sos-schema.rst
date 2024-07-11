==========
sos-schema
==========

:Date: 20 July 2023

.. contents::
   :depth: 3
..

NAME
===========

sos-schema - Manage SOS Schema

SYNOPSIS
===============

sos-schema [OPTION...]

DESCRIPTION
==================

The **sos-schema** command is used to manage SOS schema. Schema define
the format of objects in a container. This format includes the
attributes in the object that will be indexed.

A schema is defined outside the container as a text file called a
**template**. The **template** file is JSON dictionary, for example:

   ::

      {
        "name": schema-name-string,
        "uuid": uuid-string
        "attrs": [
          {
            "name": attribute-name,
            "type": attribute-type,
          },
          . . .
          {
            "name": join-attr-name,
            "type": "JOIN",
            "index": {},
            "join_attrs": [
              attr-name,
              attr-name,
              . . .
            ]
          }
        ]
      }

In the template above, the last attribute is called a *JOIN*. A *JOIN*
attribute occupies no space in the object, but allows a collection of
attributes to be used as a key in an index.

The *UUID* is a Universally Unique ID computed using **libuuid**. The
attribute names, types, and indices are used to compute this value.

A template may also contain multiple schema definitions.

   ::

      {
        "schemas" : [
          { schema-dictionary },
          { schema-dictionary },
          . . .
        ]
      }

OPTIONS
==============

--path PATH
   Specifies the **PATH** to the container.

--query
   Query the schema defined in the container. Use with the --verbose
   option to provide detail information for each schema.

--add PATH
   Adds a single schema defined in the *template file PATH* to the
   container.

--export PATH
   Export all schema defined in the container to a multi-schema
   *template* file at **PATH**.

--import PATH
   Add all schema defined in multi-schema *template* file at **PATH** to
   the container.

--verbase
   When querying schema provide detail information of the schema
   contents. Absent this option, only the schema name are printed.

--schema NAME
   Used with the --query option to print information only for the schema
   *NAME*.

--uuid UUID
   Used with the --query option to print information only for the schema
   with the Universally Unique ID *UUID*.

EXAMPLES
===============

Query Schema
------------

   ::

      $ sos-schema --path database --query
      12354914-a519-48ea-a410-d2e86ca5dc4b        0 vmstat
      d2641326-77a2-48cf-99d3-83a37dbdf65e        0 meminfo
      3ac614f6-ec43-4498-8bc6-b5c58a7e1f0d        0 lustre_client

In this example, the first column is the *UUID*, the second is the
schema *generation* number, and the third is the schema name. The
*generation* number is incremented whenever indices are added to or
removed from the schema. See the **sos-index** command for information
on how indices can be added and removed.

Query Verbose
-------------

   ::

      $ sos-schema --path database --query --verbose --schema meminfo
      d2641326-77a2-48cf-99d3-83a37dbdf65e        0 meminfo
      Id   Type                     Indexed      Name                            
      ---- ------------------------ ------------ --------------------------------
         0 TIMESTAMP                             timestamp
         1 UINT64                                component_id
         2 UINT64                                job_id
         3 UINT64                                app_id
         4 UINT64                                MemTotal
         5 UINT64                                MemFree
         . . .
        51 JOIN                                  time_job_comp [timestamp+job_id+component_id]
        52 JOIN                                  time_comp_job [timestamp+component_id+job_id]
        53 JOIN                                  job_comp_time [job_id+component_id+timestamp]
        54 JOIN                                  job_time_comp [job_id+timestamp+component_id]
        55 JOIN                                  comp_time_job [component_id+timestamp+job_id]
        56 JOIN                                  comp_job_time [component_id+job_id+timestamp]

Query Verbose
-------------

Query the details for a single schema.

   ::

      $ sos-schema --path database --query --verbose --schema meminfo
      d2641326-77a2-48cf-99d3-83a37dbdf65e        0 meminfo
      Id   Type                     Indexed      Name                            
      ---- ------------------------ ------------ --------------------------------
         0 TIMESTAMP                             timestamp
         1 UINT64                                component_id
         2 UINT64                                job_id
         3 UINT64                                app_id
         4 UINT64                                MemTotal
         5 UINT64                                MemFree
         . . .
        51 JOIN                                  time_job_comp [timestamp+job_id+component_id]
        52 JOIN                                  time_comp_job [timestamp+component_id+job_id]
        53 JOIN                                  job_comp_time [job_id+component_id+timestamp]
        54 JOIN                                  job_time_comp [job_id+timestamp+component_id]
        55 JOIN                                  comp_time_job [component_id+timestamp+job_id]
        56 JOIN                                  comp_job_time [component_id+job_id+timestamp]

Add a Single Schema
-------------------

Add a single schema to the container.

   ::

      $ sos-schema --path database --add schema-template.json

Export All Schema in a Container
--------------------------------

Export all schema in a container to a JSON template file. This is useful
for adding schema defined in one container to another.

   ::

      $ sos-schema --path database --export multi-schema-template.json

Import Schema
-------------

Import all schema defined in a JSON template file to a container.

   ::

      $ sos-schema --path database --import multi-schema-template.json

ENVIRONMENT
==================

ODS_LOG_MASK
------------

This environment variable specifies what log messages are printed by the
SOS libraries. The value is a bit mask as follows:

Value \| Description                               
* 0 - No messages are logged                        
* 1 - **Fatal** errors (i.e. the process will exit) 
* 2 - **Errors**                                    
* 4 - **Warnings**                                  
* 8 - **Informational** messages                    
* 16 - **Debug** messages                           
* 255 - **All** messages are logged                 

SEE ALSO
===============

sos-index(8), sos-part(8), sos-monitor(8), sos-import-csv(8)
