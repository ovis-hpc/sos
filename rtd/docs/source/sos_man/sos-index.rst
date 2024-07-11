=========
sos-index
=========

:Date: 17 Mar 2023

.. contents::
   :depth: 3
..

NAME
==========

sos-index - Manage SOS indices

SYNOPSIS
==============

sos-index [OPTION...]

DESCRIPTION
=================

The **sos-index** command is used to query, add, remove and verify
indices.

Query
-----

The -Q option will print the index name for all indices in the
container.

Add
---

The -A option will add an index for an attribute in a schema. The index
may already exist; if it does not it will be created.

Clients that share the container become aware of the new index (based on
a schema generation number), and begin adding objects to the index.

Remove
------

The -R option will remove an index for a schema attribute. Clients that
share the container become aware of the index removal (based on a schema
generation number), and cease adding objects to the index.

The -R option does not remove the index files themselves; the index can
be added back with the -A option and will contain all objects indexed
prior to index removal.

Verify
------

When verifying an index, if corruption is detected, the name of the
partition containing the corrupted index is printed. It is also possible
to have the underlying index library print specific consistency errors.
This is enabled with the --verbose option.

OPTIONS
=============

-Q,--query
   Print the name of all indices in the container.

Add an index to the schema specified by the -s option for the attribute
specified by the -a option.

Remove an index from the schema specified by the -s option for the
attribute specified by the -a option.

-V,--verify
   Verify the consistency of one or more indices in a container. This
   includes all index instances in all partitions attached to the
   container. If the --index NAME option is specified, only the index
   NAME will be verified.

-p,--path PATH 
   The **PATH** to the SOS container.

-n,--index NAME 
   The optional **NAME** of the index to verify. If not specified, all
   indices will be verified.

-s,--schema NAME
   The schema **NAME**, required with the -A and -R options.

-a,--attr NAME
   The attribute **NAME** to index, required with the -A and -R options.

-v,--verbose 
   This option can be used with the -V option to control the level of
   error messages printed when a corruption error is detected. If
   specified once, the partition name containing the corrupted index is
   printed. If specified more than once, both the partition name and the
   index specific corruption errors are printed.

EXAMPLES
==============

Query Example
-------------

   ::

      $ sos-index --path=/dev/shm/SOS/ldms_data --query
      meminfo_tom_job_comp_time
      meminfo_tom_timestamp
      netdev2_tom_time_comp
      slurm2_tom_time_comp
      vmstat_tom_job_comp_time
      vmstat_tom_timestamp

Add Example
-----------

   ::

      $ sos-index --path=/dev/shm/SOS/ldms_data --add --path /dev/shm/SOS/ldms_data -s meminfo -a instance

Remove Example
--------------

   ::

      $ sos-index --path=/dev/shm/SOS/ldms_data --rem --path /dev/shm/SOS/ldms_data -s meminfo -a instance

Verify All Indices
------------------

   ::

      $ sos-index --path=/dev/shm/SOS/ldms_data --verify
      Verifying index 'meminfo_tom_job_comp_time' ... OK
      Verifying index 'meminfo_tom_timestamp' ... OK
      Verifying index 'netdev2_tom_time_comp' ... OK
      Verifying index 'slurm2_tom_time_comp' ... OK
      Verifying index 'vmstat_tom_job_comp_time' ... OK
      Verifying index 'vmstat_tom_timestamp' ... OK

Verify A Single Index
---------------------

   ::

      $ sos-index --path=/dev/shm/SOS/ldms_data --verify --index  meminfo_tom_job_comp_time
      Verifying index 'meminfo_tom_job_comp_time' ... OK

SEE ALSO
==============

sos-part(8), sos-schema(8), sos-monitor(8), sos-import-csv(8)
