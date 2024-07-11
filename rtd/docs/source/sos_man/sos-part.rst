========
sos-part
========

:Date: 17 Mar 2023

.. contents::
   :depth: 3
..

NAME
=========

sos-part - Manage SOS partitions

SYNOPSIS
=============

sos-part [OPTION...] [NAME]

DESCRIPTION
================

The **sos-part** command is used to manage SOS partitions. Partitions
are collections of structured files that contain SOS objects. The files
that comprise a partition exist in a common subdirectory in a local
filesystem.

Partitions exist separately from a container and may be *attached* to
zero or more containers. When *attached* the partition is included in
container object query and object index insertion.

Querying Partitions
-------------------

The set of partitions that are attached to a container may be queried
with the --query when combined with the --cont option. If the --cont
option is not specified, the --query must include the --path option to
specify the filesystem path to the partition.

--fmt
   Specifies the desired output format of the query data: *json* or
   *table* (default).

Creating Partitions
-------------------

Partitions are created with the --create option. After creation, the
partition exists, but is not part of any container. The --attach option
is used to *attach* the partition to one or more containers.

A container maintains a *state* that informs the container about how the
partition should be treated. The container partition *state* is one of
*PRIMARY, ACTIVE*, or *OFFLINE*. Only one partition in the **container**
can be in the *PRIMARY* state. A partition in this state is the default
partition for object insertion and is included in object query requests.

A partition in the *ACTIVE* state included in object queries and can be
the target of object insertion if the application so chooses on an
object by object basis.

A partition in the *OFFLINE* state is present in the container but
cannot be the target of object insertion and is not included in object
queries.

--mode MODE
   Specifies the **OCTAL** parmission bits to apply for partition
   access. See the open(3) system call for a description of these bits.

--user NAME
   Specifies the user **USER** that owns the partition. This **USER**
   must exist on the system.

--group NAME
   Specifies the group **GROUP** that owns the partition. This **GROUP**
   must exist on the system.

--desc DESC
   Specifies a description string that will be used to annotate the
   partition. The value does not affect the behavior of the partition
   but is intended to provided administrators information on the
   contents of the partition.

Attaching Partitions
--------------------

Partitions are attached to a container with the --attach option. The
initial state of the partition is *OFFLINE*. See the --state option for
instruction on how to make the new partition *ACTIVE* or *PRIMARY*.

--attach
   Requests that a partition be attached to a container. The initial
   state of the partition is *OFFLINE*.

--cont PATH
   Specifies the **PATH** to the container to which the partition will
   be attached.

--path PATH
   Specifies the **PATH** to the partition to attach.

Setting Partition State
-----------------------

Set the state of a partitions in a container to ono of *PRIMARY*,
*ACTIVE*, or *OFFLINE*.

--state STATE-NAME
   Specifies the state for the partition in the container. The
   **STATE-NAME** is one of *PRIMARY*, *ACTIVE*, or *OFFLINE*.

--cont PATH
   Specifies the **PATH** to the container containing the partition.

--name PART-NAME
   Specifies the **PART-NAME** in the container that refers to the
   partition.

Detaching Partitions
--------------------

Partions can be *detached* from a container when the data they contain
is no longer need. Detaching a partition does not remove the partition
files or any data from the partition.

The *PRIMARY* partition in a container cannot be removed.

--detach
   Requests that a partition be detached from a container.

--cont PATH
   Specifies the **PATH** to the container from which the partition will
   be removed.

--name PART-NAME
   Specifies the partition name to detach from the container.

--set

--remap-schema

--show-schema

--reindex

--reindex-status-count

--verbose

EXAMPLES
=============

Query Example
-------------

   ::

      $ sos-part --path=/dev/shm/SOS/ldms_data --query
      meminfo_tom_job_comp_time
      meminfo_tom_timestamp
      netdev2_tom_time_comp
      slurm2_tom_time_comp
      vmstat_tom_job_comp_time
      vmstat_tom_timestamp

Verify All Indices
------------------

   ::

      $ sos-part --path=/dev/shm/SOS/ldms_data --verify
      Verifying index 'meminfo_tom_job_comp_time' ... OK
      Verifying index 'meminfo_tom_timestamp' ... OK
      Verifying index 'netdev2_tom_time_comp' ... OK
      Verifying index 'slurm2_tom_time_comp' ... OK
      Verifying index 'vmstat_tom_job_comp_time' ... OK
      Verifying index 'vmstat_tom_timestamp' ... OK

Verify A Single Index
---------------------

   ::

      $ sos-part --path=/dev/shm/SOS/ldms_data --verify --index  meminfo_tom_job_comp_time
      Verifying index 'meminfo_tom_job_comp_time' ... OK

ENVIRONMENT
================

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
=============

sos-index(8), sos-schema(8), sos-monitor(8), sos-import-csv(8)
