===========
sos-monitor
===========

:Date: 20 July 2023

.. contents::
   :depth: 3
..

NAME
============

sos-monitor - Show Index Insert Data for a Schema

SYNOPSIS
================

sos-monitor --path PATH --schema NAME [--refresh INT]

DESCRIPTION
===================

The **sos-monitor** command is a curses application that shows index
insert rates for all indices defined in the specified schema. Typing any
key will cause the application to exit.

OPTIONS
===============

--path PATH
   Specifies the *PATH* to the container.

--schema NAME
   Specifies the schema *NAME* to monitor.

--refresh INT
   Specifies that the window is updated every *INT* seconds. The default
   interval is 1 second.

EXAMPLES
================

   ::

      $ sos-monitor --path database --schema meminfo

..

   ::

      Monitoring schema meminfo in container database at 2023-07-20 10:22:45.192586

      Name              Cardinality      Inserts/s        Duplicates       Inserts/s        Size             Bytes/s         
      time_job_comp           3,962,974               0               0               0     953,745,408               0
      time_comp_job           3,962,974               0               0               0     912,850,944               0
      job_comp_time           3,962,974               0               0               0     907,608,064               0
      job_time_comp           3,962,974               0               0               0     936,968,192               0
      comp_time_job           3,962,974               0               0               0   1,227,948,032               0
      comp_job_time           3,962,974               0               0               0   1,004,601,344               0

SEE ALSO
================

sos-index(8), sos-part(8), sos-schema(8), sos-import-csv(8)
