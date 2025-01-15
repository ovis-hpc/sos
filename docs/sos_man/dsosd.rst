=====
dsosd
=====

:Date: 14 Mar 2022

.. contents::
   :depth: 3
..

NAME
======

dsosd - Start an instance of the D/SOS RPC service on a node

SYNOPSIS
==========

dsosd

DESCRIPTION
=============

The dsosd command starts the D/SOS RPC service on a node. The D/SOS RPC
service registers the D/SOS program number with the with the RPC
portmapper on the TCP and UDP transports. Only a single instance of the
daemon should be run at a time.

ENVIRONMENT
=============

The following environment variables may be used to affect the
-------------------------------------------------------------

configuration of the dsosd daemon.

DSOSD_DIRECTORY The path to a JSON formatted file that maps container
   names to local filesystem paths.

DSOSD_SERVER_ID A logical name for this dsosd instance. If not
   specified, the hostname (as determined by gethostname) will be used.
   This name is used to determine which sections of the directory file
   apply to this dsosd instance.

DSOSD_LOG_LEVEL The log level at which log messages are written to
   standard out.

SEE ALSO
==========

dsosql(8), dsosd_directory(7)
