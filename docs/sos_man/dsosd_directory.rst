===============
dsosd_directory
===============

:Date: 15 May 2023

.. contents::
   :depth: 3
..

DESCRIPTION
=======================

The dsosd directory file maps container names to local filesystem path
names. The directory is formatted as JSON and has a separate section for
each dsosd instance. The format of the file is as follows:

   ::

      {
        <server-id> : {
           <container-name> : <container-path>,
           <container-name> : <container-path>,
           . . .
        },
        <server-id> : {
           <container-name> : <container-path>,
           <container-name> : <container-path>,
           . . .
        },
        . . .
      }

By default the *<server-id>* is the hostname, but this can be overridden
with the *DSOSD_SERVER_ID* environment variable specified when starting
**dsosd**.

The *<container name>* is the *path* parameter provided to the
*sos_container_open()* function. The *<container-path>* is the
filesysystem path to the SOS container.

EXAMPLE
===================

   ::

      {
        "orion-01" : {
          "ldms-current" : "/storage/orion-01/sos/ldms-current",
          "ldms-january" : "/storage/orion-01/sos/january/ldms-current"
        },
        "orion-02" : {
          "ldms-current" : "/storage/orion-02/sos/ldms-current",
          "ldms-january" : "/storage/orion-02/sos/january/ldms-current"
        },
        "orion-03" : {
          "ldms-current" : "/storage/orion-03/sos/ldms-current",
          "ldms-january" : "/storage/orion-03/sos/january/ldms-current"
        },
        "orion-04" : {
          "ldms-current" : "/storage/orion-04/sos/ldms-current",
          "ldms-january" : "/storage/orion-04/sos/january/ldms-current"
        }
      }

ENVIRONMENT
=======================

The following environment variables may be used to affect the
-------------------------------------------------------------

configuration of the dsosd daemon.

DSOSD_DIRECTORY The path to a JSON formatted file that maps container
   names to local filesystem paths.

DSOSD_SERVER_ID A logical name for this dsosd instance. If not
   specified, the hostname (as determined by gethostname) will be used.
   This name is used to determine which sections of the directory file
   apply to this dsosd instance.

SEE ALSO
====================

dsosd(8)
