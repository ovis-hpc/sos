==============
sos-import-csv
==============

:Date: May 14 2019

.. contents::
   :depth: 3
..

NAME
===============

sos-import-csv - Import data from a comma-separated text file into a SOS
Container

SYNOPSIS
===================

sos-import-csv --path SOS-PATH --csv CSV-FILE--schema SCHEMA-FILE --map
MAP-FILE [ --sep SEP-STR --status ]

DESCRIPTION
======================

The **sos-import-csv** commannd parses CSV text data and imports this
data into a SOS container. There are three files read by the
**sos-import-csv command: the CSV text file containing the data to**
import, a *schema-file* that defines the type of object created for each
line of the CSV text file, and a *map-file* that specifies how object
attribute values are obtained from columns in the CSV file.

OPTIONS
==================

**--path** *SOS-PATH*
   | 
   | Specifies the path to the SOS container. The container must already
     exist.

**--schema** *SCHEMA*
   | 
   | The name of the SOS schema used to create objects. The schema must
     exist in the container.

**--csv** *CSV-FILE*
   | 
   | The path to the CSV text file containing the data to import.

**--map** *MAP-FILE*
   | 
   | The path to a JSON formatted text file specifying how columns in
     the CSV file map to attributes in the object schema.

**--sep** *SEP-STR*
   | 
   | A string specifying the characters that will be interpretted as
     column separators. By default this is a comma (",").

**--status**
   | 
   | If the *status* option is present, import progress status will be
     provided as the CSV file is processed.

CSV file format
==========================

The CSV text file must contain newline terminated lines of text. Each
line may contain multiple columns separated by the *separator*
character. By default, the character separating each column is a comma,
however, this can be overriden with the **--sep** command line option.

Map File Format
==========================

The map file contains a single *list* object where each element in the
list is an *action-specification*. An *action-specification* is an
object that tells the **sos-import-cmd** what to do with each line of
the input CSV file.

Each *action-specification* contains two attributes, a *target*, and a
*source*. The *target* specifies the attribute in the object to be
assigned a value. The value of the *target* attribute can be a string,
in which it is the name of the object attribute from the schema, or an
integer in which case is the attribute id.

The *source-specification* defines the value that will be assigned to
the target attribute and is specified as a JSON object. The JSON object
contains only one of four possible attribute names: "value", "column",
"list", and "range".

If the attribute named "value" is present, the value of the "value"
attribute is assigned to the target attribute. This is useful for
assigning values to object attributes that are not contained in the CSV
file.

If the attribute named "column" is present, the value of the "column"
attribute is an ordinal column number in the CSV file. In this case, the
value contained in that column of the text file is cast to the target
attribute type and assigned to the target attribute. If a conversion
from the column text string to the target attribute type is not
possible, an error is given and processing continues.

If the attribute named "list" is present, the target attribute must be
an array, and the value of the "list" attribute is a JSON list of
columns that will be assigned to each element in the target array
beginning with 0. Text to target attribute type conversion is performed
for each column specifed in the list.

If the attribute named "range" is present, the target attribute must be
an array and the value of the "range" attribute is a list of two column
numbers. All values in the range from the first column to the last
column are assigned to the elements of the target list attribute. The
range is inclusive.Text to target attribute type conversion is performed
for each column in the range.

::


   [
       { "target" : "component_id", "source" : { "value" : 10000 } },
       { "target" : 0, "source" : { "column" : 0 } },
       { "target" : 2, "source" : { "list" : [ 1, 3, 5 } },
       { "target" : 3, "source" : { "range" : [ 1, 5 } }
   ]

SEE ALSO
===================

sos_cmd(8), sos-schema(7)
