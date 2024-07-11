DSOS Quickstart
####################

Introduction
***************

The Distributed Scalable Object Store (DSOS) (pronounced "dee-sôs") is a layer on top of SOS to enable distributed, parallel ingests and queries. DSOS is intended to be used to use SOS databases across multiple devices as a unified database. Users setup a file, referred to as the cluster configuration file in this context, which names all of the nodes where a SOS database is expected. Using python API or the command line interface dsosql, users can query these SOS databases for data in the same schema. DSOS interfaces are installed alongside SOS, starting with SOS v4, with no additional enable arguments required.

Dsosql
********

For demonstration purposes, let's assume we have two nodes, node1 ande node2, with a SOS database at /storage/sos/database. 
Our cluster configuration file, let's call it dsos.conf, would simply be

.. code-block:: console

  node1
  node2

Dsosql expects the path to this dsos.conf and the database path for correct functionality. These can be entered as options in to dsosql using the -a and -o options, respectively. They can also be entered after dropping into the dsosql shell, like ldmsd_controller, commands to dsosql can be entered after going into a shell or by echo'ing them into the utility. 

.. code-block:: console

  >dsosql -a dsos.conf -o /storage/sos/database
  Attaching to cluster dsos.conf ... OK
  Opening the container /storage/sos/database ... OK
  dsosql: show_part regex .*
  Name                     Description                                   UID      GID Permission
  ------------------------ ---------------------------------------- -------- -------- -------------
  default                  default                                  33       33       -rw-rw---

  #or

  >dsosql
  dsosql: attach path dsos.conf
  Attaching to cluster dsos.conf ... OK
  dsosql: open path /storage/sos/database
  Opening the container /storage/sos/database ... OK
  dsosql: show_part regex .*
  Name                     Description                                   UID      GID Permission
  ------------------------ ---------------------------------------- -------- -------- -------------
  default                  default                                  33       33       -rw-rw---


  >echo "show_part regex .*" | dsosql -a dsos.conf -o /storage/sos/database
  Attaching to cluster dsos.conf ... OK
  Opening the container /storage/sos/database ... OK
  Name                     Description                                   UID      GID Permission
  ------------------------ ---------------------------------------- -------- -------- -------------
  default                  default                                  33       33       -rw-rw---

Commands available in dsosql are attach, create_part, create_schema, help, import, open, select, set, show, show_part, and show_schema. 


Select Syntax and Options
*************************

The select statement is used to set parameters for querying the sos databases attached in the dsosql setup.
The query language is sql-like. The basic format follows the structure of "select {metrics} from {schema}".
The schema field is for a defined schema in the database to be queried. Only a single schema can be selected in a query.
Metrics can be a single metric in a schema, a list of metrics separated by commas, or an asterisk to request all metrics in the schema be returned that match the filters.
Additional filters can be added using a "where {filters}" syntax after the schema selection.
Filters can be conjoined using "and" and "or" and paranthesis can be used to further join filter statements.
When filters are used to match on strings, the string must have quotes encasing it so the select interpreter knows it is not referencing another metric.

The index to use for the query can be specified using "order_by {index}". Dsosql will give a best effort to optimize the query based on the filters given in the select statement if the order_by is not specified.
Choosing an index for your query can vastly change the performance. Best practices are to use an index that matches the most specifying metric in your query filter.

A resampling window can also be added to the query if the schema has timestamp as a metric using "resample {window}".
This will return the averages of values for lengths of time with size specified by the window, which must be an integer.


Python API
**********

Like dsosql, python expects a dsos.conf path and a database path. A Sos.Session object is initialized, opened, and then a query setup to begin querying data out of the database. The query initialization expects a max rows returned value for the resultant data object, which will be a pandas DataFrame with columns consisting of the metrics queried and the metrics comprising the index queried. The max rows, or query block size, can typically be set at 1024*1024 though changing block sizes will affect performance.

.. code-block:: python

    import pandas as pd
    from sosdb import Sos
    sess = Sos.Session("dsos.conf")
    cont = sess.open("/storage/sos/database")
    query = Sos.SqlQuery(cont,1024*1024)
    query.select('select Active from meminfo') 
    df = query.next()

The query.next() can be run multiple times to get more data matching the query. The next() will return None if no further data matches the query. A function to return all data matching a query can be written as:

.. code-block:: python

     def get_all_data(self, query):
        df = query.next()
        if df is None:
           return None
        res = df.copy(deep=True)
        while df is not None:
            df = query.next()
            res = pd.concat([df, res])
        del df
        return res

To manually add a record to a DSOS database we can use the insert_df function for a sos container object. 
The dataframe inserted must have rows that match the types and length of the schema being inserted into, otherwise an error will be raised. 
The data will be round robin-ed into the SOS containers referenced in the dsos.conf. 

.. code-block:: python
   
    import pandas as pd
    from sosdb import Sos
    sess = Sos.Session("dsos.conf")
    cont = sess.open("/storage/sos/database")
    schema = cont.schema_by_name('meminfo')
    in_df = {DATAFRAME OF RECORD(S) TO BE INSERTED}
    cont.insert_df(schema,in_df)
       
To update a record in a DSOS database, the update needs to be bounded by a transaction begin and end to prevent data corruption.
Create a key to find the record to be updated, change the values desired, and then update the record.

.. code-block:: python
    import pandas as pd
    from sosdb import Sos
    sess = Sos.Session("dsos.conf")
    cont = sess.open("/storage/sos/database")
    schema = cont.schema_by_name('meminfo')
    attr = schema['job_time_comp']
    key = attr.key(JOB,TIME,COMP)
    obj = attr.find(key)
    cont.transaction_begin()
    obj['component_id'] = 13
    cont.obj_update(obj)
    cont.transaction_end()

