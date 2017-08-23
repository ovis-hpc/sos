from __future__ import print_function
import numpy as np
from sosdb import Sos
import datetime as dt

class Pipeline(object):
    """Implements a generic analysis pipeline interface to SosDB.

    Using an Pipeline subclass is intended to be very simple as follows:

    a = MyPipeline()
    a.set_input("/NVME/0/E2E/Today/metric_data")
    a.set_select(dt.datetime.now() - dt.timedelta(hours=1), # An hour ago ...
                 dt.datetime.now(),                         # ... to now
                 [i for i in range(0,256)]                  # component id list
                )
    a.set_output("/OVIS_DATA/Analysis/results")
    a.process()                 # analyse all the data and write the
                                # results to the output SosDB

    It is also possible to access stages in the pipeline:

    a.select(9)                 # select the data for component 9

    # Query the data from the input container:
    # -- cnt is the number of samples actually written to the result
    # -- result is a 2D array of the result [ [ col_1, col_2,...],...],
    cnt, data = a.query()

    The dimension of the array returned by query is
    [ window-size, column-count ]. The window size is, by default 1024,
    and is the number of data samples processed at a time. This is to
    allow very large data sets to be handled without exhausting
    memory.

    As an example if there are 1500 samples that match the select
    criteria, query() would return the following:

    (1024, [ [ timestamp-value, stall-value, flit-value ], ... ])

    The len() of the returned array is 1024, and the len(array[0]) ==
    3. Calling query() a second time with the keyword parameter
    cont=True would return the remaining samples:

    For example:

    (cnt, ary) = a.query(cont = True)

    Then cnt == 476, i.e. 1500 - 1024 and ary would contain the
    remaining samples. Note that the array size would still be 1024,
    therefore the cnt result needs to be consulted to determine how
    many samples were actually written to the array. It is important
    to honor this value in the transform() function to avoid getting
    unexpected results if numpy math functions process trailing
    zeroes as sample data.

    # Use the transform method to process the data
    cnt, result = a.transform(cnt, data)

    # Display the processed data on the terminal
    a.text_output(cnt, result)

    Note that the process() function effectively performs all of these
    steps for you, sending the data to the destination SosDB.

    The intended design pattern is for the developer to create a
    subclass of the Pipeline base class and overload the 'transform'
    method.

    In the __init__ method of the subclass the user must define the
    following attributes:

    inp_schema_name -- The name of the input schema

    inp_query_attrs -- The attributes in the input schema that will be
    part of the analysis and are returned in the queried data.

    out_schema_name -- The name of the schema in the results container

    out_schema_template -- An array of attribute definitions. See
    help(Sos.Schema.from_template())

    The reason that these are defined as part of __init__ and are
    immutable for the class is because changing these values would
    require rewriting the transform() method and therefore they are
    part of the object definition itself.

    Here is an example sub-class:

    from Analysis import Pipeline
    import datetime as dt
    import numpy as np

    class MyPipeline(Pipeline):
        def __init__(self):
            Pipeline.__init__(self)
            self.inp_schema_name = "metric_set_nic"
            self.inp_query_attrs = [
                    "timestamp",
                    "component_id",
                    "AR_NIC_NETMON_ORB_EVENT_CNTR_REQ_STALLED",
                    "AR_NIC_NETMON_ORB_EVENT_CNTR_REQ_FLITS"
            ]
            self.out_schema_name = "cntr_req_dy_dt"
            self.out_schema_template = [
                    { "name" : "timestamp", "type" : "timestamp", "index" : {} },
                    { "name" : "component_id", "type" : "timestamp" },
                    { "name" : "AR_NIC_NETMON_ORB_EVENT_CNTR_REQ_STALLED_dy_dt",
                      "type" : "double" },
                    { "name" : "AR_NIC_NETMON_ORB_EVENT_CNTR_REQ_FLITS_dy_dt",
                      "type" : "double" },
                    { "name" : "AR_NIC_NETMON_ORB_EVENT_CNTR_REQ_FLITS_PER_STALL_dy_dt",
                      "type" : "double" },
                    { "name" : "comp_time", "type" : "join",
                      "join_attrs" : [ "component_id", "timestamp" ],
                      "index" : {} }
            ]

        def transform(self, cnt, nda, cont=False):
            # Compute dx/dy for stalls and flits
            stalls_dt = np.gradient(nda[:cnt,2])
            flits_dt = np.gradient(nda[:cnt,3])

            # filts_dt may contains zeroes, so tell numpy to ignore divide by zero errors
            with np.errstate(divide='ignore', invalid='ignore'):
                spf = stalls_dt / flits_dt
                spf[spf == np.inf] = 0
                spf = np.nan_to_num(spf)

            # The return result describes a 'row' in the database. The
            # 'comp_time' attribute is a JOIN and it's value is
            # computed from the values of component_id and timestamp, so
            # it is not set explicitly and therefore not contained in the result
            return (cnt, [
                    [nda, 0],          # timestamp
                    [nda, 1],          # component_id
                    [stalls_dt, None], # dstalls_dt
                    [flits_dt, None],  # dflits_dt
                    [spf, None]        # stalls_per_flit
                    ])
    """
    def __init__(self):
        self.dt_fmt = "%Y/%m/%d %H:%M:%S"
        self.schema = None
        self.in_cont = None
        self.out_cont = None
        self.window = 1024
        self.order = 'index'

    def set_input(self, path):
        self.in_cont = Sos.Container(path, Sos.PERM_RO)
    def get_input(self):
        return self.in_cont

    def set_output(self, path):
        self.out_cont = Sos.Container()
        try:
            self.out_cont.open(path, Sos.PERM_RW)
        except:
            self.out_cont.create(path)
            self.out_cont.open(path, Sos.PERM_RW)
            self.out_cont.part_create("RESULTS")
            part = self.out_cont.part_by_name("RESULTS")
            part.state_set("primary")
        self.out_schema = self.out_cont.schema_by_name(self.out_schema_name)
        if not self.out_schema:
            if self.out_schema_template is None:
                raise ValueError("A schema template must be provided.". \
                                 format(self.out_schema_name))
            self.out_schema = Sos.Schema()
            self.out_schema.from_template(self.out_schema_name, self.out_schema_template)
            self.out_schema.add(self.out_cont)

    def get_output(self):
        return self.out_cont

    def set_window(self, window):
        self.window = window

    def set_select(self, start, end, comp_list,
                   index="comp_time", count=None, order=None):
        """Set the database selection criteria

        Positional Arguments:
        -- A date object specifying the start time for analysis.
        -- A date object specifying the end time for analysis. Samples
        -- An array of integers specifying the components for which data
           will be returned
        -- An array of strings specifying the metrics in the schema to return

        Keyword Arguments:
        index -- The name of the attribute to use as an indx. By default
                 this is the "comp_time" attribute.
        """
        if not self.in_cont:
            raise IOError("set_input() must be called before calling set_select()\n")

        self.in_schema = self.in_cont.schema_by_name(self.inp_schema_name)
        if self.in_schema is None:
            raise ValueError("The schema {0} was not found.".format(inp_schema_name))

        self.start = start
        self.end = end
        self.comp_list = comp_list
        self.index = index
        self.count = count
        if order:
            self.order = order

    def get_input_schema(self):
        return self.in_schema

    def select(self, comp_id):
        self.timestamp = self.in_schema.attr_by_name("timestamp")
        self.filter = Sos.Filter(self.in_schema.attr_by_name(self.index))
        self.filter.add_condition(self.timestamp,
                                  Sos.COND_GE,
                                  self.start.strftime(self.dt_fmt))
        self.filter.add_condition(self.timestamp,
                                  Sos.COND_LE,
                                  self.end.strftime(self.dt_fmt))
        self.comp_id = self.in_schema.attr_by_name("component_id")
        self.filter.add_condition(self.comp_id,
                                  Sos.COND_EQ,
                                  str(comp_id))

    def transform(self, cnt, nda, cont=False):
        """Perform the desired transform on the input data

        This method should be overloaded by analysis sub-classes.

        The result from this function describes a row in the output
        and consist of a tuple containing the count of rows in the
        result array, and the result array itself, i.e.
        (row_count, result_array).

        The result_array is interpreted as an array of column
        specifications as follows:

        [
            [ array, <number> | None ],    # first column in result
            ...
            [ array, <number> | None ]     # last column in result
        ]

        If the column-spec contains an integer for the 2nd element, it
        is assumed that the value to be placed in the row is contained
        in array[row-number][col-no]. If the col-no is

        Positional Arguments:
        -- The number of samples present in the input array
        -- The input array

        Key Arguments:
        first -- If True, this is the first call to transform in the pipeline
        last  -- If True, this is the last call to transform in the pipeline
        """
        # return a result spec for the input
        return (cnt, [[nda,i] for i in range(0, len(nda[0]))])

    def process(self):
        """Run the analysis pipeline

        The analysis pipe line calls select(), and then query(),
        transform() and output() in a loop until all samples matching
        the select criteria have been processed.
        """
        for comp_id in self.comp_list:
            self.select(comp_id)
            first = True
            last = False
            cont = False
            while True:
                cnt, data = self.query(cont=cont)
                if cnt == 0:
                    break
                if cnt < self.window:
                    last = True
                cont = True
                (res_cnt, res) = self.transform(cnt, data, first=first, last=last)
                self.output(res_cnt, res)
                if cnt < self.window:
                    break

    def query(self, cont=False, order='index', window=None):
        """Get the next window of data to process from SosDB

        Keyword Arguments:
        cont   -- If true, continue at the next sample, do not restart the iterator
        order  -- The row/column order of returned results see Sos.Filer.as_ndarray()
        window -- The size of the sample window. The default is 1024 samples
        """
        if window is None:
            window = self.window
        cnt, nda = self.filter.as_ndarray(window,
                                          shape=self.inp_query_attrs,
                                          cont=cont, order=order)
        return cnt, nda

    def output(self, cnt, res):
        if self.out_cont:
            self.sosdb_output(cnt, res)
        else:
            self.text_output(cnt, res)

    def text_output(self, res_cnt, res):
        """Send the result output to stdout"""
        for i in range(0, res_cnt):
            print("[{0:4}] -> ".format(i), end=" ")
            for col in res:
                ary = col[0]
                idx = col[1]
                if idx is not None:
                    print("{1:12}".format(i, ary[i][idx]), end=" ")
                else:
                    print("{1:12}".format(i, ary[i]), end=" ")
            print("")

    def sosdb_output(self, res_cnt, res):
        """Send the result output to SosDB"""
        for i in range(0, res_cnt):
            obj = self.out_schema.alloc()
            attr_id = 0
            for col in res:
                ary = col[0]
                idx = col[1]
                if idx is not None:
                    obj[attr_id] = ary[i][idx]
                else:
                    obj[attr_id] = ary[i]
                attr_id += 1
            obj.index_add()
