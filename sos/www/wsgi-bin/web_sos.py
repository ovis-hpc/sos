import os
import web
import os
import datetime
import tempfile
from numpy import array
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import StringIO, Image
from sos import *

urls = (
    '/table', 'SosTable',
    '/info', 'SosInfo',
    '/container', 'SosContainer',
    '/schema', 'SosSchema',
    '/graph', 'SosGraph',
    '/metric_graph', 'SosGraph2',
    '/directory', 'SosDir',
    )

class SosDir:
    def open_test(self, path):
        try:
            c = SOS.Container(str(path))
            c.close()
            return True
        except:
            return False

    def GET(self):
        """
        Given the sos_root, search the directory structure to find all
        of the containers available for use. Note that even if the
        directory is there, this will skip the container if the
        requesting user does not have access rights.

        The OVIS store is organized like this:
        {sos_root}/{container_name}
        """
        carray = []
        try:
            dirs = os.listdir(sos_root)
            # Check each subdirectory for files that constitute a container
            for ovc in dirs:
                try:
                    ovc_path = sos_root + '/' + ovc
                    try:
                        files = os.listdir(ovc_path)
                        if '.__schemas.OBJ' in files:
                            if self.open_test(ovc_path):
                                carray.append([ ovc ])
                    except Exception as e:
                        carray.append([ str(e) ])
                        pass
                except:
                    carray.append([ str(e) ])
                    pass
            return render.table_json('directory', [ 'name' ], carray, len(carray))
        except Exception as e:
            # return render.table_json('directory', [ 'name' ], [], 0)
            return render.error(str(e))

class SosRequest(object):
    """
    This base class handles the 'container', 'encoding' and 'schema',
    'start', and 'count' keywords. For DataTables compatability, 'iDisplayStart'
    is a synonym of 'start' and 'iDisplayCount' is a synonym of 'count'.
    """
    JSON = 0
    TABLE = 1
    def __init__(self):
        self.encoding_ = self.JSON
        self.container_ = None
        self.schema_ = None
        self.start = 0
        self.count = 10

    def container(self):
        return self.container_

    def encoding(self):
        return self.encoding_

    def schema(self):
        return self.schema_

    def parse_request(self, input):
        #
        # Open the container or get it from our directory
        #
        if 'container' in input:
            container = input.container
            self.container_ = SOS.Container(str(sos_root + '/' + container),
                                            mode=SOS.Container.RO)
        if not self.container():
            raise SOS.Error("The 'container' clause is mandatory")

        #
        # Encoding
        #
        if 'encoding' in input:
            if input.encoding.lower() == 'table':
                self.encoding_ = self.TABLE
        else:
            self.encoding_ = self.JSON

        #
        # Schema
        #
        if 'schema' in input:
            if self.container():
                self.schema_ = self.container().schema(input.schema)

        #
        # iDisplayStart (dataTable), start
        #
        if 'start' in input:
            self.start = int(input.start)
        # overrides start if specified
        if 'iDisplayStart' in input:
            self.start = int(input.iDisplayStart)

        #
        # iDisplayLength (dataTables), count
        #
        if 'count' in input:
            self.count = int(input.count)
        # overrides count if specified
        if 'iDisplayLength' in input:
            self.count = int(input.iDisplayLength)

class SosContainer(SosRequest):
    """
    Return a list of the schemas that are defined in the specified container.
    """
    def GET(self):
        parms = web.input()
        try:
            self.parse_request(parms)
        except Exception as e:
            return render.error("Exception: " + str(e))

        rows = []
        for name, schema in self.container().schemas.iteritems():
            row = [ name, schema.attr_count() ]
            rows.append(row)
        if self.encoding() == self.TABLE:
            return render.table("Container Schemas", [ "Schema Name", "Attribute Count" ], rows, len(rows))
        web.header('Content-Type', 'application/json')
        return render.table_json("schemas", [ "schema_name", "attr_count" ], rows, len(rows))

class SosSchema(SosRequest):
    """
    Return all of the attributes and attribute meta-data for the
    specified schema.
    """
    def GET(self):
        parms = web.input()
        try:
            self.parse_request(parms)
        except Exception as e:
            return render.error("parse_request: " + str(e))
        if not self.schema():
            return render.error("A 'schema' clause must be specified.\n")
        rows = []
        for attr_name, attr in self.schema().attrs.iteritems():
            row = [ attr_name, attr.attr_id(), attr.type_str(), attr.indexed() ]
            if attr.indexed():
                row.append(attr.iterator().cardinality())
                row.append(attr.iterator().duplicates())
                row.append(attr.iterator().minkey())
                row.append(attr.iterator().maxkey())
            else:
                row += [ "", "", "", "" ]
            rows.append(row)
        if self.encoding() == self.JSON:
            web.header('Content-Type', 'application/json')
            return render.table_json("attrs",
                                     [ "name", "id", "sos_type",
                                       "indexed", "card", "dups",
                                       "min_key", "max_key" ],
                                     rows, len(rows));
        return render.table("Attribute Table",
                            [ "Name", "Id", "Type",
                              "Indexed", "Card", "Dups", "MinKey", "MaxKey" ],
                            rows, len(rows))

class SosInfo:
    def GET(self):
        rows = [ [session.containerName,
                 session.indexName,
                 session.recordNo,
                 session.pos ]
        ]
        return render.table("Sesion",
                            [ "Container Name", "Index Name", "Record no", "Position" ],
                            rows, len(rows))

class SosQuery(SosRequest):
    """
    This is the base class for the SosTable and SosGraph classes. It handles
    all of the query preparation, such as creating the filter, advancing to
    the first matching element, etc. The SosTable and SosGraph should see the
    exact same set of records, i.e. only the presentation will change,
    specifically JSON vs. PNG
    """
    def __init__(self):
        super( SosQuery, self ).__init__()

    def reset(self):
        session.indexName = self.index_name
        session.containerName = self.container().name()
        session.pos = None
        session.recordNo = None

    def parse_query(self):
        self.parms = web.input()
        try:
            self.parse_request(self.parms)
        except Exception as e:
            return (1, "Exception in parse_request: {0}".format(e), None)

        if not self.schema():
            return (1, "A 'schema' clause must be specified.", None)

        if not 'index' in self.parms:
            return (1, "An 'index' clause must be specified.", None)

        #
        # Open an iterator on the container
        #
        self.index_attr = None
        try:
            self.index_name = self.parms.index
            if self.index_name != session.indexName:
                self.reset()
            if self.container().name() != session.containerName:
                self.reset()
            self.index_attr = self.schema().attr(self.index_name)
            self.iter_ = self.index_attr.iterator()
            self.filt = SOS.Filter(self.iter_)
            if 'unique' in self.parms:
                self.unique = True
                self.filt.unique()
            else:
                self.unique = False
            self.card = self.iter_.cardinality()
            if self.unique:
                self.card = self.card - self.iter_.duplicates()
        except Exception as e:
            return (1, "The attribute {0} was not found "
                    "in the schema.".format(str(e) + self.index_name), None)

        if 'x_axis' in self.parms:
            self.x_axis = self.parms.x_axis
        else:
            self.x_axis = self.index_name

        #
        # Parse the select clause. The view_cols contains the index as it's first element.
        #
        self.view_cols = []
        if 'select' in self.parms:
            for attr_name in self.parms.select.split(','):
                if attr_name != self.index_name:
                    self.view_cols.append(attr_name)
        else:
            for attr_name, attr in self.schema().attrs.iteritems():
                if attr_name != self.index_name:
                    self.view_cols.append(attr_name)
        #
        # Parse the where clause
        #
        #
        # A filter is an array of conditions
        #
        if 'where' in self.parms:
            try:
                where = self.parms.where
                conds = where.split(',')
                for cond in conds:
                    tokens = cond.split(':')
                    if len(tokens) < 3:
                        return (1, "Invalid where clause '{0}', "
                                "valid syntax is attr:cmp_str:value".format(cond), None)
                    attr_name = tokens[0]
                    attr = self.schema().attr(attr_name)
                    if not attr:
                        return (1, "The attribute {0} was not found "
                                "in the schema {1}".format(attr_name, self.schema().name()), None)
                    cmp_str = tokens[1].lower()
                    value_str = None
                    for s in tokens[2:]:
                        if value_str:
                            value_str = value_str + ':' + s
                        else:
                            value_str = s
                    self.filt.add(attr, tokens[1], value_str)
            except Exception as e:
                return (1, "Exception processing where clause: {0}".format(str(e)), None)

        obj = None
        if self.start == 0:
            self.reset()
            obj = self.filt.begin()
        elif self.start + self.count >= self.card:
            self.reset()
            obj = self.filt.end()
            skip = self.card % self.count
            while obj and skip > 0:
                obj = self.filt.prev()
                skip = skip - 1
        else:
            if session.pos:
                self.filt.set(session.pos)
                skip = self.start - session.recordNo
            else:
                self.filt.begin()
                skip = self.start
            obj = self.filt.obj()
            while obj and skip != 0:
                if skip > 0:
                    obj = self.filt.next()
                    skip = skip - 1
                else:
                    obj = self.filt.prev()
                    skip = skip + 1

        rc, pos = self.filt.pos()
        if rc == 0:
            session.pos = pos
            session.recordNo = self.start

        return (0, None, obj)

class SosTable(SosQuery):
    def GET(self):
        rc, msg, obj = self.parse_query()
        if rc != 0:
            return render.error(msg);

        tbl_hdr = [ 'RecNo', self.index_name ]
        for attr_name in self.view_cols:
            tbl_hdr.append(attr_name)

        rows = []
        count = 0
        while obj is not None and count < self.count:
            row = [ session.recordNo + count, obj.values[self.index_name] ]
            for attr_name in self.view_cols:
                if attr_name == self.index_name:
                    continue
                try:
                    value = str(obj.values[attr_name])
                except:
                    value = "bad_name"
                row.append(value)
            rows.append(row)
            count = count + 1
            obj = self.filt.next()

        if self.encoding() == self.TABLE:
            rows.append([ "{0} rows".format(count) ])
            return render.table(self.schema().name(), tbl_hdr, rows, self.card)

        web.header('Content-Type', 'application/json')
        return render.table_json(self.schema().name(), tbl_hdr, rows, self.card)

class SosGraph2:
    def __init__(self):
        session.indexName = None
        session.containerName = None
        session.pos = None
        session.recordNo = None

    def GET(self):
        parms = web.input()
        if 'container' in parms:
            container = parms.container
        if 'job_id' in parms:
            job_id = parms.job_id
        if 'metric_name' in parms:
            metric_name = parms.metric_name
        if 'start' in parms:
            start = parms.start
        if 'end' in parms:
            end = parms.end
        if 'duration' in parms:
            duration = parms.duration
        else:
            duration = 3600
        web.header('Content-Type', 'text/html')
        return render.metric_graph(container, job_id, metric_name, start, end, duration)

class SosGraph(SosQuery):
    def GET(self):
        rc, msg, obj = self.parse_query()
        if rc != 0:
            return render.error(msg);

        x_axis = []
        series = {}
        for attr_name in self.view_cols:
            series[attr_name] = []

        count = 0
        maxDt = datetime.datetime.fromtimestamp(0)
        minDt = datetime.datetime.now()
        while obj and count < self.count:
            t = obj.values[self.x_axis]
            dt = datetime.datetime.fromtimestamp(t.seconds()) + datetime.timedelta(seconds=count)
            # dt = datetime.datetime.now() + datetime.timedelta(seconds=count)
            if dt < minDt:
                minDt = dt
            if dt > maxDt:
                maxDt = dt
            x_axis.append(dt)
            for attr_name in self.view_cols:
                if attr_name == self.index_name:
                    continue
                try:
                    value = float(obj.values[attr_name])
                except:
                    value = 0.0
                series[attr_name].append(value)
            count = count + 1
            obj = self.filt.next()

        figure = Figure(figsize=(14,5))
        axis = figure.add_axes([0.1, 0.3, 0.8, .6])
        mfc = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k' ]
        ls = []
        for c in mfc:
            ls.append(c + 'o-')
        line_no = 0
        # axis.set_autoscalex_on(True)
        # axis.set_xlim(auto=True)
        # axis.set_xbound(lower=minDt, upper=maxDt)
        axis.set_xmargin(.01)
        for attr_name in self.view_cols:
            if line_no >= len(mfc):
                line_no = 0
            line,  = axis.plot(x_axis, series[attr_name],
                               ls[line_no], markersize=6, markerfacecolor=mfc[line_no],
                               label=attr_name, linewidth=1)
            line_no = line_no + 1
        axis.legend()
        axis.set_title(self.schema().name())
        axis.set_ylabel('Binkus: {0} recs'.format(len(x_axis)))
        minutes = mdates.MinuteLocator()
        seconds = mdates.SecondLocator()
        dateFmt = mdates.DateFormatter("%H")
        axis.grid(True)
        figure.autofmt_xdate()

        canvas = FigureCanvasAgg(figure)
        imgdata = StringIO.StringIO()
        canvas.print_png(imgdata, dpi=150)
        web.header('Content-Type', 'image/png')
        web.header('Cache-Control', 'no-store, no-cache, must-revalidate');
        return imgdata.getvalue()

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()

sos_root = '/NVME/0/SOS_ROOT'
rootdir = os.path.abspath(os.path.dirname(__file__)) + '/../'
render = web.template.render(rootdir+'templates/')
app = web.application(urls, globals(), autoreload=False)
application = app.wsgifunc()
context = {
    'containerName' : None,
    'indexName' : None,
    'recordNo' : None,
    'pos' : None
}
session = web.session.Session(app, web.session.DiskStore('sessions'), context)
