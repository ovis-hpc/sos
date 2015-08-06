import sys
import os
import math
import web
import sos
import SOS
import job
import datetime
import tempfile
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter
import StringIO, Image

urls = (
    '/plot', 'JobPlot'
    )

class MyFormatter(Formatter):
    def __init__(self, fmt='%H:%M:%S'):
        self.fmt = fmt

    def __call__(self, x, pos=0):
        dt = datetime.datetime.fromtimestamp(x)
        # sys.stderr.write('%d %s\n'%(x, dt.strftime(self.fmt)))
        return dt.strftime(self.fmt)

class MyYFormatter(Formatter):
    def __init__(self, fmt='%.2f%s'):
        self.fmt = fmt

    def __call__(self, x, pos=0):
        if (x < 1.0e3):
            return self.fmt%(x, ' ')
        if (x < 1.0e6):
            return self.fmt%(x / 1.0e3, 'K')
        if (x < 1.0e9):
            return self.fmt%(x / 1.0e6, 'M')
        if (x < 1.0e12):
            return self.fmt%(x / 1.0e9, 'G')
        return self.fmt%(x / 1.0e12, 'T')

class OldJobPlot(object):
    def __init__(self):
        self.container = None
        self.sampleSchema = None
        self.startAttr = None
        self.endAttr = None
        self.sampleFilt = None

    def GET(self):
        start_dt = datetime.datetime.now()
        input = web.input()
        container = input.container
        self.container = SOS.Container(str(sos_root + '/' + container),
                                       mode=SOS.Container.RO)

        mfc = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k' ]
        ls = []
        for c in mfc:
            ls.append(c + 'o-')

        x_axis = []

        if 'job_id' in input:
            job_id = input.job_id
        else:
            return(5, 'A job_id must be specified.')

        self.iter = JobIterator(self.container, "Sample", "JobTime")
        sample = self.iter.sup(job_id)
        if not job:
            return (1, 'The specified job was not found')

        col_dict = {}
        for i in range(0, len(sample)):
            col_dict[sample.name(i)] = i

        plot_cols = []
        series = []
        mu = []
        sigma = []
        plot_names = []
        if 'plot' in input:
            for col in input.plot.split(','):
                try:
                    i = col_dict[col]
                    plot_cols.append(i)
                    series.append([])
                    plot_names.append(col)
                    mu.append(0.0)
                    sigma.append(0.0)
                except:
                    return (3, 'The specified metric is not present in the sample')
        else:
            return (3, 'The plot clause must be specified.')

        diff = 'diff' in input
        if diff:
            # Skip first sample
            sample = self.iter.next_sample()
        while sample is not None:
            dt = datetime.datetime.fromtimestamp(sample[0].comp_mean_xi)
            x_axis.append(sample[0].comp_mean_xi)
            plot_no = 0
            for metric in plot_cols:
                if diff:
                    series[plot_no].append(sample[metric].diff_xi)
                else:
                    series[plot_no].append(sample[metric].comp_mean_xi)
                    plot_no = plot_no + 1
            nextSample = self.iter.next_sample()
            if not nextSample:
                plot_no = 0
                for metric in plot_cols:
                    mu[plot_no] = sample[metric].time_mean_xi
                    std = sample[metric].sum_xi_sq \
                          - (2.0 * sample[metric].time_mean_xi * sample[metric].sum_xi) \
                          + (sample[metric].i * (sample[metric].time_mean_xi * sample[metric].time_mean_xi))
                    std = math.sqrt(std)
                    sigma[plot_no] = std
                    plot_no = plot_no + 1
            sample = nextSample
        figure = Figure(figsize=(10,2.5),facecolor='w')
        if len(plot_cols) == 1:
            textstr = '$\mu=%.2f$\n$\sigma=%.2f$'%(mu[0], sigma[0])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axis = figure.add_axes([0.1, 0.225, 0.875, 0.65], axisbg='w')
            axis.text(0.025, 0.95, textstr, transform=axis.transAxes,
                      fontsize=10, verticalalignment='top', bbox=props)
        else:
            axis = figure.add_axes([0.1, 0.225, 0.5, 0.65], axisbg='w')
        # axis = figure.add_axes(axisbg='w')
        # axis.set_axis_bgcolor('black')
        axis.tick_params(labelsize=10, direction='out')
        plot_no = 0
        for y_axis in series:
            line,  = axis.plot(x_axis, y_axis,
                               ls[plot_no], markersize=6, markerfacecolor=mfc[plot_no],
                               label=plot_names[plot_no], linewidth=1)
            plot_no = plot_no + 1
        if plot_no == 1:
            axis.legend(loc='best', fancybox=True, framealpha=0.5)
        else:
            axis.legend(bbox_to_anchor=(1.25, 0.0, .5, 1.0), fancybox=True, framealpha=0.5)
        # axis.legend(bbox_to_anchor=(1, 1),
        # bbox_transform=plt.gcf().transFigure,
        # fancybox=True, framealpha=0.5)
        # axis.legend(bbox_to_anchor=(.5, .5, .5, .5), bbox_transform=plt.gcf().transFigure)
        axis.set_title(job.JobName() + "[" + job.UserName() + "]", fontsize=10)
        axis.set_ylabel('{0} records'.format(len(x_axis)))
        axis.xaxis.set_major_formatter(MyFormatter())
        axis.grid(True)
        figure.autofmt_xdate()
        dur = datetime.datetime.now() - start_dt
        secs = dur.seconds + (dur.microseconds / 1.0e6);
        textstr = 'render time=%.4f(s)\njob size=%.0f'%(secs, job.JobSize())
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        axis.text(0.70, 1.16, textstr, transform=axis.transAxes,
                  fontsize=10, verticalalignment='top', bbox=props)

        canvas = FigureCanvasAgg(figure)
        imgdata = StringIO.StringIO()
        canvas.print_png(imgdata, dpi=150)
        web.header('Content-Type', 'image/png')
        web.header('Cache-Control', 'no-store, no-cache, must-revalidate');
        return imgdata.getvalue()

class JobIterator(object):
    def __init__(self, container, schemaName, attrName, order=None):
        self.container_ = container.container
        self.schema_ = sos.sos_schema_by_name(self.container_, schemaName)
        self.attr_ = sos.sos_schema_attr_by_name(self.schema_, attrName)
        self.iter_ = sos.sos_attr_iter_new(self.attr_)

    def key_set(self, key, val):
        sos.sos_attr_key_from_str(self.attr_, key, val)

    def key(self, size=0):
        return sos.sos_attr_key_new(self.attr_, 0)

    def put(self, obj):
        sos.sos_obj_put(obj)

    def begin(self):
        rc = sos.sos_iter_begin(self.iter_)
        if rc:
            return None
        return sos.sos_iter_obj(self.iter_)

    def next(self):
        rc = sos.sos_iter_next(self.iter_)
        if rc:
            return None
        return sos.sos_iter_obj(self.iter_)

    def prev(self):
        rc = sos.sos_iter_prev(self.iter_)
        if rc:
            return None
        return sos.sos_iter_obj(self.iter_)

    def end(self):
        rc = sos.sos_iter_end(self.iter_)
        if rc:
            return None
        return sos.sos_iter_obj(self.iter_)

    def inf(self, key):
        rc = sos.sos_iter_inf(self.iter_, key)
        if rc:
            return None
        return sos.sos_iter_obj(self.iter_)

    def sup(self, key):
        rc = sos.sos_iter_sup(self.iter_, key)
        if rc:
            return None
        return sos.sos_iter_obj(self.iter_)

class JobPlot(object):
    def __init__(self):
        self.container = None
        self.sampleSchema = None
        self.startAttr = None
        self.endAttr = None
        self.sampleFilt = None

    def GET(self):
        start_dt = datetime.datetime.now()
        input = web.input()
        container = input.container
        self.container = SOS.Container(str(sos_root + '/' + container),
                                       mode=SOS.Container.RO)

        mfc = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k' ]
        ls = []
        for c in mfc:
            ls.append(c + 'o-')

        x_axis = []

        if 'job_id' in input:
            job_id = int(input.job_id)
        else:
            return(5, 'A job_id must be specified.')

        if 'duration' in input:
            duration = int(input.duration)
        else:
            duration = 3600

        if 'start' in input:
            start_secs = int(float(input.start))
        else:
            start_secs = 0

        self.iter = JobIterator(self.container, "Sample", "JobTime")
        sample_key = self.iter.key()

        # sys.stderr.write('start %d\n'%(start_secs))
        self.iter.key_set(sample_key, str((job_id << 32) | start_secs))
        sample_obj = self.iter.sup(sample_key)
        if not sample_obj:
            return (1, 'The specified job was not found')
        sample = job.job_sample(sample_obj)
        if start_secs == 0:
            start_secs = float(sample.JobTime.secs)
        series = {}
        metric_name = str(input.plot)
        metric_id = sample.idx(metric_name)
        x_axis_comp = sample.CompId
        while sample_obj is not None:
            sample = job.job_sample(sample_obj)
            # sys.stderr.write('%d==%d time %d\n'%(sample.JobTime.id, job_id, sample.JobTime.secs))
            if sample.JobTime.id != job_id:
                break

            comp_id = sample.CompId
            cur_secs = float(sample.JobTime.secs)
            if cur_secs - start_secs > duration:
                break

            if comp_id == x_axis_comp:
                x_axis.append(cur_secs)

            if comp_id not in series:
                y_axis = []
                series[comp_id] = y_axis
            else:
                y_axis = series[comp_id]

            y_axis.append(sample[metric_id])
            sos.sos_obj_put(sample_obj)
            sample_obj = self.iter.next()

        dur = datetime.datetime.now() - start_dt
        secs0 = dur.seconds + (dur.microseconds / 1.0e6);

        figure = Figure(figsize=(10,2.5),facecolor='w')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axis = figure.add_axes([0.1, 0.225, 0.875, 0.65], axisbg='w')
        x_axis_len = len(x_axis)
        axis.tick_params(labelsize=10, direction='out')
        plot_no = 0
        for comp_id in series:
            y_axis = series[comp_id]
            y_axis_len = len(y_axis)
            axis_len = min(x_axis_len, y_axis_len)
            plot_no = plot_no + 1
            if plot_no > len(ls) - 1:
                plot_no = 0
            line,  = axis.plot(x_axis[:axis_len], y_axis[:axis_len],
                               ls[plot_no], markersize=6, markerfacecolor=mfc[plot_no],
                               linewidth=1)
        axis.legend(loc='best', fancybox=True, framealpha=0.5)
        axis.set_title(metric_name + '[' + str(len(series)) + ']', fontsize=10)
        axis.set_ylabel('{0} records'.format(len(x_axis)))
        axis.xaxis.set_major_formatter(MyFormatter())
        axis.yaxis.set_major_formatter(MyYFormatter())
        axis.grid(True)
        figure.autofmt_xdate()
        dur = datetime.datetime.now() - start_dt
        secs = dur.seconds + (dur.microseconds / 1.0e6);
        textstr = 'Render Time: %.2f / %.2f (s)'%(secs0, secs)
        axis.text(0.050, 1.1, textstr, transform=axis.transAxes,
                  fontsize=10, verticalalignment='top', bbox=props)
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
