import sys
import os
import math
import web
import argparse
from datetime import datetime
import tempfile
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter
import StringIO, Image

from sos import *
from bwx import *

urls = (
    '/plot', 'WwwJobPlot'
    )

class MyXFormatter(Formatter):
    def __init__(self, fmt='%H:%M:%S'):
        self.fmt = fmt

    def __call__(self, x, pos=0):
        dt = datetime.fromtimestamp(x)
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

class JobPlot(object):
    def __init__(self, container, job_id, metric, start=0, duration=3600, output='gui'):
        self.containerName = container
        self.metricName = metric
        self.jobId = job_id
        self.startSecs = start
        self.duration = duration
        self.output = output

    def plot(self):
        start_dt = datetime.now()
        mfc = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k' ]
        ls = []
        for c in mfc:
            ls.append(c + 'o-')

        x_axis = []

        self.container = SOS.Container(self.containerName,
                                       mode=SOS.Container.RO)
        self.iter = SOS.Iterator(self.container, "Sample", "JobTime")
        sample_key = self.iter.key()

        self.iter.key_set(sample_key, str((self.jobId << 32) | self.startSecs))
        sample_obj = self.iter.sup(sample_key)
        if not sample_obj:
            return (1, 'There are no samples for the specified job')

        sample = bwx.job_sample(sample_obj)
        if self.startSecs == 0:
            self.startSecs = float(sample.JobTime.secs)
        series = {}
        metric_id = sample.idx(self.metricName)
        x_axis_comp = sample.CompId
        while sample_obj is not None:
            sample = bwx.job_sample(sample_obj)
            if sample.JobTime.id != self.jobId:
                break

            comp_id = sample.CompId
            cur_secs = float(sample.JobTime.secs)
            if cur_secs - self.startSecs > self.duration:
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

        dur = datetime.now() - start_dt
        secs0 = dur.seconds + (dur.microseconds / 1.0e6);

        if self.output == 'gui':
            figure = plt.figure(figsize=(10,2.5),facecolor='w')
        else:
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
        axis.set_title(self.metricName + '[' + str(len(series)) + ']', fontsize=10)
        axis.set_ylabel('{0} records'.format(len(x_axis)))
        axis.xaxis.set_major_formatter(MyXFormatter())
        axis.yaxis.set_major_formatter(MyYFormatter())
        axis.grid(True)
        figure.autofmt_xdate()
        dur = datetime.now() - start_dt
        secs = dur.seconds + (dur.microseconds / 1.0e6);
        textstr = 'Render Time: %.2f / %.2f (s)'%(secs0, secs)
        axis.text(0.050, 1.1, textstr, transform=axis.transAxes,
                  fontsize=10, verticalalignment='top', bbox=props)
        if self.output == 'gui':
            # print("Presenting Job Plot")
            plt.show()
            return None

        # print("Rending Job Plot as PNG image file")
        canvas = FigureCanvasAgg(figure)
        imgdata = StringIO.StringIO()
        canvas.print_png(imgdata, dpi=150)
        web.header('Content-Type', 'image/png')
        web.header('Cache-Control', 'no-store, no-cache, must-revalidate');
        return imgdata.getvalue()

class WwwJobPlot(object):
    def __init__(self):
        self.container = None

    def GET(self):
        args = web.input()
        if 'container' not in args:
            return render.error("The 'container' parameter must be specified.")
        container = sos_root + '/' + str(args.container)

        if 'job_id' not in args:
            return render.error("The 'job_id' parameter must be specified.")
        job_id = int(args.job_id)

        if 'metric_name' not in args:
            return render.error("The 'metric_name' parameter must be specified.")
        metric_name = str(args.metric_name)

        if 'duration' in args:
            duration = int(args.duration)
        else:
            duration = 3600

        if 'start' in args:
            start_secs = int(float(args.start))
        else:
            start_secs = 0

        job = JobPlot(container, job_id, metric_name,
                      start_secs, duration, 'www')
        return job.plot()

    def GET_(self):
        start_dt = datetime.now()
        args = web.input()
        container = args.container
        self.container = SOS.Container(str(sos_root + '/' + container),
                                       mode=SOS.Container.RO)

        mfc = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k' ]
        ls = []
        for c in mfc:
            ls.append(c + 'o-')

        x_axis = []

        if 'job_id' in args:
            job_id = int(args.job_id)
        else:
            return(5, 'A job_id must be specified.')

        if 'duration' in args:
            duration = int(args.duration)
        else:
            duration = 3600

        if 'start' in args:
            start_secs = int(float(args.start))
        else:
            start_secs = 0

        self.iter = SOS.Iterator(self.container, "Sample", "JobTime")
        sample_key = self.iter.key()

        self.iter.key_set(sample_key, str((job_id << 32) | start_secs))
        sample_obj = self.iter.sup(sample_key)
        if not sample_obj:
            raise Exception('The specified job {0} was not found at start {1}'.
                            format(job_id, start_secs))
        sample = bwx.job_sample(sample_obj)
        if start_secs == 0:
            start_secs = float(sample.JobTime.secs)

        series = {}
        # sys.stderr.write("ARGS {0}\n".format(str(args)))
        metric_name = str(args.metric_name)
        metric_id = sample.idx(metric_name)
        x_axis_comp = sample.CompId
        while sample_obj is not None:
            sample = bwx.job_sample(sample_obj)
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

        dur = datetime.now() - start_dt
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
        axis.xaxis.set_major_formatter(MyXFormatter())
        axis.yaxis.set_major_formatter(MyYFormatter())
        axis.grid(True)
        figure.autofmt_xdate()
        dur = datetime.now() - start_dt
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
    parser = argparse.ArgumentParser(description="Plot a Job's metric data.")
    parser.add_argument("--mode", default='web',
                        help="Specify 'www' for wsgi app, or 'gui' for real-time")
    parser.add_argument("--container",
                        help="Specify the container path")
    parser.add_argument("--job_id", type=int,
                        help="Specify the Job Id")
    parser.add_argument("--metric-name", default="current_freemem",
                        help="Specify the metric name to be plotted")
    parser.add_argument("--start-time",
                        help="Specify the start time, format is YYYY/MM/DD HH:MM:SS")
    parser.add_argument("--duration",
                        type=int, default=3600,
                        help="Specify the duration, format is seconds ")
    args = parser.parse_args()
    if args.mode == 'gui':
        if args.start_time:
            dt = datetime.strptime(args.start_time, "%Y/%m/%d %H:%M:%S")
            start_secs = int(dt.strftime("%s"))
        else:
            start_secs = 0
        jobPlot = JobPlot(args.container, args.job_id, args.metric_name,
                          start_secs, args.duration, args.mode)
        jobPlot.plot()
        sys.exit(0)

    app = web.application(urls, globals())
    app.run()

sos_root = '/NVME/0/SOS_ROOT'
rootdir = os.path.abspath(os.path.dirname(__file__)) + '/../'
render = web.template.render(rootdir+'templates/')
app = web.application(urls, globals(), autoreload=False)
application = app.wsgifunc()
