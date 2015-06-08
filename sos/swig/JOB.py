#!/usr/bin/python
import time
import sys
import SOS
from job import *
import os

class Iterator:
    def __init__(self, container, order=None):
        self.container = container
        if order is None:
            self.order = JOB_ORDER_BY_TIME
        else:
            self.order = order
        self.iter = job_iter_new(container.container, self.order)

    def begin(self):
        obj = job_iter_begin_job(self.iter)
        if obj is not None:
            return SOS.Object(obj)
        return None

    def next(self):
        obj = job_iter_next_job(self.iter)
        if obj is not None:
            return SOS.Object(obj)
        return None

    def prev(self):
        obj = job_iter_next_job(self.iter)
        if obj is not None:
            return SOS.Object(obj)
        return None

    def end(self):
        obj = J.job_iter_next_job(self.iter)
        if obj is not None:
            return SOS.Object(obj)
        return None

    def begin_sample(self):
        self.sample = job_iter_begin_sample(self.iter, JOB_METRIC_VAL)
        return self.sample

    def next_sample(self):
        if self.sample and self.sample.status == JOB_ITER_END:
            return None
        self.sample = job_iter_next_sample(self.iter)
        return self.sample

    def find(self, jobId):
        obj = job_iter_find_job_by_id(self.iter, int(jobId))
        if obj is not None:
            return SOS.Object(obj)
        return None

    def __del__(self):
        if self.iter:
            job_iter_free(self.iter);

if __name__ == "__main__":
    import datetime
    c = SOS.Container("/DATA/bwx")
    SOS.Object.def_fmt = SOS.Object.table_fmt
    jobIter = Iterator(c, JOB_ORDER_BY_TIME)
    job = jobIter.begin()
    print(job.table_header())
    while job is not None:
        print(job)
        job = jobIter.next()
        sample = jobIter.begin_sample()
        print(sample.name(0))
        while sample is not None and sample.status != JOB_ITER_END:
            sample = jobIter.next_sample()
            dt = datetime.datetime.fromtimestamp(sample[0].comp_mean_xi)
            print("{0} {1}".format(dt, sample[0].comp_mean_xi))
