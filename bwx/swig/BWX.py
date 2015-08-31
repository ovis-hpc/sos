#!/usr/bin/python
import time
import sys
import sos.sos
import sos.SOS
import os

class Iterator(object):
    def __init__(self, container, schemaName, attrName, order=None):
        self.container = container
        self.schema = self.container.schema(schemaName)
        self.attr = self.schema.attr(attrName)

    def key(self):
        return sos.SOS.Key(self.attr)

    def begin(self):
        return self.attr.iterator().begin()

    def next(self):
        return self.attr.iterator().next()

    def prev(self):
        return self.attr.iterator().prev()

    def end(self):
        return self.attr.iterator().end()

    def inf(self, key):
        return self.attr.iterator().inf(key)

    def sup(self, key):
        return self.attr.iterator().sup(key)

def set_er_up():
    sos.SOS.Object.def_fmt = SOS.Object.table_fmt

    container = sos.SOS.Container("/NVME/0/SOS_ROOT/BWX_Job_Data")
    job_iter = Iterator(container, "Job", "Id")
    sample_iter = Iterator(container, "Sample", "JobTime")
    return (job_iter, sample_iter)

if __name__ == "__main__":
    import datetime

    sos.SOS.Object.def_fmt = sos.SOS.Object.table_fmt

    container = sos.SOS.Container("/NVME/0/SOS_ROOT/BWX_Job_Data")
    job_iter = Iterator(container, "Job", "Id")
    sample_iter = Iterator(container, "Sample", "JobTime")
    sample_key = sample_iter.key()

    job = job_iter.begin()
    print(job.table_header())
    while job is not None:
        print(job)
        jobtime = int(job.Id) << 32
        sample_key.set(str(jobtime))
        sample = sample_iter.sup(sample_key)
        while sample and (int(sample.JobTime) >> 32) == job.Id:
            print(sample.current_freemem)
            sample = sample_iter.next()
        job = job_iter.next()

