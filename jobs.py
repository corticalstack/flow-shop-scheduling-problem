import logging
import logger as lg


class Jobs:
    """

    """

    def __init__(self):
        self.quantity = 0
        self.joblist = []
        self.job_total_units = []
        self.logger = logging.getLogger()

    def add(self, jobs, machines):
        job_times = [int(n) for n in jobs.split()]
        for ji, jt in enumerate(job_times):
            try:
                self.joblist[ji].append(jt)
            except IndexError:
                self.joblist.append([jt])

    def set_job_total_units(self):
        self.job_total_units = [sum(j) for j in self.joblist]
        if logging.DEBUG >= self.logger.level:
            for ji, j in enumerate(self.job_total_units):
                lg.message(logging.DEBUG, 'Job {} allocated {} time units'.format(ji, j))
