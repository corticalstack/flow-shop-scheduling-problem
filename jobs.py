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

    def times(self, permutation, machines, solver):
        total_idle_time = 0
        fitness, _ = solver.calculate_fitness(permutation, 1)  # set machine assigned jobs to best permutation

        lg.message(logging.INFO, 'Job\tStart Time\tFinish Time\tIdle Time')
        for pi, p in enumerate(permutation):
            start_time = 0
            end_time = 0
            idle_time = 0
            for ji, j in enumerate(machines.assigned_jobs):
                if ji == 0:
                    start_time = j[pi][1]
                    end_time = j[pi][2]
                    continue
                idle_time += j[pi][1] - end_time
                end_time = j[pi][2]
            lg.message(logging.INFO, '{}\t\t{}\t\t{}\t\t{}'.format(str(p), str(start_time), str(end_time),
                                                                   str(idle_time)))
            total_idle_time += idle_time
        lg.message(logging.INFO, 'Jobs total idle time is {}'.format(total_idle_time))


