import logging
import logger as lg


class Machines:
    def __init__(self):
        self.quantity = 0
        self.assigned_jobs = []
        self.loadout_times = []
        self.lower_bounds_taillard = []

    def set_loadout_times(self, jobs):
        for m in range(self.quantity):
            loadout = sum(i[m] for i in jobs.joblist)
            self.loadout_times.append(loadout)
            lg.message(logging.DEBUG, 'Machine {} loaded with {} time units'.format(m, loadout))

    def set_lower_bounds_taillard(self, jobs, ilb):
        for m in range(self.quantity):
            lb = self.loadout_times[m]
            minimum_before_machine_start = []
            minimum_after_machine_start = []
            for j in jobs.joblist:
                if m > 0:
                    minimum_before_machine_start.append(sum(j[:m]))
                if m < self.quantity:
                    minimum_after_machine_start.append(sum(j[m+1:]))
            if minimum_before_machine_start:
                lb += min(minimum_before_machine_start)
            if minimum_after_machine_start:
                lb += min(minimum_after_machine_start)
            self.lower_bounds_taillard.append(lb)
            lg.message(logging.DEBUG, 'Machine {} Taillard lower bound is {} time units'.format(m, lb))

        lg.message(logging.INFO, 'Calculated Taillard benchmark instance lower bound (max) is {} time units'.format(
            max(self.lower_bounds_taillard)))

        if max(self.lower_bounds_taillard) != ilb:
            lg.message(logging.WARNING, 'Calculated Taillard benchmark instance ({}) not equal to lower bound in '
                                        'benchmark instance file ({})'.format(max(self.lower_bounds_taillard), ilb))

    def times(self, permutation, solver):
        total_idle_time = 0
        fitness = solver.calculate_fitness(permutation)  # set machine assigned jobs to best permutation
        lg.message(logging.INFO, 'Machine\tStart Time\tFinish Time\tIdle Time')

        # Calculate idle time from list tuples as start time(m+1) - finish time(m). Include last machine start time
        for mi, m in enumerate(self.assigned_jobs):
            finish_time = m[-1][2]
            idle_time = sum([x[1]-x[0] for x in zip([x[2] for x in m], [x[1] for x in m[1:] + [(0, m[-1][2], 0)]])])
            total_idle_time += idle_time
            lg.message(logging.INFO, '{}\t\t{}\t\t{}\t\t{}'.format(str(mi), str(m[0][1]), str(finish_time),
                                                                   str(idle_time)))
        lg.message(logging.INFO, 'Machines total idle time is {}'.format(total_idle_time))
