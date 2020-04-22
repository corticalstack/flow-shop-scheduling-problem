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
