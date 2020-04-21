class Machines:
    def __init__(self):
        self.quantity = 0
        self.assigned_jobs = []
        self.loadout_times = []
        self.lower_bounds_taillard = []

    def set_loadout_times(self, jobs):
        for m in range(self.quantity):
            self.loadout_times.append(sum(i[m] for i in jobs.joblist))

    def set_lower_bounds_taillard(self, jobs):
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
