class Jobs:
    """

    """

    def __init__(self):
        self.quantity = 0
        self.joblist = []

    def add(self, jobs, machines):
        job_times = [int(n) for n in jobs.split()]
        for ji, jt in enumerate(job_times):
            try:
                self.joblist[ji].append(jt)
            except:
                self.joblist.append([jt])
