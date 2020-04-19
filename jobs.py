class Jobs:
    """

    """

    def __init__(self):
        self.quantity = 0
        self.joblist = []

    def add(self, j):
        job_times = [int(n) for n in j.split()]
        self.joblist.append(job_times)
