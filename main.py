from jobs import Jobs
from machines import Machines
from fsspSolver import *
from stats import Stats
from visualisation import Visualisation
import os
import sys
import logging
from datetime import datetime
import logger as lg

import plotly.figure_factory as ff
import matplotlib.font_manager as font_manager
script_name = os.path.basename(sys.argv[0]).split('.')
import time



class Fssp:
    """
    Flow Shop Scheduling Problem
    """
    def __init__(self):
        self.sample_runs = 5  # 30
        self.algorithms = [{'Id': 'SA', 'Enabled': True},
                           {'Id': 'GA', 'Enabled': True}]
        self.fitness_trend = {}

        instances = ['taillard_20_10_i1']
        #instances = ['taillard_20_5_i1']
        for inst in instances:

            # New job and machine instance for each benchmark instance
            self.jobs = Jobs()
            self.machines = Machines()
            self.load_instance(inst)
            self.jobs.set_job_total_units()
            self.machines.set_loadout_times(self.jobs)
            self.machines.set_lower_bounds_taillard(self.jobs)

            for alg in self.algorithms:
                if alg['Enabled']:
                    self.fitness_trend[alg['Id']] = []
                    cls = globals()[alg['Id']]
                    solver = cls(self.jobs, self.machines)
                    #solver.brute_force_generate_all_permutations()
                    for i in range(self.sample_runs):

                        # Invoke class dynamically
                        fitness, permutation, trend = solver.solve()
                        Visualisation.plot_fitness_trend(trend)

                        self.fitness_trend[alg['Id']].append(fitness)

            Stats.basic(self.fitness_trend)
            #Stats.wilcoxon(self.fitness_trend)


    def load_instance(self, inst):
        filename = 'instances/' + inst + '.txt'
        with open(filename, 'r') as f:
            line = f.readlines()
            for i, job_detail in enumerate(line):
                job_detail = job_detail.strip('\n')
                if i == 0:
                    self.jobs.quantity, self.machines.quantity = [int(n) for n in job_detail.split()]
                else:
                    self.jobs.add(job_detail, self.machines.quantity)



if __name__ == "__main__":
    log_filename = str(script_name[0] + ('_') + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

    logging.basicConfig(filename='logs/' + log_filename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    fssp = Fssp()

