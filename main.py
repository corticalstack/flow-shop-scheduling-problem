from jobs import Jobs
from machines import Machines
from fsspSolver import *
from stats import Stats
from visualisation import Visualisation
import os
import sys
import logging
import time
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
        lg.message(logging.INFO, 'Starting flow shop scheduling problem')
        self.sample_runs = 30  # 30
        self.algorithms = [{'Id': 'SA', 'Enabled': True},
                           {'Id': 'GA', 'Enabled': True}]
        self.fitness_trend = {}
        self.instance_lower_bound = 0
        self.instance_upper_bound = 0  # Best known minimisation

        instances = ['taillard_20_10_i1']
        #instances = ['taillard_20_5_i1']
        for inst in instances:
            lg.message(logging.INFO, 'Processing benchmark instance {}'.format(inst))
            # New job and machine instance for each benchmark instance

            self.jobs = Jobs()
            self.machines = Machines()
            self.load_instance(inst)

            self.jobs.set_job_total_units()
            self.machines.set_loadout_times(self.jobs)
            self.machines.set_lower_bounds_taillard(self.jobs, self.instance_lower_bound)

            for alg in self.algorithms:
                if alg['Enabled']:
                    lg.message(logging.INFO, 'Processing algorithm {}'.format(alg['Id']))
                    self.fitness_trend[alg['Id']] = []
                    cls = globals()[alg['Id']]
                    solver = cls(self.jobs, self.machines)

                    lg.message(logging.INFO, 'Executing {} sample runs'.format(self.sample_runs))
                    alg_runs_time_to_complete = 0
                    for i in range(self.sample_runs):
                        # Invoke class dynamically
                        alg_run_start_time = time.time()
                        fitness, permutation, trend = solver.solve()
                        alg_runs_time_to_complete += time.time() - alg_run_start_time
                        lg.message(logging.INFO, 'Run {} fitness is {} with permutation {}'.format(i, fitness,
                                                                                                   permutation))
                        Visualisation.plot_fitness_trend(trend)

                        self.fitness_trend[alg['Id']].append(fitness)

                    Stats.basic(self.fitness_trend[alg['Id']])
                    lg.message(logging.INFO, 'Completed algorithm {} in {}s'.format(alg['Id'], round(
                        alg_runs_time_to_complete, 2)))

            Visualisation.plot_fitness_trend_all_algs(self.fitness_trend)
            #Stats.wilcoxon(self.fitness_trend)

        lg.message(logging.INFO, 'Flow shop scheduling problem completed')

    def load_instance(self, inst):
        filename = 'instances/' + inst + '.txt'
        with open(filename, 'r') as f:
            line = f.readlines()
            for i, job_detail in enumerate(line):
                job_detail = job_detail.strip('\n')
                if i == 0:
                    self.jobs.quantity, self.machines.quantity = [int(n) for n in job_detail.split()]
                elif i == 1:
                    self.instance_upper_bound, self.instance_lower_bound = [int(n) for n in job_detail.split()]
                else:
                    self.jobs.add(job_detail, self.machines.quantity)

        lg.message(logging.INFO, 'Benchmark problem with {} machines and {} jobs'.format(self.machines.quantity,
                                                                                         self.jobs.quantity))
        lg.message(logging.INFO, 'Taillard benchmark instance lower bound is {}'.format(self.instance_lower_bound))
        lg.message(logging.INFO, 'Taillard benchmark instance best known upper bound is {}'.format(
            self.instance_upper_bound))


if __name__ == "__main__":
    log_filename = str('fssp_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

    logging.basicConfig(filename='logs/' + log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Disable matplotlib font manager logger
    logging.getLogger('matplotlib.font_manager').disabled = True

    fssp = Fssp()

