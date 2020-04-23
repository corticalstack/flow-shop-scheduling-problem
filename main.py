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

script_name = os.path.basename(sys.argv[0]).split('.')
import time


class Fssp:
    """
    Flow Shop Scheduling Problem
    """
    def __init__(self):
        lg.message(logging.INFO, 'Starting flow shop scheduling problem')
        self.sample_runs = 1  # 30
        self.algorithms = [{'Id': 'SA', 'Enabled': True},
                           {'Id': 'GA', 'Enabled': False}]
        self.best_candidate_fitness = 9999
        self.best_candidate_permutation = []
        self.fitness_trend = {}
        self.instance_lower_bound = 0  # Approximated best
        self.instance_upper_bound = 0  # Best known minimisation

        instances = [{'Id': 'taillard_20_5_i1', 'Enabled': False},
                     {'Id': 'taillard_20_10_i1', 'Enabled': True},
                     {'Id': 'taillard_50_10_i1', 'Enabled': False},
                     {'Id': 'taillard_100_10_i1', 'Enabled': False}]

        for inst in instances:
            if not inst['Enabled']:
                continue

            lg.message(logging.INFO, 'Processing benchmark problem instance {}'.format(inst['Id']))
            # New job and machine instance for each benchmark instance

            self.jobs = Jobs()
            self.machines = Machines()
            self.visualisation = Visualisation()
            self.load_instance(inst['Id'])

            self.jobs.set_job_total_units()
            self.machines.set_loadout_times(self.jobs)
            self.machines.set_lower_bounds_taillard(self.jobs, self.instance_lower_bound)

            for alg in self.algorithms:
                if alg['Enabled']:
                    lg.message(logging.INFO, 'Processing algorithm {}'.format(alg['Id']))
                    self.best_candidate_fitness = 9999
                    self.best_candidate_permutation = []
                    self.fitness_trend[alg['Id']] = []
                    cls = globals()[alg['Id']]
                    solver = cls(self.jobs, self.machines)

                    lg.message(logging.INFO, 'Executing {} sample runs'.format(self.sample_runs))
                    alg_runs_time_to_complete = 0
                    for i in range(self.sample_runs):
                        # Invoke class dynamically
                        alg_run_start_time = time.time()
                        fitness, permutation, trend = solver.solve()
                        if fitness < self.best_candidate_fitness:
                            self.best_candidate_fitness = fitness
                            self.best_candidate_permutation = permutation

                        alg_runs_time_to_complete += time.time() - alg_run_start_time
                        lg.message(logging.INFO, 'Run {} fitness is {} with permutation {}'.format(i, fitness,
                                                                                                   permutation))
                        self.visualisation.plot_fitness_trend(trend)

                        self.fitness_trend[alg['Id']].append(fitness)

                    Stats.basic(self.fitness_trend[alg['Id']])
                    Stats.taillard_compare(self.instance_lower_bound, self.instance_upper_bound,
                                           self.best_candidate_fitness)

                    self.visualisation.plot_gantt(self.best_candidate_permutation, self.machines, self.jobs, solver)

                    lg.message(logging.INFO, 'Completed algorithm {} in {}s'.format(alg['Id'], round(
                        alg_runs_time_to_complete, 2)))
                    lg.message(logging.INFO, 'Machine times for best fitness of {}'.format(self.best_candidate_fitness))
                    self.machines.times(self.best_candidate_permutation, solver)

                    lg.message(logging.INFO, 'Job times for best fitness of {} with permutation {}'.format(
                        self.best_candidate_fitness, self.best_candidate_permutation))
                    self.jobs.times(self.best_candidate_permutation, self.machines, solver)

            self.visualisation.plot_fitness_trend_all_algs(self.fitness_trend)
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

        lg.message(logging.INFO, '{} machines and {} jobs'.format(self.machines.quantity, self.jobs.quantity))
        lg.message(logging.INFO, 'Taillard lower bound is {}'.format(self.instance_lower_bound))
        lg.message(logging.INFO, 'Taillard best known upper bound is {}'.format(
            self.instance_upper_bound))


if __name__ == "__main__":
    log_filename = str('fssp_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

    logging.basicConfig(filename='logs/' + log_filename, level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)8s] %(message)s')

    # Disable matplotlib font manager logger
    logging.getLogger('matplotlib.font_manager').disabled = True

    fssp = Fssp()

