from jobs import Jobs
from machines import Machines
from fsspSolver import *
from stats import Stats
from visualisation import Visualisation
import os
import sys
from collections import OrderedDict
import logging
import time
from datetime import datetime
import logger as lg
import plotly.figure_factory as ff
import copy

script_name = os.path.basename(sys.argv[0]).split('.')
import time


class Fssp:
    """
    Flow Shop Scheduling Problem
    """
    def __init__(self):
        lg.message(logging.INFO, 'Starting flow shop scheduling problem')
        self.sample_runs = 5  # 30
        self.optimizers = [{'Id': 'SA', 'Enabled': True},
                           {'Id': 'GA', 'Enabled': True}]

        # Simplify declaration of new optimizers with standard stats template that is added to each
        self.stats_template = {'BestCF': 99999, 'BestCP': [], 'lb_diff_pct': 0, 'ub_diff_pct': 0, 'AvgCts': 0, 'Ft': []}

        self.instance_lb = 0  # Approximated best
        self.instance_ub = 0  # Best known minimisation

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
            self.machines.set_lower_bounds_taillard(self.jobs, self.instance_lb)

            for oi, opt in enumerate(self.optimizers):
                if opt['Enabled']:
                    lg.message(logging.INFO, 'Processing algorithm {}'.format(opt['Id']))
                    self.add_stats_template(oi)

                    cls = globals()[opt['Id']]
                    solver = cls(self.jobs, self.machines)  # Instantiate solver class dynamically

                    lg.message(logging.INFO, 'Executing {} sample runs'.format(self.sample_runs))
                    total_cts = 0
                    for i in range(self.sample_runs):

                        # Aggregate time of this run's solver execution to total
                        alg_run_start_time = time.time()
                        run_best_fitness, run_best_permutation, run_fitness_trend = solver.solve()
                        total_cts += time.time() - alg_run_start_time

                        lg.message(logging.INFO, 'Run {} best fitness is {} with permutation {}'.format(
                            i, run_best_fitness, run_best_permutation))

                        if run_best_fitness < self.optimizers[oi]['BestCF']:
                            self.optimizers[oi]['BestCF'] = run_best_fitness
                            self.optimizers[oi]['BestCP'] = run_best_permutation

                        # Log best fitness for this run to see trend over execution runs
                        self.optimizers[oi]['Ft'].append(run_best_fitness)

                        self.visualisation.plot_ft(run_fitness_trend)  # Plot run specific trend

                    # Log optimizer average completion time seconds
                    self.optimizers[oi]['AvgCts'] = total_cts / self.sample_runs

                    self.optimizers[oi]['lb_diff_pct'], self.optimizers[oi]['ub_diff_pct'] = Stats.taillard_compare(
                        self.instance_lb, self.instance_ub, self.optimizers[oi]['BestCF'])

                    self.visualisation.plot_gantt(self.optimizers[oi]['BestCP'], self.machines, self.jobs, solver)

                    lg.message(logging.INFO, 'Machine times for best fitness {}'.format(self.optimizers[oi]['BestCF']))
                    self.machines.times(self.optimizers[oi]['BestCP'], solver)

                    lg.message(logging.INFO, 'Job times for best fitness of {} with permutation {}'.format(
                        self.optimizers[oi]['BestCF'], self.optimizers[oi]['BestCP']))
                    self.jobs.times(self.optimizers[oi]['BestCP'], self.machines, solver)

            self.visualisation.plot_ft_all_optimizers(self.optimizers)
            Stats.summary(self.optimizers)

        lg.message(logging.INFO, 'Flow shop scheduling problem completed')

    def add_stats_template(self, oi):
        stats_template = copy.deepcopy(self.stats_template)
        self.optimizers[oi].update(stats_template)

    def load_instance(self, inst):
        filename = 'instances/' + inst + '.txt'
        with open(filename, 'r') as f:
            line = f.readlines()
            for i, job_detail in enumerate(line):
                job_detail = job_detail.strip('\n')
                if i == 0:
                    self.jobs.quantity, self.machines.quantity = [int(n) for n in job_detail.split()]
                elif i == 1:
                    self.instance_ub, self.instance_lb = [int(n) for n in job_detail.split()]
                else:
                    self.jobs.add(job_detail, self.machines.quantity)

        lg.message(logging.INFO, '{} machines and {} jobs'.format(self.machines.quantity, self.jobs.quantity))
        lg.message(logging.INFO, 'Taillard lower bound is {}'.format(self.instance_lb))
        lg.message(logging.INFO, 'Taillard best known upper bound is {}'.format(self.instance_ub))


if __name__ == "__main__":
    log_filename = str('fssp_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

    logging.basicConfig(filename='logs/' + log_filename, level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)8s] %(message)s')

    # Disable matplotlib font manager logger
    logging.getLogger('matplotlib.font_manager').disabled = True

    fssp = Fssp()

