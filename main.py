from utils.stats import Stats
import os
import sys
from datetime import datetime
import copy
from config.config import *
from problems.fssp import *
from optimizers.sa import *


script_name = os.path.basename(sys.argv[0]).split('.')
import time


class HeuristicOptimizer:
    """
    Heuristic Optimizer
    """
    def __init__(self):
        lg.message(logging.INFO, 'Starting heuristic optimizer')

        self.cfg = Config()
        self.visualisation = Visualisation()

        # Simplify declaration of new optimizers with standard stats template that is added to each
        self.opt_runtime_stats = {'BestCF': 999999999, 'BestCP': [], 'lb_diff_pct': 0, 'ub_diff_pct': 0, 'AvgCts': 0,
                                  'Ft': []}
        self.add_opt_runtime_stats()

        self.instance_lb = 0  # Approximated best
        self.instance_ub = 0  # Best known minimisation

        for pid in self.cfg.settings['problems']:
            if not self.cfg.settings['problems'][pid]['enabled']:
                continue
            lg.message(logging.INFO, 'Processing {}'.format(self.cfg.settings['problems'][pid]['description']))

            for iid in self.cfg.settings['benchmarks'][pid]['instances']:
                if not self.cfg.settings['benchmarks'][pid]['instances'][iid]['enabled']:
                    continue
                lg.message(logging.INFO, 'Optimizing {} benchmark problem instance {}'.format(
                    self.cfg.settings['benchmarks'][pid]['type'], iid))

                for oid in self.cfg.settings['optimizers']:
                    if not self.cfg.settings['optimizers'][oid]['enabled']:
                        continue

                    self.solve_problem(pid, iid, oid)

            self.visualisation.plot_ft_all_optimizers(self.cfg.settings['optimizers'])
            Stats.summary(self.cfg.settings['optimizers'])

        lg.message(logging.INFO, 'Flow shop scheduling problem completed')

    def solve_problem(self, pid, iid, oid):
        lg.message(logging.INFO, 'Executing optimizer {}'.format(oid))
        cls = globals()[pid]
        problem = cls(iid, self.cfg.settings['general']['computational_budget_base'],
                                                        self.cfg.settings['optimizers'][oid]['generator'])

        lg.message(logging.INFO, 'Executing {} sample runs'.format(self.cfg.settings['general']['runs_per_optimizer']))
        total_cts = 0
        cls = globals()[oid]
        optimizer = cls(problem)
        for i in range(self.cfg.settings['general']['runs_per_optimizer']):

            # Aggregate time of this run's solver execution to total
            alg_run_start_time = time.time()
            run_best_fitness, run_best_permutation, run_fitness_trend = optimizer.solve()
            total_cts += time.time() - alg_run_start_time

            lg.message(logging.INFO, 'Run {} best fitness is {} with permutation {}'.format(
                i, run_best_fitness, run_best_permutation))

            if run_best_fitness < self.cfg.settings['optimizers'][oid]['BestCF']:
                self.cfg.settings['optimizers'][oid]['BestCF'] = run_best_fitness
                self.cfg.settings['optimizers'][oid]['BestCP'] = run_best_permutation

            # Log best fitness for this run to see trend over execution runs
            self.cfg.settings['optimizers'][oid]['Ft'].append(run_best_fitness)

            self.visualisation.plot_ft(run_fitness_trend)  # Plot run specific trend

        # Log optimizer average completion time seconds
        self.cfg.settings['optimizers'][oid]['AvgCts'] = total_cts / self.cfg.settings['general']['runs_per_optimizer']

        self.cfg.settings['optimizers'][oid]['lb_diff_pct'], self.cfg.settings['optimizers'][oid]['ub_diff_pct'] = \
            Stats.taillard_compare(self.instance_lb, self.instance_ub, self.cfg.settings['optimizers'][oid]['BestCF'])

        problem.on_completion(self.cfg.settings['optimizers'][oid]['BestCP'])

    def add_opt_runtime_stats(self):
        for oid in self.cfg.settings['optimizers']:
            if not self.cfg.settings['optimizers'][oid]['enabled']:
                continue
            stats_template = copy.deepcopy(self.opt_runtime_stats)
            self.cfg.settings['optimizers'][oid].update(stats_template)


if __name__ == "__main__":
    log_filename = str('ho_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

    logging.basicConfig(filename='logs/' + log_filename, level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)8s] %(message)s')

    # Disable matplotlib font manager logger
    logging.getLogger('matplotlib.font_manager').disabled = True

    ho = HeuristicOptimizer()

