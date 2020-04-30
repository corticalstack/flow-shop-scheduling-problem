from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utils import logger as lg
import math
import numpy as np
import random
random.seed(42)  # Seed the random number generator


class SA(Optimizer):
    def __init__(self, problem):
        Optimizer.__init__(self)
        self.problem = problem
        self.temp = 0
        self.temp_threshold = 1
        lg.message(logging.DEBUG, 'Temperature threshold set to {}'.format(self.temp_threshold))

        self.cost_initial_temp = 0
        self.init_temp_weight = 0.035
        self.initial_temp = self.set_initial_temperature(self.problem.initial_sample)
        lg.message(logging.DEBUG, 'Initial temperature set to {}'.format(self.initial_temp))

        self.cooling_rate = 0.99
        lg.message(logging.DEBUG, 'Cooling rate set to {}'.format(self.cooling_rate))

    def solve(self):
        self.problem.budget['remaining'] = self.problem.budget['total'] - len(self.problem.initial_sample)
        self.fitness_trend = []
        self.temp = self.initial_temp
        self.best = Particle()

        self.best.permutation = getattr(self.problem, 'generator_' + self.problem.generator)()
        self.best.fitness = self.problem.evaluator(self.best.permutation)
        self.anneal()
        return self.best.fitness, self.best.permutation, self.fitness_trend

    def set_initial_temperature(self, sample):
        candidates = []
        for candidate in sample:
            fitness = self.problem.evaluator(candidate)
            candidates.append(fitness)

        #self.cost_initial_temp = self.comp_budget_total - self.remaining_budget
        it = int(np.mean(candidates)) * self.init_temp_weight
        lg.message(logging.INFO, 'Initial temperature set to {}'.format(it))
        return it

    def neighbour_solution(self):
        # This does a local search by swapping two random jobs
        new_candidate_permutation = self.best.permutation.copy()
        tasks = random.sample(range(0, len(new_candidate_permutation)), 2)
        new_candidate_permutation[tasks[0]], new_candidate_permutation[tasks[1]] = \
            new_candidate_permutation[tasks[1]], new_candidate_permutation[tasks[0]]
        return new_candidate_permutation

    # Could always add alternative neighbour generators here and just take the fittest one

    def anneal(self):
        while self.problem.budget['remaining'] > 0 and (self.temp > self.temp_threshold):
            new_candidate = self.neighbour_solution()  #JP to check if get neighbour solution OK/valid

            new_fitness = self.problem.evaluator(new_candidate)
            loss = self.best.fitness - new_fitness
            probability = math.exp(loss / self.temp)

            rr = random.random()
            if (new_fitness < self.best.fitness) or (rr < probability):
                lg.message(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.best.fitness, new_fitness))
                if rr < probability:
                    lg.message(logging.DEBUG, 'Random {} less than probability {}'.format(rr, probability))
                self.best.fitness = new_fitness
                self.best.permutation = new_candidate
                self.fitness_trend.append(self.best.fitness)

            self.temp *= self.cooling_rate

        lg.message(logging.DEBUG, 'Computational budget remaining is {}'.format(self.problem.budget['remaining']))
        lg.message(logging.DEBUG, 'Completed annealing with temperature at {}'.format(self.temp))
