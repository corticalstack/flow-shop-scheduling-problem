import math
import random
random.seed(42)  # Seed the random number generator
import itertools
import numpy as np
from operator import itemgetter
import logging
import logger as lg

import inspyred
from inspyred.ec import terminators
from time import time
from random import Random


class Particle:
    def __init__(self, fitness_default=999999999):
        self.fitness_default = fitness_default
        self.permutation = []
        self.fitness = fitness_default


class FsspSolver:
    def __init__(self, jobs, machines):
        self.logger = logging.getLogger()

        self.jobs = jobs
        self.machines = machines

        # While working with real-world applications may require a customised computational budget, when dealing with benchmark problems is a common practice to use
        # a fixed multipicative budget factor (usually = 5000 or 10000 fitness evaluations). Becnhmark functions can be usually tested at different dimensionality values and the budget factor is used to allocate a budget proportional to the number of desing variables.

        self.comp_budget_base = 6000
        self.comp_budget_total = self.comp_budget_base * self.jobs.quantity
        self.remaining_budget = self.comp_budget_total
        lg.message(logging.INFO, 'Computational budget is {}'.format(self.comp_budget_total))

        self.initial_candidate_size = 1
        self.best = None
        self.fitness_trend = []

    def generate_solution(self):
        candidate = list(range(0, self.jobs.quantity))
        np.random.shuffle(candidate)
        return candidate

    def calculate_fitness(self, candidate, remaining_budget):
        self.machines.assigned_jobs = []
        for i in range(0, self.machines.quantity):
            self.machines.assigned_jobs.append([])

        for ji, j in enumerate(candidate):
            start_time = 0
            end_time = 0
            for mi, mt in enumerate(self.jobs.joblist[j]):
                if self.machines.assigned_jobs[mi]:
                    if mi == 0:
                        start_time = self.machines.assigned_jobs[mi][-1][2]
                    else:
                        curr_job_prev_task_end = self.machines.assigned_jobs[mi][-1][2]
                        prev_job_task_end = self.machines.assigned_jobs[mi-1][-1][2]
                        start_time = max(curr_job_prev_task_end, prev_job_task_end)

                end_time = start_time + mt
                self.machines.assigned_jobs[mi].append((j, start_time, end_time))
                start_time = end_time

        return self.machines.assigned_jobs[-1][-1][2], remaining_budget - 1


class SA(FsspSolver):
    def __init__(self, jobs, machines):
        FsspSolver.__init__(self, jobs, machines)
        self.temp = 0
        self.temp_threshold = 1
        lg.message(logging.DEBUG, 'Temperature threshold set to {}'.format(self.temp_threshold))

        self.cost_initial_temp = 0
        self.init_temp_weight = 0.035
        self.initial_temp = self.set_initial_temperature()
        lg.message(logging.DEBUG, 'Initial temperature set to {}'.format(self.initial_temp))

        self.cooling_rate = 0.99
        lg.message(logging.DEBUG, 'Cooling rate set to {}'.format(self.cooling_rate))

    def solve(self):
        self.remaining_budget = self.comp_budget_total - self.cost_initial_temp
        self.fitness_trend = []
        self.temp = self.initial_temp
        self.best = Particle()
        self.best.permutation = self.generate_solution()
        self.best.fitness, self.remaining_budget = self.calculate_fitness(self.best.permutation, self.remaining_budget)
        self.anneal()
        return self.best.fitness, self.best.permutation, self.fitness_trend

    def set_initial_temperature(self):
        candidate = list(range(0, self.jobs.quantity))
        num_perms = int(math.pow(self.jobs.quantity, 2))
        perms = []
        for i in range(num_perms):
            c = candidate.copy()
            random.shuffle(c)
            perms.append(c)

        candidates = []
        for candidate in perms:
            fitness, self.remaining_budget = self.calculate_fitness(candidate, self.remaining_budget)
            candidates.append(fitness)

        self.cost_initial_temp = self.comp_budget_total - self.remaining_budget
        it = int(np.mean(candidates)) * self.init_temp_weight
        lg.message(logging.INFO, 'Initial temperature set to {}'.format(it))
        return it

    def neighbour_solution(self):
        # This does a local search by swapping two random jobs
        new_candidate_permutation = self.best.permutation.copy()
        tasks = random.sample(range(0, self.jobs.quantity), 2)
        new_candidate_permutation[tasks[0]], new_candidate_permutation[tasks[1]] = \
            new_candidate_permutation[tasks[1]], new_candidate_permutation[tasks[0]]
        return new_candidate_permutation

    # Could always add alternative neighbour generators here and just take the fittest one

    def anneal(self):
        #self.temperature = math.sqrt(self.temperature)
        while self.remaining_budget > 0 and (self.temp > self.temp_threshold):
            new_candidate = self.neighbour_solution()  #JP to check if get neighbour solution OK/valid

            new_fitness, self.remaining_budget = self.calculate_fitness(new_candidate, self.remaining_budget)
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

        lg.message(logging.DEBUG, 'Computational budget remaining is {}'.format(self.remaining_budget))
        lg.message(logging.DEBUG, 'Completed annealing with temperature at {}'.format(self.temp))


class GA(FsspSolver):
    def __init__(self, jobs, machines):
        FsspSolver.__init__(self, jobs, machines)
        self.parents = []
        self.children = []
        self.initial_candidate_size = 5 # JP Test with population twice size of number dimensions
        lg.message(logging.DEBUG, 'Initial candidate size set to {}'.format(self.initial_candidate_size))

        self.number_parents = 3
        lg.message(logging.DEBUG, 'Number of parents set to {}'.format(self.number_parents))

        self.number_children = 5
        lg.message(logging.DEBUG, 'Number of children set to {}'.format(self.number_children))

        self.current_generation = 1
        self.candidate_id = 0
        self.candidate_fitness = []
        self.population = []
        self.best_cf = 999999999
        self.best_cp = []

    def solve(self):
        self.population = []
        self.remaining_budget = self.comp_budget_total
        self.fitness_trend = []
        self.best_cf = 999999999
        self.best_cp = []
        self.evolve()
        return self.best_cf, self.best_cp, self.fitness_trend

    def evolve(self):
        for i in range(self.initial_candidate_size):
            particle = Particle()
            particle.permutation = self.generate_solution()
            self.population.append(particle)

        while self.remaining_budget > 0:
            self.candidate_fitness = []
            for ci, candidate in enumerate(self.population):
                if candidate.fitness == candidate.fitness_default:
                    candidate.fitness, self.remaining_budget = self.calculate_fitness(candidate.permutation, self.remaining_budget)

            # Sort population by fitness ascending
            self.population.sort(key=lambda x: x.fitness, reverse=False)

            if self.population[0].fitness < self.best_cf:
                lg.message(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.best_cf, self.population[0].fitness))
                self.best_cf = self.population[0].fitness
                self.best_cp = self.population[0].permutation
                self.fitness_trend.append(self.population[0].fitness)

            self.parents = self.parent_selection()

            self.children = self.parent_crossover()

            self.children_mutate()

            self.population = self.update_population()

        lg.message(logging.DEBUG, 'Computational budget remaining is {}'.format(self.remaining_budget))

    def update_population(self):
        new_pop = []
        for p in self.parents:
            particle = Particle()
            particle.fitness = self.population[p].fitness
            particle.permutation = self.population[p].permutation
            new_pop.append(particle)

        # Add children to population
        for c in self.children:
            particle = Particle()
            particle.permutation = c
            new_pop.append(particle)

        return new_pop

    def parent_selection(self):
        # Fitness proportionate selection (FPS), assigning probabilities to individuals acting as parents depending on their
        # fitness
        max_fitness = sum([particle.fitness for particle in self.population])
        fitness_proportionate = [particle.fitness / max_fitness for particle in self.population]

        pointer_distance = 1 / self.number_parents
        start_point = random.uniform(0, pointer_distance)
        points = [start_point + i * pointer_distance for i in range(self.number_parents)]

        # Add boundary points
        points.insert(0, 0)
        points.append(1)

        parents = []

        fitness_aggr = 0
        for fi, fp in enumerate(fitness_proportionate):
            if len(parents) == self.number_parents:
                break
            fitness_aggr += fp
            for pi, p in enumerate(points):
                if p < fitness_aggr < points[pi+1]:
                    parents.append(fi)
                    points.pop(0)
                    break

        return parents

    def parent_crossover(self):
        children = []
        for i in range(self.number_children):
            crossover_point = random.randint(1, self.jobs.quantity - 1)
            child = self.population[self.parents[0]].permutation[:crossover_point]
            for c in self.population[self.parents[1]].permutation:
                if c not in child:
                    child.append(c)
            children.append(child)

        return children

    def children_mutate(self):
        """
        Swap 2 tasks at random
        """
        # Swap positions of the 2 job tasks in the candidate
        for i in range(self.number_children):
            # Generate 2 task numbers at random, within range
            tasks = random.sample(range(0, self.jobs.quantity), 2)
            self.children[i][tasks[0]], self.children[i][tasks[1]] = \
                self.children[i][tasks[1]], self.children[i][tasks[0]]



class PSO(FsspSolver):
    def __init__(self, jobs, machines):
        FsspSolver.__init__(self, jobs, machines)
        self.initial_candidate_size = self.jobs.quantity * 2
        lg.message(logging.DEBUG, 'Swarm size to {}'.format(self.initial_candidate_size))

        self.current_generation = 1
        self.candidate_id = 0
        self.candidate_fitness = []
        self.population = []

        self.min = 0
        self.max = 4
        self.velocity_clip = (-4, 4)

        self.weight = 0.5 # Inertia
        self.local_c1 = 2.1
        self.global_c2 = 2.1

        self.max_velocity = 4
        self.min_velocity = -4
        self.gbest = None
        self.initial_candidate_size = 30
        self.global_best_fitness = 999999999
        self.global_best_permutation = []

    def generate_solution(self):
        candidate = []
        for j in range(self.jobs.quantity):
            candidate.append(round(self.min + (self.max-self.min) * random.uniform(0, 1), 2))
        return candidate

    def solve(self):
        self.population = []
        self.remaining_budget = self.comp_budget_total
        self.fitness_trend = []
        self.global_best = None
        self.global_best_fitness = 999999999
        self.global_best_permutation = []

        self.evolve()
        return self.global_best_fitness, self.global_best_permutation, self.fitness_trend

    def evolve(self):

        # Iniitalise population
        for i in range(self.initial_candidate_size):
            particle = Particle()
            particle.permutation_continuous = self.generate_solution()  # Generate random permutation
            particle.permutation = self.transform_continuous_permutation(particle)
            particle.fitness, self.remaining_budget = self.calculate_fitness(particle.permutation,
                                                                              self.remaining_budget)
            particle.prev_fitness = particle.fitness  # Set the personal (local) best fitness
            particle.prev_permutation = particle.permutation  # Set the personal (local) best permutation
            particle.prev_permutation_continuous = particle.permutation_continuous  # Set the personal (local) best permutation
            particle.velocity = [round(self.min_velocity + (self.max_velocity - self.min_velocity) * random.uniform(0, 1), 2) for j in range(self.jobs.quantity)]
            self.population.append(particle)

        self.population.sort(key=lambda x: x.fitness, reverse=False)
        self.global_best = self.population[0]

        while self.remaining_budget > 0:
            self.candidate_fitness = []
            for ci, candidate in enumerate(self.population):
                candidate.fitness, self.remaining_budget = self.calculate_fitness(candidate.permutation,
                                                                                  self.remaining_budget)

                # Evaluate fitness and set personal (local) best
                if candidate.fitness < candidate.prev_fitness:
                    candidate.prev_fitness = candidate.fitness
                    candidate.prev_permutation = candidate.permutation
                    candidate.prev_permutation_continuous = candidate.permutation_continuous



            # Set global best
            self.population.sort(key=lambda x: x.fitness, reverse=False)
            self.global_best = self.population[0]

            if self.global_best.fitness < self.global_best_fitness:
                lg.message(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.global_best_fitness, self.global_best.fitness))
                self.global_best_fitness = self.global_best.fitness
                self.global_best_permutation = self.global_best.permutation
                self.fitness_trend.append(self.global_best_fitness)

            for ci, candidate in enumerate(self.population):
                # Update velocity
                self.velocity(candidate)

            self.perturb_permutation()

        lg.message(logging.DEBUG, 'Computational budget remaining is {}'.format(self.remaining_budget))

    def transform_continuous_permutation(self, particle):
        # Get smallest position value
        spv = sorted(range(len(particle.permutation_continuous)), key=lambda i: particle.permutation_continuous[i], reverse=False)
        return spv

    def perturb_permutation(self):
        for ci, candidate in enumerate(self.population):
            if ci == 0:
                continue
            for ji, j in enumerate(candidate.permutation):
                candidate.permutation_continuous[ji] += candidate.velocity[ji]
            #print(candidate.permutation_continuous)
            candidate.permutation = self.transform_continuous_permutation(candidate)

    def velocity(self, particle):
        for pi, p in enumerate(particle.permutation_continuous):
            # exp_inertia = self.weight + particle.velocity[pi]
            # exp_local = self.local_c1 * random.uniform(0, 1) * (particle.lbest_fitness - particle.fitness)
            # exp_global = self.global_c2 * random.uniform(0, 1) * (self.global_best.fitness - particle.fitness)
            # particle.velocity[pi] = round(exp_inertia + exp_local + exp_global, 3)
            # particle.velocity[pi] = self.clamp(particle.velocity[pi])


            particle.velocity[pi] = (particle.permutation_continuous[pi] + self.weight * (particle.permutation_continuous[pi] - particle.prev_permutation_continuous[pi]) +
            self.local_c1 * random.random() * (particle.prev_permutation_continuous[pi] - particle.permutation_continuous[pi]) +
            self.global_c2 * random.random() * (self.global_best.permutation_continuous[pi] - particle.permutation_continuous[pi]))


            # xi = p[pi]
            # inertia = 0.5
            # xpi is the previous version of the population
            # pbi is the best of this candidate
            # nbest is the global best
            # JP - Need to store an array of the previous


    def clamp(self, n):
        return max(min(self.max_velocity, n), self.min_velocity)

class PSOI(FsspSolver):
    def __init__(self, jobs, machines):
        FsspSolver.__init__(self, jobs, machines)
        self.initial_candidate_size = self.jobs.quantity * 2
        lg.message(logging.DEBUG, 'Swarm size to {}'.format(self.initial_candidate_size))

        self.current_generation = 1
        self.candidate_id = 0
        self.candidate_fitness = []
        self.population = []

        self.min = 0
        self.max = 4
        self.velocity_clip = (-4, 4)

        self.weight = 0.9 # Inertia
        self.local_c1 = 1.49445
        self.global_c2 = 0.01

        self.max_velocity = 4
        self.min_velocity = -4
        self.gbest = None
        self.initial_candidate_size = 5
        self.global_best_fitness = 999999999
        self.global_best_permutation = []

    @staticmethod
    def generate_solution(random, args):
        candidate = []
        for j in range(args['jobs'].quantity):
            candidate.append(round(0 + (4-0) * random.uniform(0, 1), 2))
        return candidate

    @staticmethod
    def calculate_fit(candidates, args):
        fitness = []
        for c in candidates:
            c = sorted(range(len(c)), key=lambda i: c[i], reverse=False)
            args['machines'].assigned_jobs = []
            for i in range(0, args['machines'].quantity):
                args['machines'].assigned_jobs.append([])

            for ji, j in enumerate(c):
                start_time = 0
                end_time = 0
                for mi, mt in enumerate(args['jobs'].joblist[j]):
                    if args['machines'].assigned_jobs[mi]:
                        if mi == 0:
                            start_time = args['machines'].assigned_jobs[mi][-1][2]
                        else:
                            curr_job_prev_task_end = args['machines'].assigned_jobs[mi][-1][2]
                            prev_job_task_end = args['machines'].assigned_jobs[mi - 1][-1][2]
                            start_time = max(curr_job_prev_task_end, prev_job_task_end)

                    end_time = start_time + mt
                    args['machines'].assigned_jobs[mi].append((j, start_time, end_time))
                    start_time = end_time

            fitness.append(args['machines'].assigned_jobs[-1][-1][2])
        return fitness

    def solve(self):
        self.remaining_budget = self.comp_budget_total
        self.fitness_trend = []
        self.global_best = None
        self.global_best_fitness = 999999999
        self.global_best_permutation = []

        self.evolve()
        return self.global_best_fitness, self.global_best_permutation, self.fitness_trend

    def evolve(self):

        rand = Random()
        rand.seed(int(time()))
        prng = Random()
        prng.seed(time())

        ea = inspyred.swarm.PSO(prng)
        ea.terminator = inspyred.ec.terminators.evaluation_termination
        ea.topology = inspyred.swarm.topologies.ring_topology
        final_pop = ea.evolve(generator=self.generate_solution,
                              evaluator=self.calculate_fit,
                              jobs=self.jobs,
                              machines=self.machines,
                              pop_size=100,
                              maximize=False,
                              max_evaluations=20000,
                              neighborhood_size=5)

        # Sort and print the best individual, who will be at index 0.
        final_pop.sort(reverse=True)
        print(final_pop[0])

        def main(prng=None, display=False):
            if prng is None:
                prng = Random()
                prng.seed(time())

            problem = inspyred.benchmarks.Ackley(2)
            ea = inspyred.swarm.PSO(prng)
            ea.terminator = inspyred.ec.terminators.evaluation_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            final_pop = ea.evolve(generator=problem.generator,
                                  evaluator=problem.evaluator,
                                  pop_size=100,
                                  bounder=problem.bounder,
                                  maximize=problem.maximize,
                                  max_evaluations=30000,
                                  neighborhood_size=5)

            if display:
                best = max(final_pop)
                print('Best Solution: \n{0}'.format(str(best)))
            return ea

        # Iniitalise population
        for i in range(self.initial_candidate_size):
            particle = Particle()
            particle.permutation_continuous = self.generate_solution()  # Generate random permutation
            particle.permutation = self.transform_continuous_permutation(particle)
            particle.fitness, self.remaining_budget = self.calculate_fitness(particle.permutation,
                                                                              self.remaining_budget)
            particle.lbest_fitness = particle.fitness  # Set the personal (local) best fitness
            particle.lbest_permutation = particle.permutation  # Set the personal (local) best permutation
            particle.velocity = [round(self.min_velocity + (self.max_velocity - self.min_velocity) * random.uniform(0, 1), 2) for j in range(self.jobs.quantity)]
            self.population.append(particle)

        self.population.sort(key=lambda x: x.fitness, reverse=False)
        self.global_best = self.population[0]

        while self.remaining_budget > 0:
            self.candidate_fitness = []
            for ci, candidate in enumerate(self.population):
                candidate.fitness, self.remaining_budget = self.calculate_fitness(candidate.permutation,
                                                                                  self.remaining_budget)

                # Evaluate fitness and set personal (local) best
                if candidate.fitness < candidate.lbest_fitness:
                    candidate.lbest_fitness = candidate.fitness
                    candidate.lbest_permutation = candidate.permutation


            # Set global best
            self.population.sort(key=lambda x: x.fitness, reverse=False)
            self.global_best = self.population[0]

            if self.global_best.fitness < self.global_best_fitness:
                lg.message(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.global_best_fitness, self.global_best.fitness))
                self.global_best_fitness = self.global_best.fitness
                self.global_best_permutation = self.global_best.permutation
                self.fitness_trend.append(self.global_best_fitness)

            for ci, candidate in enumerate(self.population):
                # Update velocity
                self.velocity(candidate)

            self.perturb_permutation()

        lg.message(logging.DEBUG, 'Computational budget remaining is {}'.format(self.remaining_budget))

    @staticmethod
    def transform_continuous_permutation(candidate):
        # Get smallest position value
        spv = sorted(range(len(candidate)), key=lambda i: candidate[i], reverse=False)
        return spv

    def perturb_permutation(self):
        for ci, candidate in enumerate(self.population):
            if ci == 0:
                continue
            for ji, j in enumerate(candidate.permutation):
                candidate.permutation_continuous[ji] += candidate.velocity[ji]
            #print(candidate.permutation_continuous)
            candidate.permutation = self.transform_continuous_permutation(candidate)

    def velocity(self, particle):
        for pi, p in enumerate(particle.permutation_continuous):
            exp_inertia = self.weight * particle.velocity[pi]
            exp_local = self.local_c1 * random.uniform(0, 1) * (particle.lbest_fitness - particle.fitness)
            exp_global = self.global_c2 * random.uniform(0, 1) * (self.global_best.fitness - particle.fitness)
            particle.velocity[pi] = round(exp_inertia + exp_local + exp_global, 3)
            particle.velocity[pi] = self.clamp(particle.velocity[pi])

        print('Particle velocity ', particle.velocity)

    def clamp(self, n):
        return max(min(self.max_velocity, n), self.min_velocity)
