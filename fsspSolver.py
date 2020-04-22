import math
import random
import itertools
import numpy as np
from operator import itemgetter
import logging
import logger as lg


class FsspSolver:
    def __init__(self, jobs, machines):
        self.logger = logging.getLogger()

        self.jobs = jobs
        self.machines = machines

        # While working with real-world applications may require a customised computational budget, when dealing with benchmark problems is a common practice to use
        # a fixed multipicative budget factor (usually = 5000 or 10000 fitness evaluations). Becnhmark functions can be usually tested at different dimensionality values and the budget factor is used to allocate a budget proportional to the number of desing variables.

        self.computational_budget_base = 5000
        self.computational_budget_total = self.computational_budget_base * 1
        lg.message(logging.INFO, 'Computational budget is {}'.format(self.computational_budget_total))

        self.initial_candidate_size = 1
        self.best_candidate_fitness = 9999
        self.best_candidate_permutation = []
        self.fitness_trend = []

    def generate_solution(self):
        candidates = []
        candidate = list(range(0, self.jobs.quantity))
        while len(candidates) < self.initial_candidate_size:
            np.random.shuffle(candidate)
            while candidate in candidates:
                np.random.shuffle(candidate)
            candidates.append(candidate.copy())
        return candidates

    def calculate_fitness(self, candidate):
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

        return self.machines.assigned_jobs[-1][-1][2]

    def brute_force_generate_all_permutations(self):
        # JP need to add count of solutions with specific fitness value
        candidate = list(range(0, self.jobs.quantity))
        all_perms = list(itertools.permutations(candidate))
        print(len(all_perms))
        best = 999999
        f_set = set()
        for p in all_perms:
            fitness = self.calculate_fitness(p)
            if fitness < best:
                print('Current best is {} for candidate {}'.format(fitness, p))
                best = fitness
            f_set.add(fitness)

        print('All possible distinct fitness values ', sorted(f_set))

    # what is diff between idle time and wait time?
    def show_machine_times(self):
        fitness = self.calculate_fitness(self.best_candidate_permutation)

        row_format = "{:>15}" * 4
        print(row_format.format('Machine', 'Start Time', 'Finish Time', 'Idle Time'))
        # [x[1][1]-(x[0][2]) for x in zip(m, m[1:] + [(0, 0, 0)])]
        for mi, m in enumerate(self.machines.assigned_jobs):
            finish_time = m[-1][2]
            idle_time = sum([x[1][1]-(x[0][2]) for x in zip(m, m[1:] + [(0, m[-1][2], 0)])])
            print(row_format.format(mi, m[0][1], finish_time, idle_time))

    def show_job_times(self):
        print('Best permutation is ', self.best_candidate_permutation)
        print('With fitness value of ', self.best_candidate_fitness)
        row_format = "{:>15}" * 4
        print(row_format.format('Job', 'Start Time', 'Finish Time', 'Idle Time'))
        ### JP - to finish calculating idle time for jobs
        for j in range(self.jobs.quantity):
            idle_time = sum([x[1][1] - (x[0][2]) for x in zip(m, m[1:] + [(0, m[-1][2], 0)])])
            print(row_format.format(self.machines.assigned_jobs[0][j][0], self.machines.assigned_jobs[0][j][1], self.machines.assigned_jobs[-1][j][2], '?'))


class SA(FsspSolver):
    def __init__(self, jobs, machines):
        FsspSolver.__init__(self, jobs, machines)
        self.temperature = 0
        self.temperature_threshold = 1
        lg.message(logging.DEBUG, 'Temperature threshold set to {}'.format(self.temperature_threshold))
        self.initial_temperature = self.set_initial_temperature()

        self.cooling_rate = 0.99
        lg.message(logging.DEBUG, 'Cooling rate set to {}'.format(self.cooling_rate))

    def solve(self):
        lg.message(logging.INFO, 'Solving...')
        self.fitness_trend = []
        self.temperature = self.initial_temperature
        self.best_candidate_permutation = self.generate_solution()[0]
        self.best_candidate_fitness = self.calculate_fitness(self.best_candidate_permutation)
        self.anneal()
        lg.message(logging.INFO, 'Solved')
        return self.best_candidate_fitness, self.best_candidate_permutation, self.fitness_trend

    def set_initial_temperature(self):
        candidate = list(range(0, self.jobs.quantity))
        #perms = list(itertools.permutations(candidate))
        perms = []
        for i in range(1000):
            c = candidate.copy()
            random.shuffle(c)
            perms.append(c)
        #random.shuffle(perms)

        total_to_sample = int(len(perms) * 0.01)  # Sample 1% of permutations
        del perms[total_to_sample:]

        candidates = []
        for candidate in perms:
            candidates.append(self.calculate_fitness(candidate))

        it = int(np.percentile(candidates, 95))
        lg.message(logging.DEBUG, 'Initial temperature set to {}'.format(it))
        return it

    def neighbour_solution(self):
        new_candidate_permutation = self.best_candidate_permutation.copy()
        tasks = random.sample(range(0, self.jobs.quantity), 2)
        new_candidate_permutation[tasks[0]], new_candidate_permutation[tasks[1]] = \
            new_candidate_permutation[tasks[1]], new_candidate_permutation[tasks[0]]
        return new_candidate_permutation

    def anneal(self):
        i = 0
        self.temperature = self.temperature / 27
        #self.temperature = math.sqrt(self.temperature)
        while self.computational_budget_total > 0 and (self.temperature > self.temperature_threshold):
            self.computational_budget_total -= 1
            new_candidate = self.neighbour_solution()  #JP to check if get neighbour solution OK/valid

            new_fitness = self.calculate_fitness(new_candidate)
            loss = self.best_candidate_fitness - new_fitness
            probability = math.exp(loss / self.temperature)

            rr = random.random()
            if (new_fitness < self.best_candidate_fitness) or (rr < probability):
                lg.message(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.best_candidate_fitness, new_fitness))
                if rr < probability:
                    lg.message(logging.DEBUG, 'Random {} less than probability {}'.format(rr, probability))
                self.best_candidate_fitness = new_fitness
                self.best_candidate_permutation = new_candidate
                self.fitness_trend.append(self.best_candidate_fitness)

            self.temperature *= self.cooling_rate

        lg.message(logging.DEBUG, 'Computational budget remaining is {}'.format(self.computational_budget_total))
        lg.message(logging.DEBUG, 'Completed annealing with temperature at {}'.format(self.temperature))


class GA(FsspSolver):
    def __init__(self, jobs, machines):
        FsspSolver.__init__(self, jobs, machines)
        self.parents = []
        self.children = []
        self.initial_candidate_size = 5
        lg.message(logging.DEBUG, 'Initial candidate size set to {}'.format(self.initial_candidate_size))

        self.number_parents = 2
        lg.message(logging.DEBUG, 'Number of parents set to {}'.format(self.number_parents))

        self.number_children = 5
        lg.message(logging.DEBUG, 'Number of children set to {}'.format(self.number_children))

        self.current_generation = 1
        self.candidate_id = 0
        self.candidate_fitness = []
        self.population = []

        #self.brute_force_generate_all_permutations()
        #self.show_machine_times()
        #self.show_job_times()

    def solve(self):
        self.fitness_trend = []
        self.best_candidate_fitness = 9999
        self.best_candidate_permutation = []
        self.evolve()
        return self.best_candidate_fitness, self.best_candidate_permutation, self.fitness_trend

    def evolve(self):
        self.population = self.generate_solution()

        while self.computational_budget_total > 0:
            self.computational_budget_total -= 1
            self.candidate_fitness = []
            for ci, candidate in enumerate(self.population):
                fitness = self.calculate_fitness(candidate)
                self.candidate_fitness.append((ci, fitness))

            # Sort candidate fitness in descending order
            self.candidate_fitness = sorted(self.candidate_fitness, key=itemgetter(1))
            if self.candidate_fitness[0][1] < self.best_candidate_fitness:
                lg.message(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.best_candidate_fitness, self.candidate_fitness[0][1]))
                self.best_candidate_fitness = self.candidate_fitness[0][1]
                self.best_candidate_permutation = self.population[self.candidate_fitness[0][0]]
                self.fitness_trend.append(self.best_candidate_fitness)

            self.parents = self.parent_selection()

            self.children = self.parent_crossover()

            self.children_mutate()

            self.population = self.update_population()

        lg.message(logging.DEBUG, 'Computational budget remaining is {}'.format(self.computational_budget_total))

    def update_population(self):
        new_pop = []
        for i in range(self.number_parents):
            new_pop.append(self.population[self.candidate_fitness[i][0]])

        # Add children to population
        new_pop.extend(self.children)
        return new_pop

    def parent_selection(self):
        # Fitness proportionate selection (FPS), assigning probabilities to individuals acting as parents depending on their
        # fitness
        max_fitness = sum(n for _, n in self.candidate_fitness)
        fitness_proportionate = [fitness[1] / max_fitness for fitness in self.candidate_fitness]

        pointer_distance = 1 / self.number_parents
        start_point = random.uniform(0, pointer_distance)
        points = [start_point + i * pointer_distance for i in range(self.number_parents)]
        parents = []

        fitness_aggr = 0
        for fi, fp in enumerate(fitness_proportionate):
            fitness_aggr += fp
            for p in points:
                if fitness_aggr > p:
                    parents.append(fi)
                    points.pop(0)
                    break

        return parents

    def parent_crossover(self):
        children = []
        for i in range(self.number_children):
            crossover_point = random.randint(1, self.jobs.quantity - 1)
            child = self.population[self.parents[0]][:crossover_point]
            for c in self.population[self.parents[1]]:
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
