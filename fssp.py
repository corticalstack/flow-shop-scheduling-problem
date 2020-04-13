import os
import sys
import logging
from datetime import datetime
import logger as lg
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import matplotlib.font_manager as font_manager
import numpy as np
script_name = os.path.basename(sys.argv[0]).split('.')
import random
import time
from operator import itemgetter


class Jobs:
    """

    """

    def __init__(self):
        self.quantity = 0
        self.joblist = []

    def add(self, j):
        job_times = [int(n) for n in j.split()]
        self.joblist.append(job_times)


class Machines:
    def __init__(self):
        self.quantity = 0
        self.assigned_jobs = []


class Solver:
    def __init__(self):
        self.something = 0


class GA(Solver):
    def __init__(self, jobs, machines):
        Solver.__init__(self)
        self.jobs = jobs
        self.machines = machines
        self.parents = []
        self.children = []
        self.initial_population_size = 5
        self.number_parents = 2
        self.number_children = 5
        self.current_generation = 1
        self.candidate_id = 0
        self.last_generation = 100000
        self.best = 9999
        #self.generate_all_permutations()

        self.population = self.initialise_population()

        for i in range(self.last_generation):
            self.candidate_fitness = []
            for ci, candidate in enumerate(self.population):
                fitness = self.calculate_fitness(candidate)
                self.candidate_fitness.append((ci, fitness))

            # Sort candidate fitness in descending order
            self.candidate_fitness = sorted(self.candidate_fitness, key=itemgetter(1))
            if self.candidate_fitness[0][1] < self.best:
                self.best = self.candidate_fitness[0][1]
                print('Best is ', self.best)

            self.parents = self.parent_selection()

            self.children = self.parent_crossover()

            self.children_mutate()

            self.population = self.update_population()

    def generate_all_permutations(self):
        import itertools
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

        print(f_set)


    def update_population(self):
        new_pop = []
        for i in range(self.number_parents):
            new_pop.append(self.population[self.candidate_fitness[i][0]])

        # Add children to population
        new_pop.extend(self.children)
        return new_pop

    def initialise_population(self):
        pop = []
        candidate = list(range(0, self.jobs.quantity))
        while len(pop) < self.initial_population_size:
            np.random.shuffle(candidate)
            while candidate in pop:
                np.random.shuffle(candidate)
            pop.append(candidate.copy())
        return pop

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


class Scheduler:
    """
    Flow Shop Scheduling Problem
    """
    def __init__(self):
       self.jobs = Jobs()
       self.machines = Machines()
       self.load_instances()
       self.lower_bound()
       self.upper_bound()
       self.ga = GA(self.jobs, self.machines)


       self.benchmarks = {}


        # self.instance = []
        #
        # #self.machines = {}
        # self.jobs = []
        # self.machines = {}
        # self.tasks = 2
        # self.num_machines = 2
        # self.results = []
        # self.population_size = 10
        # self.job_colour = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red', 4: 'tab:purple',
        #                    5: 'tab:brown', 6: 'tab:pink', 7: 'tab:grey', 8: 'tab:olive', 9: 'tab:cyan', 10: 'tab:cyan'}
        #
        # #self.load_benchmarks()
        #
        # self.solve()
        # #self.load_machines(self.instances['abz5']['joblist'])
        # self.load_machines_2(self.benchmarks['abz5']['joblist'])
        # self.plot_gantt()
        # #self.fifo()
        # #self.experiments()

    def load_instances(self):
        instances = ['taillard_20_10_i1']
        for inst in instances:
            filename = 'instances/' + inst + '.txt'
            with open(filename, 'r') as f:
                line = f.readlines()
                for i, job_detail in enumerate(line):
                    job_detail = job_detail.strip('\n')
                    if i == 0:
                        self.machines.quantity, self.jobs.quantity = [int(n) for n in job_detail.split()]
                    else:
                        self.jobs.add(job_detail)

    def lower_bound(self):
        jobs = [sum(p) for p in self.jobs.joblist]
        print('Lower bound is {} for job {}'.format(min(jobs), jobs.index(min(jobs))))

    def upper_bound(self):
        jobs = [sum(p) for p in self.jobs.joblist]
        print('Upper bound is {} for job {}'.format(max(jobs), jobs.index(max(jobs))))

    def solve(self):
        budget = 5
        #Consider permuations
        num_machines = len(self.instance)
        random.shuffle(self.instance)
        #perm = range(len(self.instances))
        #random.shuffle(perm)

        schedule = [i for i in list(range(10)) for _ in range(10)]
        random.shuffle(schedule)
        for m in range(self.num_machines):
            for t in range(self.tasks):
                if m not in self.machines:
                    self.machines[m] = []
                self.machines[m].append(self.instance[t][m])


    def load_machines(self, joblist):
        for i, j in enumerate(joblist):
            start_s = 0
            for m, t in j:
                if m not in self.machines:
                    self.machines[m] = []
                self.machines[m].append((start_s, int(t)))
                start_s += int(t) + 1

    def load_machines_2(self, joblist):
        best_sol_by_job = [9, 4, 2, 1, 7, 5, 0, 1, 3, 2, 9, 6, 4, 7, 0, 1, 0, 0, 9, 3, 1, 8, 8, 5, 7, 7, 4, 5, 1, 3, 6, 2, 2,
                    7, 9, 8, 4, 1, 0, 4, 1, 9, 6, 2, 8, 5, 5, 7, 9, 0, 5, 6, 8, 9, 6, 1, 2, 3, 9, 8, 3, 7, 2, 3, 7, 5,
                    1, 8, 0, 2, 4, 3, 5, 9, 0, 4, 5, 3, 8, 6, 3, 6, 7, 5, 7, 4, 6, 2, 8, 1, 4, 6, 3, 2, 0, 4, 8, 6, 0,
                    9]

        job_copy = joblist
        for bs in best_sol_by_job:
            machine, time = job_copy[bs][0]
            job_copy[bs].pop(0)
            if machine not in self.machines:
                self.machines[machine] = []
            start = sum(i for _, i in self.machines[machine]) + 1
            self.machines[machine].append((start, int(time)))

    def plot_gantt(self):
        ypos_gmax = 0
        bar_height = 20
        yticks = []
        ylabels = []
        job_legend = {}

        fig, ax = plt.subplots()

        for mi, m in enumerate(self.machines):
            ypos_mmax = 0
            ypos_mmin = 9999
            for ji, j in enumerate(self.machines[m]):
                ypos = (((int(m) + 1) * (10 * bar_height)) + 100 * int(m)) - (ji*bar_height)
                if ypos > ypos_mmax:
                    ypos_mmax = ypos
                elif ypos < ypos_mmin:
                    ypos_mmin = ypos
                xstart = j[0]
                xlength = j[1]
                #print('Machine ', m, '  Y ', ypos, '  Start ', xstart, '   Length ', xlength)
                ax.broken_barh([(xstart, xlength)], (ypos, bar_height), facecolors=self.job_colour[ji],
                               label='Job ' + str(ji) if ji not in job_legend else '')
                if ji not in job_legend:
                    job_legend[ji] = ji

            if ypos_mmax > ypos_gmax:
                ypos_gmax = ypos_mmax

            yticks.append(int(ypos_mmax + ypos_mmin) / 2)
            ylabels.append('Machine ' + m)

        ax.set_ylim(0, ypos_gmax + 40)
        ax.set_xlim(0, 1000)

        ax.set_xlabel('Seconds Since Scheduling Start')

        major_ticks = np.arange(0, 1001, 100)
        minor_ticks = np.arange(0, 1001, 25)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        # Set legend
        font = font_manager.FontProperties(size='small')
        ax.legend(loc=1, prop=font, numpoints=1)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        plt.show()

    def experiments(self):
        for benchmark in self.benchmarks:
            pass

    def calculate_results(self):
        pass
        # Calculate standard deviation

        # Calculate average


    def output_results(self):
        for result in self.results:
            pass


if __name__ == "__main__":
    log_filename = str(script_name[0] + ('_') + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

    logging.basicConfig(filename='logs/' + log_filename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    scheduler = Scheduler()

