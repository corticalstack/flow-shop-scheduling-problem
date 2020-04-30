import logging
from utils import logger as lg


class GA:
    def __init__(self):
        self.parents = []
        self.children = []
        self.initial_candidate_size = 5  # JP Test with population twice size of number dimensions
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
