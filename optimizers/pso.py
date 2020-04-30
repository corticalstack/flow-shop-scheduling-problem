

class PSO:
    """
    Particle Swarm Optimization algorithm.
    """
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
