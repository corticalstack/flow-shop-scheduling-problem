import logging


class Optimizer:
    def __init__(self):
        self.logger = logging.getLogger()
        self.initial_candidate_size = 1
        self.best = None
        self.fitness_trend = []
