import statistics
from scipy.stats import ranksums
import logging
import logger as lg


class Stats:
    @staticmethod
    def basic(trend):
        stddev = round(statistics.pstdev(trend), 3)
        mean = round(statistics.mean(trend), 3)
        minf = min(trend)
        maxf = max(trend)
        lg.message(logging.INFO, 'Minimum (best) is {}'.format(minf))
        lg.message(logging.INFO, 'Maximum (worst) is {}'.format(maxf))
        lg.message(logging.INFO, 'Standard deviation is {}'.format(stddev))
        lg.message(logging.INFO, 'Avg is {}'.format(mean))

    @staticmethod
    def taillard_compare(lb, ub, alg_fitness):
        diff = round(((alg_fitness - lb) / lb) * 100, 2)
        lg.message(logging.INFO, 'Difference between Taillard approximate lb ({}) and best fitness ({}) is {}%'.format(
            lb, alg_fitness, diff))

        diff = round(((alg_fitness - ub) / ub) * 100, 2)
        lg.message(logging.INFO, 'Difference between Taillard best known ub ({}) and best fitness ({}) is {}%'.format(
            ub, alg_fitness, diff))

    @staticmethod
    def wilcoxon(trend):
        alpha = 0.05
        # This method print on screen a + or a - sign according to the outcome of the Wilcoxon test.
        # The test is performed by taking into consideration two realisations of the same optimisation process performed with two different optimisers: the reference and the comparison algorithm.
        # The distribution of the final results, obtained by applying the reference for a fixed number of runs, is compared with that one obtained by applying the comparsion algorithm for the same amout of runs.


        zstat, pvalue = ranksums(self.sa.fitness_trend['SA'], self.ga.fitness_trend['GA'])
        print(zstat, pvalue)
        # // Mann Whitney U-Test (Wilcoxon Rank-Sum)
        w = '='
        if pvalue < alpha:
            if statisticss.mean(self.sa.fitness_trend['SA']) < statistics.mean(self.ga.fitness_trend['GA']):
                w = '+'
            else:
                w = '-'
        print('Wilcoxon = ', w)
