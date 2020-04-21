import statistics
from scipy.stats import ranksums


class Stats:
    @staticmethod
    def basic(trend):
        for alg in trend:
            stddev = statistics.pstdev(trend[alg])
            mean = statistics.mean(trend[alg])
            minf = min(trend[alg])
            maxf = max(trend[alg])

            print('{} Min (best) is {}'.format(alg, minf))
            print('{} Max (worst) is {}'.format(alg, maxf))
            print('{} Standard deviation is {}'.format(alg, stddev))
            print('{} Mean is {}'.format(alg, mean))

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
