import statistics
from scipy.stats import ranksums


class Stats:
    @staticmethod
    def lower_bound(joblist):
        jobs = [sum(p) for p in joblist]
        print('Lower bound is {} for job {}'.format(min(jobs), jobs.index(min(jobs))))

    @staticmethod
    def upper_bound(joblist):
        jobs = [sum(p) for p in joblist]
        print('Upper bound is {} for job {}'.format(max(jobs), jobs.index(max(jobs))))

    @staticmethod
    def basic(trend):
        for alg in trend:
            stddev = statistics.pstdev(trend[alg])
            mean = statistics.mean(trend[alg])
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
