import statistics
from scipy.stats import ranksums
import logging
import logger as lg
import statistics
from collections import OrderedDict


class Stats:
    @staticmethod
    def summary(optimizers):
        lg.message(logging.INFO, 'Basic Statistics')
        reference_sample = []
        comparison_sample = []
        alpha = 0.05
        summary_results = OrderedDict()

        oi = 0  # Counter for enabled optimizers, with first as ref
        for opt in optimizers:
            if not opt['Enabled']:
                continue
            stdev = round(statistics.pstdev(opt['Ft']), 3)
            mean = round(statistics.mean(opt['Ft']), 3)
            minf = min(opt['Ft'])
            maxf = max(opt['Ft'])
            wts = ' '  # Wilcoxon test symbol
            if oi == 0:
                reference_sample = opt['Ft']
                ref_mean = mean
            elif oi > 0:  # Wilcoxon is pairwise comparison so makes no sense without at least 1 pair
                comparison_sample = opt['Ft']
                zstat, pvalue = ranksums(reference_sample, comparison_sample)
                wts = '='
                if pvalue < alpha:
                    if ref_mean > mean:
                        wts = '-'
                    else:
                        wts = '+'
            summary_results[opt['Id']] = {'minf': minf, 'maxf': maxf, 'mean': mean, 'stdev': stdev, 'wts': wts,
                                           'AvgCts': opt['AvgCts'], 'lb_diff_pct': opt['lb_diff_pct'], 'ub_diff_pct':
                                               opt['ub_diff_pct']}
            oi += 1

        lg.message(logging.INFO, 'Optimiser\tMin Fitness\tMax Fitness\tAvg Fitness\tStDev\tWilcoxon\tLB Diff %\tUB Diff %\tAvg Cts')
        for k, v in summary_results.items():
            lg.message(logging.INFO, '{}\t\t{}\t\t{}\t\t{}\t\t{}\t{}\t\t{}\t\t{}\t\t{}'.format(
                str(k), str(v['minf']), str(v['maxf']), str(v['mean']), str(v['stdev']), str(v['wts']),
                str(v['lb_diff_pct']), str(v['ub_diff_pct']), str(round(v['AvgCts'], 3))))

    @staticmethod
    def taillard_compare(lb, ub, alg_fitness):
        return round(((alg_fitness - lb) / lb) * 100, 2), round(((alg_fitness - ub) / ub) * 100, 2)

