import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualisation:
    def __init__(self):
        self.job_colour = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red', 4: 'tab:purple',
                           5: 'tab:brown', 6: 'tab:pink', 7: 'tab:grey', 8: 'tab:olive', 9: 'tab:cyan', 10: 'tab:cyan'}

    @staticmethod
    def plot_fitness_trend(trend):
        df = pd.DataFrame(trend)
        g = sns.relplot(kind="line", data=df)
        plt.show()

    @staticmethod
    def plot_fitness_trend_all_algs(trend):
        df = pd.DataFrame(trend)
        g = sns.relplot(kind="line", data=df)
        plt.show()

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
                ypos = (((int(m) + 1) * (10 * bar_height)) + 100 * int(m)) - (ji * bar_height)
                if ypos > ypos_mmax:
                    ypos_mmax = ypos
                elif ypos < ypos_mmin:
                    ypos_mmin = ypos
                xstart = j[0]
                xlength = j[1]
                # print('Machine ', m, '  Y ', ypos, '  Start ', xstart, '   Length ', xlength)
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
