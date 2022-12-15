import os
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd

from . import OutputMethod

import numpy as np


class SinglePlotOutputMethod(OutputMethod):
    """
    self defined output method designed for single plot output method
    """

    def __init__(
            self,
            *engines,
            figure_params_along_time=None,
            figure_params_along_trace=None,
            output_path=None,
            filename=None,
            contain_raw_file=True,
            display=False,
    ):
        """
        :param figure_params_along_time: figure parameters of along_time results
        :param figure_params_along_trace: figure parameters of along_trace results
        :param engines: engines to be tracked
        :param output_path: it set, whether to save results (figure is saved as default)
        :param contain_raw_file: if true, save results in a .xlsx file
        :param display: it set, display or show the figure
        """
        OutputMethod.__init__(self, *engines)
        self.figure_params_along_time = figure_params_along_time
        self.figure_params_along_trace = figure_params_along_trace
        self.output_path = output_path
        self.filename = filename
        self.contain_raw_file = contain_raw_file
        self.display = display

        self.along_time_results = None
        self.along_trace_results = None

    def _update(self, engine, results):
        self.engine = engine
        if isinstance(results, np.ndarray) and len(results.shape) == 2:
            self.along_time_results = results
            optima = np.array(np.max(results, axis=1), ndmin=2).T
            self.along_trace_results = optima if not isinstance(self.along_trace_results, np.ndarray) else \
                np.concatenate((self.along_trace_results, optima), axis=1)

    def _finalize(self):
        plt.figure(0)
        plt.title(self.figure_params_along_time['title'])
        plt.xlabel(self.figure_params_along_time['x_label'])
        plt.ylabel(self.figure_params_along_time['y_label'])

        plt.figure(1)
        plt.title(self.figure_params_along_trace['title'])
        plt.xlabel(self.figure_params_along_trace['x_label'])
        plt.ylabel(self.figure_params_along_trace['y_label'])

        if self.engine.solution == -1:
            plt.figure(0)
            plt.plot(self.along_time_results.T)
            plt.figure(1)
            plt.plot(self.along_trace_results.T)
        else:
            # show the result of correct key guess
            for i in range(self.along_time_results.shape[0]):
                if i != self.engine.solution:
                    plt.figure(0)
                    plt.plot(self.along_time_results[i, :], color='tab:gray')
                    plt.figure(1)
                    plt.plot(self.along_trace_results[i, :], color='tab:gray')
            plt.figure(0)
            plt.plot(self.along_time_results[self.engine.solution, :], color='red')
            plt.figure(1)
            plt.plot(self.along_trace_results[self.engine.solution, :], color='red')

        if self.output_path:
            plot_path1 = os.sep.join([self.output_path, 'along_time', 'plot'])
            plot_path2 = os.sep.join([self.output_path, 'along_trace', 'plot'])
            if not os.path.exists(plot_path1):
                os.makedirs(plot_path1)
            if not os.path.exists(plot_path2):
                os.makedirs(plot_path2)
            plt.figure(0)
            plt.savefig(os.sep.join([plot_path1, self.filename + '.png']))
            plt.figure(1)
            plt.savefig(os.sep.join([plot_path2, self.filename + '.png']))
            if self.contain_raw_file:
                raw_file_path1 = os.sep.join([self.output_path, 'along_time', 'tables'])
                raw_file_path2 = os.sep.join([self.output_path, 'along_trace', 'tables'])
                if not os.path.exists(raw_file_path1):
                    os.makedirs(raw_file_path1)
                if not os.path.exists(raw_file_path2):
                    os.makedirs(raw_file_path2)
                raw_data1 = pd.DataFrame(self.along_time_results)
                raw_data2 = pd.DataFrame(self.along_trace_results)
                writer1 = pd.ExcelWriter(os.sep.join([raw_file_path1, self.filename + '.xlsx']))
                writer2 = pd.ExcelWriter(os.sep.join([raw_file_path2, self.filename + '.xlsx']))
                raw_data1.to_excel(writer1, self.filename, float_format='%.5f')  # 2nd param is sheet name
                raw_data2.to_excel(writer2, self.filename, float_format='%.5f')  # 2nd param is sheet name
                writer1.close()
                writer2.close()

        if self.display:
            plt.figure(0)
            plt.show()
            plt.figure(1)
            plt.show()


class MultiplePlotsOutputMethod(OutputMethod):
    """
    self defined output method designed for multiple output method
    """

    def __init__(
            self,
            *engines,
            figure_params_along_time=None,
            figure_params_along_trace=None,
            output_path=None,
            filename=None,
            contain_raw_file=True,
            display=True,
    ):
        """
        :param figure_params: basic params to plot figures (dictionary (embedded list))
        ex: {'title': ['cmi+mi', 'cmi+pv'] , 'x_label': ['time', 'time'], 'y_label': ['mi', 'pv']}
        :param engines: engines to be tracked
        :param output_path: it set, save the figure to output_path (list of paths)
        :param display: it set, display or show the figure
        """
        OutputMethod.__init__(self, *engines)
        self.figure_params_along_time = figure_params_along_time
        self.figure_params_along_trace = figure_params_along_trace
        self.output_path = output_path
        self.filename = filename
        self.contain_raw_file = contain_raw_file
        self.display = display

        self.along_time_results = [None, None]
        self.along_trace_results = [None, None]

    def _update(self, engine, results):
        self.engine = engine
        if isinstance(results, tuple) and len(results) == 2:
            for i in range(2):
                self.along_time_results[i] = results[i]
                optima = np.array(np.max(results[i], axis=1), ndmin=2).T
                self.along_trace_results[i] = optima if not isinstance(self.along_trace_results, np.ndarray) else \
                    np.concatenate((self.along_trace_results, optima), axis=1)

    def _finalize(self):
        fig, axs = [], []
        fig.append(plt.subplots(1, 2, figsize=(16, 6))[0])
        fig.append(plt.subplots(1, 2, figsize=(16, 6))[0])
        axs.append(plt.subplots(1, 2, figsize=(16, 6))[1])
        axs.append(plt.subplots(1, 2, figsize=(16, 6))[1])
        for i in range(2):
            figure_params = self.figure_params_along_time if i == 0 else self.figure_params_along_trace
            results = self.along_time_results if i == 0 else self.along_trace_results
            for j in range(2):
                axs[i][j].set_title(figure_params['title'][i])
                axs[i][j].set_xlabel(figure_params['x_label'][i])
                axs[i][j].set_ylabel(figure_params['y_label'][i])
                if self.engine.solution == -1:
                    axs[i][j].plot(results[j].T)
                for k in range(results[j].shape[0]):
                    if k != self.engine.solution:
                        axs[i][j].plot(results[j][k, :], color='tab:gray')
                axs[i][j].plot(results[j][self.engine.solution, :], color='red')

        if self.output_path:
            plot_path1 = os.sep.join([self.output_path, 'along_time', 'plot'])
            plot_path2 = os.sep.join([self.output_path, 'along_trace', 'plot'])
            if not os.path.exists(plot_path1):
                os.makedirs(plot_path1)
            if not os.path.exists(plot_path2):
                os.makedirs(plot_path2)
            for i in range(2):
                fig[i].savefig(os.sep.join([plot_path1, self.filename + '.png']))

                if self.contain_raw_file:
                    raw_file_path1 = os.sep.join([self.output_path, 'along_time', 'tables'])
                    raw_file_path2 = os.sep.join([self.output_path, 'along_trace', 'tables'])
                    if not os.path.exists(raw_file_path1):
                        os.makedirs(raw_file_path1)
                    if not os.path.exists(raw_file_path2):
                        os.makedirs(raw_file_path2)
                    raw_data1 = pd.DataFrame(self.along_time_results[i])
                    raw_data2 = pd.DataFrame(self.along_trace_results[i])
                    writer1 = pd.ExcelWriter(os.sep.join([raw_file_path1, self.filename + '_' + str(i) + '.xlsx']))
                    writer2 = pd.ExcelWriter(os.sep.join([raw_file_path2, self.filename + '_' + str(i) + '.xlsx']))
                    raw_data1.to_excel(writer1, self.filename, float_format='%.5f')  # 2nd param is sheet name
                    raw_data2.to_excel(writer2, self.filename, float_format='%.5f')  # 2nd param is sheet name
                    writer1.close()
                    writer2.close()

        if self.display:
            for i in range(2):
                fig[i].show()


class Incremental_Batch_OutputMethod(OutputMethod):
    """
        self defined output method designed for single output method
        """

    def __init__(
            self,
            *engines,
            figure_params=None,
            output_path=None,
            filename=None,
            contain_raw_file=True,
            display=True,
    ):
        """
        :param figure_params: basic params to plot figures (dictionary)
        ex: {'title': 'cmi', 'x_label': 'time', 'y_label': 'mi'}
        :param engines: engines to be tracked
        :param output_path: it set, whether to save results (figure is saved as default)
        :param contain_raw_file: if true, save results in a .xlsx file
        :param display: it set, display or show the figure
        """
        OutputMethod.__init__(self, *engines)
        self.figure_params = figure_params
        self.output_path = output_path
        self.filename = filename
        self.contain_raw_file = contain_raw_file
        self.display = display

        self.stored_results = None

    def _update(self, engine, results):
        print(results)

    def _finalize(self):
        pass
