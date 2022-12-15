import os
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd

from . import OutputMethod

import numpy as np


class Single_Result_OutputMethod(OutputMethod):
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
            along_time=True,
            along_trace=True,
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

        self.total_results = None
        self.batch_results = []

    def _update(self, engine, results):
        plt.title(self.figure_params['title'])
        plt.xlabel(self.figure_params['x_label'])
        plt.ylabel(self.figure_params['y_label'])

        if isinstance(results, np.ndarray) and len(results.shape) == 1:
            self.batch_results.append(np.max(results))
        elif isinstance(results, np.ndarray) and len(results.shape) == 2:
            self.batch_results.append(np.max(results, axis=1))

    def _finalize(self):
        if self.display:
            plt.show()

    def from_output_method(self, output_method):
        pass


class Multiple_Results_OutputMethod(OutputMethod):
    """
    self defined output method designed for multiple output method
    """

    def __init__(
            self,
            *engines,
            figure_params=None,
            output_path=None,
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
        self.figure_params = figure_params
        self.output_path = output_path
        self.display = display

    def _update(self, engine, results):
        if isinstance(results, tuple) and len(results) == 2:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for i in range(2):
                axs[i].set_title(self.figure_params['title'][i])
                axs[i].set_xlabel(self.figure_params['x_label'][i])
                axs[i].set_ylabel(self.figure_params['y_label'][i])
                for j in range(results[i].shape[0]):
                    if j != engine.solution:
                        axs[i].plot(results[i][j, :], color='tab:gray')
                axs[i].plot(results[i][engine.solution, :], color='red')

        if self.output_path:
            fig.savefig(self.output_path)
        if self.display:
            fig.show()

    def _finalize(self):
        pass

    def from_output_method(self, output_method):
        pass


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
