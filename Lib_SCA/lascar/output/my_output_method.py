from math import ceil

import matplotlib.pyplot as plt
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
            display=True,
    ):
        """
        :param figure_params: basic params to plot figures (dictionary)
        ex: {'title': 'cmi', 'x_label': 'time', 'y_label': 'mi'}
        :param engines: engines to be tracked
        :param output_path: it set, save the figure to output_path
        :param display: it set, display or show the figure
        """
        OutputMethod.__init__(self, *engines)
        self.figure_params = figure_params
        self.output_path = output_path
        self.display = display

    def _update(self, engine, results):
        plt.title(self.figure_params['title'])
        plt.xlabel(self.figure_params['x_label'])
        plt.ylabel(self.figure_params['y_label'])

        if isinstance(results, np.ndarray) and len(results.shape) == 1:
            plt.plot(results)
        elif isinstance(results, np.ndarray) and len(results.shape) == 2:
            if engine.solution == -1:
                plt.plot(results.T)
            else:
                # show the result of correct key guess
                for i in range(results.shape[0]):
                    if i != engine.solution:
                        plt.plot(results[i, :], color='tab:gray')
                plt.plot(results[engine.solution, :], color='red')

            if self.output_path:
                plt.savefig(self.output_path)
            if self.display:
                plt.show()

        if self.output_path:
            plt.savefig(self.output_path)
        if self.display:
            plt.show()

    def _finalize(self):
        pass

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
            fig, axs = plt.subplot(1, 2, figsize=(16, 6))
            for i in range(2):
                axs[i].set_title(self.figure_params['title'][i])
                axs[i].set_xlabel(self.figure_params['x_label'][i])
                axs[i].set_ylabel(self.figure_params['y_label'][i])
                for j in range(results[i].shape[0]):
                    if j != engine.solution:
                        axs[i].plot(results[i, :], color='tab:gray')
                axs[i].plot(results[engine.solution, :], color='red')

        if self.output_path:
            fig.savefig(self.output_path)
        if self.display:
            fig.show()

    def _finalize(self):
        pass

    def from_output_method(self, output_method):
        pass
