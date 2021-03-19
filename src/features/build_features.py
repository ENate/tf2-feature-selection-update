import numpy as np
import pandas as pd


class AllDatasets(object):

    """Pre-process, organize, normalize all training data sets"""

    def __init__(self, input_data=None):
        """Initialize input data"""
        self.input_data = input_data
    
    def toy_data_set(self, n_row):
        x_inputs_1 = np.random.rand(n_row)
        x_inputs_2 = np.random.rand(n_row)
        x_inputs_3 = np.random.rand(n_row)
        y_classifier = np.array([1 if (x_inputs_1[i] + x_inputs_2[i] + (x_inputs_3[i])/3 + np.random.randn(1) > 1) else 0 for i in range(n_row)])
        y_cts = x_inputs_1 + x_inputs_2 + x_inputs_3/3 + np.random.randn(n_row)
        dat = np.array([x_inputs_1, x_inputs_2, x_inputs_3]).transpose()
        return dat, y_classifier, y_cts

    

if __name__ == "__main__":
    pass