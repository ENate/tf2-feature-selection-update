import os
import sys
import imp
import numpy as np
import unittest
import tensorflow as tf
from tensorflow import keras
__path__ = [os.path.dirname(os.path.abspath(__file__))]
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from unittest import TestCase
from .training_params_functions import ModelTrainingParameters
from .adapted_lm_algorithm import ParseArgs

class ModelTrainingParametersTests(TestCase):

    def setUp(self):
        self.trainingModel = ModelTrainingParameters(3, [3, 2], 2)
        self.parseArgs = ParseArgs(3, [3, 2], 2)

    # Test the build network method
    def test_build_network(self):
        self.my_parser = [4, 3]
        self.trainingModel.hidden_layers = [4, 3]
        # self.args_new.my_parser.mlp_hid_structure
        self.trainingModel.input_features = 4
        self.trainingModel.num_classes = 2
        self.neurons_cnt, self.lst_shape, self.lst_sizes, self.hid_layer = self.trainingModel.build_network()
        # check return types as int (number of neurons), list(layers, shapes and sizes)
        self.assertEqual(type(self.neurons_cnt), int)
        self.assertEqual(type(self.lst_shape), list)
        self.assertEqual(type(self.lst_sizes), list)
        self.assertEqual(len(self.lst_shape), len(self.lst_sizes))

    # test build network using tensorflow keras
    def test_tf_keras_mlp_model(self):
        """testing the tensorflow keras build network function
        """
        self.trainingModel.input_features = keras.Input(shape=(3, ), name="Inputs")
        self.trainingModel.hidden_layers = [3, 4]
        self.trainingModel.classify_shapes = 3
        self.model_types = self.trainingModel.tf_keras_mlp_model(self.trainingModel.input_features)
        print(self.model_types)
        self.assertNotEqual(type(self.model_types), tf.Tensor)

    def test_loss_function_from_scratch(self): 
        """Function to test entropy loss for the MLP model prior to training
        """
        set_pts, input_n, class_val = 100, 3, 2
        x1 = np.random.randn(set_pts, input_n)
        x2 = np.random.randn(set_pts, 2)
        n_neurons, shape_lst, sizes_lst, l_hidden = self.trainingModel.build_network()
        dict_params = {'choose_val': 1, 'wb_shapes': shape_lst, 'l_hidden': l_hidden, 'wb_sizes': sizes_lst, 'nclasses': class_val, 'n': input_n}
        self.initializer = tf.initializers.GlorotNormal(43)
        params_0 = tf.Variable(self.initializer([n_neurons], dtype=tf.float64))
        loss_from_scratch = self.trainingModel.loss_function_from_scratch(dict_params, params_0, x1, x2)
        print(loss_from_scratch)
    
    def test_loss_function_keras(self):
        """Construct the loss function and arrange

        Args:
            model (Keras class): The keras model object class.

        """
        # build a keras loss using return values from build_network method.
        # Generate a toy data set
        self.set_points, self.row_try, self.n_row = 100, 10, 3
        x1 = np.random.randn(self.set_points, self.n_row)
        x2 = np.random.randn(self.set_points, 2)
        x3_predict = np.random.rand(self.row_try, self.n_row)
        self.input_features_dim = keras.Input(shape=(self.n_row, ), name="Inputs_test")
        self.keras_model = self.trainingModel.mlp_function_keras(self.trainingModel.input_features_model)
        self.history, self.prediction = self.trainingModel.loss_function_keras(self.keras_model, x1, x2, x3_predict)
        # self.assertEqual(self.model, 4)

# Define a class to pytest parse_arg values

class ArgParseTestCase(TestCase):

    def setUp(self):
        """A setup for argparse!"""
        self.args_class = ParseArgs(2, [4, 3], 3)

    def test_version(self):
        sys.argv[1:]=['-v']
        with self.assertRaises(SystemExit):
            self.args_class.parse_command_line_argument_function()
            # testing version message requires redirecting stdout
    # similarly for a misc_opts test

class DatasetTests(TestCase):
    """Pre-processing and testing data set functions"""
    def setUp(self):
        """Testing the datasets"""
        pass
        


if __name__=='__main__':
    unittest.main()