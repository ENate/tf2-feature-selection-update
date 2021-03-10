import sys
import pytest
import unittest
import tensorflow as tf
from tensorflow import keras

from unittest import TestCase
from .training_params_functions import ModelTrainingParameters
from .adapted_lm_algorithm import ParseArgs


@pytest.fixture
def training_models():
    """
    tests parameter initializations
    """
    return ModelTrainingParameters(3, 2)


# @pytest.mark.usefixtures("training_models")
class ModelTrainingParametersTests(TestCase):

    def setUp(self):
        self.trainingModel = ModelTrainingParameters(3, [3, 2], 2)

    # Test the build network method
    def test_build_network(self):
        self.my_parser = [4, 3]
        self.trainingModel.hidden_layers = [4, 3]
        # self.args_new.my_parser.mlp_hid_structure
        self.trainingModel.input_features = 4
        self.trainingModel.num_classes = 2
        self.neurons_cnt, self.lst_shape, self.lst_sizes, self.hid_layer = self.trainingModel.build_network(self.trainingModel.input_features, self.trainingModel.hidden_layers, self.trainingModel.num_classes)
        # check return types as int (number of neurons), list(layers, shapes and sizes)
        self.assertEqual(type(self.neurons_cnt), int)
        self.assertEqual(type(self.lst_shape), list)
        self.assertEqual(type(self.lst_sizes), list)

    # test build network using tensorflow keras
    def test_tf_keras_mlp_model(self):
        """testing the tensorflow keras build network function
        """
        self.trainingModel.input_features = keras.Input(shape=(3, 2), name="Inputs")
        self.trainingModel.hidden_layers = [3, 4]
        self.trainingModel.classify_shapes = 3
        self.model_types = self.trainingModel.tf_keras_mlp_model(self.trainingModel.input_features)
        self.assertEqual(type(self.model_types), tf.python.keras.engine.training.Model)

    def test_loss_function_keras(self, model):
        """Construct the loss function and arrange

        Args:
            model (Keras): The keras model object class.
        """
        pass

# Define a class to pytest parse_arg values

class ArgParseTestCase(TestCase):

    def setUp(self):
        """A setup for argparse!"""
        self.args_class = ParseArgs([4, 3])

    def test_version(self):
        sys.argv[1:]=['-v']
        with self.assertRaises(SystemExit):
            self.args_class.parse_command_line_argument_function()
            # testing version message requires redirecting stdout
    # similarly for a misc_opts test


if __name__=='__main__':
    unittest.main()