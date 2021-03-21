import os
import sys
import numpy as np
import tensorflow as tf
from unittest import TestCase
# from tensorflow import keras
__path__ = [os.path.dirname(os.path.abspath(__file__))]
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from train_model import TrainModel


class TestTrainModel(TestCase):

    """Testing train model function"""

    def setUp(self):
        """Initialize all parameters"""
        self.input_test = 3
        self.layer_test_val = [3, 2]
        self.classes_test = 2
        self.init_tf_tensor = 3 # tf.keras.Input(shape=(3,))
        self.trainModel = TrainModel(self.input_test, self.layer_test_val, self.init_tf_tensor, self.classes_test)
    
    def test_fit_function_keras(self):
        """Organize function to train model

        Args:
            model (int): A model object from the tf.keras layer class.
        """
        # Initialize the x_, y_ and x_training and testing data.
        x_scale = np.random.rand(10, self.input_test)
        y_scale = np.random.rand(10, self.classes_test)
        x_valid = np.random.rand(10, self.input_test)
        self.history, self.prediction = self.trainModel.fit_function_keras(x_scale, y_scale, x_valid, self.init_tf_tensor)
        self.assertEqual(self.prediction, tf.Tensor)
        self.assertEqual(self.history, tf.Tensor)