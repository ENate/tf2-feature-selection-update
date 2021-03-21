# -*- coding: utf-8 -*-
import os 
import sys
__path__ = [os.path.dirname(os.path.abspath(__file__))]
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from training_params_functions import ModelTrainingParameters


class TrainModel:

    """Define training models for features and compare with dropout"""

    def __init__(self, input_num, layers_num, init_tf_tensor, classes_num = None):
        """Initialize parameters for training"""
        self.classes_num = classes_num
        self.input_num = input_num
        self.layers_num = layers_num
        self.init_tf_tensor = init_tf_tensor
        self.modelTrainingParams = ModelTrainingParameters(self.input_num, self.layers_num, self.classes_num)

    def fit_function_keras(self, x_train_scale, y_train_scale, x_val_scale, init_tf_tensor):
        """Construct model to train keras loss using dropout

        Args:
            model_tf (tf.functional): A deep neural network built using functional API
        """
        # Call the keras model function to train
        model_tf = self.modelTrainingParams.mlp_function_keras(init_tf_tensor)
        model_tf.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        self.history=model_tf.fit(x_train_scale, y_train_scale, epochs=30, batch_size=10, verbose=1, validation_split=0.2)
        self.predictions = model_tf.predict(x_val_scale)
        print('Working.....')
        return self.history, self.predictions