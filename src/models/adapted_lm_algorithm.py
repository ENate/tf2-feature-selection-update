import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
sys.path.append("../features/")
sys.path.append("../data/")
sys.path.append(".")
__path__ = [os.path.dirname(os.path.abspath(__file__))]
from build_features import AllDatasets
from make_dataset import data_one_preprocessing
from training_params_functions import ModelTrainingParameters

SEED=42
INITIALIZERS = {'xavier': tf.initializers.GlorotNormal(seed=SEED), 'rand_uniform': tf.initializers.random_normal(seed=SEED), 'rand_normal': tf.initializers.random_uniform(seed=SEED)}
# how frequently log is written and checkpoint saved
LOG_INTERVAL_IN_SEC = 0.05
# variants of tensor flow built-in optimizers
TF_OPTIMIZERS = {'sgd': tf.keras.optimizers.SGD, 'adam': tf.keras.optimizers.Adam}

class ParseArgs:

    """A parse arguments initialization class"""

    def __init__(self, mlp_num_features, mlp_hidden_layers, mlp_num_classes):
        """Initialize elements to define in parse args function"""
        self.num_mlp_hidden_layers = mlp_hidden_layers
        self.mlp_num_features = mlp_num_features
        self.mlp_classes = mlp_num_classes
        self.allDatasets = AllDatasets(4)
        self.modelTrainingParameters = ModelTrainingParameters(self.mlp_num_features, self.num_mlp_hidden_layers, self.mlp_classes)

    
    def parse_command_line_argument_function(self):
        """
        Define a parse argument function to initialize parameters
        """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--n_classes', action='store', type=int, required=False, default=2)
        self.parser.add_argument('--eval_steps', action='store', help='Number of steps to run for at each checkpoint', default=1, type=int)
        self.parser.add_argument('--activation', action='store', help='nonlinear activation function', type=str, choices=['relu', 'sigmoid', 'tanh'], default='tanh', required=False)
        self.parser.add_argument('--num_epochs', action='store', help='Number of epochs for iteration', default=100, type=int, required=False)
        self.parser.add_argument('--batch_size', action='store', help='Number of batch sizes', default=40, type=int, required=False)
        self.parser.add_argument('--mlp_hid_structure', action='store', help='Number of hidden layers for MLP', default=self.num_mlp_hidden_layers, type=object, required=False)
        self.parser.add_argument('--num_features', action='store', help='Number of input features',default=self.mlp_num_features, type=int, required=False)
        self.parser.add_argument('--optimizer', action='store', help='optimization algorithms', type=str, choices=['sgd', 'adam', 'lm'], default='lm', required=False)
        self.parser.add_argument('--initializer', action='store', help='trainable variables initializer', type=str, choices=['rand_normal', 'rand_uniform', 'xavier'], default='xavier', required=False)
        self.parser.add_argument('--step_delta_n', action='store', help='Delta time change', default=0, type=float, required=False)
        self.parser.add_argument('--time_delta_n', action='store', help='Delta time change', default=0, type=float, required=False)
        self.parser.add_argument('--choose_flag', action='store', help='Choice of algorithm to run', default=1, type=int, required=False)
        self.parser.add_argument('--LOG_INTERVAL_IN_SEC', action='store', help='Time interval to print results', default=0.05, type=float, required=False)
        self.training_args = self.parser.parse_args()
        self.initializer = INITIALIZERS[self.training_args.initializer]
        return self.training_args, self.initializer

        # define function to return parameter sizes, shapes and model structure
    def create_training_parameters(self):
        """A function to create and return training parameters for building ANN from scratch
        # See init function for the following parameters.
        Args:
            num_features ([type]): defines number of input features
            num_layers (list): Contains number of neurons at each layer
            num_classes (int): Represents number of labels in the output.
        """
        neurons, shapes, sizes, layers_hidden = self.modelTrainingParameters.build_network()
        print('The number of neurons are: ', neurons)
        return neurons, shapes, sizes, layers_hidden

    def gather_all_training_parameters(self):
        """Gather training parameters including data types.
        """
        # Call the build network function to construct parameter shapes and sizes.
        model_neurons, model_shapes, model_sizes, layers_hidden = self.modelTrainingParameters.build_network()
        self.hyper_params, self.initializer = ParseArgs(self.mlp_num_features, self.num_mlp_hidden_layers, self.mlp_classes).parse_command_line_argument_function()
        # Assemble final, big parameter dictionary for training
        second_kwarg = {'wb_sizes': model_sizes, 'wb_shapes': model_shapes, 'num_neurons': model_neurons,
                    'initializer': self.initializer, 'n': self.mlp_num_features, 'nhidden': layers_hidden,
                    'nclasses': self.hyper_params.n_classes, 'choose_val': 1}
        return second_kwarg, self.hyper_params, self.initializer

    def analyze_losses_before_training(self, in_obj):
        """Build different models: from scratch and keras models. Construct the loss
        and analyze before training.
        """
        modelKeras0 = self.modelTrainingParameters.tf_keras_mlp_model(in_obj)
        modelKeras = self.modelTrainingParameters.mlp_function_keras(3)
        return modelKeras, modelKeras0

    def load_all_datasets(self, four_value):
        """Load pre-processed features and outputs prior to training
        """
        all_dat, all_y, dataset = self.allDatasets.toy_data_set(four_value)
        print(all_dat.shape)
        return all_dat, all_y, dataset

if __name__ == "__main__":
    parseArgs = ParseArgs(2, [3, 2], 2)
    training_arguments, _ = parseArgs.parse_command_line_argument_function()
    neurons, num_shapes, num_sizes, num_layers = parseArgs.create_training_parameters()
    print('The number of parameters is: ', neurons)
    print('The sizes of parameters each layer are: ', num_sizes)
    print('The parameter shapes are: ', num_shapes)
    print(len(num_shapes), len(num_sizes))
    # check and call the tf keras model
    feat_obj = tf.keras.Input(shape=(2, ))
    kerasMosel, kerasMosel2 = parseArgs.analyze_losses_before_training(feat_obj)
    print(kerasMosel2.summary())
    # First toy data set
    d, y, y_cts = parseArgs.load_all_datasets(4)
    print(d.shape)
    print(y.shape)
    print(y_cts.shape)
