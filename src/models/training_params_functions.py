import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class ModelTrainingParameters:

    def __init__(self, input_dim, layers_dim, y_classes):
        """
        Define and initializing parameters
        """
        self.input_features = input_dim
        self.hidden_layers = layers_dim
    
    def build_network(self, in_feats, h_layers, n_classes):
        """Defines the inputs, number of hidden layers and classes to create mlp from scratch
        Taves in dict values for h_layers as user-defines dict.
        Args:
            in_feats (init): number of inputs
            h_layers (list): hidden layer nodes
            n_classes (int): number or output classes

        Returns:
            int: number of neurons
            list: number of shapes per layer
            list: number of trainable parameters in each layer
            list: number of hidden layers
        """
        self.mlp_structure = [int(in_feats)] + h_layers + [int(n_classes)]
        self.classify_shapes = []
        for idx in range(len(h_layers + [int(n_classes)])):
            self.classify_shapes.append((self.mlp_structure[idx], self.mlp_structure[idx + 1]))
            self.classify_shapes.append((1, self.mlp_structure[idx + 1]))
        wb_sizes_classif = [hclassif * wclassif for hclassif, wclassif in self.classify_shapes]
        neurons_cnt_classif = sum(wb_sizes_classif)
        print("The total number of trainable parameter is: ", neurons_cnt_classif)
        return neurons_cnt_classif, self.classify_shapes, wb_sizes_classif, h_layers

    def tf_keras_mlp_model(self, in_feats_dim): # , h_layer_dim, class_dim):
        """Create a neural network MLP using tensorflow keras

        Args:
            in_feats_dim (int): input feature dimension
            h_layer_dim (list): number of hidden layers
            class_dim (int): Number of output classes
        """
        x_inputs = layers.Dense(2, activation="relu", name="hidden_1")(in_feats_dim)
        x_inputs = layers.Dense(3, activation="relu", name="hidden_2")(x_inputs)
        outputs = layers.Dense(4, activation="softmax", name="predictions")(x_inputs)
        model = keras.Model(inputs=in_feats_dim, outputs=outputs)
        print(model.summary())
        return model

    def loss_function_from_scratch(self, sec_kwargs, param_values, choose_val, w_sizes, p_kwargs):
        """Function to create entropy loss for the MLP model prior to training

        Args:
            sec_kwargs (object): A dictionary of parameters via comman line with defaults in argparse
            param_values (Array): An array or matrix of weight parameters
            choose_val (int): Chooses which loss function to construct: 1 for classifier and 2 for regressor
            w_sizes (Array object): An array of weight sizes in each layer
            p_kwargs (dict object): User defined parameters types
        """
        self.x_classify = tf.Variable(tf.float64, shape=[None, sec_kwargs['n']])
        self.y_classify = tf.Variable(tf.float64, shape=[None, sec_kwargs['n_classes']])
        classify_tensors_split = tf.split(param_values, sec_kwargs['wb_sizes'], 0)
        w_classify = classify_tensors_split[0:][::2] # split tensors
        b_classify = classify_tensors_split[0:][::2]
        classify_y_hat = self.x_classify # initialize for convenience
        for k in range(sec_kwargs['l_hidden']):
            classify_y_hat = tf.math.sigmoid(tf.matmul(classify_y_hat, w_classify[i]) + b_classify[i])
        classify_y_hat = tf.matmul(classify_y_hat, w_classify[-1]) + b_classify
        if choose_val == 1:
            self.optimal_loss = tf.reduce_sum(tf.math.sigmoid_cross_entropy)
        return self.optimal_loss
    

    def mlp_function_keras(self, input_dimension):
        """Create the tf.keras mlp structure in keras before training

        Args:
            model (object): A loss function created using the keras functional API
        """
        self.model = tf.Sequential()
        self.model.add(keras.layers.Dense(12, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        self.model.add(keras.layers.Dense(8, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='linear'))
        self.model.summary()
        return self.model

    def loss_function_keras(self, x_train_scale, y_train_scale, x_val_scale):
        """Creates the loss for keras model

        Args:
            model (int): create loss from tf.keras  
        """
        model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        history=model.fit(x_train_scale, y_train_scale, epochs=30, batch_size=150, verbose=1, validation_split=0.2)
        predictions = model.predict(x_val_scale)
        return history, predictions

    def fit_function_keras(self, model):
        """Organize function to train model

        Args:
            model (int): A model object from the tf.keras layer class.
        """
        pass


if __name__ == "__main__":
    in_feats_dim = keras.Input(shape=(3, 2), name="Inputs")
    mlp_hid = [3, 2]
    n_in, n_class = 3, 4
    ModelTrainingParameters(n_in, mlp_hid, n_class).tf_keras_mlp_model(in_feats_dim)
    ModelTrainingParameters(n_in, mlp_hid, 4).build_network(n_in, mlp_hid, n_class)

    
