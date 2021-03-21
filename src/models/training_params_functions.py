import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model


class ModelTrainingParameters:

    def __init__(self, input_dim, layers_dim, out_classes):
        """
        Define and initializing parameters
        """
        self.input_features_model = input_dim
        self.hidden_layers = layers_dim
        self.n_classes = out_classes
    
    def build_network(self):
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
        self.mlp_structure = [int(self.input_features_model)] + self.hidden_layers + [int(self.n_classes)]
        self.classify_shapes = []
        for idx in range(len(self.hidden_layers + [int(self.n_classes)])):
            self.classify_shapes.append((self.mlp_structure[idx], self.mlp_structure[idx + 1]))
            self.classify_shapes.append((1, self.mlp_structure[idx + 1]))
        wb_sizes_classif = [hclassif * wclassif for hclassif, wclassif in self.classify_shapes]
        neurons_cnt_classif = sum(wb_sizes_classif)
        print("The total number of trainable parameters is: ", neurons_cnt_classif)
        return neurons_cnt_classif, self.classify_shapes, wb_sizes_classif, self.hidden_layers

    def tf_keras_mlp_model(self, k_inputs):
        """Create a neural network MLP using tensorflow keras

        Args:
            in_feats_dim (int): input feature dimension
            h_layer_dim (list): number of hidden layers
            class_dim (int): Number of output classes
        """
        x_inputs = layers.Dense(3, activation="relu", name="hidden_1")(k_inputs)
        x_inputs = layers.Dense(2, activation="relu", name="hidden_2")(x_inputs)
        outputs = layers.Dense(2, activation="softmax", name="predictions")(x_inputs)
        model = Model(inputs=k_inputs, outputs=outputs)
        return model
    
    # create parameters
    def create_parameters(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True,)
        return self.w

    def loss_function_from_scratch(self, sec_kwargs, param_values, x_classify, y_classify):
        """Function to create entropy loss for the MLP model prior to training

        Args:
            sec_kwargs (object): A dictionary of parameters via comman line with defaults in argparse
            param_values (Array): An array or matrix of weight parameters
            choose_val (int): Chooses which loss function to construct: 1 for classifier and 2 for regressor
            w_sizes (Array object): An array of weight sizes in each layer
            p_kwargs (dict object): User defined parameters types
        """
        # Variable initialization
        self.x_classify = x_classify # tf.Variable(tf.float64, shape=[None, sec_kwargs['n']])
        self.y_classify = y_classify # tf.Variable(tf.float64, shape=[None, sec_kwargs['nclasses']])
        classify_tensors_split = tf.split(param_values, sec_kwargs['wb_sizes'], 0)
        
        # reshape weights and biases
        for i in range(len(classify_tensors_split)):
            classify_tensors_split[i] = tf.reshape(classify_tensors_split[i], sec_kwargs['wb_shapes'][i])
        
        # identify weights and biases
        w_classify = classify_tensors_split[0:][::2] # split tensors
        b_classify = classify_tensors_split[1:][::2]

        # Build models
        classify_y_hat = x_classify # initialize for convenience
        for k in range(len(sec_kwargs['l_hidden'])):
            classify_y_hat = tf.keras.activations.sigmoid(tf.matmul(classify_y_hat, w_classify[k]) + b_classify[k])
        classify_y_hat = tf.matmul(classify_y_hat, w_classify[-1]) + b_classify[-1]
        
        # Build loss function
        if sec_kwargs['choose_val'] == 1:
            self.optimal_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y_classify, classify_y_hat))
        print(self.optimal_loss)
        return self.optimal_loss
    

    def mlp_function_keras(self, init_tensor):
        """Create the tf.keras mlp structure in keras before training

        Args:
            model (object): A loss function created using the keras functional API
        """
        self.model = tf.keras.models.Sequential()
        self.model.add(keras.layers.Dense(3, input_dim=init_tensor, kernel_initializer='normal', activation='relu'))
        self.model.add(keras.layers.Dense(2, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='linear'))
        return self.model

    def loss_function_keras(self, model, x_train_scale, y_train_scale, x_val_scale):
        """Creates the loss for keras model

        Args:
            model (int): create loss from tf.keras  
        """
        model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        self.history=model.fit(x_train_scale, y_train_scale, epochs=30, batch_size=10, verbose=1, validation_split=0.2)
        self.predictions = model.predict(x_val_scale)
        print('Working.....')
        return self.history, self.predictions


if __name__ == "__main__":
    in_feats_dim = keras.Input(shape=(3, 2), name="Inputs")
    mlp_hid = [3, 2]
    n_in, n_class = 2, 2
    model = ModelTrainingParameters(n_in, mlp_hid, n_class).tf_keras_mlp_model(in_feats_dim)
    sec_model = ModelTrainingParameters(n_in, mlp_hid, n_class).mlp_function_keras(4)
    a, b, c, d = ModelTrainingParameters(n_in, mlp_hid, n_class).build_network()

    print(b)
    print(model.summary())
    print(sec_model.summary())

    
