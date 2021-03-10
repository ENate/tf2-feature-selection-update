import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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

if __name__ == "__main__":
    in_feats_dim = keras.Input(shape=(3, 2), name="Inputs")
    mlp_hid = [3, 2]
    n_in, n_class = 3, 4
    ModelTrainingParameters(n_in, mlp_hid, n_class).tf_keras_mlp_model(in_feats_dim)
    ModelTrainingParameters(n_in, mlp_hid, 4).build_network(n_in, mlp_hid, n_class)
    
     

    
