import argparse


class ParseArgs(object):

    """A parse arguments initialization class"""

    def __init__(self, mlp_hidden_layers):
        """Initialize elements to define in parse args function"""
        self.num_mlp_hidden_layers = mlp_hidden_layers

    
    def parse_command_line_argument_function(self):
        """
        define a parse argument function to initialize parameters
        """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--mlp_hid_structure', action='store', type=object, required=True)
        self.parser.add_argument('--optimizer', action='store', type=object, required=False)
        self.training_args = self.parser.parse_args()
        print(self.training_args)
        return self.training_args


if __name__ == "__main__":
    ParseArgs([3, 2])