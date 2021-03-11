import argparse

def make_argparse():
    """An argparse function"""
    parsed = argparse.ArgumentParser()
    parsed.add_argument('--time_delta_n', action='store', type=int, required=True)
    arguments = parsed.parse_args()
    return arguments

def function_call_argparse():
    def_values = make_argparse()
    print(def_values.time_delta_n)

function_call_argparse()