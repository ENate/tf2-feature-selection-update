import argparse

def make_argparse():
    """An argparse function"""
    parsed = argparse.ArgumentParser()
    parsed.add_argument('--time_delta_n', help='Delta time change', default=0, type=float)
    arguments = parsed.parse_args()
    return arguments

def function_call_argparse():
    def_values = make_argparse()
    print(def_values.time_delta_n)