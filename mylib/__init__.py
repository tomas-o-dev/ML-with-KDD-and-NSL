"""
# mylib  __init.py__

# This file lists the "public-facing" (importable) functions in the
# files in this directory. The __pycache__ directory is automatically
# created the first time any of them are imported.

# To add new functions to the library, (1) put the file of definitions 
# in this directory, (2) add a new line below with the function names, 
# (3) add the names to the __all__ list (order does not matter), and 
# (4) delete the __pycache__ directory.

# If you modify a function definition in any of the files here, be sure to 
# delete the __pycache__ directory so the new definitions will be imported

"""

from .mylib import show_labels_dist, show_metrics, bias_var_metrics
# from .stmc_tests import GroupResult, run_nemenyi 

__all__ = ['show_labels_dist'
#           'run_friedman',
#           'run_nemenyi',
#           'run_shaffer',
           'bias_var_metrics',
           'show_metrics']