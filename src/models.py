# CODE FOR ALL THE PORTFOLIO MODELS

from numpy import ones

"""
   Equal weighted model

   param n : number of columns
   returns : [n - 1, 1] array, where each element is 1/n
"""
def ew(n):
    return 1/n * ones((n - 1, 1))