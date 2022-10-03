from enum import Enum

class Policy(str, Enum):
    EW = 'ew'
    MINIMUN_VAR = 'minvar'
    MINIMUM_VAR_CONSTRAINED = 'minvar-constrained'