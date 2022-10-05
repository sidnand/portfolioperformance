from enum import Enum

class Policy(str, Enum):
    EW = 'Equal Weight'
    MINIMUM_VAR = 'Minimum Variance'
    MINIMUM_VAR_CONSTRAINED = 'Minimum Variance with Shortsell Constraints'
    MINIMUM_VAR_GENERALIZED_CONSTRAINED = 'Minimum Variance with General Constraints'