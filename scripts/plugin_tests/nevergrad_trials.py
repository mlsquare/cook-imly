'''
Notes-
1) What is Candidate?
2) What are ask and tell methods for?


How to contribute?
1) Modules that hold different types of optimizers
    + optimizerlib, oneshot, recast and DE
    
'''

# _OptimizerClass -> base.Optimizer

# ParametrizedOptimizerClass -> base.ParametrizedFamily ## This is to create some sort of 'parameterized' subclass of OptimizerClass

# 		_optimizer_class = _OptimizerClass

# OptimizerClass = ParametrizedOptimizerClass().with_name("OptimizerClass", register=True) ## This is basically for registry

#  --- OR ---

#  OptimizerClass -> base.Optimizer
#  			with the decorator @registry.register


# -------------------------------------------------------------------------------------

# Candidate -- To hold args, kwargs and data of objective function
# candidate_maker -- To create a `Candidate` instance. Why not create it directly?

