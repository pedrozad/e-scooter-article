import numpy as np
from pymoo.operators.sampling.rnd import BinaryRandomSampling


## My BinaryOperator
class BinaryRandomSamplingCustom(BinaryRandomSampling):
    def __init__(self, df_individual) -> None:
        """
        This abstract class represents any sampling strategy that can be used to create an initial population or
        an initial search point.
        """
        self.individual_custom = df_individual
        super().__init__()
    
    # Custom approach. Erase the first individuals
    def _first_one(self, problem, n_samples):
        val = np.random.random((n_samples, problem.n_var))
        counter = 0
        for indv in self.individual_custom.itertuples():
            new_ind = eval(getattr(indv,"individuals"))
            val[counter] = new_ind
            counter += 1 
        return (val < 0.5).astype(bool)
        
        
    
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)

