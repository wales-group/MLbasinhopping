import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from MLbasinhopping.regressionModels import BaseTheanoModel, RegressionSystem

class GaussianBasis(BaseTheanoModel):
        
    def Y(self, X):
        
        params = self.params.get_value()
        assert (len(params) % 3) - 1 == 0
        
        model = 0
        for o in range(len(params)/3):
            amplitude = self.params[3*o]
            mu = self.params[3*o+1]
            sig = self.params[3*o+2]
            model = model + amplitude * T.exp(-(X-mu)**2/(2.*sig**2))
    
        return model

class SinBasis(BaseTheanoModel):
    
    def Y(self, X):

        return sum(self.YbasisFunctions(X))
    
    def YbasisFunctions(self, X):
        
        functions = []
        
        params = self.params.get_value()
        assert (len(params) % 3) == 0
        
        for o in range(len(params)/3):
            a = self.params[3*o]
            f = self.params[3*o+1]
            b = self.params[3*o+2]
            model_i = a * T.sin(f*X + b)  
            functions.append(model_i)
        
        return functions
    
    def predict_models(self, X):
        pass
            
class SinModel(BaseTheanoModel):
    
    def Y(self, X):
        
        return T.sin(self.params[0]*X + self.params[1])
    
class HarmonicModel(BaseTheanoModel):
    
    def Y(self, X):
        
        return self.params[0] * (X-self.params[1])**2 + self.params[2]
    
