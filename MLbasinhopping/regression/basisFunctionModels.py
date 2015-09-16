import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from MLbasinhopping.regression.models import RegressionModel, RegressionSystem

class BasisFunctionModel(RegressionModel):
    
    def Y(self, X):
        return sum(self.YbasisFunctions(X))
    
    def YbasisFunctions(self, X):
        raise NotImplementedError

class GaussianBasis(BasisFunctionModel):
        
    def YbasisFunctions(self, X):
        
        functions = []
        params = self.params.get_value()
        assert (len(params) % 3) == 0
        
        for o in range(len(params)/3):
            amplitude = self.params[3*o]
            mu = self.params[3*o+1]
            sig = self.params[3*o+2]
            model_i = amplitude * T.exp(-(X-mu)**2/(2.*sig**2))
            functions.append(model_i)
            
        return functions

class SinBasis(BasisFunctionModel):
    
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
    
    def predict_models(self, xval):
        
        self.x_to_predict.set_value(xval)    
        
        functions = self.YbasisFunctions(self.x_to_predict)
        
        return [f.eval() for f in functions]
        
            
class SinModel(RegressionModel):
    
    def Y(self, X):
        
        return T.sin(self.params[0]*X + self.params[1])
    
class HarmonicModel(RegressionModel):
    
    def Y(self, X):
        
        return 0.5 * self.params[0] * (X-self.params[1])**2 + self.params[2]
    
