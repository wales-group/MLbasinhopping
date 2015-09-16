
class BaseModel(object):
    
    def cost(self, coords):
        return NotImplementedError
    def costGradient(self, coords):
        return NotImplementedError
    def costGradientHessian(self, coords):
        return NotImplementedError