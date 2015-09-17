import numpy as np
import unittest
import logging
import sys

from MLbasinhopping.utils import run_basinhopping, run_double_ended_connect, make_disconnectivity_graph
from MLbasinhopping.regression.models import RegressionSystem
from MLbasinhopping.regression.basisFunctionModels import *

def visualize_solutions(system, db, xvals, tvals, real_params):    
    for m in db.minima():
        system.model.params.set_value(m.coords)
        xs = np.arange(0.,3.0*np.pi,0.01)
#         xs = np.atleast_2d(xs)
        curve = system.model.predict(xs)
        plt.plot(xs, curve, '-')
        
    
    plt.plot(xvals,tvals,'x')
    system.model.params.set_value(real_params)
    plt.plot(np.arange(0.,3*np.pi,0.01), system.model.predict(np.arange(0.,3*np.pi,0.01)), '-')
    plt.show()
    
class CheckCorrectOutput(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345)
        
        # generate some test data
        xvals = 3.0*np.pi*np.random.random(100)
        testx = 3.0*np.pi*np.random.random(100)
        
        # parameters values which specify the model 
        real_params = np.random.random(2)
        real_params = real_params * np.pi
          
        # this is the model we use to fit the data
        model = SinModel(input_data = xvals, 
                                target_data = None, # generate samples automatically
                                starting_params=real_params,
                                testX = testx,
                                testt = None # generate samples automatically
                                )
    
        tvals = model.t.get_value()
        testt = model.testt.get_value()
    
        self.system = RegressionSystem(model)
        
    #     db = system.create_database("SinModel"+str(real_params[0])+"_"+str(real_params[1])+".sqlite")
        self.db = self.system.create_database()
        
        # run basin-hopping on this landscape
        nsteps = 10
        self.system, self.db = run_basinhopping(self.system, nsteps, self.db)
        
        # draw various best-fits from BH run
#         visualize_solutions(system, db, xvals, tvals, real_params)
           
        # connect minima
    #     run_double_ended_connect(system, db)
            
        # connect minima
    #     make_disconnectivity_graph(system, db)
    
    def test_minima(self):
        
        Nminima = len(self.db.minima())
        self.assertEqual(Nminima, 5, msg="Nminima: "+str(Nminima)+" != 5")
        m0 = self.db.minima()[0]
        self.assertAlmostEqual(m0.energy, 48.7613706646)

    def test_costGradient(self):
        
        pot = self.system.get_potential()
        coords = np.random.random(self.system.model.params.get_value().shape)
        
        c, g = pot.getEnergyGradient(coords)
        
        self.assertAlmostEqual(c, 126.055885792)
        self.assertAlmostEqual(g[0], 259.29652069)
        
        log = logging.getLogger("CheckCorrectOutput.test_costGradient")
        log.debug("cost, grad=\n"+str(c)+"\n"+str(g))
        
logging.basicConfig(stream=sys.stderr)
logging.getLogger("CheckCorrectOutput.test_costGradient").setLevel(logging.DEBUG)
suite1 = unittest.TestLoader().loadTestsFromTestCase(CheckCorrectOutput)
# suite2 = unittest.TestLoader().loadTestsFromTestCase(FindingInterestingParamsTestCase)

suite = unittest.TestSuite(tests=[suite1])
unittest.TextTestRunner(verbosity=2).run(suite)