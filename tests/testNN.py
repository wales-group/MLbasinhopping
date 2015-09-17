import numpy as np
import unittest
import logging
import sys

from MLbasinhopping.utils import run_basinhopping, run_double_ended_connect, make_disconnectivity_graph

from MLbasinhopping.NN.models import NNSystem, NNModel

class CheckCorrectOutput(unittest.TestCase):

    def setUp(self):
        
        np.random.seed(12345)
        
        ndata = 500
        n_hidden = 10
        p = 2
        L2_reg=np.power(1.0e1, -p)
    #     L1_reg=np.power(1.0e1, -p)
    #     L2_reg=0.0
        L1_reg=0.0
        bias_reg = 0.0
        
        model = NNModel(ndata=ndata, n_hidden=n_hidden, L1_reg=L1_reg, L2_reg=L2_reg, bias_reg=bias_reg)
        self.system = NNSystem(model)

        self.db = self.system.create_database()
         
#         run_basinhopping(system, nsteps, db)     

        # connect minima
#         run_double_ended_connect(system, db)
            
        # connect minima
#         make_disconnectivity_graph(system, db)
    
    def test_minima(self):
        
        quench = self.system.get_minimizer()
        coords = np.random.random(self.system.model.nparams)
        ret = quench(coords)
        self.assertAlmostEqual(ret.energy, 3.750895922, msg="Error: Quenched energy = "+str(ret.energy))

#         self.assertEqual(Nminima, 5, msg="Nminima: "+str(Nminima)+" != 5")


#     def test_costGradient(self):
#         
#         pot = self.system.get_potential()
#         coords = np.random.random(self.system.model.params.get_value().shape)
#         
#         c, g = pot.getEnergyGradient(coords)
#         
#         self.assertAlmostEqual(c, 126.055885792)
#         self.assertAlmostEqual(g[0], 259.29652069)
#         
#         log = logging.getLogger("CheckCorrectOutput.test_costGradient")
#         log.debug("cost, grad=\n"+str(c)+"\n"+str(g))
        
logging.basicConfig(stream=sys.stderr)
logging.getLogger("CheckCorrectOutput.test_costGradient").setLevel(logging.DEBUG)
suite1 = unittest.TestLoader().loadTestsFromTestCase(CheckCorrectOutput)
# suite2 = unittest.TestLoader().loadTestsFromTestCase(FindingInterestingParamsTestCase)

suite = unittest.TestSuite(tests=[suite1])
unittest.TextTestRunner(verbosity=2).run(suite)