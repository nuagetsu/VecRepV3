import os

from oct2py import octave
import numpy as np
import pprint
g = np.array([[1, 0.5, 0.75], [0.5, 1, 0.25], [0.75, 0.25, 1]])
_ = octave.addpath("C:/Users/WShihEe/PycharmProjects/VecRepV3/src/matlab_functions")
octave.push("n", 3)
octave.push("r_rank", 3)
octave.push("G", g)

# For fixed diagonal constraint
octave.eval("I_e = [1:1:n]'; J_e = I_e; k_e = length(I_e);")


# To generate the bound e,l & u
octave.eval("e = ones(n,1);  ConstrA.e = e; ConstrA.Ie = I_e; ConstrA.Je = J_e;")

# Set options
octave.eval("OPTIONS.tau    = 0; OPTIONS.tolrel = 1.0e-5;")

# Execute function
octave.eval("[X,INFOS] = PenCorr(G,ConstrA,r_rank,OPTIONS);", verbose=False)
X = octave.pull("X")
pprint.pprint(X)