from scipy.optimize import least_squares
import numpy as np

a=np.linspace(1,2,4)

def exp(b):
    return(np.zeros(4))

np.savetxt("test save.txt",least_squares(exp,[0,1]).x)
