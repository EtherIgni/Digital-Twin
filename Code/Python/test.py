from scipy.optimize import least_squares
import numpy as np

a=np.array([[1,2,-2],
            [1,1,1],
            [2,2,1]])

b=np.tril(a,-1)
c=np.triu(a,1)
d=np.diag(np.diag(a))

e=np.linalg.inv(d)@(b+c)

f=np.linalg.eig(e).eigenvalues

g=np.max(np.abs(f))

print(g)




a=np.array([[3,1,1],
            [-1,3,1],
            [-1,-1,3]])

b=np.tril(a,-1)
c=np.triu(a,1)
d=np.diag(np.diag(a))

e=np.linalg.inv(d)@(b+c)

e=np.linalg.inv(d-c)@(b)

f=np.linalg.eig(e).eigenvalues

g=np.max(np.abs(f))

print(g)
