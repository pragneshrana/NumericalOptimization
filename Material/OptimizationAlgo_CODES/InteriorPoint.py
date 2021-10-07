#simple lp

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
# min cx
# x >= 0
# Ax = b





def newtonDec(df, dx):
	return np.dot(df,dx)

# assumes that x + alpha*dx can be made positive
def linesearch(x, dx):
   alpha = 1.
   while not np.all( x + alpha*dx > 0):
   		alpha *= 0.1
   return alpha

# min cx

def solve_lp2(A, b, c, gamma, xstart=None):
	#x = np.ones(A.shape[1])
	#lam = np.zeros(b.shape)
	xsize = A.shape[1]
	if xstart is not None:
		x = xstart
	else:
		#xlam = np.ones(xsize + b.size)
		x = np.ones(xsize) # xlam[:xsize]
		#lam = xlam[xsize:]
	while True :
		print("Iterate")
		H = sparse.bmat( [[ sparse.diags(gamma / x**2)   ,   A.T ],
		                  [ A  ,                         0 ]]  )

		dfdx = c - gamma / x #+  A.T@lam 
		dfdlam = A@x - b
		df = np.concatenate((dfdx, dfdlam))#np.zeros(b.size))) # dfdlam))
		#np.concatenate( , A@x - b)
		dxlam = linalg.spsolve(H,df)
		dx = - dxlam[:xsize]
		lam = dxlam[xsize:]

		alpha = linesearch(x,dx)
		x += alpha * dx
		#lam += dlam
		if newtonDec(dfdx,dx) >= -1e-10:
			print("stop")
			break

	return x, lam


def solve_lp(A,b,c, xstart=None):
	gamma = 1.0
	xsize = A.shape[1]
	x = np.ones(xsize)
	for i in range(8):
		x, lam = solve_lp2(A, b, c, gamma, xstart=x)
		gamma *= 0.1
	return x, lam


N = 12
A = np.ones(N).reshape(1,-1)
b = np.ones(1)*2
c = np.zeros(N)
c[0] = -1


#print(solve_lp(A,b,c, 0.000001))
print(solve_lp(A,b,c))




def BB(A, b, c, best, xhint = None):
	picked = np.zeros(xsize)
	picked[pickvar] = 1
	Anew = sparse.hstack((A, picked))
	bnew = np.concatenate((b,choice))
	x, lam = 
	np.dot(c,x)
	if lp_solve(Anew, bnew, c) < best:
		best, x = BB(Anew, bnew , c, best, xhint)
	return best, x




'''
#min  cx + gamma * ln(x)
# s.t. Ax = b

# cx + gamma * ln(x) + lambda (Ax - b)

#gradient 
delx = c + gamma * 1/x + lambda A
dellam = Ax - b
# hess
dlx = A
dxl = A.T
dxx = - gamma (1/x**2)


H @ (x l) = (delx dell)
'''
