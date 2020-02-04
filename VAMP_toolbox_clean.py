# Toolbox implementing Oracle_VAMP for the paper "Asymptotic error for convex penalized linear regression beyond Gaussian matrices" - Under review for COLT 2020

# Basic damping is implemented in both SE and DR-VAMP recursions.

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import ortho_group
from scipy import special

# Necessary functions and densities for eigenvalue distributions and state evolution equations

# soft-thresholding

def soft_thresh(x,thresh):
	n = len(x)
	res = np.zeros(n)
	
	for i in range(n):
		
		if np.abs(x[i])>thresh:
			res[i] = x[i]-np.sign(x[i]-thresh)*thresh
			
		if np.abs(x[i])<=thresh :
			res[i] = 0

	return(res)

# sampling of Haar matrices, faster than scipy ortho_group and sufficient for conisdered setup. Based on same method as scipy ortho_group, see scipy userguide.

def Haar_sample(n):
    #We first generate a m x m orthogonal matrix by taking the QR decomposition of a random m x m gaussian matrix
    gaussian_matrix = np.random.normal(0, 1., (n,n))
    O, R = np.linalg.qr(gaussian_matrix)
    #Then we multiply on the right by the signs (or phases in the complex case) of the diagonal of R to really get the Haar measure
    D = np.diagonal(R)
    Lambda = D / np.abs(D)   # normalization
    O = np.multiply(O, Lambda)
    return(O)

# distribution of state evolution variables, careful tau is a variance

def gaussian(x,tau):
	return((1/np.sqrt(2*np.pi*tau))*np.exp(-x**2/(2*tau)))

# gauss-bernoulli distribution on the teacher weights x0

def gauss_bernoulli(rho,n):
	
	x = np.zeros(n)
	
	for i in range(n):
		
		indicator = np.random.random()
		
		if indicator > rho :
			x[i] = 0
		
		if indicator <= rho : 
			x[i] = np.random.randn()
			
	return(x)

# shorthands for error functions

def erfc(x):
	return(special.erfc(x))

def erf(x):
	return(special.erf(x))


# Auxiliary function for integration in state evolution for l1 problem according to Appendix D.

def SE_aux_l1(x,reg,tau1,gamma1,my_eps):
	A = reg/gamma1
	my_e = 1/(1+my_eps/gamma1)
	res = 0.5*x**2*(erf((A-x)/(np.sqrt(2*tau1)))+erf((A+x)/(np.sqrt(2*tau1))))
	res = res+x**2-2*my_e*x**2+my_e**2*(tau1+(reg/gamma1)**2+x**2)
	res = res+1/np.sqrt(2*np.pi)*my_e*np.sqrt(tau1)*(np.exp(-(reg/gamma1-x)**2/(2*tau1))*(-my_e*reg/gamma1+(my_e-2)*x)-np.exp(-(reg/gamma1+x)**2/(2*tau1))*(my_e*reg/gamma1+(my_e-2)*x))
	res = res-0.5*(my_e**2*(tau1+(reg/gamma1-x)**2)+2*my_e*(reg/gamma1-x)*x+x**2)*erf((reg/gamma1-x)/(np.sqrt(2*tau1)))
	res = res-0.5*(x**2-2*my_e*x*(reg/gamma1+x)+my_e**2*(tau1+(reg/gamma1+x)**2))*erf((reg/gamma1+x)/(np.sqrt(2*tau1)))
	return(res)

# Marcenko-Pastur density

# careful, scaling (aspect ratio) is done when integrating

def MP(x,alpha):
	a = (1-np.sqrt(1/alpha))**2
	b = (1+np.sqrt(1/alpha))**2
	res = np.sqrt(max(0,(x-a))*max(0,b-x))/(2*np.pi*x)
	return(res)

# rescaled uniform for prescribing the spectrum for rotationally invariant data matrix. Additional argument can be added to shift asymptotic error peak :
# add a new parameter unif(x,asp,s), and write a = (s-asp)**2, b = (s+asp)**2. You're done !
    
def unif(x,asp):
	a = (1-asp)**2
	b = (1+asp)**2
	if asp<=1:
		
		if a<=np.sqrt(x)<=b:
			res = 1/(2*(b-a))*1/np.sqrt(x)
		else:
			res = 0

	if asp>1:
		
		if a<=np.sqrt(x)<=b:
			res = 1/(2*(b-a))*1/asp*1/np.sqrt(x)
		else:
			res = 0            
	return(res)      

def scale(asp): # scaling for delta spectral measure
	if asp<=1:
		res  = 1
	else :
		res = 1/asp
	return(res)

# Building orthogonally invariant random matrix with squared uniform eigenvalue distribution

def build_matrix(a,b,n,d,asp,select):
	
	V = Haar_sample(n)
	O = Haar_sample(d)
	if select == 1:
		eigen = np.random.uniform(a,b,min(n,d))
	if select == 0:
		eigen = np.ones(min(n,d))
	D_prev = np.diag(eigen)
	comp = np.zeros((max(n,d)-min(n,d),min(n,d)))
	
	if n>=d :
		
		D = np.concatenate((D_prev,comp),0)
	
	if n<d:
		
		D = np.concatenate((D_prev,np.transpose(comp)),1)
		
	F = V@D@np.transpose(O)
	return(F,D)
	
# Functions to run state evolution of VAMP for elastic-net problem, according to Theorem 2.
# One iteration of VAMP_SE_l1 for rotationally invariant matrix with suqared uniform spectrum

def VAMP_it_delta(gamma1,tau1,asp,rho,reg,delta0,my_eps):
	
	my_e = 1/(1+my_eps/gamma1)
	alpha1 = (1-rho)*my_e*special.erfc(reg/(gamma1*np.sqrt(2*tau1)))+rho*special.erfc(reg/(gamma1*np.sqrt(2*(1+tau1))))
	eta1 = gamma1/alpha1
	gamma2 = eta1-gamma1
	E1_inter = (1-rho)*my_e**2*(erfc(reg/gamma1/np.sqrt(2*tau1))*((reg/gamma1)**2+tau1)-np.exp(-(reg/gamma1)**2/(2*tau1))*np.sqrt(2*tau1/np.pi)*reg/gamma1)
	E1 = E1_inter+rho*integrate.quad(lambda p : gaussian(p,1)*SE_aux_l1(p,reg,tau1,gamma1,my_eps),-10,10)[0]
	
	tau2 = 1/((1-alpha1)**2)*(E1-alpha1**2*tau1)
	alpha2 = max(0,1-asp)+asp*scale(asp)*gamma2/(1+gamma2) 
	eta2 = gamma2/alpha2
	E2 = max(0,1-asp)*tau2+scale(asp)*asp*(delta0+tau2*gamma2**2)/(1+gamma2)**2
	# update
	gamma1 = eta2-gamma2
	tau1 = 1/((1-alpha2)**2)*(E2-alpha2**2*tau2)
	
	return(alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2)
	

# One damped iteration of VAMP_SE_l1 for rotationally invariant matrix with uniform spectrum of parameters a,b

def VAMP_it_delta_damped(alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2,damp_se,asp,rho,reg,delta0,my_eps):
	
	my_e = 1/(1+my_eps/gamma1)
	alpha1 = damp_se*((1-rho)*my_e*special.erfc(reg/(gamma1*np.sqrt(2*tau1)))+rho*special.erfc(reg/(gamma1*np.sqrt(2*(1+tau1)))))+(1-damp_se)*alpha1
	eta1 = gamma1/alpha1
	gamma2 = eta1-gamma1
	E1_inter = (1-rho)*my_e**2*(erfc(reg/gamma1/np.sqrt(2*tau1))*((reg/gamma1)**2+tau1)-np.exp(-(reg/gamma1)**2/(2*tau1))*np.sqrt(2*tau1/np.pi)*reg/gamma1)
	E1 = damp_se*(E1_inter+rho*integrate.quad(lambda p : gaussian(p,1)*SE_aux_l1(p,reg,tau1,gamma1,my_eps),-10,10)[0])+(1-damp_se)*E1
	
	tau2 = damp_se*1/((1-alpha1)**2)*(E1-alpha1**2*tau1)+(1-damp_se)*tau2
	alpha2 = damp_se*(max(0,1-asp)+scale(asp)*asp*gamma2/(1+gamma2))+(1-damp_se)*alpha2 
	eta2 = gamma2/alpha2
	E2 = damp_se*(max(0,1-asp)*tau2+scale(asp)*asp*(delta0+tau2*gamma2**2)/(1+gamma2)**2)+(1-damp_se)*E2
		
	# update
	gamma1 = eta2-gamma2
	tau1 = damp_se*1/((1-alpha2)**2)*(E2-alpha2**2*tau2)+(1-damp_se)*tau1
	
	return(alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2)


# Run the iterations for rotationally invariant matrix with uniform spectrum (row-orthogonal matrix)

def VAMP_SE_l1_delta(asp,rho,delta0,reg,niter,damp_se,my_eps):
	
	# Storing vectors
	
	tau1_vec = []
	tau2_vec = []
	gamma1_vec = []
	gamma2_vec = []
	E1_vec = []
	E2_vec = []
	alpha1_vec = []
	alpha2_vec = []
	eta1_vec = []
	eta2_vec = []
	
	# Initialization
	
	tau1 = np.random.rand()
	gamma1 = np.random.rand()
	
	alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2 = VAMP_it_delta(gamma1,tau1,asp,rho,reg,delta0,my_eps) # first iteration for possible damping
	
	tau1_vec.append(tau1)
	tau2_vec.append(tau2) 
	gamma1_vec.append(gamma1)
	gamma2_vec.append(gamma2)
	E1_vec.append(E1)
	E2_vec.append(E2)
	alpha1_vec.append(alpha1)
	alpha2_vec.append(alpha2)
	eta1_vec.append(eta1)
	eta2_vec.append(eta2)
	
	for i in range(niter-1):
		
		alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2 = VAMP_it_delta_damped(alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2,damp_se,asp,rho,reg,delta0,my_eps)
		
		#store for plot
		tau1_vec.append(tau1)
		tau2_vec.append(tau2) 
		gamma1_vec.append(gamma1)
		gamma2_vec.append(gamma2)
		E1_vec.append(E1)
		E2_vec.append(E2)
		alpha1_vec.append(alpha1)
		alpha2_vec.append(alpha2)
		eta1_vec.append(eta1)
		eta2_vec.append(eta2)
		
	return(alpha1_vec,alpha2_vec,eta1_vec,eta2_vec,gamma1_vec,gamma2_vec,E1_vec,E2_vec,tau1_vec,tau2_vec)


	# One iteration of VAMP_SE_l1 for rotationally invariant matrix with uniform spectrum of parameters a,b

def VAMP_it_unif(gamma1,tau1,asp,rho,reg,delta0,my_eps,f):
	
	my_e = 1/(1+my_eps/gamma1)
	alpha1 = (1-rho)*my_e*special.erfc(reg/(gamma1*np.sqrt(2*tau1)))+rho*special.erfc(reg/(gamma1*np.sqrt(2*(1+tau1))))
	eta1 = gamma1/alpha1
	gamma2 = eta1-gamma1
	E1_inter = (1-rho)*my_e**2*(erfc(reg/gamma1/np.sqrt(2*tau1))*((reg/gamma1)**2+tau1)-np.exp(-(reg/gamma1)**2/(2*tau1))*np.sqrt(2*tau1/np.pi)*reg/gamma1)
	E1 = E1_inter+rho*integrate.quad(lambda p : gaussian(p,1)*SE_aux_l1(p,reg,tau1,gamma1,my_eps),-10,10)[0]
	
	tau2 = 1/((1-alpha1)**2)*(E1-alpha1**2*tau1)
	alpha2 = max(0,1-asp)+integrate.quad(lambda p: asp*f(p)*gamma2/(p+gamma2),0,np.inf)[0] 
	eta2 = gamma2/alpha2
	E2 = max(0,1-asp)*tau2+integrate.quad(lambda p : asp*f(p)*(delta0*p+tau2*gamma2**2)/(p+gamma2)**2,0,np.inf)[0]
	# update
	gamma1 = eta2-gamma2
	tau1 = 1/((1-alpha2)**2)*(E2-alpha2**2*tau2)
	
	return(alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2)
	

# One damped iteration of VAMP_SE_l1 for rotationally invariant matrix with uniform spectrum of parameters a,b

def VAMP_it_unif_damped(alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2,damp_se,asp,rho,reg,delta0,my_eps,f):
	
	my_e = 1/(1+my_eps/gamma1)
	alpha1 = damp_se*my_e*((1-rho)*special.erfc(reg/(gamma1*np.sqrt(2*tau1)))+rho*special.erfc(reg/(gamma1*np.sqrt(2*(1+tau1)))))+(1-damp_se)*alpha1
	eta1 = gamma1/alpha1
	gamma2 = eta1-gamma1
	E1_inter = (1-rho)*my_e**2*(erfc(reg/gamma1/np.sqrt(2*tau1))*((reg/gamma1)**2+tau1)-np.exp(-(reg/gamma1)**2/(2*tau1))*np.sqrt(2*tau1/np.pi)*reg/gamma1)
	E1 = damp_se*(E1_inter+rho*integrate.quad(lambda p : gaussian(p,1)*SE_aux_l1(p,reg,tau1,gamma1,my_eps),-10,10)[0])+(1-damp_se)*E1
	
	tau2 = damp_se*1/((1-alpha1)**2)*(E1-alpha1**2*tau1)+(1-damp_se)*tau2
	alpha2 = damp_se*(max(0,1-asp)+integrate.quad(lambda p: asp*f(p)*gamma2/(p+gamma2),0,np.inf)[0])+(1-damp_se)*alpha2
	eta2 = gamma2/alpha2
	E2 = damp_se*(max(0,1-asp)*tau2+integrate.quad(lambda p : asp*f(p)*(delta0*p+tau2*gamma2**2)/(p+gamma2)**2,0,np.inf)[0])+(1-damp_se)*E2
		
	# update
	gamma1 = eta2-gamma2
	tau1 = damp_se*1/((1-alpha2)**2)*(E2-alpha2**2*tau2)+(1-damp_se)*tau1
	
	return(alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2)


# Run the iterations for rotationally invariant matrix with uniform spectrum

def VAMP_SE_l1_unif(asp,rho,delta0,reg,niter,damp_se,my_eps,select):
	
	# selecting a Marcenko-Pastur density or a specifically prescribed one (here the squared uniform one)
	if select == 0:
		f = lambda p : MP(p,asp)
	if select == 1:
		f = lambda p : unif(p,asp)
	# Storing vectors for convergence control/plots. take the last value for fixed point
	
	tau1_vec = []
	tau2_vec = []
	gamma1_vec = []
	gamma2_vec = []
	E1_vec = []
	E2_vec = []
	alpha1_vec = []
	alpha2_vec = []
	eta1_vec = []
	eta2_vec = []
	
	# Initialization
	
	tau1 = np.random.rand()
	gamma1 = np.random.rand()
	
	alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2 = VAMP_it_unif(gamma1,tau1,asp,rho,reg,delta0,my_eps,f) # first iteration for possible damping
	
	tau1_vec.append(tau1)
	tau2_vec.append(tau2) 
	gamma1_vec.append(gamma1)
	gamma2_vec.append(gamma2)
	E1_vec.append(E1)
	E2_vec.append(E2)
	alpha1_vec.append(alpha1)
	alpha2_vec.append(alpha2)
	eta1_vec.append(eta1)
	eta2_vec.append(eta2)
	
	for i in range(niter-1):
		
		alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2 = VAMP_it_unif_damped(alpha1,alpha2,eta1,eta2,gamma1,gamma2,E1,E2,tau1,tau2,damp_se,asp,rho,reg,delta0,my_eps,f)
		
		#store for plot
		tau1_vec.append(tau1)
		tau2_vec.append(tau2) 
		gamma1_vec.append(gamma1)
		gamma2_vec.append(gamma2)
		E1_vec.append(E1)
		E2_vec.append(E2)
		alpha1_vec.append(alpha1)
		alpha2_vec.append(alpha2)
		eta1_vec.append(eta1)
		eta2_vec.append(eta2)
		
	return(alpha1_vec,alpha2_vec,eta1_vec,eta2_vec,gamma1_vec,gamma2_vec,E1_vec,E2_vec,tau1_vec,tau2_vec)

# Oracle_VAMP as described in Section 3, needs to be ran after solving the SE equations.

def DR_VAMP(A1,A2,V,tau2,F,niter2,damp_dr,x0,y,reg,my_eps): # returns the solution to the problem, the iterations on B2 and a convergence control vecot on B2 iterates
	
	
	d = np.shape(F)[1]
	my_e = 1/(1+my_eps/A1)
	# initialization
	B2 = np.random.rand(d)
	# Storing values
	B2_conv = np.zeros(niter2)
	M = np.linalg.inv(np.transpose(F)@F+A2*np.identity(d))  # can be replaced with SVD-type update for faster algo, see Rangan et al. VAMP
	B2_mat = np.zeros((d,niter2))

	# Iterations
	
	for i in range(niter2) :
	
		#print(i)
	
		B2_mat[:,i] = B2
		w = 1/(A1*V)*(M@(np.transpose(F)@y+B2)-V*B2)
		B2_new = damp_dr*1/V*(my_e*soft_thresh(w,reg/A1)-A1*V*w)+(1-damp_dr)*B2
		B2_conv[i] = 1/d*np.linalg.norm(B2_new-B2,2)
		B2 = B2_new
	
	
	lip = np.zeros(niter2-1)

	for i in range(niter2-1):
		lip[i] = B2_conv[i+1]/B2_conv[i]
	
	conv = np.zeros(niter2)

	for i in range(niter2):
		conv[i] = np.linalg.norm(B2_mat[:,i]-B2_mat[:,-1])
		
	sol = M@(np.transpose(F)@y+B2)     # problem solution   
	return(sol,B2_mat,B2_conv)
