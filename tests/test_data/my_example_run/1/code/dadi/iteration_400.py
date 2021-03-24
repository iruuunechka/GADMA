import dadi
import numpy as np

def model_func(params, ns, pts):
	t1, nu11, s1, t2, nu21, nu22, m2_12, m2_21 = params
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	phi = dadi.Integration.one_pop(phi, xx, T=t1, nu=nu11)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	phi = dadi.Integration.two_pops(phi, xx, T=t2, nu1=nu21, nu2=nu22, m12=m2_12, m21=m2_21)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

data = dadi.Spectrum.from_file('/home/katenos/Workspace/popgen/temp/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['YRI', 'CEU']
pts = [20, 30, 40]
ns = data.sample_sizes

p0 = [1.0367551793885217, 1.1455588979827178, 0.4681877673127339, 0.2822630912846048, 1.8891613166890753, 0.48594935040547094, 1.6282602829124666, 0.9962509098456426]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
theta0 = 0.37976
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))
