import dadi

def nu(x):
    print(x)

def model_func(params, ns, pts):
    '''
    Some model
    '''
    nu(",,first line does not contain p_ids")
    nuB, nuF, TB, TF = params
    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, TB, nuB)
    phi = dadi.Integration.one_pop(phi, xx, TF, nuF)

    fs = dadi.Spectrum.from_phi(phi, ns, (xx,))
    return fs
