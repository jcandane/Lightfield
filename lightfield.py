import numpy as np 

π = np.pi 
c = 137.035999206

def rotation_matrix(θ):
    """
    GIVEN: θ (angle in degrees)
    GET:   R (the 3 rotation matrices)
    """

    θ *= np.pi/180 ## degrees to radians
    R_yz = np.array([[1.,        0.,         0.], 
                     [0., np.cos(θ), -np.sin(θ)], 
                     [0., np.sin(θ), np.cos(θ)]])
    R_xz = np.array([[ np.cos(θ), 0., np.sin(θ)], 
                     [        0., 1.,        0.], 
                     [-np.sin(θ), 0., np.cos(θ)]])
    R_xy = np.array([[np.cos(θ), -np.sin(θ), 0.], 
                     [np.sin(θ),  np.cos(θ), 0.], 
                     [0.       ,         0., 1.]])

    return np.asarray([R_yz, R_xz, R_xy])


class lightfield(object):
    ''' A class of Spatially Homogeneous Light-Field Gaussian Pulses '''
    def __init__(self, k=np.array([1., 0., 0.]), w=1., E0=1., b=0.0, dt=0.01, T=10, Γ=np.inf, t0=0., ϕ=0.):
        π = np.pi
        c = 137.035999206

        self.k  = k
        self.t0 = t0
        self.w  = w
        self.Γ  = Γ
        self.b  = b

        self.dt = dt
        self.T  = T
        
        self.ϕu = ϕ
        self.ϕd = ϕ
        self.Eu = E0
        self.Ed = E0

        self.D      = None
        self.E_stx  = None
        self.E_sωx  = None
        self.A_stx  = None
        self.A_sωx  = None
        self.B_stx  = None
        self.B_sωx  = None        
        self.F_stuv = None
        self.F_sωuv = None
    
    def fft(self, X):
        return np.sqrt(2*π)/(len(self.t) * self.dω) * np.fft.fftshift( np.fft.fft((X).real , axis=1) )

    def gaussianpulse(self):
        return np.exp( (1j*self.b - 4*np.log(2)/(self.Γ**2) ) * (self.t - self.t0)**2 - 1j * self.w * (self.t - self.t0) )

    def get_E(self, D=None):
        if D is None and self.D is None:
            self.D = np.array([[self.Eu * np.exp(1j*self.ϕu), 0.], [0., self.Ed * np.exp(1j*self.ϕd)]])

        self.t  = np.arange(0., self.T, self.dt)
        self.dω = 2*π / (self.T-self.dt)
        self.Ω  = π / self.dt
        self.ω  = np.arange(-self.Ω, self.Ω+self.dω/2, self.dω)

        basis      = rotation_matrix(90) @ (k/np.linalg.norm(k))
        subbasis   = np.where( np.sum( np.abs( basis - k[None, :] ), axis=1) != 0)[0]
        self.e_sx  = np.array([basis[subbasis[0]], basis[subbasis[1]]]) ## unit ortho. basis vectors
        self.E_stx = self.gaussianpulse()[None, :,None] * (self.D @ self.e_sx)[:,None,:]
        self.E_sωx = self.fft(self.E_stx)
        return self.E_stx, self.E_sωx

    def get_A(self):
        if self.E_sωx is None:
            self.get_E()
        self.A_stx = c/(1j * self.w) * self.E_stx
        self.A_sωx = c/(1j * self.w) * self.E_sωx
        return self.A_stx, self.A_sωx
    
    def get_F(self):
        if self.E_sωx is None:
            self.get_E()
        flip  = np.array([[0.,1.],[1.,0.]])
        self.B_stx = self.gaussianpulse()[None, :,None] * (self.D @ (flip @ self.e_sx))[:,None,:] / c
        self.B_sωx = self.fft(self.B_stx)

        F_stuv     = np.zeros( (2,len(self.t),4,4) , dtype=np.complex128)
        F_stuv[:,:,0,1:4] = self.E_stx
        F_stuv[:,:,1,2] = -self.B_stx[:,:,2]
        F_stuv[:,:,1,3] =  self.B_stx[:,:,1]
        F_stuv[:,:,2,3] =  self.B_stx[:,:,0]
        F_stuv         -= F_stuv.swapaxes(2,3)
        self.F_stuv = F_stuv

        F_sωuv     = np.zeros( (2,len(self.t),4,4) , dtype=np.complex128)
        F_sωuv[:,:,0,1:4] = self.E_sωx
        F_sωuv[:,:,1,2] = -self.B_sωx[:,:,2]
        F_sωuv[:,:,1,3] =  self.B_sωx[:,:,1]
        F_sωuv[:,:,2,3] =  self.B_sωx[:,:,0]
        F_sωuv         -= F_sωuv.swapaxes(2,3)
        self.F_sωuv = F_sωuv
        return self.F_stuv, self.F_sωuv

