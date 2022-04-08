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
    def __init__(self, k=np.array([1., 0., 0.]), w=1., E0=1e-5, b=0.0, dt=0.01, T=10, Γ=np.inf, t0=0.):
        π = np.pi
        c = 137.035999206

        self.k  = k   ## unit wave-vector
        self.t0 = t0  ## time delay
        self.w  = w   ## frequency
        self.Γ  = Γ   ## FWHM
        self.E0 = E0  ## E-field amplitude
        self.b  = b   ## chirp parameter

        self.dt = dt  ## time-step
        self.T  = T   ## duration
        
        self.D      = None  ## density-matrix
        self.E_stx  = None  ## real-time E-field
        self.E_sωx  = None  ## frequency E-field
        self.A_stx  = None  ## real-time A-field
        self.A_sωx  = None  ## frequency A-field
        self.B_stx  = None  ## real-time B-field
        self.B_sωx  = None  ## frequency B-field      
        self.F_stuv = None  ## real-time F-field
        self.F_sωuv = None  ## frequency F-field
    
    def fft(self, X):
        return np.sqrt(2*π)/(len(self.t) * self.dω) * np.fft.fftshift( np.fft.fft((X).real , axis=1) )

    def gaussianpulse(self):
        return self.E0 * np.exp( (1j*self.b - 4*np.log(2)/(self.Γ**2) ) * (self.t - self.t0)**2 - 1j * self.w * (self.t - self.t0) )

    def get_E(self, D=None):
        if D is None and self.D is None:
            self.D = np.array([[ 1., 0.], [0., 0.]])

        self.t  = np.arange(0., self.T, self.dt)
        self.dω = 2*π / (self.T-self.dt)
        self.Ω  = π / self.dt
        self.ω  = np.arange(-self.Ω, self.Ω+self.dω/2, self.dω)
        self.k  = self.k/np.linalg.norm(self.k)

        basis      = rotation_matrix(90) @ self.k
        subbasis   = np.where( np.sum( np.abs( basis - self.k[None, :] ), axis=1) != 0)[0]
        self.e_sx  = np.array([basis[subbasis[0]], basis[subbasis[1]]]) ## unit ortho. basis vectors
        self.E_stx = self.gaussianpulse()[None, :,None] * (self.D @ self.e_sx)[:,None,:]
        self.E_sωx = self.fft(self.E_stx)
        return np.sum(self.E_stx, axis=0).real, np.sum(self.E_sωx, axis=0).real

    def get_A(self):
        if self.E_sωx is None:
            self.get_E()
        self.A_stx = c/(1j * self.w) * self.E_stx
        self.A_sωx = c/(1j * self.w) * self.E_sωx
        return np.sum(self.A_stx, axis=0).real, np.sum(self.A_sωx, axis=0).real
    
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
        return np.sum(self.F_stuv, axis=0).real, np.sum(self.F_sωuv, axis=0).real

