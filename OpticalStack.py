import numpy as np
from numpy.linalg import inv

class OpticalStack:
    '''
    The OpticalStack class, where each instance represents an optical stack that contains
    a semi-definite air layer at the top spanning z < 0, optical layers with finite thicknesses below the vacuum
    layer, and a semi-definite substrate layer extending to z=∞ at the bottom.
    '''
    def __init__(self, wavelength, initial_state=0):
        '''
        Initializes the OpticalStack, given the wavelength in nm.
        Set initial_state=0 (default) for LCP (1, 0) and initial_state=1 for RCP (0, 1).
        '''
        self.n_list = np.array([1], dtype=complex)       # list of complex refractive indices n, from top to bottom
                                # the vacuum layer with n=1 is already included

        self.d_list = [0]       # list of layer thicknesses. The vacuum layer and substrate layer
                                # have thickness = 0 by default.

        self.VBs = [0]  # list of Verdet constant times magnetic field, VB, from top to bottom layer
                                  # in units of μrad/mm = 1e-12rad/nm
        
        self.flatnesses = [1]       # list of layer flatnesses (default to be 1) to account for
                                    # imperfect reflections. Normally only needed for the substrate layer

        self.sigmas = np.array([0, 0], dtype=complex)  # list of longitudinal and Hall conductivities of
                                                            # each 2D layer, in units of e^2/hbar (1/137)

        self.wavelength = wavelength        # wavelength in nm
        self.initial_state = initial_state        # initial state. 0 for (1, i) and 1 for (1, -i).
    
        return

    def insert_layer(self, n, d, VB=0, sigma_xx=0+0j, sigma_xy=0+0j):
        '''
        Inserts a layer with complex refractive index n, thickness d (in nm), and optional Verdet constant
        times magnetic field (VB, in units of μrad/mm) and conductivities (in units of e^2/hbar). 
        IMPORTANT: the substrate layer has to be manually inserted with thickness = 0 at last. 
        Complex numbers are entered as a+bj.
        '''
        self.n_list = np.append(self.n_list, np.array([n]))
        self.d_list.append(d)
        self.VBs.append(VB * 1e-12)
        self.sigmas = np.vstack((self.sigmas, np.array([sigma_xx, sigma_xy], dtype=complex) / 137))

        return 0
    


    def generate_M_normal(self):
        '''
        Generate the M matrix to solve for the field vectors, assuming normal incidence. The convention is to 
        evaluate fields at the bottom of each layer. The transmitted field in the substrate layer doesn't follow 
        this convention, but we don't care about it anyway.
        Returns the generated M matrix.
        '''
        N = len(self.d_list) - 2        # the total number of layers with finite thickness
        self.M = np.zeros((2*N + 4, 2*N + 4), dtype=complex)
        self.w = np.zeros(2*N + 4, dtype=complex)
        self.w[0] = 1                       # w = (1, 0, 0, ..., 0)
                                            # v = (E_0^+, E_0^-, E_1^+, E_1^-, ..., E_{N+1}^+, E_{N+1}^-)
                                            # such that M*v = w is the equation to be solved
        self.M[0][0] = 1     # encodes E_0^+ = 1 in M*v = w
        curr_row = 1        # pointer for the current row of M being processed
        k = 2 * np.pi / self.wavelength       # wavevector in vacuum, in nm^-1
        polarization = (-1) ** self.initial_state       # +1 for (1, i), -1 for (1, -i)
        for i in range(N + 1):
            n_i, n_i1 = self.n_list[i], self.n_list[i+1]
            sigmas_i, sigmas_i1 = self.sigmas[i], self.sigmas[i+1]
            VB_i1 = self.VBs[i+1]       # VB of the (i+1)th layer, 

            # to compute reflection coefficients r_i_i1 and r_i1_i
            # r_i_i1 = (self.n_list[i] - self.n_list[i+1]) / (self.n_list[i] + self.n_list[i+1])      # r_i_i+1
            # r_i1_i = -1 * r_i_i1                                                      # r_i+1_i
            r_i_i1_xx = 1.0 / ((n_i + n_i1 + 4*np.pi*sigmas_i1[0])**2 + (4*np.pi*sigmas_i1[1])**2)
            r_i_i1_xy = r_i_i1_xx * (-8*np.pi*n_i*sigmas_i1[1])
            r_i_i1_xx *= n_i**2 - (n_i1 + 4*np.pi*sigmas_i1[0])**2 - (4*np.pi*sigmas_i1[1])**2
            r_i_i1 = r_i_i1_xx + polarization * 1j * r_i_i1_xy      # r_i_i+1 = r_xx +- ir_xy

            r_i1_i_xx = 1.0 / ((n_i1 + n_i + 4*np.pi*sigmas_i[0])**2 + (4*np.pi*sigmas_i[1])**2)
            r_i1_i_xy = r_i1_i_xx * (-8*np.pi*n_i1*sigmas_i[1])
            r_i1_i_xx *= n_i1**2 - (n_i + 4*np.pi*sigmas_i[0])**2 - (4*np.pi*sigmas_i[1])**2
            r_i1_i = r_i1_i_xx + polarization * 1j * r_i1_i_xy      # r_i+1_i = r_xx +- ir_xy


            # to compute transmission coefficients t_i_i1 and t_i1_i
            # t_i_i1 = 2 * self.n_list[i] / (self.n_list[i] + self.n_list[i+1])         # t_i_i+1
            # t_i1_i = 2 * self.n_list[i+1] / (self.n_list[i] + self.n_list[i+1])       # t_i+1_i
            t_i_i1_xx = 1.0 / ((n_i + n_i1 + 4*np.pi*sigmas_i1[0])**2 + (4*np.pi*sigmas_i1[1])**2)
            t_i_i1_xy = r_i_i1_xy
            t_i_i1_xx *= 2 * n_i * (n_i + n_i1 + 4*np.pi*sigmas_i1[0])
            t_i_i1 = t_i_i1_xx + polarization * 1j * t_i_i1_xy

            t_i1_i_xx = 1.0 / ((n_i1 + n_i + 4*np.pi*sigmas_i[0])**2 + (4*np.pi*sigmas_i[1])**2)
            t_i1_i_xy = r_i1_i_xy
            t_i1_i_xx *= 2 * n_i1 * (n_i1 + n_i + 4*np.pi*sigmas_i[0])
            t_i1_i = t_i1_i_xx + polarization * 1j * t_i1_i_xy

            
            # The three lines below describe the first equation
            self.M[curr_row][2*i] = r_i_i1
            self.M[curr_row][2*i+1] = -1
            self.M[curr_row][2*i+3] = t_i1_i * np.exp(1j * (self.n_list[i+1] * k + polarization * VB_i1) * self.d_list[i+1])
            curr_row += 1
            # The three lines below describe the second equation
            self.M[curr_row][2*i] = t_i_i1 * np.exp(1j * (self.n_list[i+1] * k + polarization * VB_i1) * self.d_list[i+1])
            self.M[curr_row][2*i+2] = -1
            self.M[curr_row][2*i+3] = r_i1_i * np.exp(2j * (self.n_list[i+1] * k + polarization * VB_i1) * self.d_list[i+1])
            curr_row += 1
        self.M[curr_row][-1] = 1    # encodes E_{N+1}^- = 0 in M*v = w

        return self.M
    
    def compute_v(self):
        '''
        Computes the field vector v = M^-1 * w. The total reflection coefficient is then v[1] / v[0].
        '''
        self.generate_M_normal()
        self.v = np.matmul(inv(self.M), self.w)

        return self.v
