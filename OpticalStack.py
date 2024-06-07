import numpy as np
from numpy.linalg import inv

class OpticalStack:
    '''
    The OpticalStack class, where each instance represents an optical stack that contains
    a semi-definite air layer at the top spanning z < 0, optical layers with finite thicknesses below the vacuum
    layer, and a semi-definite substrate layer extending to z=∞ at the bottom.
    '''
    def __init__(self, wavelength, incident_angle=0):
        '''
        Initializes the OpticalStack, given the wavelength in nm. Normal incidence by default.
        '''
        self.n_list = np.array([1], dtype=complex)       # list of complex refractive indices n, from top to bottom
                                # the vacuum layer with n=1 is already included

        self.d_list = [0]       # list of layer thicknesses. The vacuum layer and substrate layer
                                # have thickness = 0 by default.

        self.VBs = [0]  # list of Verdet constant times magnetic field, VB, from top to bottom layer
                                  # in units of μrad/mm

        self.conductivities = np.array([0], dtype=complex)  # list of longitudinal and Hall conductivities of
                                                            # each layer, in units of e^2/hbar (1/137)

        self.wavelength = wavelength        # wavelength in nm
        self.incident_angle = incident_angle        # incident angle, in deg
    
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
        self.VBs.append(VB)
        self.conductivities = np.append(self.conductivities, np.array([sigma_xx, sigma_xy], dtype=complex))

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
        for i in range(N + 1):
            r_i_i1 = (self.n_list[i] - self.n_list[i+1]) / (self.n_list[i] + self.n_list[i+1])      # r_i_i+1
            r_i1_i = -1 * r_i_i1                                                      # r_i+1_i
            t_i_i1 = 2 * self.n_list[i] / (self.n_list[i] + self.n_list[i+1])         # t_i_i+1
            t_i1_i = 2 * self.n_list[i+1] / (self.n_list[i] + self.n_list[i+1])       # t_i+1_i
            
            
            # The three lines below describe the first equation
            self.M[curr_row][2*i] = r_i_i1
            self.M[curr_row][2*i+1] = -1
            self.M[curr_row][2*i+3] = t_i1_i * np.exp(1j * self.n_list[i+1] * k * self.d_list[i+1])
            curr_row += 1
            # The three lines below describe the second equation
            self.M[curr_row][2*i] = t_i_i1 * np.exp(1j * self.n_list[i+1] * k * self.d_list[i+1])
            self.M[curr_row][2*i+2] = -1
            self.M[curr_row][2*i+3] = r_i1_i * np.exp(2j * self.n_list[i+1] * k * self.d_list[i+1])
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
