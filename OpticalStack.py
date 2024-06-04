import numpy

class OpticalStack:
    '''
    The OpticalStack class, where each instance represents an optical stack that contains
    a semi-definite air layer at the top spanning z < 0, optical layers with finite thicknesses below the air
    layer, and a semi-definite substrate layer extending to z=∞ at the bottom.
    '''
    def __init__(self, wavelength, incident_angle=0):
        '''
        Initializes the OpticalStack, given the wavelength in nm. Normal incidence by default.
        '''
        self.n_list = [1]       # list of complex refractive indices n, from top to bottom
                                # the air layer with n=1 is already included

        self.t_list = [0]       # list of layer thicknesses. The air layer and substrate layer
                                # have thickness = 0 by default.

        self.angles = [[0, 0]]  # list of [Faraday, Kerr] angles (in μrad) associated with each layer, from top to
                                # bottom

        self.wavelength = wavelength        # wavelength in nm
        self.incident_angle = incident_angle        # incident angle, in deg
    
        return 0

    def insert_layer(self, n, t, phi_F=0, phi_K=0):
        '''
        Inserts a layer with complex refractive index n, thickness t (in nm), and optional Faraday angle phi_F (in 
        μrad) and Kerr angle phi_K (in μrad).
        '''
        self.n_list.append(n)
        self.t_list.append(t)
        self.angles.append([phi_F, phi_K])
        return 0
    
    
    
