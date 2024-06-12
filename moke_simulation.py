from OpticalStack import OpticalStack
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def monolayer_test(n, wavel=1550):
    '''
    Test the reflection simulation against theory for a single layer with refractive index n suspended in the air,
    for various thicknesses.
    '''
    k = 2*np.pi / wavel
    d_vals = np.linspace(start=0, stop=wavel/2, num=100)
    R_vals_theory = np.abs((1/n - n) * np.sin(n*k*d_vals) / ((n+1/n)*np.sin(n*k*d_vals) + 2j * np.cos(n*k*d_vals)))**2
    R_vals = []
    
    for d in d_vals:
        os = OpticalStack(wavelength=wavel)
        os.insert_layer(n=n, d=d)
        os.insert_layer(n=1, d=0)
        v = os.compute_v()
        R_vals.append(np.abs(v[1])**2)

    plt.plot(d_vals, R_vals, label="Code")
    plt.plot(d_vals, R_vals_theory, label="Theory")
    plt.plot(d_vals, R_vals - R_vals_theory, label="Deviation")
    plt.xlabel("d (nm)")
    plt.ylabel("Reflectance")
    plt.title("Reflectance vs thin film thickness with n = {}".format(n))
    plt.legend()
    plt.show()


def VB_test(VB, d, wavel=1550):
    '''
    Test the Faraday rotation due to Faraday effect under magnetic field.
    VB is in μrad/mm = 1e-12 rad/nm. d is in nm.
    '''
    theta_F = 2 * VB * 1e-12 * d        # The Faraday angle due to two propagations through the Faraday layer,
                                        # in rad
    k = 2 * np.pi / wavel
    dyna_phase = 2 * k * d          # the dynamical phase due to wave propagation in n=1
    os_l = OpticalStack(wavelength=wavel, initial_state=0)
    os_r = OpticalStack(wavelength=wavel, initial_state=1)
    os_l.insert_layer(n=1, d=d, VB=VB)
    os_r.insert_layer(n=1, d=d, VB=VB)
    os_l.insert_layer(n=1e7, d=0)      # substrate with total reflection
    os_r.insert_layer(n=1e7, d=0)
    vector_l, vector_r = os_l.compute_v()[:2], os_r.compute_v()[:2]
    print("For VB={}μrad/mm, d={}mm,".format(VB, d/1e+6))
    print("LCP (1, i): ", vector_l)
    print("where the second term should be {}".format(-1 * np.exp(1j * (dyna_phase + theta_F))))
    print("RCP (1, -i): ", vector_r)
    print("where the second term should be {}".format(-1 * np.exp(1j * (dyna_phase - theta_F))))
    print("and the resultant Kerr angle is {:.2f} μrad.".format(np.angle(vector_l[1] / vector_r[1]) / 2 * 1e+6))


def compute_twisted_stack(n_bscco, n_SiO2, n_Si, d1, d2, d_SiO2, sigma_xx, sigma_xy):
    '''
    Computes the phase accumulated for LCP and RCP light upon reflection off of a
    air-bscco1-bscco2(twisted)-SiO2-Si stack.
    sigma_xx refers to the optical conductivity (in unit of Ω^-1*cm^-1) of BSCCO, and sigma_xy refers
    to the optical Hall conductance (in unit of e^2/hbar = 1/137) of the twisted interface.
    '''
    # 1e^2/hbar = 2.4316e-4 Ω^-1
    def build_stack(stack):
        '''
        Builds the air-bscco1-bscco2-SiO2-Si optical stack.
        '''
        assert isinstance(stack, OpticalStack)
        stack.insert_layer(n=n_bscco, d=d1, sigma_xx=sigma_xx * d1 * 1e-7 / 2.4316e-4)
        stack.insert_layer(n=1, d=3.2, sigma_xx=sigma_xx * 3.2 * 1e-7 / 2.4316e-4, sigma_xy=sigma_xy)       # ignore the dielectric
                                                                                                                  # env of interface
        stack.insert_layer(n=n_bscco, d=d2, sigma_xx=sigma_xx * d2 * 1e-7 / 2.4316e-4)
        stack.insert_layer(n=n_SiO2, d=d_SiO2)
        stack.insert_layer(n=n_Si, d=0)
        
        return stack
    
    lcp_stack = build_stack(OpticalStack(wavelength=1550, initial_state=0))
    rcp_stack = build_stack(OpticalStack(wavelength=1550, initial_state=1))
    lcp_vector = lcp_stack.compute_v()[:2]
    rcp_vector = rcp_stack.compute_v()[:2]
    return lcp_vector, rcp_vector


def twisted_stack_test(d1, d2, d_SiO2, interface_sigma_xy):
    '''
    Simulate the reflection upon a twisted BSCCO stack with two BSCCO films of thicknesses d1, d2 (in nm).
    The sigma_xy of the twisted interface is in units of e^2/hbar.
    Returns the measured Kerr angle (half the phase difference) in μrad and reflectance of RCP (~LCP).
    '''
    n_bscco = np.sqrt(-0.846 + 2.212j)
    n_SiO2 = 1.45
    n_Si = 3.45
    bscco_sigma_xx = 700 * (-1j) ** (1.447 - 2)     # longitudinal conductivity of bscco in units of Ω^-1cm^-1
    l_vector, r_vector = compute_twisted_stack(n_bscco=n_bscco, n_SiO2=n_SiO2, n_Si=n_Si, d1=d1, d2=d2, 
                                               d_SiO2=d_SiO2, sigma_xx=bscco_sigma_xx, sigma_xy=interface_sigma_xy)
    r = l_vector[1] / r_vector[1]
    reflectance_rcp = np.abs(r_vector[1]) ** 2
    # print("The magnitude of r (LCP/RCP) is {:.4f} and the phase of r is {:.4f} μrad, reading θ_K = {:.4f} μrad.".format(np.abs(r), np.angle(r) * 1e6, np.angle(r) * 1e6 / 2))
    sigma_xx, sigma_xy = bscco_sigma_xx * 3.2e-7 / 2.4316e-4 / 137, interface_sigma_xy / 137
    kerr_angle_theory = np.real(np.arctan(sigma_xy / (sigma_xx + 2*np.pi*(sigma_xx**2 + sigma_xy**2))))
    # print("The theoretical Kerr angle is {:.4f} μrad.".format(kerr_angle_theory * 1e6))
    # print("Reflectance of RCP is {:.4f}.".format(reflectance_rcp))

    return np.angle(r) * 1e6 / 2, reflectance_rcp

def main():
    '''
    The main function to be run.
    '''
    # monolayer_test(1+0.01j)
    # VB_test(VB=10, d=2e+6)
    d1_vals = np.linspace(start=2, stop=50, num=100)
    kerr_angles = []
    reflectances = []
    for d1 in d1_vals:
        ang, ref = twisted_stack_test(d1=d1, d2=30, d_SiO2=280, interface_sigma_xy=1e-4)
        kerr_angles.append(ang)
        reflectances.append(ref)
    # twisted_stack_test(d1=10, d2=10, d_SiO2=10, interface_sigma_xy=1e-4)

    plt.xlabel("Top layer thickness (nm)")
    plt.plot(d1_vals, kerr_angles, label="Kerr angle")
    plt.plot(d1_vals, reflectances, label="Reflectance")
    plt.title("d2=30nm, d_SiO2=280nm, σ_xy=1e-4 / 137")
    plt.legend()
    plt.show()

    return 0


if __name__ == '__main__':
    main()