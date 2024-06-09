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
    os_l.insert_layer(n=1e10, d=0)      # substrate with total reflection
    os_r.insert_layer(n=1e10, d=0)
    vector_l, vector_r = os_l.compute_v()[:2], os_r.compute_v()[:2]
    print("For VB={}μrad/mm, d={}nm,".format(VB, d))
    print("LCP (1, i): ", vector_l)
    print("where the second term should be {}".format(-1 * np.exp(1j * (dyna_phase + theta_F))))
    print("RCP (1, -i): ", vector_r)
    print("where the second term should be {}".format(-1 * np.exp(1j * (dyna_phase - theta_F))))

    return

def main():
    '''
    The main function to be run.
    '''
    # monolayer_test(1+0.01j)
    VB_test(VB=1000, d=10e+6)
    return 0


if __name__ == '__main__':
    main()