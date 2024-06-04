from OpticalStack import OpticalStack
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def main():
    '''
    The main function to be run.
    '''
    wavel = 1550
    k = 2*np.pi / wavel
    n = 10
    d_vals = np.linspace(start=0, stop=wavel/2, num=100)
    R_vals_theory = np.abs((1/n - n) * np.sin(n*k*d_vals) / ((n+1/n)*np.sin(n*k*d_vals) + 2j * np.cos(n*k*d_vals)))**2
    R_vals = []
    
    for d in d_vals:
        os = OpticalStack(wavelength=wavel)
        os.insert_layer(n=n, d=d)
        os.insert_layer(n=1, d=0)
        os.generate_M_normal()
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
    
    return 0


if __name__ == '__main__':
    main()