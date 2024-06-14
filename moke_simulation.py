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

def compute_quartz_gold_stack(VB, d_SiO2=2, wavel=1550):
    '''
    Computes the Kerr angle and reflectance upon a quartz-gold stack under magnetic field, given VB in μrad/mm.
    Returns (Kerr angle, reflectance).
    '''
    n_Au = 0.524 + 10.742j
    n_SiO2 = 1.42
    os_l = OpticalStack(wavelength=wavel, initial_state=0)
    os_r = OpticalStack(wavelength=wavel, initial_state=1)
    os_l.insert_layer(n=n_SiO2, d=d_SiO2, VB=VB)
    os_r.insert_layer(n=n_SiO2, d=d_SiO2, VB=VB)
    os_l.insert_layer(n=n_Au, d=0)
    os_r.insert_layer(n=n_Au, d=0)
    r_l, r_r = os_l.compute_v()[1], os_r.compute_v()[1]
    r = r_l / r_r
    # print("Amplitude ratio is {:.3f}".format(np.abs(r)))

    return np.angle(r) * 1e6 / 2, np.abs(r_l)**2

def plot_field_dependence():
    '''
    Plots the Kerr angles and reflectances vs magnetic field for quartz-gold stack.
    '''
    field_vals = np.linspace(start=10, stop=100, num=100)
    kerr_angles = []
    reflectances = []
    for H in field_vals:
        VB = 0.6 * H      # VB of quartz under field
        ang, ref = compute_quartz_gold_stack(VB=VB, d_SiO2=1e6)
        kerr_angles.append(ang)
        reflectances.append(ref)
    kerr_angles = np.array(kerr_angles)
    reflectances = np.array(reflectances)
    fig, axs = plt.subplots(1, 2, layout='constrained')
    ax_R = axs[0].twinx()

    axs[0].set_xlabel("Magnetic field (G)")
    axs[0].set_ylabel("Kerr angle (μrad)")
    ax_R.set_ylabel("Reflectance")
    axs[1].set_xlabel("Magnetic field (G)")
    axs[1].set_ylabel("Kerr angle * reflectance (μrad)")

    p1 = axs[0].plot(field_vals, kerr_angles, label="Kerr angle")
    p2 = ax_R.plot(field_vals, reflectances, label="Reflectance", color="red")
    p3 = axs[1].plot(field_vals, kerr_angles * reflectances, label="Product", color="orange")
    axs[0].legend(handles=p1+p2, loc="best")
    axs[1].legend()
    fig.suptitle("Field dependence of reflection on 1mm quartz on Au")
    plt.show()

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
        stack.insert_layer(n=n_bscco, d=d1, bottom_sigma_xx=sigma_xx * 3.2 * 1e-7 / 2.4316e-4, bottom_sigma_xy=sigma_xy)
        stack.insert_layer(n=n_bscco, d=d2)
        stack.insert_layer(n=n_SiO2, d=d_SiO2)
        stack.insert_layer(n=n_Si, d=0)
        
        return stack
    
    lcp_stack = build_stack(OpticalStack(wavelength=1550, initial_state=0))
    rcp_stack = build_stack(OpticalStack(wavelength=1550, initial_state=1))
    lcp_vector = lcp_stack.compute_v()[:2]
    rcp_vector = rcp_stack.compute_v()[:2]
    return lcp_vector, rcp_vector


def twisted_stack_test(d1, d2, d_SiO2, interface_sigma_xy, print_logs=False):
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
    if print_logs:
        print("The magnitude of r (LCP/RCP) is {:.4f} and the phase of r is {:.4f} μrad, reading θ_K = {:.4f} μrad.".format(np.abs(r), np.angle(r) * 1e6, np.angle(r) * 1e6 / 2))
    sigma_xx, sigma_xy = bscco_sigma_xx * 3.2e-7 / 2.4316e-4 / 137, interface_sigma_xy / 137
    kerr_angle_theory = np.real(np.arctan(sigma_xy / (sigma_xx + 2*np.pi*(sigma_xx**2 + sigma_xy**2))))
    if print_logs:
        print("The theoretical Kerr angle is {:.4f} μrad.".format(kerr_angle_theory * 1e6))
        print("Reflectance of RCP is {:.4f}.".format(reflectance_rcp))

    return np.angle(r) * 1e6 / 2, reflectance_rcp

def bscco_thickness_dependence(d_SiO2, interface_sigma_xy):
    '''
    Plots Kerr angle, reflectance, and their product vs d1 and d2, given d_SiO2 and sigma_xy.
    '''
    # monolayer_test(1+0.01j)
    # VB_test(VB=10, d=2e+6)
    # twisted_stack_test(d1=10, d2=10, d_SiO2=280, interface_sigma_xy=1e-4, print_logs=True)

    # d_SiO2 = 280
    d1_vals = d2_vals = np.linspace(start=10, stop=51, num=50)
    D1, D2 = np.meshgrid(d1_vals, d2_vals)
    
    kerr_angles = []
    reflectances = []
    for i in range(50):
        for j in range(50):
            ang, ref = twisted_stack_test(d1=d1_vals[i], d2=d2_vals[j], d_SiO2=d_SiO2, interface_sigma_xy=interface_sigma_xy)
            kerr_angles.append(ang)
            reflectances.append(ref)
    
    kerr_angles = np.array(kerr_angles).reshape(50, 50)
    reflectances = np.array(reflectances).reshape(50, 50)
    fig, axs = plt.subplots(1, 3, figsize=(16, 9), layout="constrained")
    p1 = axs[0].contourf(D1, D2, kerr_angles, 50, cmap='viridis')
    cbar1 = fig.colorbar(p1)
    cbar1.ax.set_ylabel("Kerr angle (μrad)")
    axs[0].set_xlabel("d1 (nm)")
    axs[0].set_ylabel("d2 (nm)")
    
    p2 = axs[1].contourf(D1, D2, reflectances, 50, cmap='viridis')
    cbar2 = fig.colorbar(p2)
    cbar2.ax.set_ylabel("Reflectance")
    axs[1].set_xlabel("d1 (nm)")
    axs[1].set_ylabel("d2 (nm)")

    p3 = axs[2].contourf(D1, D2, kerr_angles * reflectances, 50, cmap='viridis')
    cbar = fig.colorbar(p3)
    cbar.ax.set_ylabel("Kerr angle * R (μrad)")
    axs[2].set_xlabel("d1 (nm)")
    axs[2].set_ylabel("d2 (nm)")

    # plt.xlabel("Top layer thickness (nm)")
    # p1 = ax1.plot(d1_vals, kerr_angles, label="Kerr angle")
    # ax1.set_ylabel("Kerr angle (μrad)")
    # p2 = ax2.plot(d1_vals, reflectances, label="Reflectance", color="red")
    # ax2.set_ylabel("Reflectance")
    # ax1.legend(handles=p1+p2, loc="best")
    fig.suptitle("d_SiO2={}nm, σ_xy={} / 137".format(d_SiO2, interface_sigma_xy))
    # plt.show()
    plt.savefig('imgs\\bscco_thickness_dependence_{}nmSiO2'.format(d_SiO2), dpi=fig.dpi)
    plt.close()

    return 0


def SiO2_thickness_dependence(d1, d2, interface_sigma_xy):
    '''
    Plots Kerr angle, reflectance, and their product vs d_SiO2, given d1, d2, sigma_xy.
    '''
    d_SiO2_vals = np.linspace(start=50, stop=360, num=100)
    kerr_angles = []
    reflectances = []
    for d_SiO2 in d_SiO2_vals:
        ang, ref = twisted_stack_test(d1=d1, d2=d2, d_SiO2=d_SiO2, interface_sigma_xy=interface_sigma_xy)
        kerr_angles.append(ang)
        reflectances.append(ref)

    kerr_angles = np.array(kerr_angles)
    reflectances = np.array(reflectances)
    fig, axs = plt.subplots(1, 2, layout='constrained')
    ax_R = axs[0].twinx()

    axs[0].set_xlabel("SiO2 thickness (nm)")
    axs[0].set_ylabel("Kerr angle (μrad)")
    ax_R.set_ylabel("Reflectance")
    axs[1].set_xlabel("SiO2 thickness (nm)")
    axs[1].set_ylabel("Kerr angle * reflectance (μrad)")

    p1 = axs[0].plot(d_SiO2_vals, kerr_angles, label="Kerr angle")
    p2 = ax_R.plot(d_SiO2_vals, reflectances, label="Reflectance", color="red")
    p3 = axs[1].plot(d_SiO2_vals, kerr_angles * reflectances, label="Product", color="orange")
    axs[0].legend(handles=p1+p2, loc="best")
    axs[1].legend()
    fig.suptitle("d1={}nm, d2={}nm, sigma_xy={} / 137".format(d1, d2, interface_sigma_xy))
    fig.savefig("imgs\\SiO2_thickness_dependence_{}nmsample".format(d1), dpi=fig.dpi)
    # plt.show()
    



    return 0


if __name__ == '__main__':
    # main(d_SiO2=280)
    # d_SiO2_vals = np.arange(start=50, stop=360, step=10)
    # for d_SiO2 in d_SiO2_vals:
    #     bscco_thickness_dependence(d_SiO2=d_SiO2, interface_sigma_xy=-5e-5 + 5e-5j)
    sample_d = 50
    SiO2_thickness_dependence(d1=sample_d, d2=sample_d, interface_sigma_xy=1e-3j)
    # plot_field_dependence()
    