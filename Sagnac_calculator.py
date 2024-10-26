'''
This file serves the purpose of calculating the Sagnac interference terms and saves manual labor.
'''
import numpy as np
import scipy.integrate as si
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


'''
The parameter space is [phi_m, omega_m, tau, phi_nr, delta_tau, delta_phase, R_ratio]
'''


def eval_Vs(V_detector, params):
    '''
    Evaluates the base case where there is only one spatial mode (two polarizations) carrying a 
    non-reciprocal phase φ_nr (in rad).
    '''
    omega_m = params[1]

    V_1fx, _ = si.quad(lambda t: V_detector(t, params) * np.cos(omega_m*t) * omega_m / np.pi, -np.pi/omega_m, np.pi/omega_m)
    V_1fy, _ = si.quad(lambda t: V_detector(t, params) * np.sin(omega_m*t) * omega_m / np.pi, -np.pi/omega_m, np.pi/omega_m)
    V_2fx, _ = si.quad(lambda t: V_detector(t, params) * np.cos(2*omega_m*t) * 2*omega_m / np.pi, -np.pi/(2*omega_m), np.pi/(2*omega_m))
    V_2fy, _ = si.quad(lambda t: V_detector(t, params) * np.sin(2*omega_m*t) * 2*omega_m / np.pi, -np.pi/(2*omega_m), np.pi/(2*omega_m))
    V_2fr = np.sqrt(V_2fx**2 + V_2fy**2)

    return [V_1fx, V_1fy, V_2fr]


def V_detector_base_case(t, params):
    [phi_m, omega_m, tau, phi_nr] = params[:4]
    return np.abs(np.exp(1j * phi_m * np.cos(omega_m*t)) + np.exp(1j * phi_m * np.cos(omega_m*t - omega_m*tau) + 1j*phi_nr))**2


def V_detector_two_modes(t, params):
    [phi_m, omega_m, tau, phi_nr, delta_tau, delta_phase, A_mode1_x, A_mode1_y, A_mode2_x, A_mode2_y] = params

    return np.abs(A_mode1_x * np.exp(1j * phi_m * np.cos(omega_m*t)) + A_mode1_y * np.exp(1j * phi_m * np.cos(omega_m*t - omega_m*tau) + 1j*phi_nr) + \
                        A_mode2_x * np.exp(1j * phi_m * np.cos(omega_m*t) + 1j*delta_phase) + A_mode2_y * np.exp(1j * phi_m * np.cos(omega_m*t - omega_m*tau - delta_tau) + 1j*phi_nr + 1j*delta_phase))**2


def investigate_delta_d():
    '''
    Investigates how a fake signal can be generated by a difference in path length between two traversal through the lens under magnetic field.
    '''
    # The parameter space is [phi_m, omega_m, tau, phi_nr, delta_tau, delta_phase, A_mode1_x, A_mode1_y, A_mode2_x, A_mode2_y]
    # phi_m_vals = np.linspace(start=0.7, stop=1, num=100)
    # phi_m = 0.851 + 0.02j
    # print(sp.jv(2, 2*phi_m) / sp.jv(1, 2*phi_m))
    # phi_nr_vals = np.linspace(start=-1e-4, stop=1e-4, num=100)
    theta_contrast_ratio = np.linspace(start=-0.5, stop=0.5, num=100)           # (θ1-θ2) / (θ1+θ2)
    lens_VHd = 390e-6        # kerr rotation of lens at 1kG of field, also θ1+θ2
    delta_theta_vals = theta_contrast_ratio * lens_VHd          # θ1-θ2
    sig_thetak_vals = []
    bg_thetak_vals = []
    phi_nr_vals = np.linspace(start=100e-6, stop=-100e-6, num=100)
    # for phi_m in phi_m_vals:
    #     params = [phi_m+0.04j, 35047e+3*2*np.pi, 1/(2*35047e+3), 800e-6, 0.001, 0.5, 0.5]
    #     [V_1fx, V_1fy, V_2fr] = eval_Vs(V_detector_two_modes, params)
    #     sig_thetak_vals.append(5e5 * np.arctan(sp.jv(2, 2*np.real(phi_m))*V_1fx / (sp.jv(1, 2*np.real(phi_m))*V_2fr)))
    #     bg_thetak_vals.append(5e5 * np.arctan(sp.jv(2, 2*np.real(phi_m))*V_1fy / (sp.jv(1, 2*np.real(phi_m))*V_2fr)))
    phi_m = 0.851 - 0.3j
    ie = 0j
    for i in range(len(delta_theta_vals)):
        above_QWP_theta = delta_theta_vals[i]
        phi_nr = phi_nr_vals[i]
        A_mode1_x = 1 + ie + (1 - ie) * above_QWP_theta
        A_mode1_y = 1 + ie - (1 - ie) * above_QWP_theta * (-0.5)
        A_mode2_x = 0
        A_mode2_y = 0
        params = [phi_m, 35047e+3*2*np.pi, 1/(2*35047e+3), 0, 0, 0, A_mode1_x, A_mode1_y, A_mode2_x, A_mode2_y]
        [V_1fx, V_1fy, V_2fr] = eval_Vs(V_detector_two_modes, params)
        sig_thetak_vals.append(5e5 * np.arctan(sp.jv(2, 2*np.real(phi_m))*V_1fx / (sp.jv(1, 2*np.real(phi_m))*V_2fr)))
        bg_thetak_vals.append(5e5 * np.arctan(sp.jv(2, 2*np.real(phi_m))*V_1fy / (sp.jv(1, 2*np.real(phi_m))*V_2fr)))

    x_data = theta_contrast_ratio * 6e3
    x_label = 'δd (μm)'

    plt.plot(x_data, sig_thetak_vals, color='red', label="Signal")
    plt.plot(x_data, bg_thetak_vals, color='blue', label="Background")
    plt.plot(x_data, 23.4 * np.ones(len(x_data)), color='orange', linestyle='dotted', label="Au without lens")
    plt.plot(x_data, -23.4 * np.ones(len(x_data)), color='orange', linestyle='dotted')

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel("θ_k measured (μrad)")
    plt.title("θ_k vs d1-d2 for at H=1kG, φ_m={}, φ_nr=0".format(phi_m))
    plt.show()


    field_vals = np.linspace(start=-1, stop=1, num=100)         # magnetic field from -1kG to +1kG
    true_signal_slope = 20e-6          # true signal slope in rad/kG
    delta_d_vals = np.linspace(start=-0.6, stop=0.6, num=10)      # δd values in mm

    cmap = plt.get_cmap('bwr')
    colors = cmap(np.linspace(0, 1, 10))
    plt.xlabel("H (kG)")
    plt.ylabel("Signal θ_k measured (μrad)")
    plt.title("Signal θ_k vs field for different δd, φ_m={}, Vd_s={}μrad/kG".format(phi_m, true_signal_slope*1e6))

    for i in range(len(delta_d_vals)):
        delta_d = delta_d_vals[i]
        delta_thetas = 65e-6 * field_vals * delta_d
        phi_nr_vals = 2 * true_signal_slope * field_vals
        sig_thetak_vals = []
        bg_thetak_vals = []
        for j in range(len(phi_nr_vals)):
            A_mode1_x = 1 + ie + (1 - ie) * delta_thetas[j]
            A_mode1_y = 1 + ie - (1 - ie) * delta_thetas[j]
            params = [phi_m, 35047e+3*2*np.pi, 1/(2*35047e+3), phi_nr_vals[j], 0, 0, A_mode1_x, A_mode1_y, 0, 0]
            [V_1fx, V_1fy, V_2fr] = eval_Vs(V_detector_two_modes, params)
            sig_thetak_vals.append(5e5 * np.arctan(sp.jv(2, 2*np.real(phi_m))*V_1fx / (sp.jv(1, 2*np.real(phi_m))*V_2fr)))
            bg_thetak_vals.append(5e5 * np.arctan(sp.jv(2, 2*np.real(phi_m))*V_1fy / (sp.jv(1, 2*np.real(phi_m))*V_2fr)))

        plt.plot(field_vals, sig_thetak_vals, label="δd={:.3f}mm".format(delta_d), color=colors[i])
        # plt.plot(field_vals, bg_thetak_vals, label="δd={:.3f}, bg".format(delta_d))
    plt.legend()
    plt.show()


def investigate_mod_dependence():
    '''
    Investigates the modulation parameter dependence of 1fx/y and 2fr.
    '''
    # The parameter space is [phi_m, omega_m, tau, phi_nr, delta_tau, delta_phase, A_mode1_x, A_mode1_y, A_mode2_x, A_mode2_y]
    def pow2phim(mod_power, mod_freq):
        '''
        Converts RF power to modulation depth phi_m, assuming 0.16rad/Vrms ratio measured in our lab (phi_m=0.92rad for RF power 25.3dBm) and using
        the factory-measured reflectance spectrum.
        '''
        omega = 2 * np.pi * mod_freq * 1e3
        omega_0 = 2 * np.pi * 35033e3
        omega_plus = 2 * np.pi * 35207e3
        omega_minus = 2 * np.pi * 34860e3
        Z_0 = 50
        C = (omega_plus**2 / omega_0**2 - 1) / (2 * Z_0 * omega_plus)
        L = 1 / (omega_0**2 * C)
        X = omega*L - 1/(omega*C)       # reactive impedance
        power_factor = 4*Z_0**2 / (4*Z_0**2 + X**2)
        P_tot = 1e-3 * 10**(mod_power/10)
        V_amp = np.sqrt(P_tot * power_factor * (Z_0**2 + X**2) / Z_0)

        return 0.16 * np.sqrt(2) * V_amp

    def unit_complex(a, b):
        '''
        Returns a complex number with Im/Re = b/a and unit magnitude.
        '''
        return (a + b*1j) / np.sqrt(a**2+b**2)
    
    mod_powers = np.arange(start=23, stop=26.2, step=0.1)
    mod_freqs = np.arange(start=35010, stop=35071, step=5)
    proper_omega_m = 35030e+3 * 2 * np.pi       # assume a value for the proper modulation frequency that gives ω_m*τ=π
    tau = np.pi / proper_omega_m
    onefx_theory = []
    onefy_theory = []
    twofr_theory = []
    A_x = 13.6             # magnitude of A_x should be ~13.6 to agree with experimental values
    A_y = 13.6
    for freq in mod_freqs:
        for pow in mod_powers:
            phi_m = pow2phim(pow, freq)
            phi_nr = 0
            params = [phi_m, 2*np.pi*freq*1e3, tau, phi_nr, 0, 0, A_x, A_y, 0, 0]
            [V_1fx, V_1fy, V_2fr] = eval_Vs(V_detector_two_modes, params)
            onefx_theory.append(V_1fx)
            onefy_theory.append(V_1fy)
            twofr_theory.append(V_2fr)

    # now we plot the data
    Pows, Freqs = np.meshgrid(mod_powers, mod_freqs)
    onefx_theory = np.array(onefx_theory).reshape(len(mod_freqs), len(mod_powers)) * 3162
    onefy_theory = np.array(onefy_theory).reshape(len(mod_freqs), len(mod_powers)) * 3162
    twofr_theory = np.array(twofr_theory).reshape(len(mod_freqs), len(mod_powers))
    z_data = [onefx_theory, onefy_theory, twofr_theory]
    cmaps = ['bwr', 'bwr', 'viridis']
    cbar_labels = ['1fx (+70dB mV)', '1fy (+70dB mV)', '2fr (mV)']
    for i in range(len(z_data)):
        if i < 2:
            continue
        plt.xlabel('Modulation power (dBm)')
        plt.ylabel('Modulation frequency (kHz)')
        plt.contourf(Pows, Freqs, z_data[i], 50, cmap=cmaps[i])
        cbar = plt.colorbar()
        cbar.set_label(cbar_labels[i])
        plt.show()
    

    return 0


def plot_smith():
    omega_step = 10e3
    omega_m_vals = 2*np.pi * np.arange(start=34000e3, stop=36000e3, step=omega_step)
    omega_0 = 2 * np.pi * 35033e3
    omega_plus = 2 * np.pi * 35207e3
    omega_minus = 2 * np.pi * 34860e3
    Z_0 = 50
    C = (omega_plus**2 / omega_0**2 - 1) / (2 * Z_0 * omega_plus)
    L = 1 / (omega_0**2 * C)
    c = 3e8 / 1.5
    l = 1.6
    def omega2Gamma(omega):
        res = 1j * (omega*L - 1/(omega*C)) / (2*Z_0 + 1j * (omega*L-1/(omega*C)))
        res *= np.exp(-1j * 2*l*omega/c)
        return res
    
    Gammas = []
    for omega in omega_m_vals:
        Gammas.append(omega2Gamma(omega))
    Gammas = np.array(Gammas, dtype=complex)

    Gamma_minus = Gammas[round((omega_minus-omega_m_vals[0])/(2*np.pi*omega_step))]
    Gamma_0 = Gammas[round((omega_0-omega_m_vals[0])/(2*np.pi*omega_step))]
    Gamma_plus = Gammas[round((omega_plus-omega_m_vals[0])/(2*np.pi*omega_step))]

    plt.plot(Gammas.real, Gammas.imag, 'b-')
    plt.scatter(Gamma_minus.real, Gamma_minus.imag, label='M2')
    plt.scatter(Gamma_0.real, Gamma_0.imag, label='M4')
    plt.scatter(Gamma_plus.real, Gamma_plus.imag, label='M3')

    plt.plot(np.linspace(-1, 1, 200), np.zeros(200), color='gray', linestyle='dotted')
    plt.plot(np.zeros(200), np.linspace(-1, 1, 200), color='gray', linestyle='dotted')
    plt.xlabel("Re(Γ)")
    plt.ylabel("Im(Γ)")
    plt.legend()
    plt.show()

    print("And we have Z_2=({})Ω, Z_4=({})Ω, and Z_3=({})Ω".format(50*(1+Gamma_minus)/(1-Gamma_minus), 50*(1+Gamma_0)/(1-Gamma_0), \
                                                                            50*(1+Gamma_plus)/(1-Gamma_plus)))

    return 0


    


if __name__ == '__main__':
    # The parameter space is [phi_m, omega_m, tau, phi_nr, delta_tau, delta_phase, A_mode1_x, A_mode1_y, A_mode2_x, A_mode2_y]
    # investigate_delta_d()
    investigate_mod_dependence()
    # plot_smith()
