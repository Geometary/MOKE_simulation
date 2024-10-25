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




if __name__ == '__main__':
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
