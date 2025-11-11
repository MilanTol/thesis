import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from config.config_cosmology import cosmo


with open('/home/milan/Desktop/thesis/code/config/config_matter_power_spectrum.yaml') as file:
    config_matter = yaml.safe_load(file.read())

z = float(config_matter['z'])

for z in [0, 1, 2]:
    
    M_min = float(config_matter['M_min'])
    M_max = float(config_matter['M_max'])
    m_min = float(config_matter['m_min'])

    #Construct k_logspace
    k_min = float(config_matter['k_min'])
    k_max = float(config_matter['k_max'])
    N_k = int(config_matter['N_k'])
    k_logspace = np.logspace(np.log10(k_min), np.log10(k_max), N_k)

    #open folder where power spectra values at redshift z are stored
    data_dir = '/home/milan/Desktop/thesis/code/data/matter_power_spectra'
    folder_name = f"power_spectrum_components,_z={z:.2f},_M_min={M_min:.1e},_M_max={M_max:.1e},_m_min={m_min:.1e}"
    folder_dir = os.path.join(data_dir, folder_name)

    #load in all power spectra arrays
    P_1h_ss_values = np.load(os.path.join(folder_dir, "P_1h_ss_values.npy"))
    P_1h_sc_values = np.load(os.path.join(folder_dir, "P_1h_sc_values.npy"))
    P_1h_self_c_values = np.load(os.path.join(folder_dir, "P_1h_self_c_values.npy"))
    P_1h_cc_values = np.load(os.path.join(folder_dir, "P_1h_cc_values.npy"))
    P_2h_ss_values = np.load(os.path.join(folder_dir, "P_2h_ss_values.npy"))
    P_2h_sc_values = np.load(os.path.join(folder_dir, "P_2h_sc_values.npy"))
    P_2h_cc_values = np.load(os.path.join(folder_dir, "P_2h_cc_values.npy"))

    P_tot_values = (P_1h_ss_values +
                    P_1h_sc_values +
                    P_1h_self_c_values +
                    P_1h_cc_values +
                    P_2h_ss_values +
                    P_2h_sc_values +
                    P_2h_cc_values)

    # #Build figure
    # plt.figure(figsize=(8, 8))
    # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_tot_values), label='P_tot')
    # # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_1h_ss_values), label='P_1h_ss')
    # # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_1h_sc_values), label='P_1h_sc')
    # # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_1h_self_c_values), label='P_1h_self_c')
    # # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_1h_cc_values), label='P_1h_cc')
    # # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_2h_ss_values), label='P_2h_ss')
    # # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_2h_sc_values), label='P_2h_sc')
    # # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_2h_cc_values), label='P_2h_cc')

    # plt.xscale('log')
    # plt.xlim(0.02, 1e4)
    # plt.ylim(-2, 4.2)
    # plt.xlabel('l')
    # plt.ylabel(r"$k^3/(2\pi^2) * P_m)$")
    # plt.legend()
    # plt.title(f'dimensionless matter power spectrum, z={z:.2f}, M_min={M_min:.1e}, M_max={M_max:.1e}, m_min={m_min:.1e}')
    # plt.show()

    #Build figure
    #plt.figure(figsize=(8, 8))
    plt.plot(k_logspace, np.log10(P_tot_values), label='P_tot')
    plt.plot(k_logspace, np.log10(cosmo.matterPowerSpectrum(k_logspace, z)), label='cosmo')
    # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_1h_ss_values), label='P_1h_ss')
    # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_1h_sc_values), label='P_1h_sc')
    # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_1h_self_c_values), label='P_1h_self_c')
    # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_1h_cc_values), label='P_1h_cc')
    # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_2h_ss_values), label='P_2h_ss')
    # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_2h_sc_values), label='P_2h_sc')
    # plt.plot(k_logspace, np.log10(k_logspace**3/(2*np.pi**2) * P_2h_cc_values), label='P_2h_cc')

    plt.xscale('log')
    plt.xlim(0.005, 2)
    plt.ylim(-1, 5)
    plt.xlabel('l')
    plt.ylabel(r"$k^3/(2\pi^2) * P_m)$")
    plt.legend()
    plt.title(f'matter power spectrum, z={z:.2f}, M_min={M_min:.1e}, M_max={M_max:.1e}, m_min={m_min:.1e}')

plt.show()