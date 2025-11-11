import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

with open('/home/milan/Desktop/thesis/code/config/config_convergence_power_spectrum.yaml') as file:
    config_convergence = yaml.safe_load(file.read())

M_min = float(config_convergence['M_min'])
M_max = float(config_convergence['M_max'])
m_min = float(config_convergence['m_min'])

#Construct l_logspace
l_min = float(config_convergence['l_min'])
l_max = float(config_convergence['l_max'])
N_l = int(config_convergence['N_l'])
l_logspace = np.logspace(np.log10(l_min), np.log10(l_max), N_l)

#open folder where power spectra values at redshift z are stored
data_dir = '/home/milan/Desktop/thesis/code/data/convergence_power_spectra'
folder_name = f"convergence_power_spectrum_components,_M_min={M_min:.1e},_M_max={M_max:.1e},_m_min={m_min:.1e}"
folder_dir = os.path.join(data_dir, folder_name)

#load in all power spectra arrays
Pk_1h_ss_values = np.load(os.path.join(folder_dir, "Pk_1h_ss_values.npy"))
Pk_1h_sc_values = np.load(os.path.join(folder_dir, "Pk_1h_sc_values.npy"))
Pk_1h_self_c_values = np.load(os.path.join(folder_dir, "Pk_1h_self_c_values.npy"))
Pk_1h_cc_values = np.load(os.path.join(folder_dir, "Pk_1h_cc_values.npy"))
Pk_2h_ss_values = np.load(os.path.join(folder_dir, "Pk_2h_ss_values.npy"))
Pk_2h_sc_values = np.load(os.path.join(folder_dir, "Pk_2h_sc_values.npy"))
Pk_2h_cc_values = np.load(os.path.join(folder_dir, "Pk_2h_cc_values.npy"))

Pk_tot_values = (Pk_1h_ss_values +
                 Pk_1h_sc_values +
                 Pk_1h_self_c_values +
                 Pk_1h_cc_values +
                 Pk_2h_ss_values +
                 Pk_2h_sc_values +
                 Pk_2h_cc_values)

#Build figure
plt.figure(figsize=(8, 8))
plt.plot(l_logspace, np.log10(l_logspace*(l_logspace + 1)/(2*np.pi) * Pk_tot_values), label='P_tot')
plt.plot(l_logspace, np.log10(l_logspace*(l_logspace + 1)/(2*np.pi) * Pk_1h_ss_values), label='P_1h_ss')
plt.plot(l_logspace, np.log10(l_logspace*(l_logspace + 1)/(2*np.pi) * Pk_1h_sc_values), label='P_1h_sc')
plt.plot(l_logspace, np.log10(l_logspace*(l_logspace + 1)/(2*np.pi) * Pk_1h_self_c_values), label='P_1h_self_c')
plt.plot(l_logspace, np.log10(l_logspace*(l_logspace + 1)/(2*np.pi) * Pk_1h_cc_values), label='P_1h_cc')
plt.plot(l_logspace, np.log10(l_logspace*(l_logspace + 1)/(2*np.pi) * Pk_2h_ss_values), label='P_2h_ss')
plt.plot(l_logspace, np.log10(l_logspace*(l_logspace + 1)/(2*np.pi) * Pk_2h_sc_values), label='P_2h_sc')
plt.plot(l_logspace, np.log10(l_logspace*(l_logspace + 1)/(2*np.pi) * Pk_2h_cc_values), label='P_2h_cc')

plt.xscale('log')
plt.xlim(10, 1e4)
plt.ylim(-8, -4)
plt.xlabel('l')
plt.ylabel(r"$log(l(l+1)/2\pi * P_\kappa)$")
plt.legend()
plt.title(f'convergence power spectrum, M_min={M_min:.1e}, M_max={M_max:.1e}, m_min={m_min:.1e}')
plt.show()

