import os
import yaml
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import interpolate

from packages.convergence_power_spectrum.Pk_from_Pm import P_convergence
from config.config_cosmology import cosmo
from packages.matter_power_spectrum import P_1h_components, P_2h_components
from packages.matter_power_spectrum import helper_functions

#
#We will first have to build all the required power spectra at various redshifts
#

#Load in configuration file for convergence_power_spectrum
with open('/home/milan/Desktop/thesis/code/config/config_convergence_power_spectrum.yaml') as file:
    config_convergence = yaml.safe_load(file.read())

#Create z_linspace, containing the various redshift values
z_min = float(config_convergence['z_min'])
z_max = float(config_convergence['z_max'])
N_z = int(config_convergence['N_z'])
z_linspace = np.linspace(z_min, z_max, N_z)

#compute matter power spectrum at each z value if not yet computed
for z in z_linspace:

    #open the matter power spectrum config file
    with open('/home/milan/Desktop/thesis/code/config/config_matter_power_spectrum.yaml') as file:
        config_matter = yaml.safe_load(file.read())

    #load in values from matter config file
    M_min = float(config_convergence['M_min'])
    M_max = float(config_convergence['M_max'])
    N_M = int(config_convergence['N_M'])
    k_min = float(config_convergence['k_min'])
    k_max = float(config_convergence['k_max'])
    N_k = int(config_convergence['N_k'])
    m_min = float(config_convergence['m_min'])

    #Store some config values of convergence config inside matter config
    config_matter['z'] = float(z)
    config_matter['M_min'] = float(M_min)
    config_matter['M_max'] = float(M_max)
    config_matter['N_M'] = int(N_M)
    config_matter['k_min'] = float(k_min)
    config_matter['k_max'] = float(k_max)
    config_matter['N_k'] = int(N_k)
    config_matter['m_min'] = float(m_min)

    # Save the updated values in config_matter file
    with open('/home/milan/Desktop/thesis/code/config/config_matter_power_spectrum.yaml', 'w') as file:
        yaml.dump(config_matter, file)

    #change cwd to data directory and store folder_name
    data_dir = '/home/milan/Desktop/thesis/code/data/matter_power_spectra'
    os.chdir(data_dir)
    folder_name = f"power_spectrum_components,_z={z:.2f},_M_min={M_min:.1e},_M_max={M_max:.1e},_m_min={m_min:.1e}"
    
    #check whether power spectrum folder already exists
    if os.path.isdir(os.path.join(data_dir, folder_name)):
        pass
    
    #if power spectrum is not stored, compute it.
    else:
        print(f"computing power spectrum at z={z}")

        #Create M and k logspace arrays
        M_logspace = np.logspace(np.log10(M_min), np.log10(M_max), N_M)
        k_logspace = np.logspace(np.log10(k_min), np.log10(k_max), N_k) 

        #redo interpolation for all helper functions with updated config values
        helper_functions.interpolate_helper_functions.interpolate()

        # wrapper function for parallel execution
        def compute_P_1h_ss(k):
            return P_1h_components.P_1h_ss(config_matter, k)
        def compute_P_1h_sc(k):
            return P_1h_components.P_1h_sc(config_matter, k)
        def compute_P_1h_self_c(k):
            return P_1h_components.P_1h_self_c(config_matter, k)
        def compute_P_1h_cc(k):
            return P_1h_components.P_1h_cc(config_matter, k)
        def compute_P_2h_ss(k):
            return P_2h_components.P_2h_ss(config_matter, k)
        def compute_P_2h_sc(k):
            return P_2h_components.P_2h_sc(config_matter, k)
        def compute_P_2h_cc(k):
            return P_2h_components.P_2h_cc(config_matter, k)

        # parallel processing
        with ProcessPoolExecutor() as executor:
            P_1h_ss_values = list(executor.map(compute_P_1h_ss, k_logspace))
            P_1h_sc_values = list(executor.map(compute_P_1h_sc, k_logspace))
            P_1h_self_c_values = list(executor.map(compute_P_1h_self_c, k_logspace))
            P_1h_cc_values = list(executor.map(compute_P_1h_cc, k_logspace))
            P_2h_ss_values = list(executor.map(compute_P_2h_ss, k_logspace))
            P_2h_sc_values = list(executor.map(compute_P_2h_sc, k_logspace))
            P_2h_cc_values = list(executor.map(compute_P_2h_cc, k_logspace))

        # convert to array
        P_1h_ss_values = np.array(P_1h_ss_values)
        P_1h_sc_values = np.array(P_1h_sc_values)
        P_1h_self_c_values = np.array(P_1h_self_c_values)
        P_1h_cc_values = np.array(P_1h_cc_values)
        P_2h_ss_values = np.array(P_2h_ss_values)
        P_2h_sc_values = np.array(P_2h_sc_values)
        P_2h_cc_values = np.array(P_2h_cc_values)
        
        print("power spectrum values computed")
        
        #Store computed data in respective data folder
        os.makedirs(folder_name) 
        folder_dir = os.path.join(data_dir, folder_name)

        #Store config file inside data_dir:
        with open(os.path.join(folder_dir, 'config_matter_power_spectrum.yaml'), 'w') as file:
            yaml.dump(config_matter, file)
        
        np.save(os.path.join(folder_dir, "P_1h_ss_values.npy"), P_1h_ss_values)
        np.save(os.path.join(folder_dir, "P_1h_sc_values.npy"), P_1h_sc_values)
        np.save(os.path.join(folder_dir, "P_1h_self_c_values.npy"), P_1h_self_c_values)
        np.save(os.path.join(folder_dir, "P_1h_cc_values.npy"), P_1h_cc_values)
        np.save(os.path.join(folder_dir, "P_2h_ss_values.npy"), P_2h_ss_values)
        np.save(os.path.join(folder_dir, "P_2h_sc_values.npy"), P_2h_sc_values)
        np.save(os.path.join(folder_dir, "P_2h_cc_values.npy"), P_2h_cc_values)



#Load power spectrum components of all zs into single lists
P_1h_ss_list = []
P_1h_sc_list = []
P_1h_self_c_list = []
P_1h_cc_list = []
P_2h_ss_list = []
P_2h_sc_list = []
P_2h_cc_list = []

P_1h_list = []
P_2h_list = []
P_tot_list = []

for z in z_linspace:
    
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

    #Compute 1h, 2h and total power spectrum values 
    P_1h_values = P_1h_ss_values + P_1h_sc_values + P_1h_self_c_values + P_1h_cc_values
    P_2h_values = P_2h_ss_values + P_2h_sc_values + P_2h_cc_values
    P_tot_values = P_1h_values + P_2h_values

    #append all components to lists
    P_1h_ss_list.append(P_1h_ss_values)
    P_1h_sc_list.append(P_1h_sc_values)
    P_1h_self_c_list.append(P_1h_self_c_values)
    P_1h_cc_list.append(P_1h_cc_values)
    P_2h_ss_list.append(P_2h_ss_values)
    P_2h_sc_list.append(P_2h_sc_values)
    P_2h_cc_list.append(P_2h_cc_values)

    P_1h_list.append(P_1h_values)
    P_2h_list.append(P_2h_values)
    P_tot_list.append(P_tot_values)

#turn all lists into arrays
P_1h_ss_array = np.array(P_1h_ss_list)
P_1h_sc_array = np.array(P_1h_sc_list)
P_1h_self_c_array = np.array(P_1h_self_c_list)
P_1h_cc_array = np.array(P_2h_cc_list)
P_2h_ss_array = np.array(P_2h_ss_list)
P_2h_sc_array = np.array(P_2h_sc_list)
P_2h_cc_array = np.array(P_2h_cc_list)

#Construct k_logspace array
k_min = float(config_convergence['k_min'])
k_max = float(config_convergence['k_max'])
N_k = int(config_convergence['N_k'])
k_logspace = np.logspace(np.log10(k_min), np.log10(k_max), N_k)

#Interpolate all arrays
P_1h_ss_interp = interpolate.RectBivariateSpline(k_logspace, z_linspace, P_1h_ss_array.T)
P_1h_sc_interp = interpolate.RectBivariateSpline(k_logspace, z_linspace, P_1h_sc_array.T)
P_1h_self_c_interp = interpolate.RectBivariateSpline(k_logspace, z_linspace, P_1h_self_c_array.T)
P_1h_cc_interp = interpolate.RectBivariateSpline(k_logspace, z_linspace, P_1h_cc_array.T)
P_2h_ss_interp = interpolate.RectBivariateSpline(k_logspace, z_linspace, P_2h_ss_array.T)
P_2h_sc_interp = interpolate.RectBivariateSpline(k_logspace, z_linspace, P_2h_sc_array.T)
P_2h_cc_interp = interpolate.RectBivariateSpline(k_logspace, z_linspace, P_2h_cc_array.T)


#change cwd to data directory and store folder_name
data_dir = '/home/milan/Desktop/thesis/code/data/convergence_power_spectra'
folder_name = f"convergence_power_spectrum_components,_M_min={M_min:.1e},_M_max={M_max:.1e},_m_min={m_min:.1e}"
folder_dir = os.path.join(data_dir, folder_name)

#check whether power spectrum folder already exists
if os.path.isdir(folder_dir):
    pass

#if power spectrum is not stored, compute it.
else:
    #Compute the convergence power spectrum
    #load in the source distribution from config
    z_sources = float(config_convergence['z_sources'])
    z_Hubble = z_sources

    #load in l_logspace values from config
    l_min = float(config_convergence['l_min'])
    l_max = float(config_convergence['l_max'])
    N_l = int(config_convergence['N_l'])
    l_logspace = np.logspace(np.log10(l_min), np.log10(l_max), N_l)

    # wrapper function for parallel execution
    def compute_Pk_1h_ss(l):
        return P_convergence(l, P_1h_ss_interp, z_sources, cosmo, z_Hubble)
    def compute_Pk_1h_sc(l):
        return P_convergence(l, P_1h_sc_interp, z_sources, cosmo, z_Hubble)
    def compute_Pk_1h_self_c(l):
        return P_convergence(l, P_1h_self_c_interp, z_sources, cosmo, z_Hubble)
    def compute_Pk_1h_cc(l):
        return P_convergence(l, P_1h_cc_interp, z_sources, cosmo, z_Hubble)
    def compute_Pk_2h_ss(l):
        return P_convergence(l, P_2h_ss_interp, z_sources, cosmo, z_Hubble)
    def compute_Pk_2h_sc(l):
        return P_convergence(l, P_2h_sc_interp, z_sources, cosmo, z_Hubble)
    def compute_Pk_2h_cc(l):
        return P_convergence(l, P_2h_cc_interp, z_sources, cosmo, z_Hubble)

    # parallel processing
    with ProcessPoolExecutor() as executor:
        Pk_1h_ss_values = list(executor.map(compute_Pk_1h_ss, l_logspace))
        Pk_1h_sc_values = list(executor.map(compute_Pk_1h_sc, l_logspace))
        Pk_1h_self_c_values = list(executor.map(compute_Pk_1h_self_c, l_logspace))
        Pk_1h_cc_values = list(executor.map(compute_Pk_1h_cc, l_logspace))
        Pk_2h_ss_values = list(executor.map(compute_Pk_2h_ss, l_logspace))
        Pk_2h_sc_values = list(executor.map(compute_Pk_2h_sc, l_logspace))
        Pk_2h_cc_values = list(executor.map(compute_Pk_2h_cc, l_logspace))

    # convert to array
    Pk_1h_ss_values = np.array(Pk_1h_ss_values)
    Pk_1h_sc_values = np.array(Pk_1h_sc_values)
    Pk_1h_self_c_values = np.array(Pk_1h_self_c_values)
    Pk_1h_cc_values = np.array(Pk_1h_cc_values)
    Pk_2h_ss_values = np.array(Pk_2h_ss_values)
    Pk_2h_sc_values = np.array(Pk_2h_sc_values)
    Pk_2h_cc_values = np.array(Pk_2h_cc_values)

    #Store computed data in respective data folder
    os.makedirs(folder_dir) 

    #Store config file in data folder
    with open(os.path.join(folder_dir, 'config_convergence_power_spectrum.yaml'), 'w') as file:
        yaml.dump(config_convergence, file)

    np.save(os.path.join(folder_dir, "Pk_1h_ss_values.npy"), Pk_1h_ss_values)
    np.save(os.path.join(folder_dir, "Pk_1h_sc_values.npy"), Pk_1h_sc_values)
    np.save(os.path.join(folder_dir, "Pk_1h_self_c_values.npy"), Pk_1h_self_c_values)
    np.save(os.path.join(folder_dir, "Pk_1h_cc_values.npy"), Pk_1h_cc_values)
    np.save(os.path.join(folder_dir, "Pk_2h_ss_values.npy"), Pk_2h_ss_values)
    np.save(os.path.join(folder_dir, "Pk_2h_sc_values.npy"), Pk_2h_sc_values)
    np.save(os.path.join(folder_dir, "Pk_2h_cc_values.npy"), Pk_2h_cc_values)




