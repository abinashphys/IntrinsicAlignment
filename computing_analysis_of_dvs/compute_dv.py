"""
The early version of this code was developed by Paul Rogozenski, but has been adapted to suit the needs of the analysis I was doing. To understand the code please read the analysis procedure

The fiducial values of each parameter of IA in the shear signal

For TATT
--------
A1 = 0.7
eta1 = -1.7
A2 = -1.36
eta2 = -2.5

For NLA 
-------
A1 = 0.7
eta1 = -1.7
A2 = 0
eta2 = 0

Analysis procedure:
-------------------
To compute the datavector for TATT we consider the arrays:
A1_arr = np.array([0.7, 0.8,0.6])
eta_1_arr = np.array([-1.7,-1.2,-2.2])
A2_arr = np.array([-1.36,-1.47,-1.25])
eta_2_arr = np.array([-2.5,-2.2,-2.8])

Thus the various permutations of these parameters produce a total of 81 datavectors.

When considering the non-higher order model resembling NLA (we take into account that A2 = 0 and eta2 = 0 so we can neglect them), 
to compute the datavectors we consider the arrays:
A1_arr = np.array([0.7, 0.9,0.8,0.6,0.52])
eta_1_arr = np.array([-1.7,-1.2,-1.35,-1.88,-1.95])
This produces atotal of 25 datavectors

"""
import sys
import numpy as np
import time 
import h5py as h5
import multiprocessing
import os

from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.model import Model
from cobaya.conventions import kinds, _timing, _params, _prior, _packages_path

# Check if a configuration file is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <config_file>")
    sys.exit(1)
configfile = sys.argv[1]

def get_model(yaml_file):
    info  = yaml_load_file(yaml_file)
    updated_info = update_info(info)
    model =  Model(updated_info[_params], updated_info[kinds.likelihood],
               updated_info.get(_prior), updated_info.get(kinds.theory),
               packages_path=info.get(_packages_path), timing=updated_info.get(_timing),
               allow_renames=False, stop_at_error=info.get("stop_at_error", False))
    return model

class CocoaModel:
     # A wrapper class for the cobaya cosmological model.
    def __init__(self, configfile, likelihood):
        self.model      = get_model(configfile)
        self.likelihood = likelihood
        
    def calculate_data_vector(self, params_values, baryon_scenario=None):        
        likelihood   = self.model.likelihood[self.likelihood]
        input_params = self.model.parameterization.to_input(params_values)
        self.model.provider.set_current_input_params(input_params)
        for (component, index), param_dep in zip(self.model._component_order.items(), 
                                                 self.model._params_of_dependencies):
            depend_list = [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(want_derived=False,
                                         dependency_params=depend_list, cached=False, **params)
        if baryon_scenario is None:
            data_vector = likelihood.get_datavector(**input_params)
        else:
            data_vector = likelihood.compute_barion_datavector_masked_reduced_dim(baryon_scenario, **input_params)
        return np.array(data_vector)


#Parameter array with fiducial values
#A1_arr = np.array([0.7, 0.9,0.8,0.6,0.52])
A1_arr = np.array([0.7, 0.8,0.6])

#eta_1_arr = np.array([-1.7])
#eta_1_arr = np.array([-1.7,-1.2,-1.35,-1.88,-1.95])
eta_1_arr = np.array([-1.7,-1.2,-2.2])

#A2_arr = np.array([-1.36])
A2_arr = np.array([-1.36,-1.47,-1.25])
#A2_arr = np.array([0])

#eta_2_arr = np.array([-2.5])
eta_2_arr = np.array([-2.5,-2.2,-2.8])
#eta_2_arr = np.array([0])

print(str(round(A1_arr[0], 1)))

def calc_cosmo(eta_1):
    # Function to calculate cosmological data vector for a given eta_1.
    likelihood = 'lsst_lens_eq_src.lsst_3x2pt'    
    cocoa_model = CocoaModel(configfile, likelihood)

    bias_fid = [0,0,0,0,0,0,0,0,0,0,0]   

    params_fid = {'logA': 3.044522437723423, 'ns': 0.97, 'H0': 70., 'omegabh2': 0.0196, 'omegach2': 0.1274,
                 'LSST_A1_1': 0.5, 'LSST_A1_2': 0.0, 'LSST_A2_1': 5.0, 'LSST_A2_2': 0.0}

    for i in range(10):
        params_fid['LSST_DZ_%d'%(i+1)] = 0.
        params_fid['LSST_BTA_%d'%(i+1)] = 0.0
        params_fid['LSST_M%d'%(i+1)] = 0.
        params_fid['LSST_B1_%d'%(i+1)] = bias_fid[i]
    reserved_eta_1 = eta_1
    for A1 in A1_arr:
        for eta_2 in eta_2_arr:        
            for A2 in A2_arr:
                if A1==0: 
                    eta_1=0
                else:
                    eta_1 = reserved_eta_1
                #if A2==0: eta_2=0
                params_fid['LSST_A1_1'] = A1
                params_fid['LSST_A1_2'] = eta_1
                params_fid['LSST_A2_1'] = A2
                params_fid['LSST_A2_2'] = eta_2
        
                #A2=0
                #eta_2=0
        
          
                save_file_name = "./dv_fid_lens_source_test_A1_"+str(A1)+"_eta1_"+str(eta_1)+"_A2_"+str(A2)+"_eta2_"+str(eta_2)
        
                if not os.path.isfile(save_file_name + ".txt"):
                
                    print("Number of dimensions: %d"%(len(params_fid)))    
                    for x in params_fid:
                        print(x + " : %2.3f"%(params_fid[x]))
                    start_time  = time.time() 
                    data_vector = cocoa_model.calculate_data_vector(params_fid)
                    end_time    = time.time()
                    print("Time taken for data vector calculation: %2.2f s "%(end_time - start_time))
        
        
        
                    with h5.File(save_file_name + ".h5", 'w') as f:
                        f['dv'] = data_vector
                        grp = f.create_group('params')
                        for x in params_fid:
                            grp[x] = params_fid[x]
                            
                    dv_file = np.array([np.arange(len(data_vector)), data_vector]).T        
                    np.savetxt(save_file_name + ".txt", dv_file)    
        
                else:
                    print("not re-computing", save_file_name)
# Parallel processing
pool = multiprocessing.Pool(5)
result = pool.map(calc_cosmo, eta_1_arr)
#calc_cosmo(0.1)