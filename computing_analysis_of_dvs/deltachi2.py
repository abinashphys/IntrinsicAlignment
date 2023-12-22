"""
This code is used to calculate the deltachi values for shear signal and GGL, 
The datavector produced by "compute_dv.py" is then compared against the datavector computed with the fiducial value of the paramters: [Final_TATT/dv_fid_lens_source_test_A1_0.7_eta1_-1.7_A2_-1.36_eta2_-2.5.txt] 
to calculate the chi-square (\( \chi^2 \)) statistic. This statistic quantifies the model's goodness of fit to the data, 
with a lower Δchi² value indicating a better fit. 
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os

colo = 'r'

datavfile1 ="Final_TATT/dv_fid_lens_source_test_A1_0.7_eta1_-1.7_A2_-1.36_eta2_-2.5.txt" # baseline datavector computed with fiducial values of the parameter
d1 = np.genfromtxt(datavfile1)[:, 1]

filename = [
    'dv_fid_lens_source_test_A1_0.7_eta1_-1.7_A2_-1.25_eta2_-2.8.txt' #example of second datavector
]

# Initialize a dictionary to store Delta chi2 values
delta_chi2_values = {}

for i in filename:
    datavfile2="/home/u8/abinashdas/cocoa/Cocoa/Datavectors/Final_TATT/"+i

    print("***")
    print(filename)
    print("===========================================================")

    covfile = "./cov_lsst_y1_lenseqsrc_Ntheta20_Ntomo10.txt"

    m = np.genfromtxt("lsst_y1_lenseqsrc_Ntheta20_Ntomo10.mask")[:, 1]


    ndata = d1.shape[0]

    ind1 = np.where(m)
    ind0 = np.where(m - 1.0) 

    data = np.genfromtxt(covfile)
    cov = np.zeros((ndata, ndata))
    for i in range(ndata):
        cov[i][i] = 1.0
    for i in range(0, len(data)):
        cov[int(data[i, 0]), int(data[i, 1])] = data[i, 2]
        cov[int(data[i, 1]), int(data[i, 0])] = data[i, 2]
        if (int(data[i, 0]) - int(data[i, 1])):
            cov[int(data[i, 0]), int(data[i, 1])] *= m[int(data[i, 0])] * m[int(data[i, 1])]
            cov[int(data[i, 1]), int(data[i, 0])] *= m[int(data[i, 0])] * m[int(data[i, 1])]

    s = np.sqrt(np.diag(cov))


    d2 = np.genfromtxt(datavfile2)[:, 1]

    ind = np.arange(0, ndata)

    temp_chis = []
    chi = 0.0
    inv = LA.inv(cov)

    for i in range(0, ndata):
        temp_chi = 0.0
        for j in range(0, ndata):
            chi += (d1[i] - d2[i]) * inv[i, j] * (d1[j] - d2[j]) * m[i] * m[j]
    print("3x2pt: Delta chi2 = %f" % (chi))

    for j in range(0, ndata):
        temp_chi = (d1[j] - d2[j]) * inv[j, j] * (d1[j] - d2[j]) * m[j] * m[j]
        if m[j] != 0: temp_chis.append(temp_chi)

    twopt_range = 2200
    cluster_range = 4120

    chi = 0.0
    inv = LA.inv(cov[:twopt_range, :twopt_range])
    for i in range(0, twopt_range):
        temp_chi = 0.0
        for j in range(0, twopt_range):
            chi += (d1[i] - d2[i]) * inv[i, j] * (d1[j] - d2[j]) * m[i] * m[j]

    print("shear: Delta chi2 = %f" % (chi))

    chi = 0.0
    inv = LA.inv(cov[twopt_range:cluster_range, twopt_range:cluster_range])
    for i in range(0, ndata - twopt_range - (ndata - cluster_range)):
        temp_chi = 0.0
        for j in range(0, ndata - twopt_range - (ndata - cluster_range)):
            chi += (d1[twopt_range + i] - d2[twopt_range + i]) * inv[i, j] * (d1[twopt_range + j] - d2[twopt_range + j]) * m[twopt_range + i] * m[twopt_range + j]

    print("ggl: Delta chi2 = %f" % (chi))

    chi = 0.0
    inv = LA.inv(cov[twopt_range:, twopt_range:])
    for i in range(0, ndata - twopt_range):
        for j in range(0, ndata - twopt_range):
            chi += (d1[twopt_range + i] - d2[twopt_range + i]) * inv[i, j] * (d1[twopt_range + j] - d2[twopt_range + j]) * m[twopt_range + i] * m[twopt_range + j]

    print("2x2pt: Delta chi2 = %f" % (chi))

    chi = 0.0
    inv = LA.inv(cov[cluster_range:, cluster_range:])
    for i in range(0, ndata - cluster_range):
        temp_chi = 0.0
        for j in range(0, ndata - cluster_range):
            chi += (d1[cluster_range + i] - d2[cluster_range + i]) * inv[i, j] * (d1[cluster_range + j] - d2[cluster_range + j]) * m[cluster_range + i] * m[cluster_range + j]

    print("wtheta: Delta chi2 = %f" % (chi))

    nxip = int(2200 / 2)
    nxim = int(2200 / 2)
    nw = 200  # 120
    nggl = ndata - nw - nxip - nxim

    delta_chi2_values[i] = {
        "3x2pt": chi2_3x2pt,  # Store the calculated value
        "shear": chi2_shear,
        "ggl": chi2_ggl,
        "2x2pt": chi2_2x2pt,
        "wtheta": chi2_wtheta
    }

# Output the results
for file, chi2_values in delta_chi2_values.items():
    print(f"Results for {file}:")
    for section, value in chi2_values.items():
        print(f"{section}: Delta chi2 = {value}")

# Append the results to a file
with open("delta_chi2_results.txt", "a") as file:
    for file_name, chi2_values in delta_chi2_values.items():
        file.write(f"Results for {file_name}:\n")
        for section, value in chi2_values.items():
            file.write(f"{section}: Delta chi2 = {value}\n")
        file.write("\n")
