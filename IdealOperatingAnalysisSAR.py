import cv2 as cv
from Useful_Tools.MATLAB_Importing import loadmat
from RF_Voltages import Calc_RF_Voltage
from Useful_Tools.General_Data_Vis import Save_Figs
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import cvxpy as cp
import sys
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer


SAR_Limit_Arr = [1,5,10,20,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
# SAR_Limit_Arr = [1,5,10]
# ****************************
ROI_xCOL_range_low=13
ROI_xCOL_range_high=23

ROI_yROW_range_low=25
ROI_yROW_range_high=35

ROI_z_slice=20


# generate realistic RF pulse
pulseDuration = 4 * (10 ** -3)  # s
sliceThickness = 42 * (10 ** -3)  # m
BWT = 8
Emp = 0.8
V_max = 114.265  # V
TR = 7.36 * (10 ** -3)  # s
# *********************************************************************
# load in MATLAB files
Adj_dictionary = loadmat('/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2021_02_19/AdjDataUser.mat')
Sys_dictionary = loadmat('/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2021_02_19/SysDataUser.mat')
Sar_dictionary = loadmat('/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2021_02_19/SarDataUser.mat')

S = np.array(Adj_dictionary['Adj']['S'])
txScaleFactor = np.array(Sys_dictionary['txScaleFactor'])
adjtra = float(Sys_dictionary['adjtra'])
ZZ = Sar_dictionary['ZZ']
ZZtype = Sar_dictionary['ZZtype']
umax = np.array(Sys_dictionary['umax'])

# get dimensions
matrixsize = int(math.sqrt(S.shape[0]))  # assuming square array replace with rows and columns, rename as N_ etc
coils = int((S.shape[1]))
slices = int((S.shape[2]))

# *********************************************************************
# get sar vops
SAR_VOP_Index = (np.where(ZZtype == 6))[0][:]  # find indicies of where in ZZ type the code = 6
SAR_VOP = np.take(ZZ, SAR_VOP_Index, axis=2)  # find corresponding VOP matrices

# *********************************************************************
# correct S which is [2809,8,52]
S_Corrected = S  # * txScaleFactor[None,:,None]
S_Corrected_Summed = np.sum(S_Corrected, axis=1)

# Cic pol, NOT for optimisation, only for plotting
S_Corrected_CP = S * txScaleFactor[None, :, None]
S_Corrected_Summed_CP = np.sum(S_Corrected_CP, axis=1)

# *********************************************************************
# CREATE ROI AROUND IAMS
# #limit S to an ROI around inner ear

# create blank arrays
image_coil = []
ROI_Square = []

# also get a test square ROI for each coil
for i in range(0, coils):
    # image for each coil
    image_coil.append((S_Corrected[:, i, :]).reshape((matrixsize, matrixsize, slices), order='C'))
    # ROI square, for each coil
    ROI_Square.append(
        (image_coil[i])[ROI_yROW_range_low:ROI_yROW_range_high, ROI_xCOL_range_low:ROI_xCOL_range_high,
        ROI_z_slice])

ROI_Square = np.array(ROI_Square)
ROI_Square_Stacked = ROI_Square.reshape((8, -1), order='F')
ROI_Square_Stacked = ROI_Square_Stacked.transpose()

Before_Av_Arr = []
After_Av_Arr = []
Before_Min_Arr = []
After_Min_Arr = []
Before_CoV_Arr = []
After_CoV_Arr = []
maxSAR_Arr = []
opt_Arr =[]
#set up plot

S_Corrected_Image_CP = S_Corrected_Summed_CP.reshape((matrixsize, matrixsize, slices), order='C')

fig, axs = plt.subplots(int((len(SAR_Limit_Arr)+1)/2), 2, figsize=(20, 20))



row = int((ROI_yROW_range_high + ROI_yROW_range_low)/2)
plt.subplot(int((len(SAR_Limit_Arr)+1)/2), 2, 1)
plt.imshow(np.abs(S_Corrected_Image_CP[:,:,ROI_z_slice]),vmin=0,vmax=0.1)
# plt.axhline(y=row,color='red')
# axs.add_patch(plt.patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
plt.title("Default CP")




# figLP, axsLP = plt.subplots(1, (len(SAR_Limit_Arr)+1), figsize=(10, 10))
# axsLP[0].imshow(np.abs(S_Corrected_Image_CP[:,:,ROI_z_slice]),vmin=0,vmax=0.1)
# #line profile through slice
# row_Image_CP =  np.abs(S_Corrected_Image_CP[row,:,ROI_z_slice])
# row_target = np.full(row_Image_CP.shape, 11.7/adjtra)
# axsLP[0].plot(row_Image_CP, 'k')



for i in range(0,len(SAR_Limit_Arr)):
    SAR_Limit = SAR_Limit_Arr[i]
    V_sq_av, AmpIntAv = Calc_RF_Voltage(pulseDuration, sliceThickness, BWT, Emp, V_max, TR)

    w = cp.Variable(8, complex=True)

    VOP = cp.Constant(SAR_VOP[:, :, 0])
    VOP1 = cp.Constant(SAR_VOP[:, :, 1])
    VOP2 = cp.Constant(SAR_VOP[:, :, 2])
    VOP3 = cp.Constant(SAR_VOP[:, :, 3])
    VOP4 = cp.Constant(SAR_VOP[:, :, 4])
    VOP5 = cp.Constant(SAR_VOP[:, :, 5])
    VOP6 = cp.Constant(SAR_VOP[:, :, 6])
    VOP7 = cp.Constant(SAR_VOP[:, :, 7])

    b = np.full((ROI_Square_Stacked[:, 1]).shape, 11.7 / adjtra)
    objective = cp.Minimize((cp.sum_squares(((ROI_Square_Stacked @ w) - b))))
    constraints = [(V_sq_av * cp.quad_form(w, VOP)) <= SAR_Limit, (V_sq_av * cp.quad_form(w, VOP1)) <= SAR_Limit,
                   (V_sq_av * cp.quad_form(w, VOP2)) <= SAR_Limit, (V_sq_av * cp.quad_form(w, VOP3)) <= SAR_Limit,
                   (V_sq_av * cp.quad_form(w, VOP4)) <= SAR_Limit, (V_sq_av * cp.quad_form(w, VOP5)) <= SAR_Limit,
                   (V_sq_av * cp.quad_form(w, VOP6)) <= SAR_Limit, (V_sq_av * cp.quad_form(w, VOP7)) <= SAR_Limit]
    # constraints = [(cp.quad_form(w,VOP)) <= SAR_Limit , (cp.quad_form(w,VOP1)) <= SAR_Limit , (cp.quad_form(w,VOP2)) <=SAR_Limit, (cp.quad_form(w,VOP3)) <= SAR_Limit, (cp.quad_form(w,VOP4)) <= SAR_Limit,  (cp.quad_form(w,VOP5)) <= SAR_Limit, (cp.quad_form(w,VOP6)) <= SAR_Limit, (cp.quad_form(w,VOP7)) <= SAR_Limit]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(w.value)
    w_opt = w.value

    # before and after average values in Square ROI
    ROI_summed_after = np.matmul(ROI_Square_Stacked, w_opt)
    ROI_summed_before = np.sum(ROI_Square_Stacked, axis=1)

    Before_Av = np.average(np.abs(ROI_summed_before))
    After_Av = np.average(np.abs(ROI_summed_after))

    # Before_Av_Arr.append(np.average(np.abs(ROI_summed_before)))
    After_Av_Arr.append(np.average(np.abs(ROI_summed_after)))
    # Before_Min_Arr.append(np.min(np.abs(ROI_summed_before)))
    After_Min_Arr.append(np.min(np.abs(ROI_summed_after)))
    # Before_CoV_Arr.append(Before_Av / ((np.std(np.abs(ROI_summed_before)))))
    After_CoV_Arr.append(((np.std(np.abs(ROI_summed_after))))/After_Av)
    opt_Arr.append(np.mean((np.matmul(ROI_Square_Stacked,w_opt) - b )**2))

    ShimCalcSAR = []
    # voltage check
    for j in range(1, 8):
        ShimCalcSAR.append((V_sq_av * (np.matmul(np.matmul((w_opt.conj().T), SAR_VOP[:, :, j]), w_opt))))
    maxSAR_Arr.append(np.max(np.array(ShimCalcSAR)))

#     # Get before image
#     S_Corrected_Image = S_Corrected_Summed.reshape((matrixsize, matrixsize, slices), order='C')
#
#     # Get optimised total image
#     S_Opt_Sum = np.sum((S_Corrected * w_opt[None, :, None]), axis=1)
#     ImageOpt = S_Opt_Sum.reshape((matrixsize, matrixsize, slices), order='C')  # 2D images slice by slice
#
#     # get optimised ROI image
#     ROI_Image_Opt = np.sum((ROI_Square * w_opt[:, None, None]), axis=0)
#     row_ImageOpt = np.abs(ImageOpt[row, :, ROI_z_slice])
#
#     plt.subplot(int((len(SAR_Limit_Arr)+1)/2), 2, (i+2) )
#     plt.imshow(np.abs(ImageOpt[:, :, ROI_z_slice]), vmin=0, vmax=0.1)
#     # axs.add_patch(
#     #     patches.Rectangle((ROI_xCOL_range_low, ROI_yROW_range_low), (ROI_xCOL_range_high - ROI_xCOL_range_low),
#     #                       (ROI_yROW_range_high - ROI_yROW_range_low), edgecolor="red", facecolor='none'))
#     plt.title("SAR limit W/kg " + str(SAR_Limit_Arr[i]))
#     plt.plot(row_ImageOpt, 'm')
#
# plt.tight_layout()
# plt.show()
# # plt.show()



fig, axs = plt.subplots(2, 2, figsize=(10, 10))

plt.subplot(2,2,1)


plt.plot(SAR_Limit_Arr,After_Av_Arr, 'r+', linestyle='--')
plt.hlines(b[0], xmin=0 , xmax=1000 ,label="target", color='blue')
plt.legend()
plt.title("Average in ROI against SAR Limit")
plt.xlabel('SAR limit W/kg')
plt.ylabel('Average in ROI (uT/V)')


plt.subplot(2,2,2)

plt.plot(SAR_Limit_Arr,After_Min_Arr, 'r+', linestyle='--')
plt.hlines(b[0], xmin=0 , xmax=1000 ,label="target", color='blue')
plt.legend()
plt.title("Minimum in ROI against SAR Limit")
plt.xlabel('SAR limit W/kg')
plt.ylabel('Minimum in ROI (uT/V)')


plt.subplot(2,2,3)

plt.plot(SAR_Limit_Arr,After_CoV_Arr, 'r+', linestyle='--')
plt.title("CoV in ROI against SAR Limit")
plt.xlabel('SAR limit W/kg')
plt.ylabel('CoV in ROI')


plt.subplot(2,2,4)
x=[0,500,1000]
y=[0,500,1000]
plt.plot(SAR_Limit_Arr,maxSAR_Arr, 'r+', linestyle='--')
plt.plot(x,y, color='blue', linestyle='--', label='y=x' )
plt.legend()
# plt.xlim(0,1000)
plt.title("Predicted SAR against SAR Limit")
plt.xlabel('SAR limit W/kg')
plt.ylabel('Predicted SAR W/kg')
plt.show()


plt.plot(SAR_Limit_Arr,opt_Arr, 'r+', linestyle='--')
plt.title("Optimisation target (|(|Sw-b|)|^2) against SAR Limit")
plt.xlabel('SAR limit W/kg')
plt.ylabel('Optimisation Target')
plt.show()







