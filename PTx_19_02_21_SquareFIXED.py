
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


#****************************
#params used on day

# ROI_xCOL_range_low=13
# ROI_xCOL_range_high=23
#
# ROI_yROW_range_low=25
# ROI_yROW_range_high=35
#
# ROI_z_slice=20

ROI_xCOL_range_low=13
ROI_xCOL_range_high=23

ROI_yROW_range_low=25
ROI_yROW_range_high=35

ROI_z_slice=20



#if cube...
# ROI_z_range_low = 35
# ROI_z_range_high = 45

#constraint params
SAR_Limit = 5

#generate realistic RF pulse
pulseDuration = 4 * (10 ** -3)  # s
sliceThickness = 42 * (10 ** -3)  # m
BWT = 8
Emp = 0.8
V_max = 114.265  #114.265  # V
TR = 7.36 * (10 ** -3)  # s
#*********************************************************************
#load in MATLAB files
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
matrixsize = int(math.sqrt(S.shape[0])) #assuming square array replace with rows and columns, rename as N_ etc
coils = int((S.shape[1]))
slices = int((S.shape[2]))

#*********************************************************************
#get sar vops
SAR_VOP_Index = (np.where(ZZtype == 6))[0][:] #find indicies of where in ZZ type the code = 6
SAR_VOP = np.take(ZZ, SAR_VOP_Index, axis=2) #find corresponding VOP matrices

#*********************************************************************
#correct S which is [2809,8,52]
S_Corrected = S #* txScaleFactor[None,:,None]
S_Corrected_Summed = np.sum(S_Corrected, axis=1)


#Cic pol, NOT for optimisation, only for plotting
S_Corrected_CP = S * txScaleFactor[None,:,None]
S_Corrected_Summed_CP = np.sum(S_Corrected_CP, axis=1)


# #*********************************************
#run this to figure out ROI params
S_Corrected_Image = S_Corrected_Summed.reshape((matrixsize,matrixsize,slices), order = 'C')
# #plot out before and after on same plot
# fig=plt.figure(figsize=(30, 30))
# columns = 10
# rows = 10
#
# ax=[]
# for i in range(0,slices):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(S_Corrected_Image[:,:,i]),vmin=0,vmax=0.025)
# ax[ROI_z_slice].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
# plt.show()



# #test to check tx scale factor being applied correctly
# S_Corrected_loop=np.zeros((2809,8,80), dtype='complex')
# for i in range(0,coils):
#     S_Corrected_loop[:,i,:]= S[:,i,:] * txScaleFactor[i]

# # test to show applying txscale factor messes up phase info
# a = (S[1000,1,40] * txScaleFactor[1]) + (S[1000,2,40] * txScaleFactor[2]) + (S[1000,3,40] * txScaleFactor[3]) + (S[1000,4,40] * txScaleFactor[4]) + (S[1000,5,40] * txScaleFactor[5]) + (S[1000,6,40] * txScaleFactor[6]) +(S[1000,7,40] * txScaleFactor[7]) + (S[1000,0,40] * txScaleFactor[0])
# phase_a = np.angle(a)

# #test summing being done as expected
# Sum_Loop=np.zeros((matrixsize*matrixsize,slices), dtype = complex)
# for j in range(0,(matrixsize*matrixsize)):
#     for i in range(0,slices):
#         Sum_Loop[j,i] = S_Corrected[j,0,i] + S_Corrected[j,1,i] + S_Corrected[j,2,i] + S_Corrected[j,3,i] + S_Corrected[j,4,i] + S_Corrected[j,5,i] +S_Corrected[j,6,i] +S_Corrected[j,7,i]
# diff = Sum_Loop - S_Corrected_Summed


#*********************************************************************
#CREATE ROI AROUND IAMS
# #limit S to an ROI around inner ear

#create blank arrays
image_coil=[]
ROI_Square=[]


#also get a test square ROI for each coil
for i in range(0,coils):
    #image for each coil
    image_coil.append((S_Corrected[:,i,:]).reshape((matrixsize,matrixsize,slices), order ='C'))
    #ROI square, for each coil
    ROI_Square.append((image_coil[i])[ROI_yROW_range_low:ROI_yROW_range_high,ROI_xCOL_range_low:ROI_xCOL_range_high,ROI_z_slice])


ROI_Square=np.array(ROI_Square)
ROI_Square_Stacked = ROI_Square.reshape((8,-1), order='F')
ROI_Square_Stacked = ROI_Square_Stacked.transpose()

# #to check ROI position
# image_coil = np.array(image_coil)
# for i in range(0,coils):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 10))
#     axs[0].imshow(np.abs(image_coil[i,:,:,ROI_z_slice]), vmin=0, vmax=0.1)
#     axs[1].imshow(np.abs(ROI_Square[i,:,:]), vmin=0, vmax=0.1)
#     plt.show()



# #•••••••••••••••••••••••••••••••••••••
# #ROI tests
# #sanity check to see is ROI Coil and STacked ROI coil have same S maps for each coil
# should_be_zeros =[]
# for i in range(0,coils):
#     diff = np.amax(ROI_Square[i,:,:]) - np.amax(ROI_Square_Stacked[i,:])
#     should_be_zeros.append(diff)
# should_be_zero = np.amax(should_be_zeros)
# #seems all good!

#•••••••••••••••••••••••••••••••••••••
# optimisation!

# AIM - value of 11.7/adjtra (target T/V) for that data set is the aim in the ROI
# b = is target T/V in ROI
# minimise this ||Sw - b||^2 subject to for each vop: w^H * VOP * w < SAR local limit and subject to for each wi: abs(wi) < umax
# https://www.cvxpy.org   one step at a time!

V_sq_av, AmpIntAv = Calc_RF_Voltage(pulseDuration, sliceThickness, BWT, Emp, V_max, TR)


w = cp.Variable(8, complex=True)

VOP = cp.Constant(SAR_VOP[:,:,0])
VOP1 = cp.Constant(SAR_VOP[:,:,1])
VOP2 = cp.Constant(SAR_VOP[:,:,2])
VOP3 = cp.Constant(SAR_VOP[:,:,3])
VOP4 = cp.Constant(SAR_VOP[:,:,4])
VOP5 = cp.Constant(SAR_VOP[:,:,5])
VOP6 = cp.Constant(SAR_VOP[:,:,6])
VOP7 = cp.Constant(SAR_VOP[:,:,7])



b = np.full((ROI_Square_Stacked[:,1]).shape, 11.7/adjtra)
objective = cp.Minimize(cp.sum_squares(((ROI_Square_Stacked @ w) - b)))
constraints = [(V_sq_av * cp.quad_form(w,VOP)) <= SAR_Limit , (V_sq_av * cp.quad_form(w,VOP1)) <= SAR_Limit , (V_sq_av * cp.quad_form(w,VOP2)) <=SAR_Limit, (V_sq_av * cp.quad_form(w,VOP3)) <= SAR_Limit, (V_sq_av * cp.quad_form(w,VOP4)) <= SAR_Limit,  (V_sq_av * cp.quad_form(w,VOP5)) <= SAR_Limit, (V_sq_av * cp.quad_form(w,VOP6)) <= SAR_Limit, (V_sq_av * cp.quad_form(w,VOP7)) <= SAR_Limit]
# constraints = [(cp.quad_form(w,VOP)) <= SAR_Limit , (cp.quad_form(w,VOP1)) <= SAR_Limit , (cp.quad_form(w,VOP2)) <=SAR_Limit, (cp.quad_form(w,VOP3)) <= SAR_Limit, (cp.quad_form(w,VOP4)) <= SAR_Limit,  (cp.quad_form(w,VOP5)) <= SAR_Limit, (cp.quad_form(w,VOP6)) <= SAR_Limit, (cp.quad_form(w,VOP7)) <= SAR_Limit]
prob = cp.Problem(objective, constraints)
result = prob.solve()
print(w.value)
w_opt = w.value

#get w_opt angle and phase
w_opt_abs = np.abs(w_opt)
w_opt_ph = (np.angle(w_opt) * 180) / np.pi

#before and after average values in Square ROI
ROI_summed_after = np.matmul(ROI_Square_Stacked, w_opt)
# ROI_summed_before = np.sum(ROI_Square_Stacked, axis =1)
ROI_summed_before = np.matmul(ROI_Square_Stacked, txScaleFactor)

Before_Av_CP = np.average(np.abs(ROI_summed_before))
After_Av = np.average(np.abs(ROI_summed_after))
Before_Min_CP = np.min(np.abs(ROI_summed_before))
After_Min = np.min(np.abs(ROI_summed_after))
Before_CoV = Before_Av_CP/((np.std(np.abs(ROI_summed_before))))
After_CoV = After_Av/((np.std(np.abs(ROI_summed_after))))

ShimCalcSAR=[]
#voltage check
for i in range(1,8):
    ShimCalcSAR.append((V_sq_av * (np.matmul(np.matmul((w_opt.conj().T) , SAR_VOP[:,:,i]) , w_opt))))
print(ShimCalcSAR)





#Get before image
S_Corrected_Image = S_Corrected_Summed.reshape((matrixsize,matrixsize,slices), order = 'C')
S_Corrected_Image_CP = S_Corrected_Summed_CP.reshape((matrixsize,matrixsize,slices), order = 'C')

#Get optimised total image
S_Opt_Sum = np.sum((S_Corrected * w_opt[None,:,None]), axis=1)
ImageOpt = S_Opt_Sum.reshape((matrixsize,matrixsize,slices), order = 'C') #2D images slice by slice

#get optimised ROI image
ROI_Image_Opt = np.sum((ROI_Square * w_opt[:,None,None]), axis=0)


# same plot for both, quad, target, solution, line profiles through, argand diagram of drives (see paper)
#
# plot it out
# https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645


#plot out before and after
# fig=plt.figure(figsize=(30, 30))
# columns = 10
# rows = 10
# ax=[]
#
# for i in range(0,slices):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(S_Corrected_Image_CP[:,:,i]),vmin=0,vmax=0.1)
# plt.figtext(0.5,0.98,"original", ha="center", va="top", fontsize=40, color="r")
# ax[ROI_z_slice].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
# plt.show()
#
# fig=plt.figure(figsize=(30, 30))
# columns = 10
# rows = 10
# ax=[]
# for j in range(0, slices):
#     ax.append(fig.add_subplot(rows, columns, j+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(ImageOpt[:,:,j]),vmin=0,vmax=0.1)
# plt.figtext(0.5,0.98,"optimised", ha="center", va="top", fontsize=40, color="r")
# ax[ROI_z_slice].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
# plt.show()


#zoom in before, target, after on one slice
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
axs[0,0].imshow(np.abs(S_Corrected_Image_CP[:,:,ROI_z_slice]),vmin=0,vmax=0.1)
axs[0,1].imshow(np.full((S_Corrected_Image[:,:,ROI_z_slice]).shape, 11.7/adjtra),vmin=0,vmax=0.1)
axs[0,2].imshow(np.abs(ImageOpt[:,:,ROI_z_slice]),vmin=0,vmax=0.1)

axs[0,0].set_title("pre opt")
axs[0,1].set_title("target value")
axs[0,2].set_title("post opt")


axs[0,0].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
axs[0,1].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
axs[0,2].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))

#line profile through slice
row = int((ROI_yROW_range_high + ROI_yROW_range_low)/2)
row_Image = np.abs(S_Corrected_Image[row,:,ROI_z_slice]) #checked it was this way round using distinctive row 52
row_ImageOpt  = np.abs(ImageOpt[row,:,ROI_z_slice])
row_Image_CP =  np.abs(S_Corrected_Image_CP[row,:,ROI_z_slice])
row_target = np.full(row_Image.shape, 11.7/adjtra)


axs[0,0].axhline(y=row,color='red')
axs[0,2].axhline(y=row,color='red')

axs[1,2].plot(row_ImageOpt, 'm', label='optimised image')
axs[1,0].plot(row_Image_CP, 'k', label='pre optimised image')
axs[1,1].plot(row_ImageOpt, 'm', label='optimised image')
axs[1,1].plot(row_Image_CP, 'k', label='before')
axs[1,1].plot(row_target, label='target')

axs[1,0].legend(loc='upper right')
axs[1,1].legend()
axs[1,2].legend(loc='upper right')

axs[1,0].set_xlabel('Pixel')
axs[1,1].set_xlabel('Pixel')
axs[1,2].set_xlabel('Pixel')

axs[1,0].set_ylabel('B1+ (uT/V)')
axs[1,1].set_ylabel('B1+ (uT/V)')
axs[1,2].set_ylabel('B1+ (uT/V)')


axs[1,1].axvline(x=ROI_xCOL_range_low, color='red',ls='--')
axs[1,1].axvline(x=ROI_xCOL_range_high, color='red', ls='--')
plt.show()




#NOW PHASE INFO
#zoom in before, target, after on one slice
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
axs[0].imshow(np.angle(S_Corrected_Image[:,:,ROI_z_slice]), vmin=-3.14159, vmax=3.14159)
axs[1].imshow(np.angle(ImageOpt[:,:,ROI_z_slice]), vmin=-3.14159, vmax=3.14159)

axs[0].set_title("pre opt phase")
axs[1].set_title("post opt phase")

axs[0].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
axs[1].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))

a=np.angle(S_Corrected_Image[:,:,ROI_z_slice])

plt.show()


#argand diagram.....
for x in w_opt:
    plt.polar([np.angle(w_opt)], [np.abs(w_opt)], marker='o')
plt.show()



# post op - units?
# fig, axs = plt.subplots(1, 2, figsize=(10, 10))
# ImageOptANGLE = (ImageOpt * 2.675 * (10**8) * (10**-6) * 184.8 * (10**-3)) * (180/np.pi)
# rowImageOptANGLE = np.abs(ImageOptANGLE[25,:,ROI_z_slice])
# axs[0].imshow(np.abs(ImageOptANGLE[:,:,ROI_z_slice]))
# axs[1].plot(rowImageOptANGLE)
# plt.show()


# # VOPS
# fig=plt.figure(figsize=(30, 30))
# columns = 4
# rows = 2
# ax=[]
#
# for i in range(0,7):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(SAR_VOP[:,:,i]))
# plt.show()
#
