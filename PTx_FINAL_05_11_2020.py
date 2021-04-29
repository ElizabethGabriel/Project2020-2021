#simple square PTx for use on 04/11/2020

import cv2 as cv
from Useful_Tools.MATLAB_Importing import loadmat
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

#*********************************************************************
#load in MATLAB files
Adj_dictionary = loadmat('/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/MATS/AdjDataUser.mat')
Sys_dictionary = loadmat('//Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/MATS/SysDataUser.mat')
Sar_dictionary = loadmat('/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/MATS/SarDataUser.mat')

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

S_Corrected_CP = S * txScaleFactor[None,:,None]
S_Corrected_Summed_CP = np.sum(S_Corrected_CP, axis=1)

##test to check tx scale factor being applied correctly
# S_Corrected_loop=np.zeros((2809,8,52), dtype='complex')
# for i in range(0,coils):
#     S_Corrected_loop[:,i,:]=S[:,i,:] * txScaleFactor[i]


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

#get a cube roughly over the IAMs region for each coil
#also get a test square ROI for each coil
for i in range(0,coils):
    #image for each coil
    image_coil.append((S_Corrected[:,i,:]).reshape((matrixsize,matrixsize,slices), order ='C'))
    #ROI square, for each coil
    ROI_Square.append((image_coil[i])[30:40,30:40,43])

ROI_Square=np.array(ROI_Square)
ROI_Square_Stacked = ROI_Square.reshape((8,-1), order='F')
ROI_Square_Stacked = ROI_Square_Stacked.transpose()

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

SAR_Limit = 1

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
constraints = [(cp.quad_form(w,VOP)) <= SAR_Limit , (cp.quad_form(w,VOP1)) <= SAR_Limit , (cp.quad_form(w,VOP2)) <=SAR_Limit, (cp.quad_form(w,VOP3)) <= SAR_Limit, (cp.quad_form(w,VOP4)) <= SAR_Limit,  (cp.quad_form(w,VOP5)) <= SAR_Limit, (cp.quad_form(w,VOP6)) <= SAR_Limit, (cp.quad_form(w,VOP7)) <= SAR_Limit]
prob = cp.Problem(objective, constraints)
result = prob.solve()
print(w.value)
w_opt = w.value



ShimCalcSAR=[]
#voltage check
for i in range(1,8):
    ShimCalcSAR.append(((np.matmul(np.matmul((w_opt.conj().T) , SAR_VOP[:,:,i]) , w_opt))))

#before and after average values in Square ROI
ROI_summed_after = np.matmul(ROI_Square_Stacked, w_opt)
ROI_summed_before = np.sum(ROI_Square_Stacked, axis =1)

Before_Av = np.average(np.abs(ROI_summed_before))
After_Av = np.average(np.abs(ROI_summed_after))
Before_Min = np.min(np.abs(ROI_summed_before))
After_Min = np.min(np.abs(ROI_summed_after))
Before_CoV = Before_Av/((np.std(np.abs(ROI_summed_before))))
After_CoV = After_Av/((np.std(np.abs(ROI_summed_after))))

#Get before image
S_Corrected_Image = S_Corrected_Summed.reshape((matrixsize,matrixsize,slices), order = 'C')
S_Corrected_Image_CP = S_Corrected_Summed_CP.reshape((matrixsize,matrixsize,slices), order = 'C')

#Get optimised total image
S_Opt_Sum = np.sum((S_Corrected * w_opt[None,:,None]), axis=1)
ImageOpt = S_Opt_Sum.reshape((matrixsize,matrixsize,slices), order = 'C') #2D images slice by slice

#get optimised ROI image
ROI_Image_Opt = np.sum((ROI_Square * w_opt[:,None,None]), axis=0)


# # # same plot for both, quad, target, solution, line profiles through, argand diagram of drives (see seminal paper)
# # #
# # # plot it out
# # # https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
# #

# #plot out before and after on same plot
# fig=plt.figure(figsize=(30, 30))
# columns = 10
# rows = 10
# ax=[]
#
# for i in range(0,slices):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(S_Corrected_Image[:,:,i]),vmin=0,vmax=0.1)
# # ax[43].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
# plt.figtext(0.5,0.98,"original", ha="center", va="top", fontsize=40, color="r")
# for j in range(slices+4, slices*2 +4):
#     ax.append(fig.add_subplot(rows, columns, j+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(ImageOpt[:,:,j-slices-4]),vmin=0,vmax=0.1)
# plt.figtext(0.5,0.5,"optimised", ha="center", va="top", fontsize=40, color="r")
# # ax[95].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
# plt.show()


# #plot it out
# plt.figure(figsize=(15,15))
# for i in range(0,slices):
#     plt.subplot(int(math.sqrt(slices)+1),int(math.sqrt(slices)+1),(i+1))
#     plt.imshow(np.abs(S_Corrected_Image[:,:,i]),vmin=0,vmax=0.1)
#     plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(15,15))
# for i in range(0,slices):
#     plt.subplot(int(math.sqrt(slices)+1),int(math.sqrt(slices)+1),(i+1))
#     plt.imshow(np.abs(ImageOpt[:,:,i]),vmin=0,vmax=0.1)
#     plt.tight_layout()
# plt.show()
#

# #ROI plot out before and after on same plot
# fig=plt.figure(figsize=(30, 30))
# columns = 8
# rows = 8
# ax=[]
#
# for i in range(0,ROI_Cube_Summed.shape[2]):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(ROI_Cube_Summed[:,:,i]),vmin=0,vmax=0.1)
# # ax[43].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
# plt.figtext(0.5,0.98,"original", ha="center", va="top", fontsize=40, color="r")
# for j in range(ROI_Cube_Summed.shape[2] + 1, (ROI_Cube_Summed.shape[2])*2 + 1):
#     ax.append(fig.add_subplot(rows, columns, j+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(ROI_Image_Opt[:,:,j-(ROI_Cube_Summed.shape[2])-1]),vmin=0,vmax=0.1)
# plt.figtext(0.5,0.5,"optimised", ha="center", va="top", fontsize=40, color="r")
# # ax[95].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
# plt.show()
#
#
# #zoom in before, target, after on one slice
# fig, axs = plt.subplots(2, 3, figsize=(10, 10))
# axs[0,0].imshow(np.abs(S_Corrected_Image[:,:,43]),vmin=0,vmax=0.1)
# axs[0,1].imshow(np.full((S_Corrected_Image[:,:,43]).shape, 11.7/adjtra),vmin=0,vmax=0.1)
# axs[0,2].imshow(np.abs(ImageOpt[:,:,43]),vmin=0,vmax=0.1)
#
# axs[0,0].set_title("pre opt")
# axs[0,1].set_title("target value")
# axs[0,2].set_title("post opt")
#
# axs[0,0].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
# axs[0,1].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
# axs[0,2].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
#
# #line profile through slice
# # row_Image = np.abs(S_Un_Stacked_Image[35,:,43]) #checked it was this way round using distinctive row 52
# # row_ImageOpt  = np.abs(ImageOpt[35,:,43])
# row_Image = np.abs(S_Corrected_Image[:,35,43]) #checked it was this way round using distinctive row 52
# row_ImageOpt  = np.abs(ImageOpt[:,35,43])
# row_target = np.full(row_Image.shape, 11.7/adjtra)
#
# axs[1,2].plot(row_ImageOpt)
# axs[1,0].plot(row_Image)
# axs[1,1].plot(row_ImageOpt, label='optimised image')
# axs[1,1].plot(row_Image, label='before')
# axs[1,1].plot(row_target, label='target')
# plt.legend()
# axs[1,1].axvline(x=30, color='red',ls='--')
# axs[1,1].axvline(x=40, color='red', ls='--')
# plt.show()



#zoom in before, target, after on one slice
#****************************
ROI_xCOL_range_low=30
ROI_xCOL_range_high=40

ROI_yROW_range_low=30
ROI_yROW_range_high=40

ROI_z_slice=43


fig, axs = plt.subplots(1, 4, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
plt1=axs[0].imshow(np.abs(S_Corrected_Image_CP[:,:,ROI_z_slice]),vmin=0,vmax=0.1)
axs[1].imshow(np.full((S_Corrected_Image[:,:,ROI_z_slice]).shape, 11.7/adjtra),vmin=0,vmax=0.1)
axs[2].imshow(np.abs(ImageOpt[:,:,ROI_z_slice]),vmin=0,vmax=0.1)

axs[0].set_title("pre opt")
axs[1].set_title("target value")
axs[2].set_title("post opt")


axs[0].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
axs[1].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))
axs[2].add_patch(patches.Rectangle((ROI_xCOL_range_low,ROI_yROW_range_low), (ROI_xCOL_range_high-ROI_xCOL_range_low), (ROI_yROW_range_high-ROI_yROW_range_low), edgecolor="red", facecolor='none'))

#line profile through slice
row = int((ROI_yROW_range_high + ROI_yROW_range_low)/2)
row_Image = np.abs(S_Corrected_Image[row,:,ROI_z_slice]) #checked it was this way round using distinctive row 52
row_ImageOpt  = np.abs(ImageOpt[row,:,ROI_z_slice])
row_Image_CP =  np.abs(S_Corrected_Image_CP[row,:,ROI_z_slice])
row_target = np.full(row_Image.shape, 11.7/adjtra)


axs[0].axhline(y=row,color='red')
axs[2].axhline(y=row,color='red')


# axs[1,1].plot(row_ImageOpt, 'm', label='optimised image')
# axs[1,1].plot(row_Image_CP, 'k', label='before')
# axs[1,1].plot(row_target, label='target')
# axs[1,1].legend()
# axs[1,1].set_xlabel("pixel")
# axs[1,1].set_ylabel("B1+ (uT/V)")

# axs[1,1].axvline(x=ROI_xCOL_range_low, color='red',ls='--')
# axs[1,1].axvline(x=ROI_xCOL_range_high, color='red', ls='--')

cbar=fig.colorbar(plt1, cax=axs[3])
cbar.set_label(u"\u03bcT/V")

# axs[1,0].axis("off")
# axs[1,2].axis("off")
# axs[1,3].axis("off")


plt.show()


fig = plt.figure()
ax = plt.axes()
ax.plot(row_ImageOpt, 'm', label='optimised image')
ax.plot(row_Image_CP, 'k', label='before')
ax.plot(row_target, label='target')
ax.legend()
ax.set_xlabel("pixel")
ax.set_ylabel("B1+ (uT/V)")

ax.axvline(x=ROI_xCOL_range_low, color='red',ls='--')
ax.axvline(x=ROI_xCOL_range_high, color='red', ls='--')
plt.show()


#NOW PHASE INFO
#zoom in before, target, after on one slice


fig, axs = plt.subplots(1, 4, figsize=(10,5 ), gridspec_kw={'width_ratios': [1, 1,1, 0.1]})
plt2=axs[0].imshow(np.angle(S_Corrected_Image[:,:,43]), vmin=-3.14159, vmax=3.14159)
axs[2].imshow(np.angle(ImageOpt[:,:,43]), vmin=-3.14159, vmax=3.14159)
axs[1].imshow(np.full((S_Corrected_Image[:,:,ROI_z_slice]).shape, 0),vmin=-3.14159, vmax=3.14159)

axs[0].set_title("pre opt")
axs[1].set_title("target")
axs[2].set_title("post opt")

axs[0].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
axs[1].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))
axs[2].add_patch(patches.Rectangle((30,30), 10, 10, edgecolor="red", facecolor='none'))

axs[0].axhline(y=row,color='red')
axs[2].axhline(y=row,color='red')


a=np.angle(S_Corrected_Image[:,:,43])


# divider = make_axes_locatable(ax)
# axs[2] = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(plt2,fraction=0.046, pad=0.04)

cbar = fig.colorbar(plt2, cax=axs[3], shrink=0.46)
cbar.set_label("Radians")

plt.show()



row_ImageOptph  = np.angle(ImageOpt[row,:,ROI_z_slice])
row_Image_CPph =  np.angle(S_Corrected_Image[row,:,ROI_z_slice])
row_targetph = np.full(row_Image.shape, 0)


fig = plt.figure()
ax = plt.axes()
ax.plot(row_ImageOptph, 'm', label='optimised image')
ax.plot(row_Image_CPph, 'k', label='before')
ax.plot(row_targetph, label='target')
ax.legend()
ax.set_xlabel("pixel")
ax.set_ylabel("Phase (radians)")

ax.axvline(x=ROI_xCOL_range_low, color='red',ls='--')
ax.axvline(x=ROI_xCOL_range_high, color='red', ls='--')
plt.show()


# #argand diagram.....
# for x in w_opt:
#     plt.polar([np.angle(w_opt)], [np.abs(w_opt)], marker='o')
# plt.show()

