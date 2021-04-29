from networkx.linalg.tests.test_algebraic_connectivity import scipy
from scipy import stats
from Useful_Tools.View_Dicom_Images import Get_Dicom_Array
from Useful_Tools.View_Dicom_Images import View_Dicom_Images, View_Dicom_Images_Colourbar, Get_Dicom_Array_Sorted
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as patches
import pydicom as dcm
import os

#load in images
#No DPs B1 maps
NO_DP_PreAngCor_MAP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/HeadNODPpreMAP/")
NO_DP_PostAngCor_MAP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/HeadNODPpostMAP/")
NO_DP_PostAngCor_anatom = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/HeadNODPpostAnat/")

#No DPs images
NO_DP_ir_fs_SPACE = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/NODP_Ax_t2_ir_fs_spc_isotropic_match3T/")
NO_DP_slabSel_SPACE_NM = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/NODP_Ax_t2_spc_slabsek_match3T_NM/")
NO_DP_slabSel_SPACE_FL = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/NODP_Ax_t2_spc_slabsek_match3T_FL/")
NO_DP_CISS = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/NODP_CISS/")

#DPs B1 maps
DP_PreAngCor_MAP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/HeadDPpreMAP/")
DP_PostAngCor_MAP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/HeadDPpostMAP/")
DP_PostAngCor_anatom = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/HeadDPpostAnat/")
# #DPs images
DP_ir_fs_SPACE = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/DP_Ax_t2_ir_fs_spc_isotropic_match3T/")
DP_slabSel_SPACE_NM = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/DP_Ax_t2_spc_slabsek_match3T_NM/")
DP_CISS = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/DPs_17_03_21/DPsCISS/")


#can use below to plot out all
# fig, axs = plt.subplots(6,5, figsize=(15, 15), facecolor='w', edgecolor='k')
# # fig.subplots_adjust(hspace = .5, wspace=.001)
# axs = axs.ravel()
# for i in range(0,30):
#     axs[i].imshow((NO_DP_PostAngCor_MAP[:,:,i]), cmap='gray')
#     # axs[i].set_title(str(250+i))
# plt.tight_layout()
# plt.show()
#
# #can use below to plot out all
# fig, axs = plt.subplots(6,5, figsize=(15, 15), facecolor='w', edgecolor='k')
# # fig.subplots_adjust(hspace = .5, wspace=.001)
# axs = axs.ravel()
# for i in range(0,30):
#     axs[i].imshow((DP_PostAngCor_MAP[:,:,i]), cmap='gray')
#     # axs[i].set_title(str(250+i))
# plt.tight_layout()
# plt.show()

# #*************************************************************************************************************************
# #B1 maps pre
# #slice 15 looks good
# row = 117 #mid way
# slice = 7
# dataset1 = NO_DP_PreAngCor_MAP
# dataset2 = DP_PreAngCor_MAP
# ROIdataset1 = 65
# ROI2dataset1 =185
# ROIdataset2 = 60
# ROI2dataset2 =190
#
#
# OneRowNODPpre = dataset1[row,:,slice]
# OneRowNODPpost = dataset2[row,:,slice]
#
# fig, axs = plt.subplots(2,2, figsize=(10, 10), facecolor='w', edgecolor='k')
#
# #plot data set1
# tempplt = axs[0,0].imshow((dataset1[:,:,slice]),vmin=0,vmax=1500)
# axs[0,0].axhline(y=row, color='red',ls='--')
# #mark on ROI
# axs[0,0].axvline(x=ROIdataset1, color='red',ls='--')
# axs[0,0].axvline(x=ROI2dataset1, color='red',ls='--')
#
#
# #plot dataset 2
# axs[0,1].imshow((dataset2[:,:,slice]),vmin=0,vmax=1500)
# axs[0,1].axhline(y=row, color='red',ls='--')
# #mark on ROI
# axs[0,1].axvline(x=ROIdataset2, color='red',ls='--')
# axs[0,1].axvline(x=ROI2dataset2, color='red',ls='--')
#
# #line plots
# axs[1,0].plot(OneRowNODPpre, label='line profile, shown in red on image')
# #mark on ROI
# axs[1,0].axvline(x=ROIdataset1, color='red',ls='--')
# axs[1,0].axvline(x=ROI2dataset1, color='red',ls='--')
#
# axs[1,1].plot(OneRowNODPpost, label='line profile, shown in red on image')
# #mark on ROI
# axs[1,1].axvline(x=ROIdataset2, color='red',ls='--')
# axs[1,1].axvline(x=ROI2dataset2, color='red',ls='--')
#
# plt.legend()
#
# #formatting
# axs[0,0].set_title('No DPs B1 map Pre Voltage Adjustment')
# axs[0,1].set_title('DPs B1 map Pre Voltage Adjustment')
# axs[1,0].set_xlim([55,190])
# axs[1,1].set_xlim([55,195])
# # cbar_ax = fig.add_axes([0.1, 0.05, 0.5, 0.05]) #((left, bottom, width, height)
# # fig.colorbar(tempplt, cax=cbar_ax, orientation='vertical')
#
# plt.show()


# # B1 maps post
# # slice 15 looks good
# row = 117 #mid way
# slice = 7
# dataset1 = NO_DP_PostAngCor_MAP
# dataset2 = DP_PostAngCor_MAP
# ROIdataset1 = 65
# ROI2dataset1 =185
# ROIdataset2 = 60
# ROI2dataset2 =190
#
# OneRowNODPpre = dataset1[row,:,slice]
# OneRowNODPpost = dataset2[row,:,slice]
#
# fig, axs = plt.subplots(2,2, figsize=(10, 10), facecolor='w', edgecolor='k')
#
# #plot data set1
# tempplt = axs[0,0].imshow((dataset1[:,:,slice]),vmin=0,vmax=1500)
# axs[0,0].axhline(y=row, color='red',ls='--')
# #mark on ROI
# axs[0,0].axvline(x=ROIdataset1, color='red',ls='--')
# axs[0,0].axvline(x=ROI2dataset1, color='red',ls='--')
#
#
# #plot dataset 2
# axs[0,1].imshow((dataset2[:,:,slice]),vmin=0,vmax=1500)
# axs[0,1].axhline(y=row, color='red',ls='--')
# #mark on ROI
# axs[0,1].axvline(x=ROIdataset2, color='red',ls='--')
# axs[0,1].axvline(x=ROI2dataset2, color='red',ls='--')
#
# #line plots
# axs[1,0].plot(OneRowNODPpre, label='line profile, shown in red on image')
# #mark on ROI
# axs[1,0].axvline(x=ROIdataset1, color='red',ls='--')
# axs[1,0].axvline(x=ROI2dataset1, color='red',ls='--')
#
# axs[1,1].plot(OneRowNODPpost, label='line profile, shown in red on image')
# #mark on ROI
# axs[1,1].axvline(x=ROIdataset2, color='red',ls='--')
# axs[1,1].axvline(x=ROI2dataset2, color='red',ls='--')
#
# plt.legend()
#
# #formatting
# axs[0,0].set_title('No DPs B1 map Post Voltage Adjustment')
# axs[0,1].set_title('DPs B1 map Post Voltage Adjustment')
# axs[1,0].set_xlim([55,190])
# axs[1,1].set_xlim([55,195])
# # cbar_ax = fig.add_axes([0.1, 0.05, 0.5, 0.05]) #((left, bottom, width, height)
# # fig.colorbar(tempplt, cax=cbar_ax, orientation='vertical')
#
# plt.show()



#and also histograms for b1 maps
rowDPs = 117 #mid way
rownoDPs = 119
slice = 7
dataset1 = NO_DP_PostAngCor_MAP
dataset2 = DP_PostAngCor_MAP


#get one slice
B1DPslice = dataset2[:,:,:]/10
B1noDPslice = dataset1[:,:,:]/10

#inner ear region
DPsIE=dataset2[113:123,54:64,6:8]/10
noDPsIE= dataset1[112:122,58:68,6:8]/10







# no dps

no_DP_ant_mask = NO_DP_PostAngCor_anatom > 800

nodps_masked = np.multiply(no_DP_ant_mask , NO_DP_PostAngCor_MAP)
nodps_masked = nodps_masked[:,:,4:28]


fig=plt.figure(figsize=(30, 30))
columns = 10
rows = 4
im = nodps_masked
ax=[]
for i in range(0,(im.shape[2])):
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].axis('off')
    plt.imshow(np.abs(im[:,:,i]), vmin=0, vmax=1800)
plt.show()


#dps


DP_ant_mask = DP_PostAngCor_anatom > 600

dps_masked = np.multiply(DP_ant_mask , DP_PostAngCor_MAP)
dps_masked = dps_masked[:,:,3:27]


fig=plt.figure(figsize=(30, 30))
columns = 10
rows = 4
im = dps_masked
ax=[]
for i in range(0,(im.shape[2])):
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].axis('off')
    plt.imshow(np.abs(im[:,:,i]), vmin=0, vmax=1800)
plt.show()



#flatten
B1DP = dps_masked.flatten()/10
B1noDP = nodps_masked.flatten()/10
DPsIEflat = DPsIE.flatten()
noDPsIEflat = noDPsIE.flatten()

#plot histograms
plt.hist(B1DP, bins = 200,  facecolor='blue')
plt.xlabel('Flip Angle')
plt.ylabel('Number of pixels')
plt.xlim(30,160)
plt.ylim(0,10000)
plt.title(r'Histogram for B1+ map with Dielectric Pads, whole map')
plt.show()
plt.hist(B1noDP, bins = 200,  facecolor='blue')
plt.xlim(30,160)
plt.ylim(0,10000)
plt.xlabel('Flip Angle')
plt.ylabel('Number of pixels')
plt.title(r'Histogram for B1+ map without Dielectric Pads, whole map')
plt.show()


# #and histograms for inner ear
# plt.hist(DPsIEflat, facecolor='blue')
# plt.xlabel('Flip Angle')
# plt.xlim(25,175)
# plt.ylabel('Number of pixels')
# plt.title(r'Histogram for B1+ map with Dielectric Pads, inner ear')
# plt.show()
#
# plt.hist(noDPsIEflat, facecolor='blue' )
# plt.xlim(25,175)
# plt.xlabel('Flip Angle')
# plt.ylabel('Number of pixels')
# plt.title(r'Histogram for B1+ map without Dielectric Pads, inner ear')
# plt.show()





# B1 maps post
# plot out
slice = 7
dataset1 = NO_DP_PostAngCor_MAP/10
dataset2 = DP_PostAngCor_MAP/10



# fig, axs = plt.subplots(1,2, figsize=(10, 10), facecolor='w', edgecolor='k')
# #plot data set1
# tempplt = axs[0].imshow((dataset1[:,:,slice]),vmin=0,vmax=1500, cmap = 'gray')
# axs[0].axis('off')
# #plot dataset 2
# axs[1].imshow((dataset2[:,:,slice]),vmin=0,vmax=1500, cmap = 'gray')
# axs[1].axis('off')
# #formatting
# axs[0].set_title('No DPs B1 map Post Voltage Adjustment')
# axs[1].set_title('DPs B1 map Post Voltage Adjustment')
# cbar_ax = fig.add_axes([0.1, 0.05, 0.5, 0.05]) #((left, bottom, width, height)
# fig.colorbar(tempplt, cax=cbar_ax, orientation='horizontal')
# plt.tight_layout()
# plt.show()



# #plot b1 maps
# fig, axs = plt.subplots(1, 3, figsize=(10,7),gridspec_kw={'width_ratios': [1, 1, 0.1]})
# # fig.suptitle('Position 1, Axial, Central Slice', x=0.5 ,y=0.99, fontsize='xx-large' )
# # fig.delaxes(axs[1,0])
# #top row, B1 maps
# P1_NO_DPim = axs[0].imshow(dataset1[:,:,slice], vmin=0, vmax=150)
# P1_LR_DPim = axs[1].imshow(dataset2[:,:,slice], vmin=0, vmax=150)
#
#
#
#
# axs[0].set_title("No DPs")
# axs[1].set_title("DPs")
#
#
# cbar=fig.colorbar(P1_NO_DPim, cax=axs[2])
# cbar.set_label("Flip angle (degrees)")
#
# plt.show()

# #*************************************************************************************************************************
# # ax space slab sel
# #slice143 looks good
#
# row1 = 270
# row2 = 265
# slice = 60
# dataset1 = NO_DP_slabSel_SPACE_NM
# dataset2 = DP_slabSel_SPACE_NM
# ROIdataset1 = 30
# ROI2dataset1 = 480
# ROIdataset2 = 30
# ROI2dataset2 =480
#
#
# OneRowNODPpre = dataset1[row1,:,slice]
# OneRowNODPpost = dataset2[row2,:,slice]
#
# fig, axs = plt.subplots(2,2, figsize=(10, 10), facecolor='w', edgecolor='k')
#
# #plot data set1
# tempplt = axs[0,0].imshow((dataset1[:,:,slice]),vmin=0,vmax=2400, cmap='gray')
# axs[0,0].axhline(y=row1, color='red',ls='--')
# #mark on ROI
# axs[0,0].axvline(x=ROIdataset1, color='blue',ls='--')
# axs[0,0].axvline(x=ROI2dataset1, color='blue',ls='--')
#
#
# #plot dataset 2
# axs[0,1].imshow((dataset2[:,:,slice]),vmin=0,vmax=2400, cmap='gray')
# axs[0,1].axhline(y=row2, color='red',ls='--')
# #mark on ROI
# axs[0,1].axvline(x=ROIdataset2, color='blue',ls='--')
# axs[0,1].axvline(x=ROI2dataset2, color='blue',ls='--')
#
# #line plots
# axs[1,0].plot(OneRowNODPpre, label='line profile, shown in red on image', color='red')
# #mark on ROI
# axs[1,0].axvline(x=ROIdataset1, color='blue',ls='--')
# axs[1,0].axvline(x=ROI2dataset1, color='blue',ls='--')
#
# axs[1,1].plot(OneRowNODPpost, label='line profile, shown in red on image', color='red')
# #mark on ROI
# axs[1,1].axvline(x=ROIdataset2, color='blue',ls='--')
# axs[1,1].axvline(x=ROI2dataset2, color='blue',ls='--')
#
# plt.legend()
#
# #formatting
# axs[0,0].set_title('No DPs SPACE Slab Selective')
# axs[0,1].set_title('DPs SPACE Slab Selective')
#
# # cbar_ax = fig.add_axes([0.1, 0.05, 0.5, 0.05]) #((left, bottom, width, height)
# # fig.colorbar(tempplt, cax=cbar_ax, orientation='vertical')
#
# plt.show()






# #*************************************************************************************************************************
# # ax space FS IR SPACE
# #slice143 looks good
#
# row = 167
# slice = 143
# dataset1 = NO_DP_ir_fs_SPACE
# dataset2 = DP_ir_fs_SPACE
# ROIdataset1 = 35
# ROI2dataset1 =230
# ROIdataset2 = 30
# ROI2dataset2 =235
#
#
# OneRowNODPpre = dataset1[row,:,slice]
# OneRowNODPpost = dataset2[row,:,slice]
#
# fig, axs = plt.subplots(2,2, figsize=(10, 10), facecolor='w', edgecolor='k')
#
# #plot data set1
# tempplt = axs[0,0].imshow((dataset1[:,:,slice]),vmin=2000,vmax=2400, cmap='gray')
# axs[0,0].axhline(y=row, color='red',ls='--')
# #mark on ROI
# axs[0,0].axvline(x=ROIdataset1, color='blue',ls='--')
# axs[0,0].axvline(x=ROI2dataset1, color='blue',ls='--')
#
#
# #plot dataset 2
# axs[0,1].imshow((dataset2[:,:,slice]),vmin=2000,vmax=2400, cmap='gray')
# axs[0,1].axhline(y=row, color='red',ls='--')
# #mark on ROI
# axs[0,1].axvline(x=ROIdataset2, color='blue',ls='--')
# axs[0,1].axvline(x=ROI2dataset2, color='blue',ls='--')
#
# #line plots
# axs[1,0].plot(OneRowNODPpre, label='line profile, shown in red on image', color='red')
# #mark on ROI
# axs[1,0].axvline(x=ROIdataset1, color='blue',ls='--')
# axs[1,0].axvline(x=ROI2dataset1, color='blue',ls='--')
#
# axs[1,1].plot(OneRowNODPpost, label='line profile, shown in red on image', color='red')
# #mark on ROI
# axs[1,1].axvline(x=ROIdataset2, color='blue',ls='--')
# axs[1,1].axvline(x=ROI2dataset2, color='blue',ls='--')
#
# plt.legend()
#
# #formatting
# axs[0,0].set_title('No DPs SPACE Fs IR')
# axs[0,1].set_title('DPs SPACE Fs IR')
#
# # cbar_ax = fig.add_axes([0.1, 0.05, 0.5, 0.05]) #((left, bottom, width, height)
# # fig.colorbar(tempplt, cax=cbar_ax, orientation='vertical')
#
# plt.show()



# # #*************************************************************************************************************************
# # CISS
#
# row1 = 270
# row2 = 265
# slice = 47
# dataset1 = NO_DP_CISS
# dataset2 = DP_CISS
# ROInoDPs = dataset1[265:275,30:40,slice]
# ROIDPs = dataset2[260:270,25:35,slice]
#
# CISSmeanNoDPs = np.mean(ROInoDPs)
# CISSmeanDPs = np.mean(ROIDPs)
#
# CISSstdevNoDPs = np.std(ROInoDPs)
# CISSstdevDPs = np.std(ROIDPs)
#
# CISSmaxNoDPs = np.max(ROInoDPs)
# CISSmaxDPs = np.max(ROIDPs)
#
# CISSminNoDPs = np.min(ROInoDPs)
# CISSminDPs = np.min(ROIDPs)
#
# CISSttest= scipy.stats.ttest_rel(ROInoDPs,ROIDPs)
#
#
#
# row1 = 270
# row2 = 265
# slice = 47
# dataset1 = NO_DP_CISS
# dataset2 = DP_CISS
# ROIdataset1 = 35
# ROI2dataset1 = 480
# ROIdataset2 = 30
# ROI2dataset2 =470
#
#
# OneRowNODPpre = dataset1[row1,:,slice]
# OneRowNODPpost = dataset2[row2,:,slice]
#
# fig, axs = plt.subplots(2,2, figsize=(10, 10), facecolor='w', edgecolor='k')
#
# #plot data set1
# tempplt = axs[0,0].imshow((dataset1[:,:,slice]),vmin=0,vmax=1000, cmap='gray')
# axs[0,0].axhline(y=row1, color='red',ls='--')
# #mark on ROI
# axs[0,0].axvline(x=ROIdataset1, color='blue',ls='--')
# axs[0,0].axvline(x=ROI2dataset1, color='blue',ls='--')
#
#
# #plot dataset 2
# axs[0,1].imshow((dataset2[:,:,slice]),vmin=0,vmax=1000, cmap='gray')
# axs[0,1].axhline(y=row2, color='red',ls='--')
# #mark on ROI
# axs[0,1].axvline(x=ROIdataset2, color='blue',ls='--')
# axs[0,1].axvline(x=ROI2dataset2, color='blue',ls='--')
#
# #line plots
# axs[1,0].plot(OneRowNODPpre, label='line profile, shown in red on image', color='red')
# #mark on ROI
# axs[1,0].axvline(x=ROIdataset1, color='blue',ls='--')
# axs[1,0].axvline(x=ROI2dataset1, color='blue',ls='--')
#
# axs[1,1].plot(OneRowNODPpost, label='line profile, shown in red on image', color='red')
# #mark on ROI
# axs[1,1].axvline(x=ROIdataset2, color='blue',ls='--')
# axs[1,1].axvline(x=ROI2dataset2, color='blue',ls='--')
#
# plt.legend()
#
# #formatting
# axs[0,0].set_title('No DPs CISS')
# axs[0,1].set_title('DPs CISS')
#
# # cbar_ax = fig.add_axes([0.1, 0.05, 0.5, 0.05]) #((left, bottom, width, height)
# # fig.colorbar(tempplt, cax=cbar_ax, orientation='vertical')
#
# plt.show()


