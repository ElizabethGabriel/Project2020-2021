from Useful_Tools.View_Dicom_Images import View_Dicom_Images
from Useful_Tools.View_Dicom_Images import Get_Dicom_Array

from Useful_Tools.View_Dicom_Images import Get_Dicom_Array_Sorted
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import pydicom as dcm
from Dicom_Server_Tools import Dicom
import os
from PTx_Tools.PTx_Square_ROI import PTx_Square
#PTx check
#Load in prediction and compare with output


S_Corrected_Image, Before_Av, After_Av, Before_Min, After_Min, Before_CoV, After_CoV, w_opt, ImageOpt, adjtra = PTx_Square('/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/MATS/AdjDataUser.mat','/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/MATS/SysDataUser.mat','/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/MATS/SarDataUser.mat')

# #0029  series PTx weights
View_Dicom_Images("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0029/",0,15)
DicomArray = Get_Dicom_Array("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0029/",0,15)


DicomArraySorted = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0029/")


DicomArray = DicomArray/10

transmitV = adjtra
#getting conversion factor

V1 = 241.0551
uVs1 = 120527
# k = V1/uVs1

V2 = 250
uVs2 = 90649

x = np.array([V1,V2])
y = np.array([uVs1,uVs2])

def best_fit_slope_and_intercept(xs, ys):
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) /
         ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))

    b = np.mean(ys) - m * np.mean(xs)

    return m, b

m,c = best_fit_slope_and_intercept(x, y)

convFact = ( m * transmitV ) + c

FAImOpt = ImageOpt * (10**-6) * 267 * (10**6) * convFact * (10**-6) * 2 * np.pi

# try normalisation

ImageOptNorm = np.abs(ImageOpt) / np.max(np.abs(ImageOpt))
DicomArrayNorm = np.abs(DicomArray)/ np.max(np.abs(DicomArray))


#plot both out

# plt.figure(figsize=(15,15))
# for i in range(0,80):
#     plt.subplot(int(math.sqrt(80)+1),int(math.sqrt(80)+1),(i+1))
#     plt.imshow(np.abs(FAImOpt[:,:,i]),vmin=0,vmax=150)
#     plt.tight_layout()
# plt.show()


# fig=plt.figure(figsize=(15,15))
# ax = plt.axes()
# for i in range(0,4):
#     plt.subplot(int(math.sqrt(40)+1),int(math.sqrt(40)+1),(i+1))
#     plt1=plt.imshow(np.abs(DicomArraySorted[:,:,i]/10),vmin=0,vmax=150)
#     plt.tight_layout()
# cbar=fig.colorbar(plt1, cax=ax)
# plt.show()

# #plot out before and after
# fig=plt.figure(figsize=(10, 10))
# columns = 8
# rows = 4
# ax=[]
#
# for i in range(8,32):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt1=plt.imshow(np.abs(DicomArraySorted[:,:,i]/10),vmin=0,vmax=150)
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.15, 0.85, 0.7, 0.05])
# cbar=fig.colorbar(plt1, cax=cbar_ax, shrink=0.5, orientation="horizontal")
# cbar.set_label("Flip Angle (degrees)")
# plt.show()




#horizontal line plot comparison
# zoom in before, target, after on one slice
rowexp = 175

fig, axs = plt.subplots(2, 4, figsize=(10, 10), gridspec_kw={'width_ratios': [1, 0.1,1, 0.1]})
plt2=axs[0,0].imshow(np.abs(FAImOpt[:, :, 43]) , vmin=0, vmax=30)
plt4=axs[0,2].imshow(DicomArray , vmin=0, vmax=150)

axs[0,0].set_title("prediction")
axs[0,2].set_title("experimental")

axs[0,0].add_patch(patches.Rectangle((30, 30), 10, 10, edgecolor="red", facecolor='none'))
# axs[1].add_patch(patches.Rectangle((30, 30), 10, 10, edgecolor="red", facecolor='none'))

# # line profile through slice
row_Dicom_Image = DicomArray[rowexp, :]
row_ImageOpt = np.abs(FAImOpt[35, :, 43])
# row_target = np.full(row_Image.shape, 11.7 / adjtra)

axs[1,0].plot(row_ImageOpt)
axs[1,2].plot(row_Dicom_Image)
axs[1,0].plot(row_ImageOpt, label='prediction')
axs[1,2].plot(row_Dicom_Image, label='experimental')

axs[1,0].set_ylim(0,20)
axs[1,2].set_ylim(0,100)

axs[1,2].set_xlim([35,210])

axs[1,0].legend(loc='upper right')
axs[1,2].legend()

axs[1,0].set_xlabel('Pixel')
axs[1,2].set_xlabel('Pixel')


axs[1,0].set_ylabel('B1+ (Flip Angle)')
axs[1,2].set_ylabel('B1+ (Flip Angle)')


axs[1,0].legend()
axs[1,2].legend()

axs[0,0].axhline(y=35, color='red',ls='--')
axs[0,2].axhline(y=rowexp, color='red', ls='--')


axs[1,1].axis("off")
axs[1,3].axis("off")

cbar = fig.colorbar(plt4, cax=axs[0,3], shrink=0.46)
cbar.set_label("degrees")

cbar = fig.colorbar(plt2, cax=axs[0,1], shrink=0.46)
cbar.set_label("degrees")


plt.show()



#vertical line plot comparison

# zoom in before, target, after on one slice
fig, axs = plt.subplots(2, 4, figsize=(10, 10), gridspec_kw={'width_ratios': [1,0.1, 1, 0.1]})
plt5=axs[0,0].imshow(np.abs(FAImOpt[:, :, 43]), vmin=0, vmax=30)
plt6=axs[0,2].imshow(DicomArray , vmin=0, vmax=150)

axs[0,0].set_title("prediction")
axs[0,2].set_title("experimental")

axs[0,0].add_patch(patches.Rectangle((30, 30), 10, 10, edgecolor="red", facecolor='none'))
# axs[1].add_patch(patches.Rectangle((30, 30), 10, 10, edgecolor="red", facecolor='none'))

# # line profile through slice
col_Dicom_Image = DicomArray[:, 165]
col_ImageOpt = np.abs(FAImOpt[:, 35, 43])
# row_target = np.full(row_Image.shape, 11.7 / adjtra)

axs[1,0].plot(col_ImageOpt)
axs[1,2].plot(col_Dicom_Image)

axs[1,0].plot(col_ImageOpt, label='prediction')
axs[1,2].plot(col_Dicom_Image, label='experimental')
axs[1,0].legend()
axs[1,2].legend()

axs[1,0].set_xlabel('Pixel')
axs[1,2].set_xlabel('Pixel')

axs[1,0].set_ylabel('B1+ (Flip Angle)')
axs[1,2].set_ylabel('B1+ (Flip Angle)')


axs[1,2].set_xlim([50,240])

axs[0,0].axvline(x=35, color='red',ls='--')
axs[0,2].axvline(x=165, color='red', ls='--')

axs[1,1].axis("off")
axs[1,3].axis("off")

cbar = fig.colorbar(plt5, cax=axs[0,1], shrink=0.46)
cbar.set_label("degrees")
cbar = fig.colorbar(plt6, cax=axs[0,3], shrink=0.46)
cbar.set_label("degrees")

plt.show()




PTxHist  =(np.abs(FAImOpt[20:40,5:45,32:40])).flatten()
ExpHist = (np.abs(DicomArraySorted[100:200,90:200,16:22])).flatten()


plt.hist(PTxHist, bins = 200,  facecolor='blue')
plt.xlabel('Flip Angle')
plt.ylabel('Number of pixels')
plt.xlim(2,196)
# plt.ylim(0,1000)
plt.title(r'PTx')
plt.show()






plt.hist(ExpHist/10, bins = 200,  facecolor='blue')
plt.xlabel('Flip Angle')
plt.ylabel('Number of pixels')
plt.xlim(2,196)
# plt.ylim(0,21000)
plt.title(r'Experimental')
plt.show()








## 0028 series PTx weights anatomical
# Dicom_Array = []
# Slice_Pos = []
# for i in range(0,39):
#     View_Dicom_Images()
#     one_slice_array = Get_Dicom_Array("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0028/",0,i)
#     # Dicom_Array.append(one_slice_array)
#     # ds = dcm.read_file("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0028/",0,i)
#     # Slice_Pos.append(ds.SliceLocation)


# #0029  series PTx weights
# # #
# path = "/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0029/"
# for i in range(0,39):
#     for f in os.listdir(path):
#         temp = Dicom(path + f)
#         print(temp)
#         ds = dcm.read_file(temp)
#     View_Dicom_Images(path,0,i)


# View_Dicom_Images("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0029/",0,15)
# array = Get_Dicom_Array("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0029/",0,15)

# for i in range(0, 40):
#     View_Dicom_Images("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0029/",0, i)

# ds = dcm.read_file("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_RFSHIM_CHANGE_TEST_0029/2020_11_05_EXP2.MR.RESEARCH_LG_MENIERES.0029.0006.2020.11.05.16.58.57.463572.27645831.IMA")
# print(ds)
#

