from Useful_Tools.View_Dicom_Images import View_Dicom_Images, View_Dicom_Images_Colourbar, Get_Dicom_Array_Sorted
import matplotlib.pyplot as plt
import numpy as np

#for P1...
#load in B1 maps
P1_NO_DP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_NODP_P1_0007/")
P1_LR_DP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_DP_LR_P1_0004/")
P1_Inf_DP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_DP_LRINFERIOR_P1_0010/")

#
# #load in anatomical images
P1_NO_DP_ant = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_NODP_P1_0006/")
P1_LR_DP_ant = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_DP_LR_P1_0003/")
P1_Inf_DP_ant = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_DP_LRINFERIOR_P1_0009/")

#ratio B1maps
RatioDPLRP1 = np.divide(P1_LR_DP , P1_NO_DP)
RatioDPinfP1 = np.divide(P1_Inf_DP , P1_NO_DP)

#for P2...
#load in B1 maps
P2_NO_DP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_NODP_P2_0017/")
P2_LR_DP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_DP_LR_P2_0014/")
P2_Inf_DP = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_DP_LRINFERIOR_P2_0020/")

# #load in anatomical images
P2_NO_DP_ant = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_NODP_P2_0016/")
P2_LR_DP_ant = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_DP_LR_P2_0013/")
P2_Inf_DP_ant = Get_Dicom_Array_Sorted("/Users/lizgabriel/PycharmProjects/PTx/Data/menieres_2020_11_05/IMAGE_AND_B1_MAP_DATA/B1_MAPPING_SINGLE_DP_LRINFERIOR_P2_0019/")

#ratio B1maps
RatioDPLRP2 = np.divide(P2_LR_DP , P2_NO_DP)
RatioDPinfP2 = np.divide(P2_Inf_DP , P2_NO_DP)



#masking background
P1_NO_DP_ant_mask = P1_NO_DP_ant > 300
P1_NO_DP_ant_mask2 = P1_NO_DP_ant > 500
P1_NO_DP_masked = np.multiply(P1_NO_DP_ant_mask , P1_NO_DP)
P1_NO_DP_masked[:,:,3] = np.multiply(P1_NO_DP_ant_mask2[:,:,3], P1_NO_DP[:,:,3])
P1_NO_DP_masked = P1_NO_DP_masked[:,:,3:30] #remove  edge slices


P1_LR_DP_ant_mask = P1_LR_DP_ant > 250
P1_LR_DP_ant_mask2 = P1_LR_DP_ant > 500
P1_LR_DP_masked = np.multiply(P1_LR_DP_ant_mask , P1_LR_DP)
P1_LR_DP_masked[:,:,0:4] = np.multiply(P1_LR_DP_ant_mask2[:,:,0:4], P1_LR_DP[:,:,0:4])
P1_LR_DP_masked = P1_LR_DP_masked[:,:,3:30] #remove high value edge slices


P2_NO_DP_ant_mask = P2_NO_DP_ant > 250
P2_NO_DP_ant_mask2 = P2_NO_DP_ant > 500
P2_NO_DP_masked = np.multiply(P2_NO_DP_ant_mask , P2_NO_DP)
P2_NO_DP_masked[:,:,0:4] = np.multiply(P2_NO_DP_ant_mask2[:,:,0:4], P2_NO_DP[:,:,0:4])
P2_NO_DP_masked = P2_NO_DP_masked[:,:,:26] #remove high value edge slices

P2_LR_DP_ant_mask = P2_LR_DP_ant > 250
P2_LR_DP_ant_mask2 = P2_LR_DP_ant > 500
P2_LR_DP_masked = np.multiply(P2_LR_DP_ant_mask , P2_LR_DP)
P2_LR_DP_masked[:,:,0:3] = np.multiply(P2_LR_DP_ant_mask2[:,:,0:3], P2_LR_DP[:,:,0:3])
P2_LR_DP_masked = P2_LR_DP_masked[:,:,:26] #remove high value edge slices

# fig=plt.figure(figsize=(30, 30))
# columns = 10
# rows = 4
# im = P1_NO_DP_masked
# ax=[]
# for i in range(0,(im.shape[2])):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(im[:,:,i]), vmin=0, vmax=1800)
# plt.show()
#
# fig=plt.figure(figsize=(30, 30))
# columns = 10
# rows = 4
# im = P2_NO_DP_masked
# ax=[]
# for i in range(0,(im.shape[2])):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(im[:,:,i]) , vmin=0, vmax=1800)
# plt.show()
#
# fig=plt.figure(figsize=(30, 30))
# columns = 10
# rows = 4
# im = P1_LR_DP_masked
# ax=[]
# for i in range(0,(im.shape[2])):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(im[:,:,i]),  vmin=0, vmax=1800)
# plt.show()
#
# fig=plt.figure(figsize=(30, 30))
# columns = 10
# rows = 4
# im = P2_LR_DP_masked
# ax=[]
# for i in range(0,(im.shape[2])):
#     ax.append(fig.add_subplot(rows, columns, i+1))
#     ax[-1].axis('off')
#     plt.imshow(np.abs(im[:,:,i]),  vmin=0, vmax=1800)
# plt.show()



#histograms
#P1
# axcentralsliceP1 = 16
DPOneSliceP1 = P1_LR_DP_masked[:,:,:]/10
noDPOneSliceP1 = P1_NO_DP_masked[:,:,:]/10


#P2
# axcentralsliceP2 = 12
DPOneSliceP2 = P2_LR_DP_masked[:,:,:]/10
noDPOneSliceP2 = P2_NO_DP_masked[:,:,:]/10

#flatten
DPP1 = DPOneSliceP1.flatten()
noDPP1 = noDPOneSliceP1.flatten()
DPP2 = DPOneSliceP2.flatten()
noDPP2 = noDPOneSliceP2.flatten()

#median
medDPP1 = np.median(DPP1[DPP1 != 0])
mednoDPP1 = np.median(noDPP1[noDPP1 != 0])
medDPP2 = np.median(DPP2[DPP2 != 0])
mednoDPP2 = np.median(noDPP2[noDPP2 != 0])

#plot histograms
plt.rcParams.update({'font.size': 17})
plt.hist(DPP1, bins = 200,  facecolor='blue')
plt.xlabel('Flip Angle')
plt.ylabel('Number of pixels')
plt.xlim(2,150)
plt.ylim(0,10000)
plt.title(r'DPs P1')
plt.show()

plt.hist(noDPP1, bins = 200,  facecolor='blue')
plt.xlim(2,150)
plt.ylim(0,10000)
plt.xlabel('Flip Angle')
plt.ylabel('Number of pixels')
plt.title(r'no DPs P1')
plt.show()

plt.hist(DPP2, bins = 200,  facecolor='blue')
plt.xlabel('Flip Angle')
plt.ylabel('Number of pixels')
plt.xlim(2,150)
plt.ylim(0,10000)
plt.title(r'DPs P2')
plt.show()

plt.hist(noDPP2, bins = 200,  facecolor='blue')
plt.xlim(2,150)
plt.ylim(0,10000)
plt.xlabel('Flip Angle')
plt.ylabel('Number of pixels')
plt.title(r'no DPs P2')
plt.show()





# #***************************************************
# #axial
# # ****************
# # Line plots
#
#
axcentralsliceP1 = 16
ax_row_line_profile = 140
ax_col_line_profile = 125
#
# # #inferior
# # axcentralsliceP1 = 4
# # ax_row_line_profile = 150
# # ax_col_line_profile = 125
#
# #rows and cols
# DPOneSliceP1 = P1_LR_DP[:,:,axcentralsliceP1]
# DPOneRowP1 = DPOneSliceP1[ax_row_line_profile,:]
# DPOneColP1 = DPOneSliceP1[:,ax_col_line_profile]
#
# noDPOneSliceP1 = P1_NO_DP[:,:,axcentralsliceP1]
# noDPOneRowP1 = noDPOneSliceP1[ax_row_line_profile,:]
# noDPOneColP1 = noDPOneSliceP1[:,ax_col_line_profile]
#
# infDPOneSliceP1 = P1_Inf_DP[:,:,axcentralsliceP1]
# infDPOneRowP1 = infDPOneSliceP1[ax_row_line_profile,:]
# infDPOneColP1 = infDPOneSliceP1[:,ax_col_line_profile]
#
# RatioDPLRP1OneSlice = RatioDPLRP1[:,:,axcentralsliceP1]
# RatioDPLRP1Row=RatioDPLRP1OneSlice[ax_row_line_profile,:]
# RatioDPLRP1Col=RatioDPLRP1OneSlice[:,ax_col_line_profile]
#
# RatioDPinfP1OneSlice = RatioDPinfP1[:,:,axcentralsliceP1]
# RatioDPinfP1Row=RatioDPinfP1OneSlice[ax_row_line_profile,:]
# RatioDPinfP1Col=RatioDPinfP1OneSlice[:,ax_col_line_profile]
#
#
#
#
# #****************************************************
# #axial P2
# #****************
# #Line plots
#
#
axcentralsliceP2 = 12
ax_row_line_profileP2 = 140
#
# # #inferior
# # axcentralsliceP2 = 1
# # ax_row_line_profileP2 = 150
#
# #rows and cols
# DPOneSlice = P2_LR_DP[:,:,axcentralsliceP2]
# DPOneRow = DPOneSlice[ax_row_line_profileP2,:]
#
# noDPOneSlice = P2_NO_DP[:,:,axcentralsliceP2]
# noDPOneRow = noDPOneSlice[ax_row_line_profileP2,:]
#
# infDPOneSlice = P2_Inf_DP[:,:,axcentralsliceP2]
# infDPOneRow = infDPOneSlice[ax_row_line_profileP2,:]
#
# RatioDPLRP2OneSlice = RatioDPLRP2[:,:,axcentralsliceP2]
# RatioDPLRP2Row=RatioDPLRP2OneSlice[ax_row_line_profileP2,:]
#
# RatioDPinfP2OneSlice = RatioDPinfP2[:,:,axcentralsliceP2]
# RatioDPinfP2Row=RatioDPinfP2OneSlice[ax_row_line_profileP2,:]
#
#
#
# # #*******
# #for B1+
# #set up figure
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# fig.suptitle('Line Plots of B1+ for an Axial Inferior Slice', y=1)
#
# #label figs
# axs[0].set_title("P1", pad=0)
# axs[1].set_title("P2")
#
# #plot lines
# axs[0].plot(DPOneRowP1/10, 'g--', label='DPs Left and Right')
# axs[0].plot(noDPOneRowP1/10, 'r--', label='No DPs')
# axs[0].plot(infDPOneRowP1/10, 'b--', label='DPs inferior')
# axs[0].legend()
#
# # limit axises
# axs[0].set_xlim([50,205])
#
# # #inferior
# # axs[0].set_xlim([90,165])
#
#
# #label axis
# axs[0].set_ylabel('B1+ (Flip Angle, degrees)')
# axs[0].set_xlabel('Pixels')
#
#
#
# #plot lines
# axs[1].plot(DPOneRow/10, 'g--', label='DPs Left and Right')
# axs[1].plot(noDPOneRow/10, 'r--', label='No DPs')
# axs[1].plot(infDPOneRow/10, 'b--', label='DPs inferior')
# axs[1].legend()
#
#
# # limit axises
# axs[1].set_xlim([50,205])
#
# #inferior
# # axs[1].set_xlim([90,165])
#
# #label axis
# axs[1].set_ylabel('B1+ (Flip Angle, degrees)')
# axs[1].set_xlabel('Pixels')
#
# # plt.subplots_adjust(top= 0.7, bottom = 0)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#
# plt.show()







# #*******
# #repeat line plots for ratio images
# #set up figure
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# fig.suptitle('Line Plots of Ratio Images for an Axial Inferior Slice', y=1)
#
# #label figs
# axs[0].set_title("P1", pad=0)
# axs[1].set_title("P2")
#
# #plot lines
# axs[0].plot(RatioDPLRP1Row, 'g--', label='Ratio DPs Left and Right/No DPs')
# axs[0].plot(RatioDPinfP1Row, 'r--', label='Ratio DPs Inferior/No DPs')
# axs[0].legend()
#
#
#
# # #limit axises
# # axs[0].set_xlim([50,205])
# # axs[0].set_ylim([0,2])
#
#
# # inferior
# axs[0].set_xlim([90,165])
# # axs[1].set_xlim([110,185])
# axs[0].set_ylim([0,3])
#
# #label axis
# axs[0].set_ylabel('Ratio')
# axs[0].set_xlabel('Pixels')
#
#
# axs[1].plot(RatioDPLRP2Row, 'g--', label='Ratio DPs Left and Right/No DPs')
# axs[1].plot(RatioDPinfP2Row, 'r--', label='Ratio DPs Inferior/No DPs')
# axs[1].legend()
#
#
# # limit axises
# # axs[1].set_xlim([50,205])
# # # axs[1].set_xlim([70,225])
# # axs[1].set_ylim([0,2])
#
# # #inferior
# axs[1].set_xlim([95,165])
# # axs[1].set_xlim([100,180])
# axs[1].set_ylim([0,3])
# # axs[1].set_ylim([0,3])
#
# #label axis
# axs[1].set_ylabel('Ratio')
# axs[1].set_xlabel('Pixels')
#
#
# plt.show()






#********************************************
#B1 and ratio plots
#central slice
#set up figure

P1_NO_DP=P1_NO_DP/10
P1_LR_DP=P1_LR_DP/10
P1_Inf_DP=P1_Inf_DP/10

fig, axs = plt.subplots(2, 4, figsize=(10,7),gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
fig.suptitle('Position 1, Axial, Central Slice', x=0.5 ,y=0.99, fontsize='xx-large' )
fig.delaxes(axs[1,0])
#top row, B1 maps
P1_NO_DPim = axs[0,0].imshow(P1_NO_DP[:,:,axcentralsliceP1], vmin=0, vmax=150)
P1_LR_DPim = axs[0,1].imshow(P1_LR_DP[:,:,axcentralsliceP1], vmin=0, vmax=150)
P1_Inf_DPim = axs[0,2].imshow(P1_Inf_DP[:,:,axcentralsliceP1], vmin=0, vmax=150)

#lines
axs[0,0].axhline(y=ax_row_line_profile, color='red',ls='--' )
# axs[0,0].axvline(x=ax_col_line_profile, color='red',ls='--')
axs[0,1].axhline(y=ax_row_line_profile, color='red',ls='--', clip_box='L1' )
axs[0,2].axhline(y=ax_row_line_profile, color='red',ls='--', clip_box='L1' )
axs[1,1].axhline(y=ax_row_line_profile, color='red',ls='--', clip_box='L1' )
axs[1,2].axhline(y=ax_row_line_profile, color='red',ls='--', clip_box='L1' )


axs[0,0].set_title("No DPs")
axs[0,1].set_title("DPs Left and Right")
axs[0,2].set_title("DPs Inferior Left and Right")

cbar=fig.colorbar(P1_Inf_DPim, cax=axs[0,3])
cbar.set_label("degrees")
#bottom row, ratio images
RatioDPLRP1im = axs[1,1].imshow((RatioDPLRP1[:,:,axcentralsliceP1]),vmin=0.5,vmax=1.8)
RatioDPinfP1im= axs[1,2].imshow((RatioDPinfP1[:,:,axcentralsliceP1]),vmin=0.5,vmax=1.8)

axs[1,1].set_title('Ratio DPs LR/No DPs')
axs[1,2].set_title('Ratio of DPs Inf/No DPs')

cbar=fig.colorbar(RatioDPinfP1im, cax=axs[1,3])
cbar.set_label("ratio")

plt.figtext(0.4,0.51,'B1+ Maps',fontsize=15)
plt.figtext(0.4,0.02,'Ratio Images',fontsize=15)

plt.show()
#
#
# #****************************************************
# #cor P1
#
# #central slice
# #set up figure
#
# centralsliceP1 = 125
#
# fig, axs = plt.subplots(2, 4, figsize=(10,8),gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
# fig.suptitle('Position 1, Coronal, Central Slice', x=0.5 ,y=0.99, fontsize='xx-large' )
# fig.delaxes(axs[1,0])
# #top row, B1 maps
# P1_NO_DPim = axs[0,0].imshow(P1_NO_DP[centralsliceP1,:,:], vmin=0, vmax=150, aspect=0.156)
# P1_LR_DPim = axs[0,1].imshow(P1_LR_DP[centralsliceP1,:,:], vmin=0, vmax=150, aspect=0.156)
# P1_Inf_DPim = axs[0,2].imshow(P1_Inf_DP[centralsliceP1,:,:], vmin=0, vmax=150, aspect=0.156)
#
# axs[0,0].set_title("No DPs")
# axs[0,1].set_title("DPs Left and Right")
# axs[0,2].set_title("DPs Inferior Left and Right")
#
# cbar=fig.colorbar(P1_Inf_DPim, cax=axs[0,3])
# cbar.set_label("degrees")
#
# #bottom row, ratio images
# RatioDPLRP1im = axs[1,1].imshow((RatioDPLRP1[centralsliceP1,:,:]),vmin=0.5,vmax=1.9, aspect=0.156)
# RatioDPinfP1im= axs[1,2].imshow((RatioDPinfP1[centralsliceP1,:,:]),vmin=0.5,vmax=1.9, aspect=0.156)
#
# axs[1,1].set_title('Ratio DPs LR/No DPs')
# axs[1,2].set_title('Ratio of DPs Inf/No DPs')
#
# cbar=fig.colorbar(RatioDPinfP1im, cax=axs[1,3])
# cbar.set_label("ratio")
#
# fig.subplots_adjust(hspace=0.8)
#
# plt.figtext(0.4,0.51,'B1+ Maps',fontsize=15)
# plt.figtext(0.4,0.02,'Ratio Images',fontsize=15)
#
# plt.subplots_adjust(top = 0.5, bottom=0.01, hspace=1.5, wspace=0.4)
# plt.show()
#
#
# # #****************************************************
# # #sag P1
# #
# # #central slice
# # #set up figure
# #
# # centralsliceP1 = 75
# #
# # fig, axs = plt.subplots(2, 4, figsize=(10,8),gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
# # fig.suptitle('Position 1, Sagital, Slice near one DP', x=0.5 ,y=0.99, fontsize='xx-large' )
# # fig.delaxes(axs[1,0])
# # #top row, B1 maps
# # P1_NO_DPim = axs[0,0].imshow(P1_NO_DP[:,centralsliceP1,:], vmin=0, vmax=1500, aspect=0.156)
# # P1_LR_DPim = axs[0,1].imshow(P1_LR_DP[:,centralsliceP1,:], vmin=0, vmax=1500, aspect=0.156)
# # P1_Inf_DPim = axs[0,2].imshow(P1_Inf_DP[:,centralsliceP1,:], vmin=0, vmax=1500, aspect=0.156)
# #
# # axs[0,0].set_title("No DPs")
# # axs[0,1].set_title("DPs Left and Right")
# # axs[0,2].set_title("DPs Inferior Left and Right")
# #
# # fig.colorbar(P1_Inf_DPim, cax=axs[0,3])
# #
# # #bottom row, ratio images
# # RatioDPLRP1im = axs[1,1].imshow((RatioDPLRP1[:,centralsliceP1,:]),vmin=0.5,vmax=1.5, aspect=0.156)
# # RatioDPinfP1im= axs[1,2].imshow((RatioDPinfP1[:,centralsliceP1,:]),vmin=0.5,vmax=1.5, aspect=0.156)
# #
# # axs[1,1].set_title('Ratio DPs LR/No DPs')
# # axs[1,2].set_title('Ratio of DPs Inf/No DPs')
# #
# # fig.colorbar(RatioDPinfP1im, cax=axs[1,3])
# #
# # fig.subplots_adjust(hspace=0.8)
# #
# # plt.figtext(0.4,0.51,'B1+ Maps',fontsize=15)
# # plt.figtext(0.4,0.02,'Ratio Images',fontsize=15)
# #
# # # fig.tight_layout(pad=5.0)
# # plt.subplots_adjust(top = 0.5, bottom=0.01, hspace=1.5, wspace=0.4)
# # plt.show()



