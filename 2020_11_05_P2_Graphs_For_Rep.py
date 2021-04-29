from Useful_Tools.View_Dicom_Images import View_Dicom_Images, View_Dicom_Images_Colourbar, Get_Dicom_Array_Sorted
import matplotlib.pyplot as plt
import numpy as np


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


#****************************************************
#axial P2
#****************
#Line plots
#
#
axcentralsliceP2 = 12
ax_row_line_profile = 140
ax_col_line_profile = 125
#
# # #inferior
# # axcentralsliceP2 = 1
# # ax_row_line_profile = 150
# # ax_col_line_profile = 125
#
# #rows and cols
# DPOneSlice = P2_LR_DP[:,:,axcentralsliceP2]
# DPOneRow = DPOneSlice[ax_row_line_profile,:]
# DPOneCol = DPOneSlice[:,ax_col_line_profile]
#
# noDPOneSlice = P2_NO_DP[:,:,axcentralsliceP2]
# noDPOneRow = noDPOneSlice[ax_row_line_profile,:]
# noDPOneCol = noDPOneSlice[:,ax_col_line_profile]
#
# infDPOneSlice = P2_Inf_DP[:,:,axcentralsliceP2]
# infDPOneRow = infDPOneSlice[ax_row_line_profile,:]
# infDPOneCol = infDPOneSlice[:,ax_col_line_profile]
#
# RatioDPLRP2OneSlice = RatioDPLRP2[:,:,axcentralsliceP2]
# RatioDPLRP2Row=RatioDPLRP2OneSlice[ax_row_line_profile,:]
# RatioDPLRP2Col=RatioDPLRP2OneSlice[:,ax_col_line_profile]
#
# RatioDPinfP2OneSlice = RatioDPinfP2[:,:,axcentralsliceP2]
# RatioDPinfP2Row=RatioDPinfP2OneSlice[ax_row_line_profile,:]
# RatioDPinfP2Col=RatioDPinfP2OneSlice[:,ax_col_line_profile]
#
# #*******
# #for B1+
# #set up figure
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))
# fig.suptitle('Line Plots of B1+ for an Axial Central Slice of Phantom in P2', y=1)
#
# #label figs
# axs[0].set_title("Horizontal", pad=0)
# axs[1].set_title("Vertical")
#
# #plot lines
# axs[0].plot(DPOneRow, 'g--', label='DPs Left and Right')
# axs[0].plot(noDPOneRow, 'r--', label='No DPs')
# axs[0].plot(infDPOneRow, 'b--', label='DPs inferior')
# axs[0].legend()
#
# axs[1].plot(DPOneCol, 'g--', label='DPs Left and Right')
# axs[1].plot(noDPOneCol, 'r--', label='No DPs')
# axs[1].plot(infDPOneCol, 'b--', label='DPs inferior')
# axs[1].legend()
#
# # limit axises
# axs[0].set_xlim([50,205])
# axs[1].set_xlim([70,225])
#
# #inferior
# # axs[0].set_xlim([90,165])
# # axs[1].set_xlim([95,180])
#
# #label axis
# axs[0].set_ylabel('B1+ (FA * 19)')
# axs[0].set_xlabel('Pixels')
#
# axs[1].set_ylabel('B1+ (FA * 10)')
# axs[1].set_xlabel('Pixels')
#
# # plt.subplots_adjust(top= 0.7, bottom = 0)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#
# plt.show()
#
#
# #*******
# #repeat line plots for ratio images
# #set up figure
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))
# fig.suptitle('Line Plots of Ratio Images for an Axial Central Slice of Phantom in P2', y=1)
#
# #label figs
# axs[0].set_title("Horizontal", pad=0)
# axs[1].set_title("Vertical")
#
# #plot lines
# axs[0].plot(RatioDPLRP2Row, 'g--', label='Ratio DPs Left and Right/No DPs')
# axs[0].plot(RatioDPinfP2Row, 'r--', label='Ratio DPs Inferior/No DPs')
# axs[0].legend()
#
# axs[1].plot(RatioDPLRP2Col, 'g--', label='Ratio DPs Left and Right/No DPs')
# axs[1].plot(RatioDPinfP2Col, 'r--', label='Ratio DPs Inferior/No DPs')
# axs[1].legend()
#
# # limit axises
# axs[0].set_xlim([50,205])
# axs[1].set_xlim([70,225])
# axs[0].set_ylim([0,2])
# axs[1].set_ylim([0,2])
#
# # #inferior
# # axs[0].set_xlim([90,160])
# # axs[1].set_xlim([100,180])
# # axs[0].set_ylim([0,3])
# # axs[1].set_ylim([0,3])
#
# #label axis
# axs[0].set_ylabel('Ratio')
# axs[0].set_xlabel('Pixels')
#
# axs[1].set_ylabel('Ratio')
# axs[1].set_xlabel('Pixels')
#
# # plt.subplots_adjust(top= 0.7, bottom = 0)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#
# plt.show()





#central slice
#set up figure

P2_NO_DP=P2_NO_DP/10
P2_LR_DP=P2_LR_DP/10
P2_Inf_DP=P2_Inf_DP/10



fig, axs = plt.subplots(2, 4, figsize=(10,7),gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
fig.suptitle('Position 2, Axial, Central Slice', x=0.5 , y=0.99, fontsize='xx-large' )
fig.delaxes(axs[1,0])

#top row, B1 maps
P2_NO_DPim = axs[0,0].imshow(P2_NO_DP[:,:,axcentralsliceP2], vmin=0, vmax=150)
P2_LR_DPim = axs[0,1].imshow(P2_LR_DP[:,:,axcentralsliceP2], vmin=0, vmax=150)
P2_Inf_DPim = axs[0,2].imshow(P2_Inf_DP[:,:,axcentralsliceP2], vmin=0, vmax=150)


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

cbar=fig.colorbar(P2_Inf_DPim, cax=axs[0,3])
cbar.set_label("degrees")
#bottom row, ratio images
RatioDPLRP2im = axs[1,1].imshow((RatioDPLRP2[:,:,axcentralsliceP2]),vmin=0.5,vmax=1.8)
RatioDPinfP2im= axs[1,2].imshow((RatioDPinfP2[:,:,axcentralsliceP2]),vmin=0.5,vmax=1.8)

axs[1,1].set_title('Ratio DPs LR/No DPs')
axs[1,2].set_title('Ratio of DPs Inf/No DPs')

cbar=fig.colorbar(RatioDPinfP2im, cax=axs[1,3])
cbar.set_label("ratio")
plt.figtext(0.4,0.51,'B1+ Maps',fontsize=15)
plt.figtext(0.4,0.02,'Ratio Images',fontsize=15)

plt.show()


#****************************************************
#cor P2

#central slice
#set up figure

centralsliceP2 = 125

fig, axs = plt.subplots(2, 4, figsize=(10,8),gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
fig.suptitle('Position 2, Coronal, Central Slice', x=0.5 ,y=0.99, fontsize='xx-large' )
fig.delaxes(axs[1,0])
#top row, B1 maps
P2_NO_DPim = axs[0,0].imshow(P2_NO_DP[centralsliceP2,:,:], vmin=0, vmax=150, aspect=0.156)
P2_LR_DPim = axs[0,1].imshow(P2_LR_DP[centralsliceP2,:,:], vmin=0, vmax=150, aspect=0.156)
P2_Inf_DPim = axs[0,2].imshow(P2_Inf_DP[centralsliceP2,:,:], vmin=0, vmax=150, aspect=0.156)

axs[0,0].set_title("No DPs")
axs[0,1].set_title("DPs Left and Right")
axs[0,2].set_title("DPs Inferior Left and Right")

cbar=fig.colorbar(P2_Inf_DPim, cax=axs[0,3])
cbar.set_label("degrees")

#bottom row, ratio images
RatioDPLRP2im = axs[1,1].imshow((RatioDPLRP2[centralsliceP2,:,:]),vmin=0.5,vmax=2.2, aspect=0.156)
RatioDPinfP2im= axs[1,2].imshow((RatioDPinfP2[centralsliceP2,:,:]),vmin=0.5,vmax=2.2, aspect=0.156)

axs[1,1].set_title('Ratio DPs LR/No DPs')
axs[1,2].set_title('Ratio of DPs Inf/No DPs')

cbar=fig.colorbar(RatioDPinfP2im, cax=axs[1,3])
cbar.set_label("ratio")

fig.subplots_adjust(hspace=0.8)

plt.figtext(0.4,0.51,'B1+ Maps',fontsize=15)
plt.figtext(0.4,0.02,'Ratio Images',fontsize=15)

# fig.tight_layout(pad=5.0)
plt.subplots_adjust(top = 0.5, bottom=0.01, hspace=1.5, wspace=0.4)
plt.show()


# #****************************************************
# #sag P2
#
# #central slice
# #set up figure
#
# centralsliceP2 = 66
#
# fig, axs = plt.subplots(2, 4, figsize=(10,8),gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
# fig.suptitle('Position 2, Sagittal, Slice near one DP', x=0.5 ,y=0.99, fontsize='xx-large' )
# fig.delaxes(axs[1,0])
# #top row, B1 maps
# P2_NO_DPim = axs[0,0].imshow(P2_NO_DP[:,centralsliceP2,:], vmin=0, vmax=1500, aspect=0.156)
# P2_LR_DPim = axs[0,1].imshow(P2_LR_DP[:,centralsliceP2,:], vmin=0, vmax=1500, aspect=0.156)
# P2_Inf_DPim = axs[0,2].imshow(P2_Inf_DP[:,centralsliceP2,:], vmin=0, vmax=1500, aspect=0.156)
#
# axs[0,0].set_title("No DPs")
# axs[0,1].set_title("DPs Left and Right")
# axs[0,2].set_title("DPs Inferior Left and Right")
#
# fig.colorbar(P2_Inf_DPim, cax=axs[0,3])
#
# #bottom row, ratio images
# RatioDPLRP2im = axs[1,1].imshow((RatioDPLRP2[:,centralsliceP2,:]),vmin=0.5,vmax=1.5, aspect=0.156)
# RatioDPinfP2im= axs[1,2].imshow((RatioDPinfP2[:,centralsliceP2,:]),vmin=0.5,vmax=1.5, aspect=0.156)
#
# axs[1,1].set_title('Ratio DPs LR/No DPs')
# axs[1,2].set_title('Ratio of DPs Inf/No DPs')
#
# fig.colorbar(RatioDPinfP2im, cax=axs[1,3])
#
# fig.subplots_adjust(hspace=0.8)
#
# plt.figtext(0.4,0.51,'B1+ Maps',fontsize=15)
# plt.figtext(0.4,0.02,'Ratio Images',fontsize=15)
#
# # fig.tight_layout(pad=5.0)
# plt.subplots_adjust(top = 0.5, bottom=0.01, hspace=1.5, wspace=0.4)
# plt.show()
