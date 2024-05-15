import pandas as pd
import numpy as np
from BaseLineCorrectionAndFiltering import polynomialBaseLineCorrect
from BasicFunctions2 import Get_Earthquake_Data,Split_periods,Spectral_Matching,Response_spectra,Spectral_misfit,PGA_correction,AIF

'''
This module is the main program for spectral matching, including PGA correction, AIF fitting and PSO algorithm.

'''
'''--------------------Import the original data-----------------------------'''
target_spectrum=pd.read_csv(r'target_spectrum.txt', header=None, sep='\t')
dt,npts,seed_accelerogram=Get_Earthquake_Data(r'scaled acceleration time series.AT2')
target_spectrum_value= target_spectrum.iloc[5:500:5,1].reset_index(drop=True)
'''---------------Spectral matching parameter setting-------------------------------'''
Periods=target_spectrum.iloc[5:500:5,0].reset_index(drop=True)
damping=0.05 #damping ratio
iteration_step=5 #The number of iterations of spectrum matching
subperiods_num=10 #Period division of target spectrum
PSO_partial=10 # The number of particle swarms
PSO_diamentions=1 # The dimension of particle swarm
PSO_steps=5 #The number of iterations of PSO algorithm

'''--------------------main program--------------------------'''
Original_accelerogram=target_spectrum.iloc[0,1]/max(abs(seed_accelerogram))*seed_accelerogram #Obtain the scaling factor of seed accelerogram
Original_accelerogram_spectrum=np.array([Response_spectra(Original_accelerogram, dt, period, damping)[0] for period in Periods]) #scale seed accelerogram
Original_average_misfit,Original_max_misfit=Spectral_misfit(target_spectrum_value,Original_accelerogram_spectrum,Periods) #calculate the average misfit and maximum misfit
Input_accelerogram=np.copy(Original_accelerogram)
subperiods, subperiod_indexs = Split_periods(Periods, subperiods_num) #split target period
for i in range(iteration_step):
    for j in range(subperiods_num):
        sub_target_spectrum=target_spectrum_value[subperiod_indexs[j][0]:subperiod_indexs[j][1]]
        sub_periods=subperiods[j]
        matched_accelerogram = Spectral_Matching(sub_target_spectrum, Input_accelerogram,sub_periods,  dt, damping, PSO_partial, PSO_diamentions, PSO_steps)
        Input_accelerogram=matched_accelerogram
    '''=================AIF==========================='''
    matched_accelerogram = AIF(Original_accelerogram, matched_accelerogram, dt)
    '''================PGA correction===================='''
    matched_accelerogram=PGA_correction(Original_accelerogram,matched_accelerogram)
    Input_accelerogram = matched_accelerogram
    matched_accelerogram_spectrum = np.array([Response_spectra(matched_accelerogram, dt, period, damping)[0] for period in Periods])
    matched_average_misfit, matched_max_misfit = Spectral_misfit(target_spectrum_value, matched_accelerogram_spectrum,Periods)
'''-----------------------------baseline correction------------------------------------------'''
Final_accelerogram, _, _ = np.array(polynomialBaseLineCorrect(matched_accelerogram * 100, dt)) / 100

