import numpy as np
from scipy.signal import savgol_filter

def erfi(z):
    z1 = -1j * z
    m, c, n = erf(z1)

    expo = c
    A = -m * 1j
    B = -n * 1j

    return A, expo, B
def erf(z):
    a0 = abs(z)
    expo = -z ** 2
    pi = np.pi
    z1 = z if z.real >= 0.0 else -z

    if a0 <= 5.8:
        cs = z1
        cr = z1
        for k in range(1, 121):
            cr = cr * z1 ** 2 / (k + 0.5)
            cs += cr
            if abs(cr / cs) < 1.0e-15:
                break
        A = 0
        B = 2.0 * cs / np.sqrt(pi)
    else:
        cl = 1.0 / z1
        cr = cl
        for k in range(1, 14):
            cr = -cr * (k - 0.5) / (z1 ** 2)
            cl += cr
            if abs(cr / cl) < 1.0e-15:
                break
        A = 1
        if np.real(expo) > 307:
            A = 0
        B = -cl / np.sqrt(pi)

    if z.real < 0.0:
        A = -A
        B = -B

    return A, expo, B
def Integ4(tau, deltaT, beta_i, beta_j, wi, wj, ti, tj, alpha):
    PI = np.pi
    wj1 = wj * np.sqrt(1 - beta_j ** 2)
    wi1 = wi * np.sqrt(1 - beta_i ** 2)
    deltaT = np.arctan(np.sqrt(1 - beta_j ** 2) / beta_j) / wj1
    tc = deltaT + ti - tj

    C1 = complex(-np.sin(tc * wi1), np.cos(tc * wi1))
    C2 = complex(np.cos(tc * wi1), -np.sin(tc * wi1))
    C3 = complex(np.sin(tc * wi1), np.cos(tc * wi1))

    A = ((alpha ** 2) * (beta_i ** 2) * (wi ** 2) - 4 * beta_i * wi * tc - 2 * (alpha ** 2) * ((wi1 ** 2) + (wj1 ** 2))) / 4
    B = -2 * beta_i * wi * (alpha ** 2) * (wi1 + wj1) / 4
    F =  complex(A,B)

    t1 = tau - tj + deltaT

    term1_r = np.sqrt(PI) * alpha * wi / (8 * np.sqrt(1 - beta_i ** 2))
    term1 = complex(0,term1_r)

    Arg1 = complex(alpha * (wi1 - wj1) / 2, (-2 * t1 + alpha ** 2 * beta_i * wi) / (2 * alpha))
    Arg2 = complex(alpha * (wi1 + wj1) / 2, (-2 * t1 + alpha ** 2 * beta_i * wi) / (2 * alpha))
    Arg3 = complex(-t1 / alpha + alpha * beta_i * wi / 2, alpha * (wi1 + wj1) / 2)
    Arg4 = complex(alpha * (wi1 - wj1) / 2, t1 / alpha - alpha * beta_i * wi / 2)

    A_1, expo_1, B_1 = erfi(Arg1)
    A_2, expo_2, B_2 = erfi(Arg2)
    A_3, expo_3, B_3 = erf(Arg3)
    A_4, expo_4, B_4 = erfi(Arg4)

    Exp1_a = (alpha ** 2) / 4 *complex((wi1 + wj1) ** 2,4 * beta_i * wi * wj1) +F
    Exp1_b = (alpha ** 2) / 4 * complex((wi1 + wj1) ** 2, 4 * beta_i * wi * wj1) + F+ expo_1

    Exp2_a = (alpha ** 2) / 4 * (wi1 - wj1) ** 2 + F
    Exp2_b = (alpha ** 2) / 4 * (wi1 - wj1) ** 2 + F + expo_2

    Exp3_a = (alpha ** 2) / 4 *complex((wi1 - wj1) ** 2 ,4 * beta_i * wi * (wi1 + wj1))+ F
    Exp3_b = (alpha ** 2) / 4 * complex((wi1 - wj1) ** 2, 4 * beta_i * wi * (wi1 + wj1)) + F+ expo_3

    Exp4_a = (alpha ** 2) / 4 *complex((wi1 + wj1) ** 2,4 * beta_i * wi * wi1) + F
    Exp4_b = (alpha ** 2) / 4 * complex((wi1 + wj1) ** 2, 4 * beta_i * wi * wi1) + F+ expo_4

    CArg1 = np.exp(Exp1_a) * A_1 + np.exp(Exp1_b) * B_1
    CArg2 = np.exp(Exp2_a) * A_2 + np.exp(Exp2_b) * B_2
    term2 = (CArg1 + CArg2) * C1

    term3 = np.exp(Exp3_a) * A_3 * C2 + np.exp(Exp3_b) * B_3 * C2

    term4 = np.exp(Exp4_a) * A_4 * C3 + np.exp(Exp4_b) * B_4 * C3

    # Evaluate integral
    cInteg = term1 * (term2 + term3 + term4)
    Integ4 = cInteg.real

    return Integ4
def Calc_C(ti, tj, beta_i, beta_j, wi, wj):
    '''
    The response to the acceleration time history ti is calculated, when the wavelet function center is tj .
    :param ti: the  time instant corresponding to the response of a SDOF with a period of ti
    :param tj: the  time instant corresponding to the reference origin (ti) of a adjustment wavelet
    :param beta_i:damping ratio
    :param beta_j:damping ratio
    :param wi:angular frequency of a SDOF system with a target period
    :param wj:angular frequency of  a adjustment wavelet
    :return:respond value
    '''
    f=wj/2/np.pi
    alpha=1.178 * f ** (-0.93)
    #calculate the difference beteween the peak acceleration and original poiont
    tmp1 = np.sqrt(1.0 - beta_j ** 2)
    deltaT=np.arctan(tmp1 / beta_j) / (wj * tmp1)
    #calculate the response of SDOF subject to wavelets
    Cij = Integ4(ti, deltaT, beta_i, beta_j, wi, wj, ti, tj, alpha) - Integ4(0., deltaT, beta_i, beta_j, wi, wj, ti, tj, alpha)
    return Cij
def Calc_C_Matrix(w,tPeak,damping):
    '''
    the function is to calculate the respond matrix
    :param w:angular frequency of a SDOF system
    :param tPeak:the time instant corresponding to the maximum respond
    :param damping:damping ratio
    :return:respond matrix
    '''
    nQ=len(tPeak)
    damping=np.ones(nQ)*damping
    C=np.zeros((nQ,nQ))
    for i in range(nQ):
        for j in range(nQ):
            C[i][j] = Calc_C(tPeak[i], tPeak[j], damping[i], damping[j], w[i], w[j])
    return C
def  Get_Earthquake_Data(inFilename):
    '''
    this function is to obtain the acceleration time series from PEER  format
    :param inFilename: the acceleration file from peer database
    :return: time interval, the length of accelerogram, accelerogram
    '''
    dt = 0.0
    npts = 0
    a = []
    inFileID = open(inFilename, 'r')
    flag = 0
    for line in inFileID:
        if line == '\n':
            continue
        elif flag == 1:
            words = line.split()
            lengthLine = len(words)
            if words[0] == '***':
                break
            else:
                for i in range(0, lengthLine):
                    a.append(float(words[i]))

        else:
            words = line.split()
            lengthLine = len(words)
            if lengthLine >= 4:
                if words[0] == 'NPTS=':
                    for word in words:
                        if word != '':
                            if flag == 1:
                                dt = float(word)
                                break
                            if flag == 2:
                                npts = int(word.strip(','))
                                flag = 0
                            if word == 'DT=' or word == 'dt':
                                flag = 1
                            if word == 'NPTS=':
                                flag = 2
                elif words[-1] == 'DT':
                    count = 0
                    for word in words:
                        if word != '':
                            if count == 0:
                                npts = int(word)
                            elif count == 1:
                                dt = float(word)
                            elif word == 'DT':
                                flag = 1
                                break
                            count += 1
    return dt, npts, np.array(a)
def Split_periods(Period,n):
    '''
    the function is to split the period of target spectrum
    :param Period: input period series
    :param n: number of split periods
    :return: subperiods ,subperiods_indexs
    '''
    Period=np.array(Period)
    subperiod_len=len(Period)//n
    subperiods=[]
    subperiods_indexs=[]
    for i in range (n):
        if i==(n-1):
            subperiod=Period[i*subperiod_len:len(Period)]
            subperiods_index=[i*subperiod_len,len(Period)]
        else:
            subperiod=Period[i*subperiod_len:(i+1)*subperiod_len]
            subperiods_index = [i*subperiod_len,(i+1)*subperiod_len]
        subperiods.append(subperiod)
        subperiods_indexs.append(subperiods_index)
    return subperiods,subperiods_indexs
def Incremental_wavelet(frequency,timeseries, central_time,damping):
    '''
    the function is to generation adjustment wavelet
    :param frequency: central frequency of wavelet
    :param timeseries: input time series of wavelet
    :param central_time: central_time on wavelet
    :param damping: damping ratio
    :return: wavelet series
    '''
    beta_j=damping
    tmp1 = np.sqrt(1.0 - beta_j ** 2)
    wj = 2 * np.pi * frequency * tmp1
    deltaT = np.arctan(tmp1 / beta_j) / (wj)
    t = timeseries
    tj = central_time
    r = 1.178 * frequency ** (-0.93) # 小波尺度参数
    wavelet_series = np.exp(-((t - tj+deltaT) / r)**2) * np.cos(wj * (t - tj+deltaT))
    return wavelet_series
def sdof_response_history(omg, damping, acc_series, dt):
    '''
    the function is to realize the response time history based on piecewise integral.
    :param omg:angular frequency of a SDOF system
    :param damping:the damping ratio of a SDOF system
    :param acc_series: input acceleration time series
    :param dt:the time interval of  input acceleration time series
    :return: the respond time series
    '''
    zeta=damping
    ag=acc_series
    omg_d = omg * np.sqrt(1.0 - zeta * zeta)
    n = len(ag)
    u = np.zeros(n)
    v = np.zeros(n)

    B1 = np.exp(-zeta * omg * dt) * np.cos(omg_d * dt)
    B2 = np.exp(-zeta * omg * dt) * np.sin(omg_d * dt)

    omg_2 = 1.0 / omg / omg
    omg_3 = 1.0 / omg / omg / omg

    for i in range(n-1):
        u_i = u[i]
        v_i = v[i]
        p_i = -ag[i]
        alpha_i = (-ag[i + 1] + ag[i]) / dt

        A0 = p_i * omg_2 - 2.0 * zeta * alpha_i * omg_3
        A1 = alpha_i * omg_2
        A2 = u_i - A0
        A3 = (v_i + zeta * omg * A2 - A1) / omg_d

        u[i + 1] = A0 + A1 * dt + A2 * B1 + A3 * B2
        v[i + 1] = A1 + (omg_d * A3 - zeta * omg * A2) * B1 - (omg_d * A2 + zeta * omg * A3) * B2

    a = -2.0 * zeta * omg * v - omg * omg * u
    return a, v, u
def Response_spectra(accelerogram, dt, period, damping):
    '''
    The function is to obtain the response spectra of input accelerogram
    :param accelerogram:input accelerogram
    :param dt:time interval of input accelerogram
    :param period:the period of an SDOF
    :param damping:the damping ratio of an SDOF
    :return:maxRSA,the time instant corresponding to the maximum respond
    '''
    omg = 2.0 * np.pi /period
    a, v, u = sdof_response_history(omg, damping, accelerogram, dt)
    maxRSA=max(a,key=abs)
    maxTimeRSA= np.argmax(np.abs(a)) * dt
    return maxRSA,maxTimeRSA
def PSO(fitness,N,D,iter):
    '''
    The function is to implement the particle swarm optimization algorithm to find the optimal relaxation coefficient.
    :param func: cost function
    :param N: The number of particle swarms
    :param D: The dimension of particle swarm
    :param iter: The number of iterations of PSO algorithm
    :return: optimal parameter
    '''
    N = N
    D = D
    T = iter
    c1 = c2 = 2
    w_max = 0.8
    w_min = 0.4
    x_max = 1
    x_min = 0
    v_max = 1
    v_min = -1
    func = fitness
    x = np.random.rand(N, D) * (x_max - x_min) + x_min
    v = np.random.rand(N, D) * (v_max - v_min) + v_min
    p = x
    p_best = np.ones((N, 1))
    for i in range(N):
        p_best[i] = func(x[i, :])
    g_best = 100
    gb = np.ones(T)
    x_best = np.ones(D)
    for i in range(T):
        for j in range(N):
            if p_best[j] > func(x[j, :]):
                p_best[j] = func(x[j, :])
                p[j, :] = x[j, :].copy()
            if g_best > p_best[j]:
                g_best = p_best[j]
                x_best = x[j, :].copy()
            w = w_max - (w_max - w_min) * i / T
            v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p[j, :] - x[j, :]) + c2 * np.random.rand(1) * (x_best - x[j, :])
            x[j, :] = x[j, :] + v[j, :]
            for ii in range(D):
                if (v[j, ii] > v_max) or (v[j, ii] < v_min):
                    v[j, ii] = v_min + np.random.rand(1) * (v_max - v_min)
                if (x[j, ii] > x_max) or (x[j, ii] < x_min):
                    x[j, ii] = x_min + np.random.rand(1) * (x_max - x_min)
        gb[i] = g_best
        misserror = func(x_best)
    return x_best
def Spectral_misfit(target_spectrum, input_spectrum, Periods):
    '''
    the aim of this function is to calculate the spectral misfit between spectra of input accelerogram and target
    :param target_spectrum: target acceleration spectrum
    :param input_spectrum: input acceleration spectrum
    :param Periods:
    :return: average misfit, max misfit
    '''
    target = np.array(target_spectrum)
    estimated = np.array(input_spectrum)
    average_misfit=np.sum((abs(np.abs(target - abs(estimated)) / target))) / len(Periods)*100
    max_misfit=max(np.abs(target-abs(estimated))/target)
    return average_misfit,max_misfit
def Spectral_Matching(target_spectrum,input_accelerogram,Periods,dt,damping,PSO_partial,PSO_diamentions,PSO_steps):
    '''
    the aim of this function is to achieve time damain spectral matching method
    :param target_spectrum: input target spectrum
    :param Periods: period of SDOF corresponding to target spectrum
    :param input_accelerogram: input ground motion accelerogram
    :param dt: interval of input accelerogram
    :param damping: damping of SDOF
    :param PSO_partial: munber of  partials
    :param PSO_diamentions: diamension of partial
    :param PSO_steps:iteration steps of partial
    :return:matched acceleration time series
    '''
    time_series=np.arange(0, len(input_accelerogram)) * dt
    target_spectrum=np.array(target_spectrum)
    input_accelerogram_spectrums, max_input_accelrogram_times = [], []
    for period in Periods:
        spectrum, max_time = Response_spectra(input_accelerogram, dt, period, damping)
        input_accelerogram_spectrums.append(spectrum)
        max_input_accelrogram_times.append(max_time)
    input_accelerogram_spectrums,max_input_accelrogram_times = np.array(input_accelerogram_spectrums),np.array(max_input_accelrogram_times)
    #-------------------------------Step1--------------------------------------------
    P = np.array([1 if p > 0 else -1 for p in input_accelerogram_spectrums])
    mis_error = np.array(target_spectrum - abs(input_accelerogram_spectrums)) * P
    # ------------------------------Step2--------------------------------------------
    C_matrix = Calc_C_Matrix(2 * np.pi / Periods, max_input_accelrogram_times, damping)
    mask = ~np.eye(C_matrix.shape[0], dtype=bool)
    C_matrix[mask] *=0.6
    # ------------------------------Step3--------------------------------------------
    b = np.linalg.inv(C_matrix).dot(mis_error)
    # ------------------------------Step4--------------------------------------------
    def Fitness( gama):
        temporary_matched_accelerogram=np.copy(input_accelerogram)
        for k in range(len(Periods)):
            wavelet_accelerogram = Incremental_wavelet(frequency=1 / Periods[k], timeseries=time_series,central_time=max_input_accelrogram_times[k], damping=damping)
            temporary_matched_accelerogram += gama*b[k]*wavelet_accelerogram
        matched_accelerogram_spectrum = np.array([Response_spectra(temporary_matched_accelerogram, dt, period, damping)[0] for period in Periods])
        average_misfit,max_misfit=Spectral_misfit(target_spectrum,matched_accelerogram_spectrum,Periods)
        return average_misfit
    best_gama = PSO(Fitness, N=PSO_partial, D=PSO_diamentions, iter=PSO_steps)
    # ------------------------------Step5--------------------------------------------
    matched_accelerogram=np.copy(input_accelerogram)
    for k in range(len(Periods)):
        wavelet_accelerogram = Incremental_wavelet(frequency=1 / Periods[k], timeseries=time_series,  central_time=max_input_accelrogram_times[k], damping=damping)
        matched_accelerogram  +=   best_gama * b[k] * wavelet_accelerogram
    return np.array(matched_accelerogram)
def Arias_intensity(input_accelerogram,dt):
    '''
    the aim of this function is to calculate the cumulative Arias intensity
    :param input_accelerogram: input acceleration time series
    :param dt: time interval
    :return: Arias intensity
    '''
    accelerogram=np.array(input_accelerogram)
    Arias_intensity=np.pi / 2 * 9.8 * np.cumsum(accelerogram ** 2) * dt
    return Arias_intensity
def Fourier_Spectral(input_accelerogram, dt):
    '''
    the aim of this function is to calculate Fourier spectrum
    :param input_accelerogram: input acceleration time series
    :param dt:time interval
    :return:frequence, the amplitude of Fourier spectrum
    '''
    Fs = 1 / dt
    n = 2 ** int(np.ceil(np.log2(len(input_accelerogram))))
    fy = np.fft.fft(input_accelerogram, n=n, axis=0)
    frequence = Fs * np.arange(0, n / 2) / n
    fourier_amplitude = 2 / n * np.abs(fy)
    return frequence, fourier_amplitude[:int(n / 2)]
def PGA_correction(input_accelerogram,matched_accelerogram):
    '''
    The aim of this function is to approximate the PGA of matched ground motion and scaled ground motion.
    :param input_accelerogram: original accelerogram
    :param matched_accelerogram: matched accelerogram after spectral matching
    :return: mached accelerogram after PGA correction
    '''
    seed_acc_his = np.array(input_accelerogram)
    pre_match_seed_his = np.array(matched_accelerogram)
    filter_pre_acc=savgol_filter(np.copy(pre_match_seed_his), window_length=12, polyorder=2)
    max_filter_pre_acc_index=np.argmax(abs(filter_pre_acc))
    for i in range(max_filter_pre_acc_index,0,-1):
        if (filter_pre_acc[i]-filter_pre_acc[i-1])*(filter_pre_acc[i-1]-filter_pre_acc[i-2])<0:
            left_index=i-1
            break
    for j in range(max_filter_pre_acc_index,len(filter_pre_acc),1):
        if (filter_pre_acc[j+2]-filter_pre_acc[j+1])*(filter_pre_acc[j+1]-filter_pre_acc[j])<0:
            right_index = j - 1
            break
    pre_match_seed_his[left_index:right_index]=filter_pre_acc[left_index:right_index]
    aim_PGA = max(abs(seed_acc_his))
    max_fit_PGA0 = max(abs(pre_match_seed_his))
    match_seed_his = np.copy(pre_match_seed_his)
    if max_fit_PGA0<aim_PGA:
        max_fit_PGA_index=np.argmax(abs(match_seed_his))
        near_max_PGA2 = max_fit_PGA_index
        for i in range(max_fit_PGA_index, 0, -1):
            if match_seed_his[max_fit_PGA_index] * (match_seed_his[i] - match_seed_his[i - 1]) < 0:
                near_max_PGA1 = i
                break
        for j in range(max_fit_PGA_index, len(match_seed_his), 1):
            if match_seed_his[max_fit_PGA_index] * (match_seed_his[j + 1] - match_seed_his[j]) > 0:
                near_max_PGA3 = j
                break
        sgn = 1 if match_seed_his[near_max_PGA2] > 0 else -1
        match_seed_his[near_max_PGA2] =aim_PGA* sgn
        for k in range(near_max_PGA1, near_max_PGA2 ):
            match_seed_his[k] = (match_seed_his[near_max_PGA2] - match_seed_his[near_max_PGA1]) / ( near_max_PGA2 - near_max_PGA1) * (k - near_max_PGA1) + match_seed_his[near_max_PGA1]
        for k2 in range(near_max_PGA2, near_max_PGA3 ):
            match_seed_his[k2] = (match_seed_his[near_max_PGA3] - match_seed_his[near_max_PGA2]) / ( near_max_PGA3 - near_max_PGA2) * (k2 - near_max_PGA3) + match_seed_his[near_max_PGA3]
    else:
        max_base_PGA=0.95 * aim_PGA
        wave_crests_index = []
        for index in range(len(pre_match_seed_his) - 2):
            if (pre_match_seed_his[index + 1] - pre_match_seed_his[index]) * (
                    pre_match_seed_his[index + 2] - pre_match_seed_his[index + 1]) < 0:
                if abs(pre_match_seed_his[index + 1]) < aim_PGA:
                    if abs(pre_match_seed_his[index + 1]) > max_base_PGA:
                        max_base_PGA = abs(pre_match_seed_his[index + 1])
                else:
                    wave_crests_index.append(index + 1)
        det_PGA=max_fit_PGA0-max_base_PGA
        near_max_PGA1=[]
        near_max_PGA2= []
        near_max_PGA3 = []
        for wave_index in wave_crests_index:
            for i in range(wave_index, 0, -1):
                if match_seed_his[wave_index] *(match_seed_his[i] - match_seed_his[i - 1]) < 0:
                    near_max_PGA1.append(i)
                    break
            for j in range(wave_index, len(match_seed_his), 1):
                if match_seed_his[wave_index] *(match_seed_his[j + 1] - match_seed_his[j]) > 0:
                    near_max_PGA3 .append(j)
                    break
            sgn = 1 if match_seed_his[wave_index] > 0 else -1
            match_seed_his[wave_index] = (((abs(match_seed_his[wave_index]) - max_base_PGA) / det_PGA) * ( aim_PGA - max_base_PGA) + max_base_PGA) * sgn
            near_max_PGA2.append(wave_index)
        for m in range(len(near_max_PGA2)):
            for k in range(near_max_PGA1[m], near_max_PGA2[m]):
                cofficient=1-(k-near_max_PGA1[m])*(1-match_seed_his[near_max_PGA2[m]]/pre_match_seed_his[near_max_PGA2[m]])/(near_max_PGA2[m]-near_max_PGA1[m])
                match_seed_his[k] =cofficient*pre_match_seed_his[k]
            for k2 in range(near_max_PGA2[m], near_max_PGA3[m]):
                cofficient=1+(1-match_seed_his[near_max_PGA2[m]] / pre_match_seed_his[near_max_PGA2[m]])*(k2-near_max_PGA3[m])/( near_max_PGA3[m] - near_max_PGA2[m])
                match_seed_his[k2] = cofficient * pre_match_seed_his[k2]
    return match_seed_his
def Calculate_effective_during(input_accelerogram,dt):
    '''
    Calculate the effective duration of ground motion
    :param input_accelerogram:input grounn motion time series
    :param dt:time interval
    :return:cumulative Arias intensity,the time instant with 5%  Arias intensity,the time instant with 95%  Arias intensity
    '''
    input_accelerogram=np.array(input_accelerogram)
    Arias=Arias_intensity(input_accelerogram,dt)
    Arias_5=max(Arias)*0.05
    Arias_95 = max(Arias) * 0.95
    Arias_5_index=np.argmin(abs(Arias-Arias_5))
    Arias_95_index = np.argmin(abs(Arias - Arias_95))
    return Arias,Arias_5_index,Arias_95_index
def AIF(original_accelerogram, matched_accelerogram, dt):
    '''
    The aim of this function is to achieve Arias intensity fitting.
    :param original_accelerogram:
    :param matched_accelerogram:
    :param dt:time intervakl
    :return:  acceleration time series similar to the original ground motion Arias intensity
    '''
    target_acc_his=np.array(original_accelerogram)
    input_acc_his=np.array(matched_accelerogram)
    target_arias_intensity,arias_intensity_5_index,arias_intensity_95_index=Calculate_effective_during(target_acc_his,dt)
    partial_target_arias_intensity = target_arias_intensity[arias_intensity_5_index:arias_intensity_95_index]
    det_partial_target_arias_intensity = partial_target_arias_intensity[-1] - partial_target_arias_intensity[0]
    input_acc_arias_intensity,arias_intensity_5_index,arias_intensity_95_index = Calculate_effective_during(input_acc_his, dt)
    partial_input_acc_arias_intensity = input_acc_arias_intensity[arias_intensity_5_index:arias_intensity_95_index]
    det_partial_input_acc_arias_intensity = partial_input_acc_arias_intensity[-1] - partial_input_acc_arias_intensity[0]
    scaled_cofficient = np.sqrt(det_partial_target_arias_intensity / det_partial_input_acc_arias_intensity)
    for i in range(arias_intensity_5_index, 0, -1):
        if (input_acc_his[i] - input_acc_his[i - 1]) * (input_acc_his[i - 1] - input_acc_his[i - 2]) < 0:
            time_min_left = i - 1
            break
    for j in range(arias_intensity_5_index, arias_intensity_95_index, 1):
        if (input_acc_his[j + 2] - input_acc_his[j + 1]) * (input_acc_his[j + 1] - input_acc_his[j]) < 0:
            time_min = j + 1
            break
    for k in range(arias_intensity_95_index, arias_intensity_5_index, -1):
        if (input_acc_his[k] - input_acc_his[k - 1]) * (input_acc_his[k - 1] - input_acc_his[k - 2]) < 0:
            time_max = k - 1
            break
    for m in range(arias_intensity_95_index, len(input_acc_his), 1):
        if (input_acc_his[m + 2] - input_acc_his[m + 1]) * (input_acc_his[m + 1] - input_acc_his[m]) < 0:
            time_max_right = m + 1
            break
    factors = []
    for n in range(len(input_acc_his)):
        if n <= time_min_left:
            factors.append(1)
        elif time_min_left < n < time_min:
            scaling_factor = scaled_cofficient + (scaled_cofficient - 1) / (time_min - time_min_left) * ( n - time_min_left)
            factors.append(scaling_factor)
        elif time_min <= n <= time_max:
            factors.append(scaled_cofficient)
        elif time_max < n < time_max_right:
            scaling_factor = scaled_cofficient - (scaled_cofficient - 1) / (time_max_right - time_max) * ( n - time_max_right)
            factors.append(scaling_factor)
        else:
            factors.append(1)
    AIF_acc_his = input_acc_his * np.array(factors)
    return AIF_acc_his
def Zero_level_upcrossing(acc,dt):
    '''
    calculate the cumulative number of zero_level_upcrossing
    :param acc:input acceleration time series
    :param dt:time interval
    :return: time series and the cumulative number of zero_level_upcrossing
    '''
    acc=np.array(acc)
    time_series = np.arange(0, len(acc)) * dt
    upcrossing_count=[]
    for i in range(len(acc)-1):
        if acc[i]*acc[i+1]<=0:
            if acc[i+1]-acc[i]>=0:
                upcrossing_count.append(1)
            else:
                upcrossing_count.append(0)
        else:
            upcrossing_count.append(0)
    cumsum_count=np.cumsum(upcrossing_count)
    return time_series[:-1],cumsum_count
