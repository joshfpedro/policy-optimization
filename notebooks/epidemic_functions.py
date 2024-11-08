import numpy as np

def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# Define the function eta() which takes input parameters theta and returns the log-odds of disease for each yard i in current time period
def eta(theta, N, y_lag, n_lag, s_lag, tI1_all, tI2_all, dist_2014, dist_2015, dist_2016, dist_2017,
        z_lag_2014, z_lag_2015, z_lag_2016, z_lag_2017, nz_lag_2014, nz_lag_2015, nz_lag_2016, nz_lag_2017,
        sz_lag_2014, sz_lag_2015, sz_lag_2016, sz_lag_2017, a_lag_2014, a_lag_2015, a_lag_2016, a_lag_2017,
        w_lag_2014, w_lag_2015, w_lag_2016, w_lag_2017, sI1_lag_2014, sI1_lag_2015, sI1_lag_2016, sI1_lag_2017,
        sI2_2014, sI2_2015, sI2_2016, sI2_2017, N_2014, N_2015, N_2016, N_2017,
        tI1_2014, tI1_2015, tI1_2016, tI1_2017, tI2_2014, tI2_2015, tI2_2016, tI2_2017):
            
        
    beta1, beta2, delta1, delta2, gamma1, gamma2, alpha1, alpha2, eta11, eta12, eta21, eta22 = theta
    
    beta1_array = np.full((N,1), beta1)
    beta2_array = np.full((N,1), beta2)
    
    auto_infection1 = delta1 * (y_lag / n_lag) * np.exp(-eta11 * s_lag)
    auto_infection2 = delta2 * (y_lag / n_lag) * np.exp(-eta12 * s_lag)
    
    dispersal1 = []
    dispersal2 = []
    
    for year in [2014, 2015, 2016, 2017]:
        
        if year == 2014:
            dist = dist_2014
            sI2 = sI2_2014
            z_lag = z_lag_2014
            nz_lag = nz_lag_2014
            sz_lag = sz_lag_2014
            a_lag = a_lag_2014
            w_lag = w_lag_2014
            sI1_lag = sI1_lag_2014
            Nz = N_2014
            tI1 = tI1_2014
            tI2 = tI2_2014
            
        elif year == 2015:
            dist = dist_2015
            sI2 = sI2_2015
            z_lag = z_lag_2015
            nz_lag = nz_lag_2015
            sz_lag = sz_lag_2015
            a_lag = a_lag_2015
            w_lag = w_lag_2015
            sI1_lag = sI1_lag_2015
            Nz = N_2015
            tI1 = tI1_2015
            tI2 = tI2_2015
        
        elif year == 2016:
            dist = dist_2016
            sI2 = sI2_2016
            z_lag = z_lag_2016
            nz_lag = nz_lag_2016
            sz_lag = sz_lag_2016
            a_lag = a_lag_2016
            w_lag = w_lag_2016
            sI1_lag = sI1_lag_2016
            Nz = N_2016
            tI1 = tI1_2016
            tI2 = tI2_2016
            
        elif year == 2017:
            dist = dist_2017
            sI2 = sI2_2017
            z_lag = z_lag_2017
            nz_lag = nz_lag_2017
            sz_lag = sz_lag_2017
            a_lag = a_lag_2017
            w_lag = w_lag_2017
            sI1_lag = sI1_lag_2017
            Nz = N_2017
            tI1 = tI1_2017
            tI2 = tI2_2017
        
        for j in range(0, Nz):
            
            dispersal_array = ((a_lag * (z_lag / nz_lag)) * (w_lag[:, j].reshape(Nz,1)))
            dispersal_array1 = dispersal_array * np.exp(-eta21 * sz_lag) * np.power(1 + dist[:, j].reshape(Nz,1), -alpha1) * sI1_lag
            dispersal_array2 = dispersal_array * np.exp(-eta22 * sz_lag) * np.power(1 + dist[:, j].reshape(Nz,1), -alpha2) * sI2
            dispersal_component1_i = gamma1 * (np.sum(dispersal_array1) - dispersal_array1[j][0])
            dispersal_component2_i = gamma2 * (np.sum(dispersal_array2) - dispersal_array2[j][0])
        
            dispersal1.append(dispersal_component1_i)
            dispersal2.append(dispersal_component2_i)
    
    dispersal1 = np.array(dispersal1).reshape(N,1)
    dispersal2 = np.array(dispersal2).reshape(N,1)

    eta = tI1_all * (beta1_array + auto_infection1 + dispersal1) + tI2_all * (beta2_array + auto_infection2 + dispersal2)
    
    return eta

def costFunction(theta, *args):
    
    neg_log_likelihood = -(1/N) * np.sum(y * eta(theta) - n * np.log(1 + np.exp(eta(theta))))

    return neg_log_likelihood

def partial(theta, *args):
    beta1, beta2, delta1, delta2, gamma1, gamma2, alpha1, alpha2, eta11, eta12, eta21, eta22 = theta

    d_beta1 = tI1_all
    d_beta2 = tI2_all

    d_delta1 = tI1_all * (y_lag / n_lag) * np.exp(-eta11 * s_lag)
    d_delta2 = tI2_all * (y_lag / n_lag) * np.exp(-eta12 * s_lag)

    d_gamma1 = []
    d_gamma2 = []
    d_alpha1 = []
    d_alpha2 = []
    d_eta21 = []
    d_eta22 = []

    for year in [2014, 2015, 2016, 2017]:
        
        if year == 2014:
            dist = dist_2014
            sI2 = sI2_2014
            z_lag = z_lag_2014
            nz_lag = nz_lag_2014
            sz_lag = sz_lag_2014
            a_lag = a_lag_2014
            w_lag = w_lag_2014
            sI1_lag = sI1_lag_2014
            Nz = N_2014
            tI1 = tI1_2014
            tI2 = tI2_2014
            
        elif year == 2015:
            dist = dist_2015
            sI2 = sI2_2015
            z_lag = z_lag_2015
            nz_lag = nz_lag_2015
            sz_lag = sz_lag_2015
            a_lag = a_lag_2015
            w_lag = w_lag_2015
            sI1_lag = sI1_lag_2015
            Nz = N_2015
            tI1 = tI1_2015
            tI2 = tI2_2015
        
        elif year == 2016:
            dist = dist_2016
            sI2 = sI2_2016
            z_lag = z_lag_2016
            nz_lag = nz_lag_2016
            sz_lag = sz_lag_2016
            a_lag = a_lag_2016
            w_lag = w_lag_2016
            sI1_lag = sI1_lag_2016
            Nz = N_2016
            tI1 = tI1_2016
            tI2 = tI2_2016
            
        elif year == 2017:
            dist = dist_2017
            sI2 = sI2_2017
            z_lag = z_lag_2017
            nz_lag = nz_lag_2017
            sz_lag = sz_lag_2017
            a_lag = a_lag_2017
            w_lag = w_lag_2017
            sI1_lag = sI1_lag_2017
            Nz = N_2017
            tI1 = tI1_2017
            tI2 = tI2_2017
        
        for i in range(0, Nz):
        
            mask = np.arange(Nz) != i # mask out the current yard i
        
            d_gamma1_i = tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1)) * sI1_lag[mask])
            d_gamma2_i = tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2)) * sI2[mask])
            
            d_alpha1_i = gamma1 * tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI1_lag[mask])
            d_alpha2_i = gamma2 * tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI2[mask])
            
            d_eta21_i = -gamma1 * tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1)) * sI1_lag[mask] * sz_lag[mask])
            d_eta22_i = -gamma2 * tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2)) * sI2[mask] * sz_lag[mask])
        
            d_gamma1.append(d_gamma1_i)
            d_gamma2.append(d_gamma2_i)
            d_alpha1.append(d_alpha1_i)
            d_alpha2.append(d_alpha2_i)
            d_eta21.append(d_eta21_i)
            d_eta22.append(d_eta22_i)

    d_gamma1 = np.array(d_gamma1).reshape(N,1)
    d_gamma2 = np.array(d_gamma2).reshape(N,1)
    d_alpha1 = np.array(d_alpha1).reshape(N,1)
    d_alpha2 = np.array(d_alpha2).reshape(N,1)
    d_eta21 = np.array(d_eta21).reshape(N,1)
    d_eta22 = np.array(d_eta22).reshape(N,1)


    d_eta11 = -tI1_all * delta1 * s_lag * (y_lag / n_lag) * np.exp(-eta11 * s_lag)
    d_eta12 = -tI2_all * delta2 * s_lag * (y_lag / n_lag) * np.exp(-eta12 * s_lag)



    grad_entries = np.array([d_beta1, d_beta2, d_delta1, d_delta2, d_gamma1, d_gamma2, d_alpha1, d_alpha2, d_eta11, d_eta12, d_eta21, d_eta22])

    return grad_entries

def partial_by_partial(theta, *args):

    beta1, beta2, delta1, delta2, gamma1, gamma2, alpha1, alpha2, eta11, eta12, eta21, eta22 = theta
    
    d_beta1 = tI1_all
    d_beta2 = tI2_all
    
    d_delta1 = tI1_all * (y_lag / n_lag) * np.exp(-eta11 * s_lag)
    d_delta2 = tI2_all* (y_lag / n_lag) * np.exp(-eta12 * s_lag)
    
    d_eta11 = -tI1_all * delta1 * s_lag * (y_lag / n_lag) * np.exp(-eta11 * s_lag)
    d_eta12 = -tI2_all * delta2 * s_lag * (y_lag / n_lag) * np.exp(-eta12 * s_lag)
    
    d_gamma1 = []
    d_gamma2 = []
    d_alpha1 = []
    d_alpha2 = []
    d_eta21 = []
    d_eta22 = []
    
    for year in [2014, 2015, 2016, 2017]:
        
        if year == 2014:
            dist = dist_2014
            sI2 = sI2_2014
            z_lag = z_lag_2014
            nz_lag = nz_lag_2014
            sz_lag = sz_lag_2014
            a_lag = a_lag_2014
            w_lag = w_lag_2014
            sI1_lag = sI1_lag_2014
            Nz = N_2014
            tI1 = tI1_2014
            tI2 = tI2_2014
            
        elif year == 2015:
            dist = dist_2015
            sI2 = sI2_2015
            z_lag = z_lag_2015
            nz_lag = nz_lag_2015
            sz_lag = sz_lag_2015
            a_lag = a_lag_2015
            w_lag = w_lag_2015
            sI1_lag = sI1_lag_2015
            Nz = N_2015
            tI1 = tI1_2015
            tI2 = tI2_2015
        
        elif year == 2016:
            dist = dist_2016
            sI2 = sI2_2016
            z_lag = z_lag_2016
            nz_lag = nz_lag_2016
            sz_lag = sz_lag_2016
            a_lag = a_lag_2016
            w_lag = w_lag_2016
            sI1_lag = sI1_lag_2016
            Nz = N_2016
            tI1 = tI1_2016
            tI2 = tI2_2016
            
        elif year == 2017:
            dist = dist_2017
            sI2 = sI2_2017
            z_lag = z_lag_2017
            nz_lag = nz_lag_2017
            sz_lag = sz_lag_2017
            a_lag = a_lag_2017
            w_lag = w_lag_2017
            sI1_lag = sI1_lag_2017
            Nz = N_2017
            tI1 = tI1_2017
            tI2 = tI2_2017
        
        for i in range(0, Nz):
        
            mask = np.arange(Nz) != i # mask out the current yard i
        
            d_gamma1_i = tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1)) * sI1_lag[mask])
            d_gamma2_i = tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2)) * sI2[mask])
            
            d_alpha1_i = gamma1 * tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI1_lag[mask])
            d_alpha2_i = gamma2 * tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI2[mask])

            d_eta21_i = -gamma1 * tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1)) * sI1_lag[mask] * sz_lag[mask])
            d_eta22_i = -gamma2 * tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2)) * sI2[mask] * sz_lag[mask])
            
            d_gamma1.append(d_gamma1_i)
            d_gamma2.append(d_gamma2_i)
            d_alpha1.append(d_alpha1_i)
            d_alpha2.append(d_alpha2_i)
            d_eta21.append(d_eta21_i)
            d_eta22.append(d_eta22_i)
    
    d_gamma1 = np.array(d_gamma1).reshape(N,1)
    d_gamma2 = np.array(d_gamma2).reshape(N,1)
    d_alpha1 = np.array(d_alpha1).reshape(N,1)
    d_alpha2 = np.array(d_alpha2).reshape(N,1)
    d_eta21 = np.array(d_eta21).reshape(N,1)
    d_eta22 = np.array(d_eta22).reshape(N,1)
        

    grad_entries = np.array([[d_beta1*d_beta1, d_beta2*d_beta1, d_delta1*d_beta1, d_delta2*d_beta1, d_gamma1*d_beta1, d_gamma2*d_beta1, d_alpha1*d_beta1, d_alpha2*d_beta1, d_eta11*d_beta1, d_eta12*d_beta1, d_eta21*d_beta1, d_eta22*d_beta1],
                            [d_beta1*d_beta2, d_beta2*d_beta2, d_delta1*d_beta2, d_delta2*d_beta2, d_gamma1*d_beta2, d_gamma2*d_beta2, d_alpha1*d_beta2, d_alpha2*d_beta2, d_eta11*d_beta2, d_eta12*d_beta2, d_eta21*d_beta2, d_eta22*d_beta2],
                            [d_beta1*d_delta1, d_beta2*d_delta1, d_delta1*d_delta1, d_delta2*d_delta1, d_gamma1*d_delta1, d_gamma2*d_delta1, d_alpha1*d_delta1, d_alpha2*d_delta1, d_eta11*d_delta1, d_eta12*d_delta1, d_eta21*d_delta1, d_eta22*d_delta1],
                            [d_beta1*d_delta2, d_beta2*d_delta2, d_delta1*d_delta2, d_delta2*d_delta2, d_gamma1*d_delta2, d_gamma2*d_delta2, d_alpha1*d_delta2, d_alpha2*d_delta2, d_eta11*d_delta2, d_eta12*d_delta2, d_eta21*d_delta2, d_eta22*d_delta2],
                            [d_beta1*d_gamma1, d_beta2*d_gamma1, d_delta1*d_gamma1, d_delta2*d_gamma1, d_gamma1*d_gamma1, d_gamma2*d_gamma1, d_alpha1*d_gamma1, d_alpha2*d_gamma1, d_eta11*d_gamma1, d_eta12*d_gamma1, d_eta21*d_gamma1, d_eta22*d_gamma1],
                            [d_beta1*d_gamma2, d_beta2*d_gamma2, d_delta1*d_gamma2, d_delta2*d_gamma2, d_gamma1*d_gamma2, d_gamma2*d_gamma2, d_alpha1*d_gamma2, d_alpha2*d_gamma2, d_eta11*d_gamma2, d_eta12*d_gamma2, d_eta21*d_gamma2, d_eta22*d_gamma2],
                            [d_beta1*d_alpha1, d_beta2*d_alpha1, d_delta1*d_alpha1, d_delta2*d_alpha1, d_gamma1*d_alpha1, d_gamma2*d_alpha1, d_alpha1*d_alpha1, d_alpha2*d_alpha1, d_eta11*d_alpha1, d_eta12*d_alpha1, d_eta21*d_alpha1, d_eta22*d_alpha1],
                            [d_beta1*d_alpha2, d_beta2*d_alpha2, d_delta1*d_alpha2, d_delta2*d_alpha2, d_gamma1*d_alpha2, d_gamma2*d_alpha2, d_alpha1*d_alpha2, d_alpha2*d_alpha2, d_eta11*d_alpha2, d_eta12*d_alpha2, d_eta21*d_alpha2, d_eta22*d_alpha2],
                            [d_beta1*d_eta11, d_beta2*d_eta11, d_delta1*d_eta11, d_delta2*d_eta11, d_gamma1*d_eta11, d_gamma2*d_eta11, d_alpha1*d_eta11, d_alpha2*d_eta11, d_eta11*d_eta11, d_eta12*d_eta11, d_eta21*d_eta11, d_eta22*d_eta11],
                            [d_beta1*d_eta12, d_beta2*d_eta12, d_delta1*d_eta12, d_delta2*d_eta12, d_gamma1*d_eta12, d_gamma2*d_eta12, d_alpha1*d_eta12, d_alpha2*d_eta12, d_eta11*d_eta12, d_eta12*d_eta12, d_eta21*d_eta12, d_eta22*d_eta12],
                            [d_beta1*d_eta21, d_beta2*d_eta21, d_delta1*d_eta21, d_delta2*d_eta21, d_gamma1*d_eta21, d_gamma2*d_eta21, d_alpha1*d_eta21, d_alpha2*d_eta21, d_eta11*d_eta21, d_eta12*d_eta21, d_eta21*d_eta21, d_eta22*d_eta21],
                            [d_beta1*d_eta22, d_beta2*d_eta22, d_delta1*d_eta22, d_delta2*d_eta22, d_gamma1*d_eta22, d_gamma2*d_eta22, d_alpha1*d_eta22, d_alpha2*d_eta22, d_eta11*d_eta22, d_eta12*d_eta22, d_eta21*d_eta22, d_eta22*d_eta22]])
    
    
    
    return grad_entries

def partial_sq(theta, *args):

    beta1, beta2, delta1, delta2, gamma1, gamma2, alpha1, alpha2, eta11, eta12, eta21, eta22 = theta
    
    # delta1 second derivatives
    
    d_delta1_d_eta11 = -tI1_all * (y_lag / n_lag) * np.exp(-eta11 * s_lag) * s_lag
    d_delta1_d_eta12 = 0
    d_delta2_d_eta11 = 0
    d_delta2_d_eta12 = -tI2_all * (y_lag / n_lag) * np.exp(-eta12 * s_lag) * s_lag
    d_gamma1_d_eta22 = 0
    d_gamma1_d_alpha2 = 0
    d_gamma2_d_eta21 = 0
    d_gamma2_d_alpha1 = 0
    d_alpha1_d_gamma2 = 0
    d_alpha1_d_eta22 = 0
    d_alpha1_d_alpha2 = 0
    d_alpha2_d_gamma1 = 0
    d_alpha2_d_eta21 = 0
    d_alpha2_d_alpha1 = 0
    d_eta11_d_delta1 = -tI1_all * s_lag * (y_lag / n_lag) * np.exp(-eta11 * s_lag)
    d_eta11_d_delta2 = 0
    d_eta11_d_eta11 = tI1_all * delta1 * (s_lag**2) * (y_lag / n_lag) * np.exp(-eta11 * s_lag)
    d_eta11_d_eta12 = 0
    d_eta12_d_delta1 = 0
    d_eta12_d_delta2 = -tI2_all * s_lag * (y_lag / n_lag) * np.exp(-eta12 * s_lag)
    d_eta12_d_eta11 = 0
    d_eta12_d_eta12 = tI2_all * delta2 * (s_lag**2) * (y_lag / n_lag) * np.exp(-eta12 * s_lag)
    d_eta21_d_gamma2 = 0
    d_eta21_d_eta22 = 0
    d_eta21_d_alpha2 = 0
    d_eta22_d_gamma1 = 0
    d_eta22_d_eta21 = 0
    d_eta22_d_alpha1 = 0
    
    # summations
    
    d_gamma1_d_eta21 = []
    d_gamma1_d_alpha1 = []
    d_gamma2_d_eta22 = []
    d_gamma2_d_alpha2 = []
    d_alpha1_d_gamma1 = []
    d_alpha1_d_eta21 = []
    d_alpha1_d_alpha1 = []
    d_alpha2_d_gamma2 = []
    d_alpha2_d_eta22 = []
    d_alpha2_d_alpha2 = []
    d_eta21_d_gamma1 = []
    d_eta21_d_eta21 = []
    d_eta21_d_alpha1 = []
    d_eta22_d_gamma2 = []
    d_eta22_d_eta22 = []
    d_eta22_d_alpha2 = []
    
    
    for year in [2014, 2015, 2016, 2017]:
    
        if year == 2014:
            dist = dist_2014
            sI2 = sI2_2014
            z_lag = z_lag_2014
            nz_lag = nz_lag_2014
            sz_lag = sz_lag_2014
            a_lag = a_lag_2014
            w_lag = w_lag_2014
            sI1_lag = sI1_lag_2014
            Nz = N_2014
            tI1 = tI1_2014
            tI2 = tI2_2014
            
        elif year == 2015:
            dist = dist_2015
            sI2 = sI2_2015
            z_lag = z_lag_2015
            nz_lag = nz_lag_2015
            sz_lag = sz_lag_2015
            a_lag = a_lag_2015
            w_lag = w_lag_2015
            sI1_lag = sI1_lag_2015
            Nz = N_2015
            tI1 = tI1_2015
            tI2 = tI2_2015
        
        elif year == 2016:
            dist = dist_2016
            sI2 = sI2_2016
            z_lag = z_lag_2016
            nz_lag = nz_lag_2016
            sz_lag = sz_lag_2016
            a_lag = a_lag_2016
            w_lag = w_lag_2016
            sI1_lag = sI1_lag_2016
            Nz = N_2016
            tI1 = tI1_2016
            tI2 = tI2_2016
            
        elif year == 2017:
            dist = dist_2017
            sI2 = sI2_2017
            z_lag = z_lag_2017
            nz_lag = nz_lag_2017
            sz_lag = sz_lag_2017
            a_lag = a_lag_2017
            w_lag = w_lag_2017
            sI1_lag = sI1_lag_2017
            Nz = N_2017
            tI1 = tI1_2017
            tI2 = tI2_2017
        
        for i in range(0, Nz):
            
            mask = np.arange(Nz) != i # mask out the current yard i
            
            d_gamma1_d_eta21_i = -tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1)) * sI1_lag[mask] * sz_lag[mask])
            d_gamma1_d_alpha1_i = tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI1_lag[mask])
            d_gamma2_d_eta22_i = -tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2)) * sI2[mask] * sz_lag[mask])
            d_gamma2_d_alpha2_i = tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI2[mask])
            d_alpha1_d_gamma1_i = tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI1_lag[mask])
            d_alpha1_d_eta21_i = -gamma1 * tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI1_lag[mask] * sz_lag[mask])
            d_alpha1_d_alpha1_i = gamma1 * tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1) * (np.log(1 + dist[:, i][mask].reshape(Nz-1, 1)))**2) * sI1_lag[mask])
            d_alpha2_d_gamma2_i = tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI2[mask])
            d_alpha2_d_eta22_i = -gamma2 * tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI2[mask] * sz_lag[mask])
            d_alpha2_d_alpha2_i = gamma2 * tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2) * (np.log(1 + dist[:, i][mask].reshape(Nz-1, 1)))**2) * sI2[mask])
            d_eta21_d_gamma1_i = -tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1)) * sI1_lag[mask] * sz_lag[mask])
            d_eta21_d_eta21_i = gamma1 * tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1)) * sI1_lag[mask] * (sz_lag[mask]**2))
            d_eta21_d_alpha1_i = -gamma1 * tI1[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta21 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha1) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI1_lag[mask] * sz_lag[mask])
            d_eta22_d_gamma2_i = -tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2)) * sI2[mask] * sz_lag[mask])
            d_eta22_d_eta22_i = gamma2 * tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2)) * sI2[mask] * (sz_lag[mask]**2))
            d_eta22_d_alpha2_i = -gamma2 * tI2[i] * np.sum((a_lag[mask] * (z_lag[mask] / nz_lag[mask])) * np.exp(-eta22 * sz_lag[mask]) * (w_lag[:, i][mask].reshape(Nz-1, 1) * -np.power(1 + dist[:, i][mask].reshape(Nz-1, 1), -alpha2) * np.log(1 + dist[:, i][mask].reshape(Nz-1, 1))) * sI2[mask] * sz_lag[mask])

            d_gamma1_d_eta21.append(d_gamma1_d_eta21_i)
            d_gamma1_d_alpha1.append(d_gamma1_d_alpha1_i)
            d_gamma2_d_eta22.append(d_gamma2_d_eta22_i)
            d_gamma2_d_alpha2.append(d_gamma2_d_alpha2_i)
            d_alpha1_d_gamma1.append(d_alpha1_d_gamma1_i)
            d_alpha1_d_eta21.append(d_alpha1_d_eta21_i)
            d_alpha1_d_alpha1.append(d_alpha1_d_alpha1_i)
            d_alpha2_d_gamma2.append(d_alpha2_d_gamma2_i)
            d_alpha2_d_eta22.append(d_alpha2_d_eta22_i)
            d_alpha2_d_alpha2.append(d_alpha2_d_alpha2_i)
            d_eta21_d_gamma1.append(d_eta21_d_gamma1_i)
            d_eta21_d_eta21.append(d_eta21_d_eta21_i)
            d_eta21_d_alpha1.append(d_eta21_d_alpha1_i)
            d_eta22_d_gamma2.append(d_eta22_d_gamma2_i)
            d_eta22_d_eta22.append(d_eta22_d_eta22_i)
            d_eta22_d_alpha2.append(d_eta22_d_alpha2_i)
        
    d_gamma1_d_eta21 = np.array(d_gamma1_d_eta21).reshape((N, 1))
    d_gamma1_d_alpha1 = np.array(d_gamma1_d_alpha1).reshape((N, 1))
    d_gamma2_d_eta22 = np.array(d_gamma2_d_eta22).reshape((N, 1))
    d_gamma2_d_alpha2 = np.array(d_gamma2_d_alpha2).reshape((N, 1))
    d_alpha1_d_gamma1 = np.array(d_alpha1_d_gamma1).reshape((N, 1))
    d_alpha1_d_eta21 = np.array(d_alpha1_d_eta21).reshape((N, 1))
    d_alpha1_d_alpha1 = np.array(d_alpha1_d_alpha1).reshape((N, 1))
    d_alpha2_d_gamma2 = np.array(d_alpha2_d_gamma2).reshape((N, 1))
    d_alpha2_d_eta22 = np.array(d_alpha2_d_eta22).reshape((N, 1))
    d_alpha2_d_alpha2 = np.array(d_alpha2_d_alpha2).reshape((N, 1))
    d_eta21_d_gamma1 = np.array(d_eta21_d_gamma1).reshape((N, 1))
    d_eta21_d_eta21 = np.array(d_eta21_d_eta21).reshape((N, 1))
    d_eta21_d_alpha1 = np.array(d_eta21_d_alpha1).reshape((N, 1))
    d_eta22_d_gamma2 = np.array(d_eta22_d_gamma2).reshape((N, 1))
    d_eta22_d_eta22 = np.array(d_eta22_d_eta22).reshape((N, 1))
    d_eta22_d_alpha2 = np.array(d_eta22_d_alpha2).reshape((N, 1))
        
        
    
    zero = np.zeros((N, 1))
    
    hess_entries = np.array([[zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero],    #d_beta1
                            [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero],    #d_beta2
                            [zero, zero, zero, zero, zero, zero, zero, zero, d_eta11_d_delta1, zero, zero, zero],    #d_delta1
                            [zero, zero, zero, zero, zero, zero, zero, zero, zero, d_eta12_d_delta2, zero, zero],    #d_delta2
                            [zero, zero, zero, zero, zero, zero, d_alpha1_d_gamma1, zero, zero, zero, d_eta21_d_gamma1, zero],    #d_gamma1
                            [zero, zero, zero, zero, zero, zero, zero, d_alpha2_d_gamma2, zero, zero, zero, d_eta22_d_gamma2],    #d_gamma2
                            [zero, zero, zero, zero, d_gamma1_d_alpha1, zero, d_alpha1_d_alpha1, zero, zero, zero, d_eta21_d_alpha1, zero],    #d_alpha1
                            [zero, zero, zero, zero, zero, d_gamma2_d_alpha2, zero, d_alpha2_d_alpha2, zero, zero, zero, d_eta22_d_alpha2],    #d_alpha2
                            [zero, zero, d_delta1_d_eta11, zero, zero, zero, zero, zero, d_eta11_d_eta11, zero, zero, zero],    #d_eta11
                            [zero, zero, zero, d_delta2_d_eta12, zero, zero, zero, zero, zero, d_eta12_d_eta12, zero, zero],    #d_eta12
                            [zero, zero, zero, zero, d_gamma1_d_eta21, zero, d_alpha1_d_eta21, zero, zero, zero, d_eta21_d_eta21, zero],    #d_eta21
                            [zero, zero, zero, zero, zero, d_gamma2_d_eta22, zero, d_alpha2_d_eta22, zero, zero, zero, d_eta22_d_eta22]])   #d_eta22
    
    
    return hess_entries

def gradient(theta, *args):

    mu = y - (n / (1 + np.exp(-eta(theta))))
    
    # Gradient 
    gradient = - (1 / N) * np.sum((partial(theta) * mu), axis=1)
    
    return gradient

def hessian(theta, *args):

    mu = y - (n / (1 + np.exp(-eta(theta))))
    
    # Hessian entries
    hessian = - (1 / N) * np.sum((partial_sq(theta) * mu - n * (partial_by_partial(theta)) * (np.exp(-eta(theta)) / (1 + np.exp(-eta(theta)))**2)), axis=2)
    hessian = hessian.reshape((12, 12))
    
    return hessian