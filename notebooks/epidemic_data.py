# epidemic_data.py
import numpy as np

# Load data
X_2014 = np.load('../data/processed/data_2014.npz')
X_2015 = np.load('../data/processed/data_2015.npz')
X_2016 = np.load('../data/processed/data_2016.npz')
X_2017 = np.load('../data/processed/data_2017.npz')

# Extract data for each year
N_2014 = X_2014['N']
N_2015 = X_2015['N']
N_2016 = X_2016['N']
N_2017 = X_2017['N']

dist_2014 = X_2014['distance']
dist_2015 = X_2015['distance']
dist_2016 = X_2016['distance']
dist_2017 = X_2017['distance']

tI1_2014 = X_2014['tI1'].reshape(N_2014, 1)
tI1_2015 = X_2015['tI1'].reshape(N_2015, 1)
tI1_2016 = X_2016['tI1'].reshape(N_2016, 1)
tI1_2017 = X_2017['tI1'].reshape(N_2017, 1)

tI2_2014 = X_2014['tI2'].reshape(N_2014, 1)
tI2_2015 = X_2015['tI2'].reshape(N_2015, 1)
tI2_2016 = X_2016['tI2'].reshape(N_2016, 1)
tI2_2017 = X_2017['tI2'].reshape(N_2017, 1)

sI2_2014 = X_2014['sI2'].reshape(N_2014, 1)
sI2_2015 = X_2015['sI2'].reshape(N_2015, 1)
sI2_2016 = X_2016['sI2'].reshape(N_2016, 1)
sI2_2017 = X_2017['sI2'].reshape(N_2017, 1)

# Load individual year data
def load_year_data(year, key):
    X = locals()[f'X_{year}']
    N = locals()[f'N_{year}']
    return X[key].reshape(N, 1)

# Load data for each year and month
for year in [2014, 2015, 2016, 2017]:
    for month in ['apr', 'may', 'jun', 'jul']:
        locals()[f'y_{month}_{year}'] = load_year_data(year, f'y_{month}')
        locals()[f'n_{month}_{year}'] = load_year_data(year, f'n_{month}')
        locals()[f'a_{month}_{year}'] = load_year_data(year, f'a_{month}')
        locals()[f's_{month}_{year}'] = load_year_data(year, f's_{month}')
        locals()[f'sI1_{month}_{year}'] = load_year_data(year, f'sI1_{month}')
        locals()[f'w_{month}_{year}'] = X_{year}[f'wind_{month}']  # Wind data is not reshaped

# Function to load and stack data for all years
def load_data(key, reshape=True):
    data = []
    for year in [2014, 2015, 2016, 2017]:
        X = locals()[f'X_{year}']
        N = locals()[f'N_{year}']
        if reshape:
            data.append(X[key].reshape(N, 1))
        else:
            data.append(X[key])
    return np.vstack(data)

# Load data for all months
y_apr = load_data('y_apr')
y_may = load_data('y_may')
y_jun = load_data('y_jun')
y_jul = load_data('y_jul')

n_apr = load_data('n_apr')
n_may = load_data('n_may')
n_jun = load_data('n_jun')
n_jul = load_data('n_jul')

a_apr = load_data('a_apr')
a_may = load_data('a_may')
a_jun = load_data('a_jun')
a_jul = load_data('a_jul')

s_apr = load_data('s_apr')
s_may = load_data('s_may')
s_jun = load_data('s_jun')
s_jul = load_data('s_jul')

sI1_apr = load_data('sI1_apr')
sI1_may = load_data('sI1_may')
sI1_jun = load_data('sI1_jun')
sI1_jul = load_data('sI1_jul')

# Load wind data (not reshaped)
w_apr = load_data('wind_apr', reshape=False)
w_may = load_data('wind_may', reshape=False)
w_jun = load_data('wind_jun', reshape=False)
w_jul = load_data('wind_jul', reshape=False)

# Calculate total N
N = N_2014 + N_2015 + N_2016 + N_2017

# Stack tI1 and tI2 for all years
tI1_all = np.vstack((tI1_2014, tI1_2015, tI1_2016, tI1_2017))
tI2_all = np.vstack((tI2_2014, tI2_2015, tI2_2016, tI2_2017))

# Create dictionaries for May-June and June-July data
may_jun_data = {
    'N': N,
    'y': y_jun,
    'n': n_jun,
    'y_lag': y_may,
    'n_lag': n_may,
    's_lag': s_may,
    'tI1_all': tI1_all,
    'tI2_all': tI2_all,
    'dist_2014': dist_2014,
    'dist_2015': dist_2015,
    'dist_2016': dist_2016,
    'dist_2017': dist_2017,
    'z_lag_2014': y_may_2014,
    'z_lag_2015': y_may_2015,
    'z_lag_2016': y_may_2016,
    'z_lag_2017': y_may_2017,
    'nz_lag_2014': n_may_2014,
    'nz_lag_2015': n_may_2015,
    'nz_lag_2016': n_may_2016,
    'nz_lag_2017': n_may_2017,
    'sz_lag_2014': s_may_2014,
    'sz_lag_2015': s_may_2015,
    'sz_lag_2016': s_may_2016,
    'sz_lag_2017': s_may_2017,
    'a_lag_2014': a_may_2014,
    'a_lag_2015': a_may_2015,
    'a_lag_2016': a_may_2016,
    'a_lag_2017': a_may_2017,
    'w_lag_2014': w_may_2014,
    'w_lag_2015': w_may_2015,
    'w_lag_2016': w_may_2016,
    'w_lag_2017': w_may_2017,
    'sI1_lag_2014': sI1_may_2014,
    'sI1_lag_2015': sI1_may_2015,
    'sI1_lag_2016': sI1_may_2016,
    'sI1_lag_2017': sI1_may_2017,
    'sI2_2014': sI2_2014,
    'sI2_2015': sI2_2015,
    'sI2_2016': sI2_2016,
    'sI2_2017': sI2_2017,
    'N_2014': N_2014,
    'N_2015': N_2015,
    'N_2016': N_2016,
    'N_2017': N_2017,
    'tI1_2014': tI1_2014,
    'tI1_2015': tI1_2015,
    'tI1_2016': tI1_2016,
    'tI1_2017': tI1_2017,
    'tI2_2014': tI2_2014,
    'tI2_2015': tI2_2015,
    'tI2_2016': tI2_2016,
    'tI2_2017': tI2_2017,
}

jun_jul_data = {
    'N': N,
    'y': y_jul,
    'n': n_jul,
    'y_lag': y_jun,
    'n_lag': n_jun,
    's_lag': s_jun,
    'tI1_all': tI1_all,
    'tI2_all': tI2_all,
    'dist_2014': dist_2014,
    'dist_2015': dist_2015,
    'dist_2016': dist_2016,
    'dist_2017': dist_2017,
    'z_lag_2014': y_jun_2014,
    'z_lag_2015': y_jun_2015,
    'z_lag_2016': y_jun_2016,
    'z_lag_2017': y_jun_2017,
    'nz_lag_2014': n_jun_2014,
    'nz_lag_2015': n_jun_2015,
    'nz_lag_2016': n_jun_2016,
    'nz_lag_2017': n_jun_2017,
    'sz_lag_2014': s_jun_2014,
    'sz_lag_2015': s_jun_2015,
    'sz_lag_2016': s_jun_2016,
    'sz_lag_2017': s_jun_2017,
    'a_lag_2014': a_jun_2014,
    'a_lag_2015': a_jun_2015,
    'a_lag_2016': a_jun_2016,
    'a_lag_2017': a_jun_2017,
    'w_lag_2014': w_jun_2014,
    'w_lag_2015': w_jun_2015,
    'w_lag_2016': w_jun_2016,
    'w_lag_2017': w_jun_2017,
    'sI1_lag_2014': sI1_jun_2014,
    'sI1_lag_2015': sI1_jun_2015,
    'sI1_lag_2016': sI1_jun_2016,
    'sI1_lag_2017': sI1_jun_2017,
    'sI2_2014': sI2_2014,
    'sI2_2015': sI2_2015,
    'sI2_2016': sI2_2016,
    'sI2_2017': sI2_2017,
    'N_2014': N_2014,
    'N_2015': N_2015,
    'N_2016': N_2016,
    'N_2017': N_2017,
    'tI1_2014': tI1_2014,
    'tI1_2015': tI1_2015,
    'tI1_2016': tI1_2016,
    'tI1_2017': tI1_2017,
    'tI2_2014': tI2_2014,
    'tI2_2015': tI2_2015,
    'tI2_2016': tI2_2016,
    'tI2_2017': tI2_2017,
}

# You can add any additional data processing or organization here if needed