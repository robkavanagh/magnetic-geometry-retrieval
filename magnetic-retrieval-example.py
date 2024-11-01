import numpy as np
from pandas import read_csv
import ultranest
from ultranest.stepsampler import SliceSampler
import h5py # Note: required for parallelising with mpiexec

rad_per_deg = np.pi / 180

# Number of steps for the step sampler
nsteps = 10

# Number of iterations for solver (gives error of 1e-10 for F)
n_solver = 5 

# Frequency bands
n_per_band = 8
freq_high = np.linspace(1500, 1300, n_per_band)[1:] # skip 1st one as its 1500 MHz
freq_low = np.linspace(1150, 950, n_per_band)


##########################################
# Data
##########################################

# Load data
data = read_csv('data.csv')
phase = data['Phase'].values
flux_high = data['1300-1500 MHz flux (mJy)'].values
flux_high_err = data['1300-1500 MHz flux error (mJy)'].values
flux_low = data['950-1150 MHz flux (mJy)'].values
flux_low_err = data['950-1150 MHz flux error (mJy)'].values
nphase = len(phase)

# To make compatible with model flux array
flux_high = np.expand_dims(flux_high, axis = 1)
flux_high_err = np.expand_dims(flux_high_err, axis = 1)
flux_low = np.expand_dims(flux_low, axis = 1)
flux_low_err = np.expand_dims(flux_low_err, axis = 1)


##########################################
# Model
##########################################

# Model for the lightcurve of 2 active field lines
# 
# Inputs
# - cos_i: Cosine of the inclination [-1, 1]
# - beta: Magnetic obliquity (deg)
# - phase_0: Rotation phase at time zero of the MeerKAT observations [0, 1]
# - alpha: Emission cone opening angle (deg)
# - dalpha: Emission cone thickness (deg)
# - theta_B_1: Magnetic co-latitude of the emission cone on AFL 1 (deg)
# - theta_B_2: Magnetic co-latitude of the emission cone on AFL 2 (deg)
# - phi_B_1: Magnetic longitude of AFL 1 (deg)
# - phi_B_2: Magnetic longitude of AFL 2 (deg)
# - F_1: Flux density of AFL 1 (mJy)
# - F_2: Flux density of AFL 2 (mJy)
# 
# Returns the flux density at each phase of the MeerKAT observations

def model(params):

	cos_i, beta, phase_0, alpha, dalpha, theta_B_1, theta_B_2, phi_B_1, phi_B_2, F_1, F_2 = params

	# Convert angles to radians
	beta *= rad_per_deg
	alpha *= rad_per_deg
	dalpha *= rad_per_deg
	theta_B_1 *= rad_per_deg
	theta_B_2 *= rad_per_deg
	phi_B_1 *= rad_per_deg
	phi_B_2 *= rad_per_deg

	# Rotation phase
	phi_rot = 2 * np.pi * (phase_0 + phase)

	# Repeated terms
	sin_i = np.sin(np.arccos(cos_i))
	sin_beta = np.sin(beta)
	cos_beta = np.cos(beta)
	sin_phi_rot = np.sin(phi_rot)
	cos_phi_rot = np.cos(phi_rot)
	f = sin_i * sin_beta
	g = cos_i * cos_beta

	flux = np.zeros(nphase)

	# Loop over each AFL
	for j in range(2):

		# Specify AFL parameters
		theta_B = [theta_B_1, theta_B_2][j]
		phi_B = [phi_B_1, phi_B_2][j]
		F = [F_1, F_2][j]

		# Repeated terms
		cos_theta_B = np.cos(theta_B)
		sin_phi_B = np.sin(phi_B)
		cos_phi_B = np.cos(phi_B)

		# Compute the beam angle from each hemisphere
		a = 3 * np.sin(theta_B) * cos_theta_B / (1 + 3 * cos_theta_B ** 2) ** 0.5
		b = (3 * cos_theta_B ** 2 - 1) / (1 + 3 * cos_theta_B ** 2) ** 0.5
		c = - sin_i * np.sin(phi_B)
		d = sin_i * cos_beta * cos_phi_B
		e = - cos_i * sin_beta * cos_phi_B
		T_1 = a * c
		T_2 = a * d
		T_3 = b * f
		T_4 = a * e
		T_5 = b * g
		gamma_N = np.arccos(T_1 * sin_phi_rot + (T_2 + T_3) * cos_phi_rot + (T_4 + T_5))
		gamma_S = np.arccos(T_1 * sin_phi_rot + (T_2 - T_3) * cos_phi_rot + (T_4 - T_5))

		# Add contribution of each AFL to the total flux
		flux += F * (np.exp(- 0.5 * ((gamma_N - alpha) / dalpha) ** 2) - np.exp(- 0.5 * ((gamma_S - alpha) / dalpha) ** 2))

	return flux


##########################################
# Function setup
##########################################

# Parameter bounds
cos_i_min = 0
cos_i_max = 1
beta_min = 0
beta_max = 90
phase_0_min = 0
phase_0_max = 1
alpha_min, alpha_max = 0, 90
dalpha_min = 0
dalpha_max = 10
theta_1500_min = 0
theta_1500_max = 71.36	# Required for emission at 950 MHz to occur on each AFL
phi_B_min = 0
phi_B_max = 360
F_min = 1
F_max = 10

# Uniform priors
def log_prior_transform(cube):

	params = cube.copy()

	params[0] = cube[0] * (cos_i_max - cos_i_min) + cos_i_min
	params[1] = cube[1] * (beta_max - beta_min) + beta_min
	params[2] = cube[2] * (phase_0_max - phase_0_min) + phase_0_min
	params[3] = cube[3] * (alpha_max - alpha_min) + alpha_min
	params[4] = cube[4] * (dalpha_max - dalpha_min) + dalpha_min
	params[5] = cube[5] * (theta_1500_max - theta_1500_min) + theta_1500_min
	params[6] = cube[6] * (theta_1500_max - theta_1500_min) + theta_1500_min
	params[7] = cube[7] * (phi_B_max - phi_B_min) + phi_B_min
	params[8] = cube[8] * (phi_B_max - phi_B_min) + phi_B_min
	params[9] = cube[9] * (F_max - F_min) + F_min
	params[10] = cube[10] * (F_max - F_min) + F_min

	return params

# Likelihood function
def log_likelihood(params):

	cos_i, beta, phase_0, alpha, dalpha, theta_1500_1, theta_1500_2, phi_B_1, phi_B_2, F_1, F_2 = params

	# Force 2nd line ahead of 1st
	if phi_B_2 < phi_B_1: return - 1e50 * abs(phi_B_2 - phi_B_1)

	else:

		##########################################
		# Solve for co-latitude of each frequency
		##########################################

		x_1500_1 = np.sin(theta_1500_1 * rad_per_deg) ** 2
		x_1500_2 = np.sin(theta_1500_2 * rad_per_deg) ** 2

		Q_high_1 = (1500 / freq_high) ** 2 * x_1500_1 ** 6 / (4 - 3 * x_1500_1)
		Q_high_2 = (1500 / freq_high) ** 2 * x_1500_2 ** 6 / (4 - 3 * x_1500_2)

		Q_low_1 = (1500 / freq_low) ** 2 * x_1500_1 ** 6 / (4 - 3 * x_1500_1)
		Q_low_2 = (1500 / freq_low) ** 2 * x_1500_2 ** 6 / (4 - 3 * x_1500_2)

		x_high_1 = np.full(n_per_band - 1, x_1500_1)
		x_high_2 = np.copy(x_high_1)
		x_low_1 = np.full(n_per_band, x_1500_1)
		x_low_2 = np.copy(x_low_1)

		for _ in range(n_solver):

			x_high_1 = x_high_1 - (x_high_1 ** 6 + 3 * Q_high_1 * x_high_1 - 4 * Q_high_1) / (6 * x_high_1 ** 5 + 3 * Q_high_1)
			x_high_2 = x_high_2 - (x_high_2 ** 6 + 3 * Q_high_2 * x_high_2 - 4 * Q_high_2) / (6 * x_high_2 ** 5 + 3 * Q_high_2)
			x_low_1 = x_low_1 - (x_low_1 ** 6 + 3 * Q_low_1 * x_low_1 - 4 * Q_low_1) / (6 * x_low_1 ** 5 + 3 * Q_low_1)
			x_low_2 = x_low_2 - (x_low_2 ** 6 + 3 * Q_low_2 * x_low_2 - 4 * Q_low_2) / (6 * x_low_2 ** 5 + 3 * Q_low_2)

		theta_high_1 = np.arcsin(x_high_1 ** 0.5) / rad_per_deg
		theta_high_2 = np.arcsin(x_high_2 ** 0.5) / rad_per_deg
		theta_high_1 = np.insert(theta_high_1, 0, theta_1500_1) # add in co-latitude at 1500 MHz
		theta_high_2 = np.insert(theta_high_2, 0, theta_1500_2)
		theta_low_1 = np.arcsin(x_low_1 ** 0.5) / rad_per_deg
		theta_low_2 = np.arcsin(x_low_2 ** 0.5) / rad_per_deg


		##########################################
		# Compute model flux for each frequency
		##########################################

		flux_high_model = np.zeros((nphase, n_per_band))
		flux_low_model = np.zeros((nphase, n_per_band))

		for j in range(n_per_band):

			params_high = cos_i, beta, phase_0, alpha, dalpha, theta_high_1[j], theta_high_2[j], phi_B_1, phi_B_2, F_1, F_2
			params_low = cos_i, beta, phase_0, alpha, dalpha, theta_low_1[j], theta_low_2[j], phi_B_1, phi_B_2, F_1, F_2

			flux_high_model[:, j] = model(params_high)
			flux_low_model[:, j] = model(params_low)


		##########################################
		# Compute likelihood
		##########################################

		log_L_high = - 0.5 * (np.log(2 * np.pi * flux_high_err ** 2) + ((flux_high - flux_high_model) / flux_high_err) ** 2).sum()
		log_L_low = - 0.5 * (np.log(2 * np.pi * flux_low_err ** 2) + ((flux_low - flux_low_model) / flux_low_err) ** 2).sum()

		return log_L_high + log_L_low


##########################################
# Sampling
##########################################

paramnames = ['cos_i', 'beta', 'phase_0', 'alpha', 'dalpha', 'theta_1500_1', 'theta_1500_2', 'phi_B_1', 'phi_B_2', 'F_1', 'F_2']

# Run the sampler
sampler = ultranest.ReactiveNestedSampler(paramnames, log_likelihood, transform = log_prior_transform, wrapped_params = [False, False, True, False, False, False, False, True, True, False, False], log_dir = 'run', resume = 'overwrite')
sampler.stepsampler = SliceSampler(nsteps = nsteps, generate_direction = ultranest.stepsampler.generate_mixture_random_direction)
sampler.run(viz_callback = False, show_status = True)

