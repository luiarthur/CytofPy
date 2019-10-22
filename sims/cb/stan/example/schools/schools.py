import pystan

# Data for STAN model
schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

# Read STAN model
sm = pystan.StanModel(file='schools.stan')

# Fit STAN model
fit = sm.sampling(data=schools_dat, iter=1000, chains=4)
