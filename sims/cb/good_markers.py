import numpy as np
import pandas as pd

dat = pd.read_csv('data/cb.csv').fillna(-5.9)
dat = dat[(dat > -6).all(1)]

# Markers near zero
near_zero = (dat
             .groupby('sample_id')
             .apply(lambda x: (abs(x) < .5).mean(0))
             .T) > .25

# Markers mostly neg
markers_mostly_neg = (dat
                      .groupby('sample_id')
                      .apply(lambda x: (x < 0).mean(0))
                      .T) > .9
# Markers mostly pos
markers_mostly_pos = (dat
                      .groupby('sample_id')
                      .apply(lambda x: (x > 0).mean(0))
                      .T) > .9

bad_markers = pd.DataFrame({'near_zero_any': near_zero.any(1),
                            'mostly_neg_all': markers_mostly_neg.all(1), 
                            'mostly_pos_all': markers_mostly_pos.all(1)})

print('markername,token')
for m in range(bad_markers.shape[0] - 1):
    marker = bad_markers.iloc[m]
    name = marker.name
    if marker['near_zero_any']:
        token = 'o'
    elif marker['mostly_neg_all']:
        token = '-'
    elif marker['mostly_pos_all']:
        token = '+'
    else:
        token = ''
    print('{},{}'.format(name, token))
