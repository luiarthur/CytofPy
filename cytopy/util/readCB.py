import torch

def readCB(cb_filepath:str):
    with open(cb_filepath, 'r') as f:
        # Read number of cells in first line
        N = f.readline()
        N = N.split()[1:]
        N = [int(n) for n in N]
        
        # Number of samples
        I = len(N)

        # Read marker names
        markers = f.readline().split()

        # Read data into separate samples
        y = []
        m = []
        for i in range(I):
            yi = []
            for _ in range(N[i]):
                line = f.readline().split()
                line = [float(obs.replace('NA', 'nan')) for obs in line]
                yi.append(line)
            y.append(torch.tensor(yi))
            m.append(torch.isnan(y[i]))

    return {'N': N, 'markers': markers, 'y': y, 'm': m}

def preprocess(data, rm_cells_below=-6.0):
    idx = []
    I = len(data['y'])
    for i in range(I):
        is_above_min = data['y'][i] > rm_cells_below
        is_nan = torch.isnan(data['y'][i])
        idx_i = (is_above_min + is_nan).prod(1)
        idx_i = idx_i.nonzero().squeeze()
        data['N'][i] = len(idx_i)
        data['y'][i] = data['y'][i][idx_i, :]
        data['m'][i] = data['m'][i][idx_i, :]
