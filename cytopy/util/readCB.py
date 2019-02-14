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
