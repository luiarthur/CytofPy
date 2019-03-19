import re
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

def get_elbos(rd, ed):
    with open('{}/{}/log.txt'.format(rd, ed), 'r') as f:
        lines = f.readlines()
        content = '\n'.join(lines)
        elbos = re.findall('(?<=elbo:\s)-\d+\.\d+', content)
        elbos = list(map(lambda e: float(e), elbos))
    return elbos


RESULTS_DIR = 'results/sim1-vae'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        RESULTS_DIR = sys.argv[1]
    else:
        print('Usage: python3 parse_elbo.py <path-to-results-dir>')
        print('Exiting...')
        sys.exit(1)

    os.makedirs('{}/elbo/'.format(RESULTS_DIR), exist_ok=True)

    EXP_DIRS = os.listdir(RESULTS_DIR)
    EXP_DIRS = list(filter(lambda ed: re.match('\d+', ed) is not None,  EXP_DIRS))
    tail = 100
    elbo_mean_max = -float('inf')
    best_sim = None
        
    for ed in sorted(EXP_DIRS):
        elbos = get_elbos(RESULTS_DIR, ed)
        elbo_mean_curr = np.mean(elbos[-tail:])

        if elbo_mean_curr > elbo_mean_max:
            elbo_mean_max = elbo_mean_curr
            best_sim = ed

    for ed in sorted(EXP_DIRS):
        elbos = get_elbos(RESULTS_DIR, ed)
        if ed != best_sim:
            plt.plot(elbos[-tail:], lw=.5)

    elbos_best = get_elbos(RESULTS_DIR, best_sim)
    plt.plot(elbos_best[-tail:], label='best sim: {}'.format(best_sim), c='black')
    plt.legend()
    plt.savefig('{}/elbo/elbos.pdf'.format(RESULTS_DIR))
    plt.close()

    with open('{}/elbo/best_elbo.txt'.format(RESULTS_DIR), 'w') as f:
        msg = 'Best elbo: {}/{}\n'.format(RESULTS_DIR, best_sim)
        f.write(msg)
