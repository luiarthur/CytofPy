import re
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

def get_metrics(rd, ed, metric='elbo'):
    with open('{}/{}/log.txt'.format(rd, ed), 'r') as f:
        lines = f.readlines()
        content = '\n'.join(lines)
        metrics = re.findall('(?<={}:\s)-\d+\.\d+'.format(metric), content)
        metrics = list(map(lambda e: float(e), metrics))
    return metrics


RESULTS_DIR = 'results/sim1-vae'
METRIC = 'elbo'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        RESULTS_DIR = sys.argv[1]
        METRIC = sys.argv[2]
    else:
        print('Usage: python3 parse_elbo.py <path-to-results-dir>')
        print('Exiting...')
        sys.exit(1)

    os.makedirs('{}/metrics/'.format(RESULTS_DIR), exist_ok=True)

    EXP_DIRS = os.listdir(RESULTS_DIR)
    EXP_DIRS = list(filter(lambda ed: re.match('\d+', ed) is not None,  EXP_DIRS))
    tail = 100
    metric_mean_max = -float('inf')
    best_sim = None
        
    for ed in sorted(EXP_DIRS):
        metrics = get_metrics(RESULTS_DIR, ed, METRIC)
        metric_mean_curr = np.mean(metrics[-tail:])

        if metric_mean_curr > metric_mean_max:
            metric_mean_max = metric_mean_curr
            best_sim = ed

    for ed in sorted(EXP_DIRS):
        metrics = get_metrics(RESULTS_DIR, ed, METRIC)
        if ed != best_sim:
            plt.plot(metrics[-tail:], lw=.5)

    metrics_best = get_metrics(RESULTS_DIR, best_sim, METRIC)
    plt.plot(metrics_best[-tail:], label='best sim: {}'.format(best_sim), c='black')
    plt.legend()
    plt.savefig('{}/metrics/{}.pdf'.format(RESULTS_DIR, METRIC))
    plt.close()

    with open('{}/metrics/best_{}.txt'.format(RESULTS_DIR, METRIC), 'w') as f:
        msg = 'Best {}: {}/{}\n'.format(METRIC, RESULTS_DIR, best_sim)
        f.write(msg)
