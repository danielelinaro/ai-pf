
import numpy as np
from pfcommon import compute_generator_inertias

if __name__ == '__main__':

    def print_list(lst, prefix=''):
        print(prefix + '[' + ', '.join(map(lambda x: f'{x:.3f}', lst)) + ']')

    area_ID = 1
    dH = 0.2
    H_min, H_max, H_step = 3, 6.3, 0.3
    H_G2, H_G3 = [], []
    for target_H in np.r_[H_min : H_max : H_step] + dH:
        tmp = compute_generator_inertias(target_H, area_ID, verbose=False)
        H_G2.append(tmp['G 02'])
        H_G3.append(tmp['G 03'])

    print_list(H_G2,  '"G 02": ')
    print_list(H_G3,  '"G 03": ')
