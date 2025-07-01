
import numpy as np


__all__ = ['SynchMachAdynLoadConstS']


class SynchMachAdynLoadConstS (object):    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


    def __str__(self):
        par_names = 'w_base', 'V_base', 'P_load', 'Q_load', 'Ggnd', 'rstr', \
            'R_gen', 'X_gen', 'tag', 'E0', 'cosn', 'I_base_gen', 'pt', 'xmdm', \
            'dpu', 'addmt', 'dkd', 'dpe', 'w_ref'
        return '\n'.join(map(str, [getattr(self, name) for name in par_names]))


    def _compute_P_load(self, t):
        if not hasattr(self, 'OU'):
            return self.P_load
        idx, = np.where(self.OU[:,0] <= t)
        if idx[-1] == t or idx[-1] == self.OU.shape[0] - 1:
            return self.OU[idx[-1], 1]
        i = idx[-1]
        return np.interp(t, self.OU[i:i+2, 0], self.OU[i:i+2, 1])

    
    def jac(self, t, y, ydot, cj, J):
        w, phi, uBr, uBi, iLr, iLi, iGr, iGi = y
        n = 1
        ubr = uBr / self.V_base
        ubi = uBi / self.V_base
        igr = iGr / self.I_base_gen
        igi = iGi / self.I_base_gen
        den = n * self.cosn * self.tag
        J[0,:] = [self.pt / (n**2 * self.tag),
                   0.0,
                   (igr / self.V_base) / den * self.w_base,
                   (igi / self.V_base) / den * self.w_base,
                   0.0,
                   0.0,
                   (ubr + 2 * self.rstr * igr) / den * self.w_base / self.I_base_gen,
                   (ubi + 2 * self.rstr * igi) / den * self.w_base / self.I_base_gen]
        J[1,:] = [0. for _ in range(8)]
        J[2,:] = [0., 0., -self.Ggnd,         0., -1.,  0., 1., 0.]
        J[3,:] = [0., 0.,         0., -self.Ggnd,  0., -1., 0., 1.]
        den = np.sqrt(3) * (uBr ** 2 + uBi ** 2) ** 2
        P_load = self._compute_P_load(t)
        J[4,:] = [0., 0.,
                  -(     P_load * (uBi**2 - uBr**2) - 2 * self.Q_load * uBr * uBi) / den,
                  -(self.Q_load * (uBr**2 - uBi**2) - 2 *      P_load * uBr * uBi) / den,
                  1., 0., 0., 0.]
        J[5,:] = [0., 0.,
                  -(self.Q_load * (uBr**2 - uBi**2) - 2 *      P_load * uBr * uBi) / den,
                  -(     P_load * (uBr**2 - uBi**2) - 2 * self.Q_load * uBr * uBi) / den,
                  0., 1., 0., 0.]
        J[6,:] = [0.,  self.E0 * np.sin(phi), 1., 0., 0., 0., self.R_gen, -self.X_gen]
        J[7,:] = [0., -self.E0 * np.cos(phi), 0., 1., 0., 0., self.X_gen,  self.R_gen]
        return 0


    def __call__(self, t, y, ydot, res):
        w, phi, uBr, uBi, iLr, iLi, iGr, iGi = y
        wdot, phidot = ydot[:2]
        # without the sqrt(3) coefficient, eqs. 4 & 5 below are not satisfied at the power
        # flow solution found by PowerFactory
        den = np.sqrt(3) * (uBr ** 2 + uBi ** 2)
        # mechanical torque
        n = w / self.w_base
        nref = 1.
        tm = self.pt / n - (self.xmdm + self.dpu * n - self.addmt)       # [p.u.]
        # damping torques
        n_ref = self.w_ref / self.w_base
        tdkd = self.dkd * (n - n_ref)
        tdpe = self.dpe / n * (n - n_ref)
        # electrical torque
        n = 1 # neglect rotor speed variations
        ut = (uBr + 1j * uBi) / self.V_base                              # [p.u.]
        it = (iGr + 1j * iGi) / self.I_base_gen                          # [p.u.]
        psistr = (ut + self.rstr * it) / (1j * n)                        # [p.u.]
        te = (it.imag * psistr.real - it.real * psistr.imag) / self.cosn # [p.u.]
        P_load = self._compute_P_load(t)
        # with damping torques
        res[0] = wdot - (tm - te - tdkd - tdpe) / self.tag * self.w_base
        # without damping torques
        #res[0] = wdot - (tm - te) / self.tag * self.w_base
        res[1] = phidot
        res[2] = iGr - (iLr + self.Ggnd * uBr)
        res[3] = iGi - (iLi + self.Ggnd * uBi)
        res[4] = iLr - (P_load * uBr + self.Q_load * uBi) / den
        res[5] = iLi - (-self.Q_load * uBr + P_load * uBi) / den
        res[6] = uBr + self.R_gen * iGr - self.X_gen * iGi - self.E0 * np.cos(phi)
        res[7] = uBi + self.X_gen * iGr + self.R_gen * iGi - self.E0 * np.sin(phi)

        return 0
    
