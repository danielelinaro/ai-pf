
import numpy as np


__all__ = ['SynchMachAdynLoadConstS']


class SynchMachAdynLoadConstS (object):    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        par_names = 'w_base', 'V_base', 'T_base', 'P_load', 'Q_load', 'Ggnd', 'rstr', \
            'R_gen', 'X_gen', 'tag', 'E0', 'phiG', 'cosn', 'I_base_gen', 'pt', 'xmdm', \
            'dpu', 'addmt'
        return '\n'.join(map(str, [getattr(self, name) for name in par_names]))

    def __call__(self, t, y, ydot, res):
        w, phi, uBr, uBi, iLr, iLi, iGr, iGi = y
        wdot, phidot = ydot[:2]
        den = np.sqrt(3) * (uBr ** 2 + uBi ** 2)
        # mechanical torque
        n = w / self.w_base
        tm = self.pt / n - (self.xmdm + self.dpu * n - self.addmt)       # [p.u.]
        Tm = tm * self.T_base                                            # [Nm]
        # electrical torque
        n = 1 # neglect rotor speed variations
        ut = (uBr + 1j * uBi) / self.V_base                              # [p.u.]
        it = (iGr + 1j * iGi) / self.I_base_gen                          # [p.u.]
        psistr = (ut + self.rstr * it) / (1j * n)                        # [p.u.]
        te = (it.imag * psistr.real - it.real * psistr.imag) / self.cosn # [p.u.]
        Te = te * self.T_base                                          # [Nm]
        if hasattr(self, 'OU'):
            idx, = np.where(self.OU[:,0] <= t)
            if idx[-1] == t or idx[-1] == self.OU.shape[0] - 1:
                P_load = self.OU[idx[-1], 1]
            else:
                i = idx[-1]
                P_load = np.interp(t, self.OU[i:i+2, 0], self.OU[i:i+2, 1])
        else:
            P_load = self.P_load
        # with damping torques
        #res[0] = wdot - (tm - te - tdkd - tdpe) / tag * self.w_base
        # without damping torques
        res[0] = wdot - (tm - te) / self.tag * self.w_base
        res[1] = phidot
        res[2] = iGr - (iLr + self.Ggnd * uBr)
        res[3] = iGi - (iLi + self.Ggnd * uBi)
        res[4] = iLr - (P_load * uBr + self.Q_load * uBi) / den
        res[5] = iLi - (-self.Q_load * uBr + P_load * uBi) / den
        res[6] = uBr + self.R_gen * iGr - self.X_gen * iGi - self.E0 * np.cos(self.phiG)
        res[7] = uBi + self.X_gen * iGr + self.R_gen * iGi - self.E0 * np.sin(self.phiG)

