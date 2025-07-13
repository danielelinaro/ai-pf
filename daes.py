
import numpy as np
from scikits.odes.sundials import ida

__all__ = ['SynchMachAdynLoadConstS']


class SynchMachAdynLoadConstS():
    algebraic_vars_idx = [2, 3, 4, 5, 6, 7]

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.res = None
        self.jac = None
        self.neq = 8
        self.exclalg_err = False

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

    def set_res(self, resfunction):
        """Function to set the resisual function as required by IDA"""
        self.res = resfunction

    def set_jac(self, jacfunction):
        """Function to set the resisual function as required by IDA"""
        self.jac = jacfunction
    

class resindex1(ida.IDA_RhsFunction):
    """ Residual function class as needed by the IDA DAE solver"""

    def set_power_system(self, power_sys):
        """Set the power system problem to solve to have access to the data"""
        self.power_sys = power_sys

    def evaluate(self, t, y, ydot, res, userdata):
        w, phi, uBr, uBi, iLr, iLi, iGr, iGi = y
        wdot, phidot = ydot[:2]
        # without the sqrt(3) coefficient, eqs. 4 & 5 below are not satisfied at the power
        # flow solution found by PowerFactory
        den = np.sqrt(3) * (uBr ** 2 + uBi ** 2)
        # parameters of the system
        w_base     = self.power_sys.w_base
        w_ref      = self.power_sys.w_ref
        pt         = self.power_sys.pt
        xmdm       = self.power_sys.xmdm
        dpu        = self.power_sys.dpu
        dkd        = self.power_sys.dkd
        dpe        = self.power_sys.dpe
        addmt      = self.power_sys.addmt
        V_base     = self.power_sys.V_base
        I_base_gen = self.power_sys.I_base_gen
        rstr       = self.power_sys.rstr
        cosn       = self.power_sys.cosn
        tag        = self.power_sys.tag
        Ggnd       = self.power_sys.Ggnd
        Q_load     = self.power_sys.Q_load
        R_gen      = self.power_sys.R_gen
        X_gen      = self.power_sys.X_gen
        E0         = self.power_sys.E0
        # mechanical torque
        n = w / w_base
        nref = 1.
        tm = pt / n - (xmdm + dpu * n - addmt)       # [p.u.]
        # damping torques
        n_ref = w_ref / w_base
        tdkd = dkd * (n - n_ref)
        tdpe = dpe / n * (n - n_ref)
        # electrical torque
        n = 1 # neglect rotor speed variations
        ut = (uBr + 1j * uBi) / V_base                              # [p.u.]
        it = (iGr + 1j * iGi) / I_base_gen                          # [p.u.]
        psistr = (ut + rstr * it) / (1j * n)                        # [p.u.]
        te = (it.imag * psistr.real - it.real * psistr.imag) / cosn # [p.u.]
        P_load = self.power_sys._compute_P_load(t)
        # with damping torques
        res[0] = wdot - (tm - te - tdkd - tdpe) / tag * w_base
        # without damping torques
        #res[0] = wdot - (tm - te) / self.tag * self.w_base
        res[1] = phidot
        res[2] = iGr - (iLr + Ggnd * uBr)
        res[3] = iGi - (iLi + Ggnd * uBi)
        res[4] = iLr - (P_load * uBr + Q_load * uBi) / den
        res[5] = iLi - (-Q_load * uBr + P_load * uBi) / den
        res[6] = uBr + R_gen * iGr - X_gen * iGi - E0 * np.cos(phi)
        res[7] = uBi + X_gen * iGr + R_gen * iGi - E0 * np.sin(phi)
        return 0


class jacindex1(ida.IDA_JacRhsFunction):

    def set_power_system(self, power_sys):
        """Set the power system problem to solve to have access to the data"""
        self.power_sys = power_sys

    def evaluate(self, t, y, ydot, res, cj, jac, userdata):
        w, phi, uBr, uBi, iLr, iLi, iGr, iGi = y
        # parameters of the system
        w_base     = self.power_sys.w_base
        pt         = self.power_sys.pt
        V_base     = self.power_sys.V_base
        I_base_gen = self.power_sys.I_base_gen
        rstr       = self.power_sys.rstr
        cosn       = self.power_sys.cosn
        tag        = self.power_sys.tag
        Ggnd       = self.power_sys.Ggnd
        Q_load     = self.power_sys.Q_load
        R_gen      = self.power_sys.R_gen
        X_gen      = self.power_sys.X_gen
        E0         = self.power_sys.E0

        n = 1
        ubr = uBr / V_base
        ubi = uBi / V_base
        igr = iGr / I_base_gen
        igi = iGi / I_base_gen
        P_load = self.power_sys._compute_P_load(t)

        jac[:,:] = 0.
        den = n * cosn * tag
        jac[0,0] = pt / (n**2 * tag)
        jac[0,2] = (igr / V_base) / den * w_base
        jac[0,3] = (igi / V_base) / den * w_base
        jac[0,6] = (ubr + 2 * rstr * igr) / den * w_base / I_base_gen
        jac[0,7] = (ubi + 2 * rstr * igi) / den * w_base / I_base_gen
        jac[2,2] = -Ggnd
        jac[2,4] = -1.
        jac[2,6] = 1.
        jac[3,3] = -Ggnd
        jac[3,5] = -1.
        jac[3,7] = 1.
        den = np.sqrt(3) * (uBr ** 2 + uBi ** 2) ** 2
        jac[4,2] = -(P_load * (uBi**2 - uBr**2) - 2 * Q_load * uBr * uBi) / den
        jac[4,3] = -(Q_load * (uBr**2 - uBi**2) - 2 * P_load * uBr * uBi) / den
        jac[4,4] = 1.
        jac[5,2] = -(Q_load * (uBr**2 - uBi**2) - 2 * P_load * uBr * uBi) / den
        jac[5,3] = -(P_load * (uBr**2 - uBi**2) - 2 * Q_load * uBr * uBi) / den
        jac[5,5] = 1.
        jac[6,1] = E0 * np.sin(phi)
        jac[6,2] = 1.
        jac[6,6] = R_gen
        jac[6,7] = -X_gen
        jac[7,1] = -E0 * np.cos(phi)
        jac[7,3] = 1.
        jac[7,6] = X_gen
        jac[7,7] = R_gen
        return 0


