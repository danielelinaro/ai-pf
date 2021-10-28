
import re
import tables
import numpy as np

__all__ = ['get_simulation_variables', 'get_simulation_time', 'get_simulation_dt',
           'get_ID', 'get_line_bus_IDs', 'normalize', 'OU', 'OU_2', 'run_load_flow',
           'print_load_flow', 'correct_Vd_Vq', 'find_element_by_name',
           'is_voltage', 'is_power', 'is_frequency', 'compute_generator_inertias',
           'BaseParameters', 'sort_objects_by_name']


class BaseParameters (tables.IsDescription):
    F0    = tables.Float64Col()
    frand = tables.Float64Col()


def sort_objects_by_name(objects):
    argsort = lambda lst: [i for i,_ in sorted(enumerate(lst), key=lambda x: x[1])]
    idx = argsort([obj.loc_name for obj in objects])
    return [objects[i] for i in idx]


def compute_generator_inertias(target_H_area, area_ID, default_H='IEEE39', S='IEEE39', areas_map='IEEE39', F0=60, verbose=True):
    if default_H == 'IEEE39':
        default_H = {'G 01': 5.0, 'G 02': 4.33, 'G 03': 4.47, 'G 04': 3.57, 'G 05': 4.33, 
                     'G 06': 4.35, 'G 07': 3.77, 'G 08': 3.47, 'G 09': 3.45, 'G 10': 4.20}
    if S == 'IEEE39':
        S = {'G 01': 10000.0, 'G 02': 700.0, 'G 03': 800.0, 'G 04': 800.0, 'G 05': 300.0,
             'G 06': 800.0, 'G 07': 700.0, 'G 08': 700.0, 'G 09': 1000.0, 'G 10': 1000.0}
    if areas_map == 'IEEE39':
        areas_map = {
            1: ['G 02', 'G 03', 'G 10'],
            2: ['G 04', 'G 05', 'G 06', 'G 07'],
            3: ['G 08', 'G 09'],
            4: ['G 01']
        }
    num, den = 0,0
    generator_names = areas_map[area_ID]
    N_generators_in_area = len(generator_names)
    for generator_name in generator_names:
        num += S[generator_name] * default_H[generator_name]
        den += S[generator_name]
    default_H_area = num / den 
    # default_E_area = num * 1e-3
    # default_M_area = 2 * num * 1e-3 / F0

    if verbose: print(f'default H area {area_ID} = {default_H_area:g} s')

    H = default_H.copy()
    dH = target_H_area - default_H_area
    S_tot = np.sum([S[generator_name] for generator_name in generator_names])
    num, den = 0,0
    for generator_name in generator_names:
        H[generator_name] += dH * S_tot / N_generators_in_area / S[generator_name]
        num += S[generator_name] * H[generator_name]
        den += S[generator_name]
        if verbose: print(f'{generator_name}: H = {H[generator_name]:.3f} s.')
    if np.abs(num / den - target_H_area) > 1e-9:
        raise Exception('Cannot find a combination of generator inertias that match the requested area inertia')
    if verbose: print(f'H area {area_ID} = {num / den:g} s') 
    
    return H


is_voltage = lambda var_name: var_name in ('m:ur', 'm:ui', 'm:u')
is_power = lambda var_name: var_name in ('m:P:bus1', 'm:Q:bus1', \
                                         'm:Psum:bus1', 'm:Qsum:bus1')
is_frequency = lambda var_name: var_name in ('m:fe', )


def find_element_by_name(elements, name):
    found = False
    for elem in elements:
        if elem.loc_name == name:
            found = True
            break
    if found: return elem
    return None


def correct_Vd_Vq(Vd, Vq, delta):
    """
       Vd - (MxN) array, where M is the number of samples and N the number of buses at which Vd is recorded
       Vq - (MxN) array, where M is the number of samples and N the number of buses at which Vq is recorded
    delta - (Mx1) array, measured in degrees
    """
    # convert delta to radians
    try:
        n = Vd.shape[1]
        delta_ref = np.tile(delta / 180 * np.pi, [n,1]).T
    except:
        delta_ref = delta / 180 * np.pi
    mod = np.sqrt(Vd**2 + Vq**2)
    angle = np.arctan2(Vq, Vd) - delta_ref
    return mod * np.cos(angle), mod * np.sin(angle)


def get_simulation_variables(res, var_name, elements=None, elements_name=None, app=None, decimation=1, full_output=False):
    if elements is None:
        if elements_name is None:
            raise Exception('You must provide one of "elements" or "elements_name"')
        if app is None:
            raise Exception('You must provide "app" if "elements_name" is passed')
        full_output = True
        elements = app.GetCalcRelevantObjects(elements_name)
    n_samples = res.GetNumberOfRows()
    variables = np.zeros((int(np.ceil(n_samples / decimation)), len(elements)))
    for i,element in enumerate(elements):
        col = res.FindColumn(element, var_name)
        if col < 0:
            raise Exception(f'Variable {var_name} is not available.')
        variables[:,i] = np.array([res.GetValue(j, col)[1] for j in range(0, n_samples, decimation)])
    if full_output:
        return np.squeeze(variables), elements
    return np.squeeze(variables)


get_simulation_time = lambda res, decimation=1: \
    np.array([res.GetValue(i, -1)[1] for i in range(0, res.GetNumberOfRows(), decimation)])


get_simulation_dt = lambda res: np.diff([res.GetValue(i, -1)[1] for i in range(2)])[0]


get_ID = lambda elem: int(re.findall('\d+', elem.loc_name)[0])


get_line_bus_IDs = lambda line: tuple(map(int, re.findall('\d+', line.loc_name)))


def normalize(x):
    n = x.shape[0]
    xm = np.tile(x.mean(axis=0), [n,1])
    xs = np.tile(x.std(axis=0), [n,1])
    return (x - xm) / xs


def OU(dt, mean, stddev, tau, N, random_state = None):
    """
    OU returns a realization of the Ornstein-Uhlenbeck process given its
    parameters.

    Parameters
    ----------
    dt : float
        Time step.
    mean : float
        Mean of the process.
    stddev : float
        Standard deviation of the process.
    tau : float
        Time constant of the autocorrelation function of the process.
    N : integer
        Number of samples.
    random_state : RandomState object, optional
        The object used to draw the random numbers. The default is None.

    Returns
    -------
    ou : array of length N
        A realization of the Ornstein-Uhlenbeck process with the above parameters.

    """
    const = 2 * stddev**2 / tau
    mu = np.exp(-dt / tau)
    coeff = np.sqrt(const * tau / 2 * (1 - mu**2))
    if random_state is not None:
        rnd = random_state.normal(size=N)
    else:
        rnd = np.random.normal(size=N)
    ou = np.zeros(N)
    ou[0] = mean
    for i in range(1, N):
        ou[i] = mean + mu * (ou[i-1] - mean) + coeff * rnd[i]
    return ou

def OU_2(dt, alpha, mu, c, N, random_state = None):
    """
    OU_2 returns a realization of the Ornstein-Uhlenbeck process given its
    parameters.

    Parameters
    ----------
    dt : float
        Time step.
    alpha : float
        Sets the autocorrelation time constant of the process (which is equal to 1/alpha).
    mu : float
        Mean of the process.
    c : float
        Together with alpha, sets the variance of the process (which is equal to c**2 / (2 * alpha)).
    N : integer
        Number of samples.
    random_state : RandomState object, optional
        The object used to draw the random numbers. The default is None.

    Returns
    -------
    ou : array of length N
        A realization of the Ornstein-Uhlenbeck process with the above parameters.

    """
    coeff = np.array([alpha * mu * dt, 1 / (1 + alpha * dt)])
    if random_state is not None:
        rnd = c * np.sqrt(dt) * random_state.normal(size=N)
    else:
        rnd = c * np.sqrt(dt) * np.random.normal(size=N)
    ou = np.zeros(N)
    ou[0] = mu
    for i in range(N-1):
        ou[i+1] = (ou[i] + coeff[0] + rnd[i]) * coeff[1]
    return ou


def run_load_flow(app, project_folder, generators, loads, buses, study_case_name, verbose=False):
    study_case = project_folder.GetContents(study_case_name)[0]
    study_case.Activate()
    if verbose: print(f'Successfully activated study case {study_case_name}.')
    load_flow = app.GetFromStudyCase('ComLdf')
    err = load_flow.Execute()
    if err:
        raise Exception('Cannot run load flow')
    if verbose: print('Successfully run load flow.')
    results = {key: {} for key in ('generators','buses','loads')}
    
    Ptot, Qtot = 0, 0
    for gen in generators:
        pq = [gen.GetAttribute(f'm:{c}sum:bus1') for c in 'PQ']
        results['generators'][gen.loc_name] = {
            'P': pq[0],
            'Q': pq[1],
            'I': gen.GetAttribute('m:I:bus1'),
            'V': gen.GetAttribute('m:U1:bus1'),    # line-to-ground voltage
            'Vl': gen.GetAttribute('m:U1l:bus1')  # line-to-line voltage
        }
        Ptot += pq[0]
        Qtot += pq[1]
    results['generators']['Ptot'] = Ptot
    results['generators']['Qtot'] = Qtot
        
    Ptot, Qtot = 0, 0
    for load in loads:
        pq = [load.GetAttribute(f'm:{c}sum:bus1') for c in 'PQ']
        results['loads'][load.loc_name] = {
            'P': pq[0],
            'Q': pq[1],
            'I': load.GetAttribute('m:I:bus1'),
            'V': load.GetAttribute('m:U1:bus1'),    # line-to-ground voltage
            'Vl': load.GetAttribute('m:U1l:bus1'),  # line-to-line voltage
        }
        Ptot += pq[0]
        Qtot += pq[1]
    results['loads']['Ptot'] = Ptot
    results['loads']['Qtot'] = Qtot
    
    power_types = ['gen','load','flow','out']
    for bus in buses:
        results['buses'][bus.loc_name] = {
            'voltage': bus.GetAttribute('m:u'),
            'V': bus.GetAttribute('m:U'),
            'Vd': bus.GetAttribute('m:u1r'),
            'Vq': bus.GetAttribute('m:u1i'),
            'Vl': bus.GetAttribute('m:Ul'),
            'P': {power_type: bus.GetAttribute(f'm:P{power_type}') for power_type in power_types},
            'Q': {power_type: bus.GetAttribute(f'm:Q{power_type}') for power_type in power_types}
        }

    return results


def print_load_flow(results):
    print('\n===== Generators =====')
    for name,data in results['generators'].items():
        if name not in ('Ptot','Qtot'):
            print(f'{name}: P = {data["P"]:7.2f} MW, Q = {data["Q"]:6.2f} MVAR, ' + 
                  f'I = {data["I"]:6.3f} kA, V = {data["V"]:6.3f} kV.')
    print(f'Total P = {results["generators"]["Ptot"]*1e-3:5.2f} GW, total Q = {results["generators"]["Qtot"]*1e-3:5.2f} GVAR')

    print('\n======= Loads ========')
    for name,data in results['loads'].items():
        if name not in ('Ptot','Qtot'):
            print(f'{name}: P = {data["P"]:7.2f} MW, Q = {data["Q"]:6.2f} MVAR, ' + 
                  f'I = {data["I"]:6.3f} kA, V = {data["V"]:8.3f} kV.')
    print(f'Total P = {results["loads"]["Ptot"]*1e-3:5.2f} GW, total Q = {results["loads"]["Qtot"]*1e-3:5.2f} GVAR')
    
    print('\n======= Buses ========')
    for name,data in results['buses'].items():
        print(f'{name}: voltage = {data["voltage"]:5.3f} pu, V = {data["Vl"]:7.3f} kV, ' + \
              f'Pflow = {data["P"]["flow"]:7.2f} MW, Qflow = {data["Q"]["flow"]:7.2f} MVA.')

