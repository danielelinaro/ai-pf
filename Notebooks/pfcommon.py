
import re
import numpy as np

__all__ = ['get_simulation_variables', 'get_simulation_time', 'get_simulation_dt',
           'get_ID', 'get_line_bus_IDs', 'normalize', 'OU', 'run_load_flow',
           'print_load_flow']


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

# def OU(dt, alpha, mu, c, N, random_state = None):
#     coeff = np.array([alpha * mu * dt, 1 / (1 + alpha * dt)])
#     if random_state is not None:
#         rnd = c * np.sqrt(dt) * random_state.normal(size=N)
#     else:
#         rnd = c * np.sqrt(dt) * np.random.normal(size=N)
#     ou = np.zeros(N)
#     ou[0] = mu
#     for i in range(N-1):
#         ou[i+1] = (ou[i] + coeff[0] + rnd[i]) * coeff[1]
#     return ou


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

