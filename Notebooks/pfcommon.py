
import numpy as np

__all__ = ['get_simulation_variables', 'get_simulation_time', 'get_simulation_dt', 'OU']

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
