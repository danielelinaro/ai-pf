
import os
import re
import tables
import numpy as np

__all__ = ['BaseParameters', 'terminal_site_name', 'SiteNode', 'LineSitesEdge',
           'get_simulation_variables', 'get_simulation_time', 'get_simulation_dt',
           'get_ID', 'get_line_bus_IDs', 'normalize', 'OU', 'OU_2', 'run_power_flow',
           'print_power_flow', 'correct_traces', 'find_element_by_name',
           'is_voltage', 'is_power', 'is_frequency', 'is_current', 
           'compute_generator_inertias', 'sort_objects_by_name', 'get_objects',
           'make_full_object_name', 'build_network_graph', 'Node', 'Edge',
           'parse_sparse_matrix_file', 'parse_Amat_vars_file', 'parse_Jacobian_vars_file',
           'combine_output_spectra']


class BaseParameters (tables.IsDescription):
    F0    = tables.Float64Col()
    srate = tables.Float64Col()


def get_objects(app, pattern, keep_out_of_service=False, sort=False):
    if pattern[:2] != '*.':
        pattern = '*.' + pattern
    objs = filter(lambda obj: keep_out_of_service or \
                  not obj.HasAttribute('outserv') or \
                  not obj.outserv, app.GetCalcRelevantObjects(pattern))
    if sort:
        D = {obj.loc_name: obj for obj in objs}
        return [D[k] for k in sorted(D.keys())]
    return list(objs)


make_full_object_name = lambda obj: '|'.join([s.split('.Elm')[0] for s in 
                                              obj.GetFullName().split(os.path.sep)
                                              if 'Elm' in s])


# get the site name from a terminal
terminal_site_name = lambda terminal: make_full_object_name(terminal).split('|')[1]


class SiteNode (object):
    """
    Class used to store information about a PowerFactory site that will be used as a node in the graph.
    """
    def __init__(self, site):
        """
        SiteNode constructor
        Parameters
        ----------
        site : PowerFactory DataObject
            The PowerFactory site object.

        Returns
        -------
        None.
        """
        self.name = site.loc_name
        self.full_name = make_full_object_name(site)
        self._coords = [site.GPSlat, site.GPSlon]
        self._lat,self._lon = self._coords
    @property
    def coords(self):
        return self._coords
    @coords.setter
    def coords(self, c):
        assert len(c) == 2
        self._coords = np.squeeze(np.array(c))
        self._lat,self._lon = self._coords
    @property
    def lat(self):
        return self._lat
    @property
    def lon(self):
        return self._lon
    def __eq__(self, o):
        if o is None:
            return False
        return self.full_name == o.full_name and self.lat == o.lat and self.lon == o.lon
    def __hash__(self):
        return hash(repr(self))


class LineSitesEdge (object):
    """
    Class used to store information about the line connecting two PowerFactory sites.
    """
    def __init__(self, line, node1, node2):
        """
        Parameters
        ----------
        line : PowerFactory DataObject
            The PowerFactory line object.
        node1,node2 : SiteNode
            The SiteNodes object to which the line is connected.

        Returns
        -------
        None.
        """
        # node1 and node2 must be objects of type SiteNode
        assert isinstance(node1,SiteNode) and isinstance(node2,SiteNode)
        self.name = line.loc_name
        self.full_name = make_full_object_name(line)
        self.coords = line.GPScoords
        self.length = line.dline if line.HasAttribute('dline') else 1e-3
        self.vrating = line.typ_id.uline
        self.terminals = [make_full_object_name(line.bus1.cterm),
                          make_full_object_name(line.bus2.cterm)]
        self.node1,self.node2 = node1,node2
        self.nodes = [self.node1, self.node2]


def find_connected_terminal(term1, conn_elm, attr_names):
    # make sure that term1 is indeed one of the terminals of conn_elm
    if conn_elm.GetAttribute(attr_names[0]).cterm != term1 and \
        conn_elm.GetAttribute(attr_names[1]).cterm != term1:
            return None
    for attr_name in attr_names:
        term2 = conn_elm.GetAttribute(attr_name).cterm
        if term2 != term1:
            return term2
    return None


class Node (object):
    def __init__(self, name, voltage, coords=[0.,0.]):
        self.name = name
        self.voltage = voltage
        self.coords = np.array(coords)
        self.lat, self.lon = coords

    def __eq__(self, o):
        return self.name == o.name and self.voltage == o.voltage and \
            self.lat == o.lat and self.lon == o.lon


class Edge (object):
    def __init__(self, name, node1, node2, length):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.length = length
        self.voltage = max(node1.voltage, node2.voltage) # somewhat arbitrarily

    def __str__(self):
        return 'Terminal 1: {} @ ({:.3f},{:.3f})\n'.format(self.node1.name,
                                                           self.node1.lat,
                                                           self.node1.lon) + \
               'Terminal 2: {} @ ({:.3f},{:.3f})\n'.format(self.node2.name,
                                                           self.node2.lat,
                                                           self.node2.lon) + \
               'Edge: {}\n'.format(self.name) + \
               'Length: {} km\n'.format(self.length) + \
               'Voltage: {} kV\n'.format(self.voltage)

    def __eq__(self, o):
        return self.name == o.name and self.node1 == o.node1 and \
            self.node2 == o.node2 and self.length == o.length and \
            self.voltage == o.voltage


def make_edges_from_terminal(term1):
    term1_name = make_full_object_name(term1)
    node1 = Node(term1_name, term1.uknom, [term1.GPSlat, term1.GPSlon])
    edges = []
    for elm in term1.GetConnectedElements():
        if (elm.HasAttribute('outserv') and elm.outserv) or \
           (elm.HasAttribute('on_off') and not elm.on_off):
            continue
        if elm.HasAttribute('bus2'):
            # a line or a breaker
            other_terms = [find_connected_terminal(term1, elm, ('bus1', 'bus2'))]
        elif elm.HasAttribute('busmv'):
            # a 3-way transformer
            other_terms = [find_connected_terminal(term1, elm, ('bushv', 'busmv')),
                           find_connected_terminal(term1, elm, ('bushv', 'buslv'))]
        elif elm.HasAttribute('bushv'):
            # a 2-way transformer
            other_terms = [find_connected_terminal(term1, elm, ('bushv', 'buslv'))]
        else:
            # print('Do not know how to deal with connected element {}'.format(elm.loc_name))
            continue
        elm_name = make_full_object_name(elm)
        # remove the elements that are None
        other_terms = filter(lambda x: x is not None, other_terms)
        for term2 in other_terms:
            term2_name = make_full_object_name(term2)
            node2 = Node(term2_name, term2.uknom, [term2.GPSlat, term2.GPSlon])
            edge = Edge(elm_name, node1, node2, elm.dline if elm.HasAttribute('dline') else 1e-3)
            edges.append(edge)

    return edges


def build_network_graph(app, verbose=False):
    from networkx import MultiGraph
    
    terminals = get_objects(app, 'ElmTerm')

    swap_nodes = lambda edge: Edge(edge.name, edge.node2, edge.node1, edge.length)
    
    edges = []
    cnt, cnt_swapped = 0, 0
    for term1 in terminals:
        for edge in make_edges_from_terminal(term1):
            edge_swapped_nodes = swap_nodes(edge)
            if edge in edges:
                cnt += 1
                if verbose: print('Edge {} already present.'.format(edge))
            elif edge_swapped_nodes in edges:
                cnt_swapped += 1
            else:
                edges.append(edge)
    if verbose:
        print('Number of edges not added: {}'.format(cnt))
        print('Number of swapped edges not added: {}'.format(cnt_swapped))

    nodes = []
    for edge in edges:
        if edge.node1 not in nodes:
            nodes.append(edge.node1)
        if edge.node2 not in nodes:
            nodes.append(edge.node2)

    G = MultiGraph()
    for e in edges:
        G.add_edge(e.node1, e.node2, weight=e.length,
                   label=e.name, voltage=e.voltage)

    return G,edges,nodes


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
            1: ['G 02', 'G 03'],
            2: ['G 04', 'G 05', 'G 06', 'G 07'],
            3: ['G 08', 'G 09', 'G 10'],
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
is_frequency = lambda var_name: var_name in ('m:fe', 's:fe')
is_current = lambda var_name: var_name in ('m:ir:bus1', 'm:iu:bus1', 'm:i:bus1')


def find_element_by_name(elements, name):
    found = False
    for elem in elements:
        if elem.loc_name == name:
            found = True
            break
    if found: return elem
    return None


def correct_traces(xre, xim, delta):
    """
      xre - (MxN) array, where M is the number of samples and N the number of buses at which xre is recorded
      xim - (MxN) array, where M is the number of samples and N the number of buses at which xim is recorded
    delta - (Mx1) array, measured in degrees
    """
    # convert delta to radians
    try:
        n = xre.shape[1]
        delta_ref = np.tile((delta - delta[0]) / 180 * np.pi, [n,1]).T
    except:
        delta_ref = (delta - delta[0]) / 180 * np.pi
    mod = np.sqrt(xre**2 + xim**2)
    angle = np.arctan2(xim, xre) - delta_ref
    return mod * np.cos(angle), mod * np.sin(angle)



def _compute_samples_interval(res, interval, dt):
    n_samples = res.GetNumberOfRows()
    start = 0 if interval[0] == 0 else int(interval[0] / dt)
    stop = n_samples if interval[1] is None else int(np.ceil(interval[1] / dt))
    return start,stop


def get_simulation_variables(res, var_name, vector, interval=(0,None), dt=None,
                             elements=None, elements_name=None, app=None,
                             decimation=1, full_output=False):
    # vector is a PowerFactory IntVec object. It can be None, in which case
    # the (much slower) GetValue function is used, instead of GetColumnValues
    if elements is None:
        if elements_name is None:
            raise Exception('You must provide one of "elements" or "elements_name"')
        if app is None:
            raise Exception('You must provide "app" if "elements_name" is passed')
        full_output = True
        elements = app.GetCalcRelevantObjects(elements_name)
    start,stop = _compute_samples_interval(res, interval, dt)
    n_samples = stop - start
    variables = np.zeros((int(np.ceil(n_samples / decimation)), len(elements)))
    for i,element in enumerate(elements):
        col = res.FindColumn(element, var_name)
        if col < 0:
            raise Exception(f'Variable {element.loc_name}:{var_name} is not available.')
        if vector is not None:
            err = res.GetColumnValues(vector, col)
            if not err:
                variables[:,i] = np.array(vector.V)[start:stop]
        else:
            variables[:,i] = np.array([res.GetValue(j,col)[1] for j in range(start, stop, decimation)])
    if full_output:
        return np.squeeze(variables), elements
    return np.squeeze(variables)


def get_simulation_time(res, vector, interval=(0,None), dt=None, decimation=1):
    start,stop = _compute_samples_interval(res, interval, dt)
    if vector is not None:
        err = res.GetColumnValues(vector, -1)
        if err:
            raise Exception('Cannot get time variable')
        return np.array(vector.V)[start:stop]
    return np.array([res.GetValue(i,-1)[1] for i in range(start, stop, decimation)])


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


def run_power_flow(app, project_folder=None, study_case_name=None, verbose=False):
    if project_folder is not None and study_case_name is not None:
        study_case = project_folder.GetContents(study_case_name)[0]
        study_case.Activate()
        if verbose: print(f'Successfully activated study case {study_case_name}.')
    power_flow = app.GetFromStudyCase('ComLdf')
    err = power_flow.Execute()
    if err:
        raise Exception('Cannot run load flow')
    if verbose: print('Successfully run load flow.')
    results = {key: {} for key in ('SMs', 'SGs', 'buses', 'loads', 'lines', 'transformers')}
    
    get_objects = lambda clss: [obj for obj in app.GetCalcRelevantObjects('*.' + clss) \
                                if not obj.outserv]
        
    # NOTE: the ``1`` in the variable names below indicates that it's a
    # positive-sequence quantity
    Ptot, Qtot = 0, 0
    for sm in get_objects('ElmSym'):
        pq = [sm.GetAttribute(f'm:{c}sum:bus1') for c in 'PQ'] # [MW,Mvar]
        results['SMs'][sm.loc_name] = {
            'P':      pq[0], # [MW]
            'Q':      pq[1], # [Mvar]
            'ur':     sm.GetAttribute('m:u1r:bus1'),   # [pu]
            'ui':     sm.GetAttribute('m:u1i:bus1'),   # [pu]
            'u':      sm.GetAttribute('m:u1:bus1'),    # [pu]
            'V':      sm.GetAttribute('m:U1:bus1'),    # [kV] line-to-ground voltage
            'Vl':     sm.GetAttribute('m:U1l:bus1'),   # [kV] line-to-line voltage
            'ir':     sm.GetAttribute('m:i1r:bus1'),   # [pu]
            'ii':     sm.GetAttribute('m:i1i:bus1'),   # [pu]
            'i':      sm.GetAttribute('m:i1:bus1'),    # [pu]
            'I':      sm.GetAttribute('m:I:bus1'),     # [kA]
            'phiu':   sm.GetAttribute('m:phiu1:bus1'), # [deg] current angle
            'phii':   sm.GetAttribute('m:phii1:bus1'), # [deg] current angle
            'cosphi': sm.GetAttribute('m:cosphi:bus1') # [deg] current angle
        }
        Ptot += pq[0]
        Qtot += pq[1]
    results['SMs']['Ptot'] = Ptot
    results['SMs']['Qtot'] = Qtot

    Ptot, Qtot = 0, 0
    for sg in get_objects('ElmGenStat'):
        pq = [sg.GetAttribute(f'm:{c}sum:bus1') for c in 'PQ']
        results['SGs'][sg.loc_name] = {
            'P': pq[0],
            'Q': pq[1]
        }
        Ptot += pq[0]
        Qtot += pq[1]
    results['SGs']['Ptot'] = Ptot
    results['SGs']['Qtot'] = Qtot

    Ptot, Qtot = 0, 0
    for load in get_objects('ElmLod'):
        pq = [load.GetAttribute(f'm:{c}sum:bus1') for c in 'PQ']
        results['loads'][load.loc_name] = {
            'P':      pq[0], # [MW]
            'Q':      pq[1], # [Mvar]
            'ur':     load.GetAttribute('m:u1r:bus1'),   # [pu]
            'ui':     load.GetAttribute('m:u1i:bus1'),   # [pu]
            'u':      load.GetAttribute('m:u1:bus1'),    # [pu]
            'V':      load.GetAttribute('m:U1:bus1'),    # [kV] line-to-ground voltage
            'Vl':     load.GetAttribute('m:U1l:bus1'),   # [kV] line-to-line voltage
            'ir':     load.GetAttribute('m:i1r:bus1'),   # [pu]
            'ii':     load.GetAttribute('m:i1i:bus1'),   # [pu]
            'i':      load.GetAttribute('m:i1:bus1'),    # [pu]
            'I':      load.GetAttribute('m:I:bus1'),     # [kA]
            'phiu':   load.GetAttribute('m:phiu1:bus1'), # [deg] current angle
            'phii':   load.GetAttribute('m:phii1:bus1'), # [deg] current angle
            'cosphi': load.GetAttribute('m:cosphi:bus1') # [deg] current angle
        }
        Ptot += pq[0]
        Qtot += pq[1]
    results['loads']['Ptot'] = Ptot
    results['loads']['Qtot'] = Qtot
    
    power_types = ['gen','load','flow','out']
    for bus in get_objects('ElmTerm'):
        try:
            results['buses'][bus.loc_name] = {
                'ur':     bus.GetAttribute('m:u1r'),     # [pu]
                'ui':     bus.GetAttribute('m:u1i'),     # [pu]
                'u':      bus.GetAttribute('m:u1'),      # [pu]
                'V':      bus.GetAttribute('m:U'),       # [kV] line-to-ground voltage
                'Vl':     bus.GetAttribute('m:Ul'),      # [kV] line-to-line voltage
                'phi':    bus.GetAttribute('m:phiu'),    # [deg]
                'phirel': bus.GetAttribute('m:phiurel'), # [deg]
                'P': {power_type: bus.GetAttribute(f'm:P{power_type}') for power_type in power_types},
                'Q': {power_type: bus.GetAttribute(f'm:Q{power_type}') for power_type in power_types}
            }
        except:
            pass

    Ptot = {'bus1': 0, 'bus2': 0}
    Qtot = {'bus1': 0, 'bus2': 0}
    for line in get_objects('ElmLne'):
        P1 = line.GetAttribute('m:Psum:bus1')
        Q1 = line.GetAttribute('m:Qsum:bus1')
        P2 = line.GetAttribute('m:Psum:bus2')
        Q2 = line.GetAttribute('m:Qsum:bus2')
        Ptot['bus1'] += P1
        Qtot['bus1'] += Q1
        Ptot['bus2'] += P2
        Qtot['bus2'] += Q2
        results['lines'][line.loc_name] = {
            'P_bus1': P1, 'Q_bus1': Q1,
            'P_bus2': P2, 'Q_bus2': Q2,
        }
    results['lines']['Ptot'] = Ptot
    results['lines']['Qtot'] = Qtot

    results['transformers']['Ptot'] = {'bushv': 0, 'buslv': 0}
    results['transformers']['Qtot'] = {'bushv': 0, 'buslv': 0}
    for trans in get_objects('ElmTr2'):
        results['transformers'][trans.loc_name] = {}
        has_attrs = True
        for pq in 'PQ':
            for hl in 'hl':
                try:
                    val = trans.GetAttribute(f'm:{pq}sum:bus{hl}v')
                    results['transformers'][trans.loc_name][f'{pq}_bus{hl}v'] = val
                    results['transformers'][f'{pq}tot'][f'bus{hl}v'] += val
                except:
                    results['transformers'].pop(trans.loc_name)
                    has_attrs = False
                    break
            if not has_attrs:
                break

    return results


def print_power_flow(results):
    print('\n===== Generators =====')
    for name in sorted(list(results['SMs'].keys())):
        data = results['SMs'][name]
        if name not in ('Ptot','Qtot'):
            print(f'{name}: P = {data["P"]:7.2f} MW, Q = {data["Q"]:6.2f} MVAR, ' + 
                  f'I = {data["I"]:6.3f} kA, V = {data["V"]:6.3f} kV.')
    P, Q = results["SMs"]["Ptot"], results["SMs"]["Qtot"]
    if P > 1e4:
        coeff, unit = 1e-3, 'G'
    else:
        coeff, unit = 1.0, 'M'
    print(f'Total P = {P*coeff:7.4f} {unit}W, total Q = {Q*coeff:7.4f} {unit}VAR')

    print('\n======= Loads ========')
    for name in sorted(list(results['loads'].keys())):
        data = results['loads'][name]
        if name not in ('Ptot','Qtot'):
            print(f'{name}: P = {data["P"]:7.2f} MW, Q = {data["Q"]:6.2f} MVAR, ' + 
                  f'I = {data["I"]:6.3f} kA, V = {data["V"]:8.3f} kV.')
    print(f'Total P = {results["loads"]["Ptot"]*coeff:7.4f} {unit}W, total Q = {results["loads"]["Qtot"]*coeff:7.4f} {unit}VAR')
    
    print('\n===== Transformers =====')
    for name in sorted(list(results['transformers'].keys())):
        data = results['transformers'][name]
        if name not in ('Ptot','Qtot'):
            P = data['P_bushv'] + data['P_buslv']
            Q = data['Q_bushv'] + data['Q_buslv']
            print(f'{name}: P = {P:7.2f} MW, Q = {Q:7.2f} MVA')
    P = results['transformers']['Ptot']['bushv'] + results['transformers']['Ptot']['buslv']
    Q = results['transformers']['Qtot']['bushv'] + results['transformers']['Qtot']['buslv']
    print(f'Total P = {P*coeff:7.2f} {unit}W, total Q = {Q*coeff:7.2f} {unit}VA.')

    print('\n======= Lines =======')
    for name in sorted(list(results['lines'].keys())):
        data = results['lines'][name]
        if name not in ('Ptot','Qtot'):
            print(f'{name}: P = {data["P_bus1"]:7.2f} MW, Q bus1 = {data["Q_bus1"]:7.2f} MVA ' + 
                  f'Q bus2 = {data["Q_bus2"]:7.2f} MVA.')
    print(f'Total P = {results["lines"]["Ptot"]["bus1"]*coeff:7.2f} {unit}W, ' + 
          f'total Q = {(results["lines"]["Qtot"]["bus1"]-results["lines"]["Qtot"]["bus2"])*coeff:7.2f} {unit}VA.')

    print('\n======= Buses ========')
    for name in sorted(list(results['buses'].keys())):
        data = results['buses'][name]
        print(f'{name}: voltage = {data["voltage"]:5.3f} pu, V = {data["Vl"]:7.3f} kV, ' + \
              f'Pflow = {data["P"]["flow"]*coeff:7.2f} {unit}W, Qflow = {data["Q"]["flow"]*coeff:7.2f} {unit}VA.')


def parse_sparse_matrix_file(filename, sparse=False, one_based_indexes=True):
    from scipy.sparse import csr_matrix
    data = np.loadtxt(filename)
    row_ind = np.asarray(data[:,0], dtype=int)
    col_ind = np.asarray(data[:,1], dtype=int)
    if one_based_indexes:
        row_ind, col_ind = row_ind-1, col_ind-1
    M = csr_matrix((data[:,2], (row_ind, col_ind)))
    if not sparse:
        return M.toarray()
    return M


def _parse_line(line):
    col_match = re.search('\d+', line)
    col = int(col_match.group()) - 1
    var_name_match = re.search('".*"', line)
    var_name = var_name_match.group()[1:-1]
    start = col_match.span()[1]
    stop = var_name_match.span()[0]
    model_name = line[start:stop].strip()
    return col,model_name,var_name


def parse_Amat_vars_file(filename):
    cols, var_names, model_names = [], [], []
    with open(filename, 'r') as fid:
        for L in fid:
            line = L.strip()
            if line != '' and ';' not in line:
                col,model_name,var_name = _parse_line(line)
                cols.append(col)
                model_names.append(model_name)
                var_names.append(var_name)
    return np.array(cols), var_names, model_names


def parse_Jacobian_vars_file(filename):
    from collections import OrderedDict
    vars_idx = OrderedDict()
    state_vars, voltages = OrderedDict(), OrderedDict()
    currents, signals = OrderedDict(), OrderedDict()
    def add_to_list(D, oname, vname):
        try:
            D[oname].append(vname)
        except:
            D[oname] = [vname]
    with open(filename, 'r') as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0:
                continue
            elif ';' in line:
                tokens = [token.lstrip() for token in line.split(';')]
                var_type = tokens[2].lower()
            else:
                ret = _parse_line(line)
                idx = ret[0]
                # obj_name = ret[1].split('\\')[-1].split('.')[0]
                obj_name = ret[1].replace('\\', '-')
                var_name = ret[2].replace(':bus1','')
                if obj_name not in vars_idx:
                    vars_idx[obj_name] = OrderedDict()
                add_to_list(vars_idx[obj_name], var_name, idx)
                if 'state' in var_type:
                    add_to_list(state_vars, obj_name, var_name)
                elif 'voltage' in var_type:
                    add_to_list(voltages, obj_name, var_name)
                elif 'current' in var_type:
                    add_to_list(currents, obj_name, var_name)
                elif 'signal' in var_type:
                    add_to_list(signals, obj_name, var_name)
    return vars_idx,state_vars,voltages,currents,signals


def combine_output_spectra(output_spectra, input_names, output_names, load_names,
                           var_names, var_types, freq, PF, bus_equiv_terms,
                           ref_freq=50.0, ref_SM_name=None):
    """
    Arguments:
     output_spectra: an ndarray with dimensions MxNxL, where M is the number of frequency
                     samples, N is the number of inputs and L is the number of outputs.
                     The inputs are the stochastic loads and the outputs are the
                     variables of the network
        input_names: the names of the inputs that should be used in the calculation of the
                     combined spectra
       output_names: the names of the outputs for which the combined spectra should be
                     computed
         load_names: the N load names for which individual TFs are available
          var_names: the L variable names for which individual TFs are available
          var_types: a list of L variable types (m:ur, s:xspeed, ...)
               freq: the frequencies at which the TFs are sampled
                 PF: the data with the power-flow solution of the power network
    bus_equiv_terms: ...
           ref_freq: the frequency at which the system operates (i.e., 50 or 60 Hz)
        ref_SM_name: the name of the slack synchronous machine

    Returns:
      an ndarray with dimensions (# outputs, # freq. samples), where each row is the
      combined spectrum for a given output
    """
    add_spectra = lambda sp: np.sqrt(np.sum(np.abs(sp)**2, axis=1))

    # make sure that things work also if load_names and var_names are NumPy arrays
    load_names_L = list(load_names)
    var_names_L = list(var_names)
    N_outputs = len(output_names)
    N_samples = output_spectra.shape[0]
    assert freq.size == N_samples
    spectra = np.zeros((N_outputs,N_samples))
    idx_inputs = [load_names_L.index(input_name) for input_name in input_names]
    for i in range(N_outputs):
        if var_types[i] in ('m:ur','m:ui','s:xspeed'):
            idx_output = var_names_L.index(output_names[i])
            spectra[i,:] = add_spectra(output_spectra[:, idx_inputs, idx_output])
        elif var_types[i] == 'U':
            tmp = '.'.join(output_names[i].split('.')[:-1])
            idx_ur = var_names_L.index(tmp + '.ur')
            idx_ui = var_names_L.index(tmp + '.ui')
            bus_name = output_names[i].split('.')[0].split('-')[-1]
            ur,ui = PF['buses'][bus_name]['ur'], PF['buses'][bus_name]['ui']
            assert ur != 0, '{}: ur,ui = ({:g},{:g})'.format(output_names[i], ur, ui)
            coeff_ur,coeff_ui = np.array([ur, ui]) / np.sqrt(ur**2 + ui**2)
            tmp = coeff_ur * output_spectra[:, idx_inputs, idx_ur] + coeff_ui * output_spectra[:, idx_inputs, idx_ui]
            spectra[i,:] = add_spectra(tmp)
        elif var_types[i] in ('m:fe', 'theta', 'omega'):
            tmp = '.'.join(output_names[i].split('.')[:-1])
            idx_ur = var_names_L.index(tmp + '.ur')
            idx_ui = var_names_L.index(tmp + '.ui')
            bus_name = output_names[i].split('.')[0].split('-')[-1]
            ur,ui = PF['buses'][bus_name]['ur'], PF['buses'][bus_name]['ui']
            assert ur != 0, '{}: ur,ui = ({:g},{:g})'.format(output_names[i], ur, ui)
            coeff_ur = -ui / ur**2 / (1 + (ui / ur)**2)
            coeff_ui =  1 / (ur * (1 + (ui/ur) **2))
            tmp = coeff_ur * output_spectra[:, idx_inputs, idx_ur] + coeff_ui * output_spectra[:, idx_inputs, idx_ui]
            if var_types[i] in ('m:fe','omega'):
                freq_2d = np.tile(freq.squeeze()[:, np.newaxis], [1, len(idx_inputs)])
                tmp = tmp * 1j * 2 * np.pi * freq_2d # Δω = jωΔθ
                if var_types[i] == 'm:fe':
                    ref_SM_idx = [i for i, name in enumerate(var_names) if ref_SM_name + '.ElmSym.speed' in name]
                    assert len(ref_SM_idx) == 1, f"Cannot find variable `speed` of object '{ref_SM_name}'"
                    ref_SM_idx = ref_SM_idx[0]
                    tmp /= 2 * np.pi * ref_freq # !!! scaling factor !!!
                    tmp += output_spectra[:, idx_inputs, ref_SM_idx]
            spectra[i,:] = add_spectra(tmp)
    return spectra
