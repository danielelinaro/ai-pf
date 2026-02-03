# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:14:10 2024

@author: Daniele Linaro
"""

import re

__all__ = ['AutomaticVoltageRegulator', 'TurbineGovernor',
           'Load', 'SynchronousMachine', 'PowerPlant', 'Bus', 'Transformer',
           'Line', 'Shunt', 'SeriesCapacitor', 'CommonImpedance', 'ACVoltageSource']


def _bus_name_to_terminal_name(bus):
    return 'bus{}'.format(int(re.findall('\d+', bus)[0]))
    #return 'term_{}'.format(bus.lower().replace(' ', '_'))


def _read_element_parameters(element, par_names=None, type_par_names=None,
                             bus_names=['bus1'], coeffs=None):
    data = {
        'name': re.sub('^[0-9]*', '', element.loc_name).replace(' ','').replace('-',''),
        'loc_name': element.loc_name,
    }
    if len(data['name']) == 0:
        data['name'] = None
    if bus_names is not None and len(bus_names) > 0:
        data['terminals'] = [element.GetAttribute(bus_name).cterm.loc_name
                             for bus_name in bus_names]
    if par_names is not None:
        for k, v in par_names.items():
            c = coeffs.get(k, 1.) if isinstance(coeffs, dict) else 1.
            data[v] = c * element.GetAttribute(k)
    if type_par_names is not None:
        for k, v in type_par_names.items():
            c = coeffs.get(k, 1.) if isinstance(coeffs, dict) else 1.
            data[v] = c * element.typ_id.GetAttribute(k)
    return data


class AutomaticVoltageRegulator (object):
    def __init__(self, avr, type_name, vrating=16.5e3):
        self.type_name = type_name.upper()
        if self.type_name not in ('IEEEEXC1',):
            raise Exception('AVR type must be "IEEEExc1"')
        par_names = {'Ka': 'ka', 'Ta': 'ta', 'Kf': 'kf',
                     'Tf': 'tf', 'Ke': 'ke', 'Te': 'te',
                     'Tr': 'tr', 'Vrmin': 'vmin', 'Vrmax': 'vmax',
                     'E1': 'e1', 'E2': 'e2', 'Se1': 'se1', 'Se2': 'se2'}
        self.type_ID = 2
        data = _read_element_parameters(avr, par_names, bus_names=None)
        for k,v in data.items():
            self.__setattr__(k, v)
        self.vrating = vrating
        self.par_names = par_names

    @property
    def fmt(self):
        s = self.name + ' {} {} poweravr type=' + str(self.type_ID)
        s += ' vrating={:e} \\\n\t\t'.format(self.vrating)
        for i,par_name in enumerate(self.par_names.values()):
            s += '{}={} '.format(par_name, self.__getattribute__(par_name))
            if (i+1) % 5 == 0 and i != len(self.par_names) - 1:
                s += '\\\n\t\t'
        return s


class TurbineGovernor (object):
    def __init__(self, gov, type_name):
        self.type_name = type_name.upper()
        if self.type_name not in ('IEEEG1', 'IEEEG3'):
            raise Exception('Governor type must be one of "IEEEG1" or "IEEEG3"')
        if self.type_name == 'IEEEG1':
            par_names = {'K': 'r', 'T1': 't1', 'T2': 't2', 'T3': 't3',
                         'K1': 'k1', 'K2': 'k2', 'T5': 't5', 'K3': 'k3',
                         'K4': 'k4', 'T6': 't6', 'K5': 'k5', 'K6': 'k6',
                         'T4': 't4', 'T7': 't7', 'K7': 'k7', 'K8': 'k8',
                         'Uc': 'uc', 'Uo': 'uo', 'Pmin': 'pmin', 'Pmax': 'pmax'}
            self.type_ID = 3
        elif self.type_name == 'IEEEG3':
            par_names = {'Tg': 'tg', 'Tp': 'tp', 'Sigma': 'sigma', 'Delta': 'delta',
                         'Tr': 'tr', 'a11': 'a11', 'a13': 'a13', 'a21': 'a21',
                         'a23': 'a23', 'Tw': 'tw', 'Uc': 'uc', 'Uo': 'uo',
                         'Pmin': 'pmin', 'Pmax': 'pmax'}
            self.type_ID = 4
        data = _read_element_parameters(gov, par_names, bus_names=None)
        for k,v in data.items():
            self.__setattr__(k, v)
        if self.type_name == 'IEEEG1':
            self.r = 1. / self.r
        self.par_names = par_names

    @property
    def fmt(self):
        s = self.name + ' {} {} '
        s += 'powertg type={} \\\n\t\t'.format(self.type_ID)
        for i,par_name in enumerate(self.par_names.values()):
            s += '{}={} '.format(par_name, self.__getattribute__(par_name))
            if (i+1) % 5 == 0 and i != len(self.par_names) - 1:
                s += '\\\n\t\t'
        return s


class SynchronousMachine (object):
    def __init__(self, sm):
        if not sm.ip_ctrl and sm.av_mode != 'constv':
            raise Exception(f'Synchronous machine {sm.loc_name} is not a PV machine and is not the slack')
        par_names = {'pgini': 'pg', 'usetp': 'vg', 'Pmax_uc': 'pmax', 'Pmin_uc': 'pmin',
                'q_min': 'qmin', 'q_max': 'qmax', 'ngnum': 'num', 'ip_ctrl': 'ref_gen'}
        type_par_names = {'sgn': 'prating', 'ugn': 'vrating', 'cosn': 'cosn', 'h': 'h',
                          'iturbo': 'rotor_type', 'rstr': 'ra', 'dpe': 'd'}
        self.model_type = sm.typ_id.model_inp
        if self.model_type == 'det':
            self.type_id = 6
            for key in ('xl','xd','xq','xrl','xrlq'):
                type_par_names[key] = key
            type_par_names['tds0']  = 'td0p'
            type_par_names['tqs0']  = 'tq0p'
            type_par_names['xds']   = 'xdp'
            type_par_names['xqs']   = 'xqp'
            type_par_names['tdss0'] = 'td0s'
            type_par_names['tqss0'] = 'tq0s'
            type_par_names['xdss']  = 'xds'
            type_par_names['xqss']  = 'xqs'
        elif self.model_type == 'cls':
            self.type_id = 2
            type_par_names['xstr']  = 'xdp'
        else:
            raise Exception('Unknown model type: "{}"'.format(self.model_type))
        data = _read_element_parameters(sm, par_names, type_par_names)
        for k,v in data.items():
            self.__setattr__(k, v)
        if self.ref_gen:
            print(f'{self.name} is the slack generator.')
        self.bus_id = int(re.findall('\d+', self.terminals[0])[0])
        self.pg /= self.prating
        self.pmax /= self.prating
        self.pmin /= self.prating
        self.prating *= self.num * 1e6
        self.vrating *= 1e3

    @property
    def fmt(self):
        s = '{} {} '.format(self.name, \
            _bus_name_to_terminal_name(self.terminals[0])) + \
            '{} {} ' + 'omega{} powergenerator type={} qlimits=no phtype=1 \\\n\t\t' \
            .format(self.bus_id, self.type_id)
        if self.ref_gen:
            s += 'slack=yes '
        else:
            s += f'pg={self.pg:g} '
        s += 'vg={:g} prating={:.6e} vrating={:.6e} \\\n\t\t' \
            .format(self.vg, self.prating, self.vrating) + \
            'qmax={:g} qmin={:g} pmax={:g} pmin={:g} \\\n\t\t' \
            'ra={:g} h={:g} d={:g} xdp={:g} ' \
            .format(self.qmax, self.qmin, self.pmax, self.pmin, self.ra, self.h, self.d, self.xdp)
        if self.type_id == 6:
            s += 'xd={:g} xq={:g} xds={:g} xqs={:g} \\\n\t\t'.format(self.xd, self.xq, self.xds, self.xqs) + \
            'td0p={:g} td0s={:g} tq0s={:g} xl={:g} \\\n\t\t'.format(self.td0p, self.td0s, self.tq0s, self.xl) + \
            'xqp={:g} tq0p={:g} '.format(self.xqp, self.tq0p)
        return s
    
    def __str__(self):
        return self.fmt.format('gnd', 'gnd')


class PowerPlant (object):
    def __init__(self, power_plant):
        self.name = power_plant.loc_name
        slots = power_plant.pblk
        elements = power_plant.pelm
        self.sm, self.avr, self.gov = None, None, None
        for slot, element in zip(slots, elements):
            if element is not None:
                slot_name = slot.loc_name.lower()
                if 'sym' in slot_name or 'sm' in slot_name:
                    self.sm = SynchronousMachine(element)
                elif 'avr' in slot_name:
                    self.avr = AutomaticVoltageRegulator(element, type_name='IEEEEXC1')
                elif 'gov' in slot_name:
                    name = element.typ_id.loc_name.lower()
                    type_name = None
                    if 'ieee' in name:
                        if 'g1' in name:
                            type_name = 'IEEEG1'
                        elif 'g3' in name:
                            type_name = 'IEEEG3'
                    if type_name is None:
                        raise Exception(f'Unknown governor type "{name}"')
                    self.gov = TurbineGovernor(element, type_name)
        if self.sm is None:
            raise Exception('A synchronous machine must be present in a power plant')
        if self.gov is None:
            raise Exception('A turbine governor must be present in a power plant')
        if self.avr is not None:
            self.avr.vrating = self.sm.vrating
                    
    def __str__(self):
        bus_id = self.sm.bus_id
        avr_str = self.avr.fmt.format(f'bus{bus_id}', f'avr{bus_id}') if self.avr is not None else ''
        if self.gov.type_name.upper() == 'IEEEG1':
            gov_str = self.gov.fmt.format(f'php{bus_id}', f'omega{bus_id}')
            sm_str = self.sm.fmt.format(f'avr{bus_id}' if self.avr is not None else 'gnd', f'php{bus_id}')
        elif self.gov.type_name.upper() == 'IEEEG3':
            gov_str = self.gov.fmt.format(f'pm{bus_id}', f'omega{bus_id}')
            sm_str = self.sm.fmt.format(f'avr{bus_id}' if self.avr is not None else 'gnd', f'pm{bus_id}')
        else:
            raise Exception(f'Unknown governor type "{self.gov.type_name}"')
        return avr_str + '\n\n' + gov_str + '\n\n' + sm_str


class ACVoltageSource (object):
    def __init__(self, vac):
        data = _read_element_parameters(vac, {'Unom': 'vb',
                                              'R1': 'rseries',
                                              'X1': 'xseries'})
        for k,v in data.items():
            self.__setattr__(k,v)
        self.vb *= 1e3

    def __str__(self):
        return '{} {:5s} vsource vdc={:.6e}' \
                .format(self.name,
                        _bus_name_to_terminal_name(self.terminals[0]),
                        self.vb)

class Load (object):
    def __init__(self, load):
        data = _read_element_parameters(load, {'plini': 'pc', 'qlini': 'qc'})
        for k,v in data.items():
            self.__setattr__(k,v)
        self.vrating = load.bus1.cterm.uknom * 1e3
        self.pc *= 1e6
        self.qc *= 1e6
            
    def __str__(self):
        return '{} {:5s} powerload utype=1 pc={:.6e} qc={:.6e} vrating={:.6e}' \
                .format(self.name,
                        _bus_name_to_terminal_name(self.terminals[0]),
                        self.pc, self.qc, self.vrating)

class Line (object):
    def __init__(self, line):
        par_names = {'dline': 'length', 'nlnum': 'num', 'R1': 'r', 'X1': 'x', 'B1': 'b'}
        type_par_names = {}
        bus_names = ['bus1', 'bus2']
        coeffs = {'B1': 1e-6} # B1 is in uS
        data = _read_element_parameters(line, par_names, type_par_names, bus_names, coeffs)
        for k,v in data.items():
            self.__setattr__(k, v)
        if self.num > 1:
            print(f'Line {self.name} has {self.num} parallel lines.')
        self.vrating = line.bus1.cterm.uknom * 1e3
        self.utype = 1

    def __str__(self):
        return '{} {:5s} {:5s} powerline utype={:d} r={:.6e} x={:.6e} b={:.6e} vrating={:.6e}' \
                .format(self.name,
                       _bus_name_to_terminal_name(self.terminals[0]),
                       _bus_name_to_terminal_name(self.terminals[1]),
                       self.utype, self.r, self.x, self.b, self.vrating)

class SeriesCapacitor (Line):
    def __init__(self, capacitor):
        par_names = {'ucn': 'vrating', 'xcap': 'x'}
        data = _read_element_parameters(capacitor, par_names, bus_names=['bus1', 'bus2'])
        for k,v in data.items():
            self.__setattr__(k, v)        
        self.r = 0.
        self.b = 0.
        self.x = -self.x
        self.vrating *= 1e3 
        self.utype = 1

class Shunt (object):
    def __init__(self, shunt):
        par_names = {'qcapn': 'q', 'ushnm': 'vrating', 'bcap': 'b', 'gparac': 'g'}
        data = _read_element_parameters(shunt, par_names)
        for k,v in data.items():
            self.__setattr__(k, v)
        self.q *= 1e6
        self.vrating *= 1e3
        self.b *= 1e-6
        self.g *= 1e-6
        self.utype = 1

    def __str__(self):
        return '{} {:5s} powershunt utype={:d} b={:.6e} g={:.6e} vrating={:.6e}' \
                .format(self.name,
                       _bus_name_to_terminal_name(self.terminals[0]),
                       self.utype, self.b, self.g, self.vrating)

class CommonImpedance (object):
    def __init__(self, impedance):
        par_names = {'Sn': 'prating', 'r_pu': 'r', 'x_pu': 'x'}
        data = _read_element_parameters(impedance, par_names, bus_names=['bus1', 'bus2'])
        for k,v in data.items():
            self.__setattr__(k, v)
        if impedance.bus1.cterm.uknom != impedance.bus2.cterm.uknom:
            print(f'Common impedance {self.name}: ' + 
                  f'bus1.vrating = {impedance.bus1.cterm.uknom} kV, ' + 
                  f'bus2.vrating = {impedance.bus2.cterm.uknom} kV')
        self.vrating = impedance.bus1.cterm.uknom * 1e3
        self.utype = 0

    def __str__(self):
        return '{} {:5s} {:5s} powerline utype={:d} r={:.6e} x={:.6e} prating={:.6e} vrating={:.6e}' \
                .format(self.name,
                       _bus_name_to_terminal_name(self.terminals[0]),
                       _bus_name_to_terminal_name(self.terminals[1]),
                       self.utype, self.r, self.x, self.prating, self.vrating)

class Transformer (object):
    def __init__(self, transformer, voltages_from='type'):
        par_names = {'ntnum': 'num', 'nntap': 'nntap', 't:dutap': 'tappc'}
        type_par_names = {'r1pu': 'r', 'x1pu': 'x', 'strn': 'prating'}
        if voltages_from == 'type':
            type_par_names['utrn_h'] = 'vh'
            type_par_names['utrn_l'] = 'vl'
        bus_names = ['buslv', 'bushv']
        data = _read_element_parameters(transformer, par_names, type_par_names, bus_names)
        if voltages_from == 'bus':
            data['vl'] = transformer.buslv.cterm.uknom
            data['vh'] = transformer.bushv.cterm.uknom
        for k,v in data.items():
            self.__setattr__(k, v)
        if self.num > 1:
            print(f'Transformer {self.name} has {self.num} parallel transformers.')
        self.a = self.vh / self.vl
        self.tappc /= 100   # percentage of voltage increase for each tap
        self.kt = 1 + self.nntap * self.tappc
        self.prating *= self.num * 1e6
        self.vrating = data['vl'] * 1e3

    def __str__(self):
        return ('{} {:5s} {:5s} powertransformer r={:.6e} x={:.6e} ' +
                'a={:.6e} kt={:.6e} prating={:.6e} vrating={:.6e}') \
                .format(self.name,
                       _bus_name_to_terminal_name(self.terminals[0]),
                       _bus_name_to_terminal_name(self.terminals[1]),
                       self.r, self.x, self.a, self.kt,
                       self.prating, self.vrating)
    
class Bus (object):
    def __init__(self, bus):
        data = _read_element_parameters(bus, {'uknom': 'vb'}, bus_names=None)
        for k,v in data.items():
            self.__setattr__(k, v)
        if self.name is None:
            self.name = 'Bus_' + bus.loc_name
        self.vb *= 1e3
        self.v0 = bus.GetAttribute('m:u')
        self.theta0 = bus.GetAttribute('m:phiu')
        self.terminal = _bus_name_to_terminal_name(self.name)

    def __str__(self):
        return '{} {:5s} powerbus vb={:.6e} v0={:.6e} theta0={:.6e}' \
                .format(self.name, self.terminal, self.vb, self.v0, self.theta0)
