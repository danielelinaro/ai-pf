
import os
import sys
import json
import powerfactory as pf

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ('-h','--help'):
            print('usage: {} <project_name>'.format(os.path.basename(sys.argv[0])))
            sys.exit(0)
        project_name = sys.argv[1]
    else:
        project_name = '\Terna_Inerzia\V2020_Rete_Sardegna_2021_06_03cr'

    app = pf.GetApplication()
    if app is None:
        print("Cannot get PowerFactory application.")
        sys.exit(1)
    print("Started PowerFactory.")

    err = app.ActivateProject(project_name)
    if err:
        print(f"Cannot activate project '{project_name}'")
        sys.exit(2)
    print(f"Activated project '{project_name}'.")
    
    D = {'Sym': {}, 'Genstat': {}}
    obj_types = 'Site','Term','Lod'
    for typ in obj_types:
        if typ != 'Site':
            D[typ] = {}
    SMs = app.GetCalcRelevantObjects('*.ElmSym')
    SGs = app.GetCalcRelevantObjects('*.ElmGenstat')
    substations = app.GetCalcRelevantObjects('*.ElmSubstat')
    terminals = app.GetCalcRelevantObjects('*.ElmTerm')
    loads = app.GetCalcRelevantObjects('*.ElmLod')
    all_objs = SMs + SGs + terminals + loads
    for typ in obj_types:
        print(f"Loading objects of type 'Elm{typ}'...")
        for obj in app.GetCalcRelevantObjects('*.Elm' + typ):
            lat,lon = obj.GPSlat,obj.GPSlon
            obj_name = None
            if typ == 'Site' and (lat != 0 or lon != 0):
                site_contents = obj.GetContents()
                for site_obj in site_contents:
                    # if '__SUBNET__' in site_obj.loc_name:
                    if site_obj in substations:
                        substation_contents = site_obj.GetContents()
                        for substation_obj in substation_contents:
                            if substation_obj in all_objs:
                                if substation_obj in SMs:
                                    key = 'Sym'
                                if substation_obj in SGs:
                                    key = 'Genstat'
                                elif substation_obj in terminals:
                                    key = 'Term'
                                elif substation_obj in loads:
                                    key = 'Lod'
                                obj_name = substation_obj.loc_name
                                outserv = substation_obj.outserv
            elif typ != 'Site':
                key = typ
                obj_name = obj.loc_name
                outserv = obj.outserv
            if lat > 0 and lon > 0 and obj_name is not None and obj_name not in D[key]:
                D[key][obj_name] = {'lat': lat, 'lon': lon, 'outserv': outserv}

    json.dump(D, open('{}_coords.json'.format(project_name.split(os.path.sep)[-1]),'w'), indent=4)
    
    SM_info = {sm.loc_name: {'h': sm.typ_id.h, 'S': sm.typ_id.sgn} for sm in SMs}
    SM_info = {name: SM_info[name] for name in  sorted([sm.loc_name for sm in SMs])}
    json.dump(SM_info, open('{}_SM_info.json'.format(project_name.split(os.path.sep)[-1]),'w'), indent=4)
        
