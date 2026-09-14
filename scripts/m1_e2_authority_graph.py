import json, zipfile
from pathlib import Path
from scripts.m1_e2_symbolic import inspect, authority_graph

ASSET=Path('external/hf-libero/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init')
OUT=Path('docs/superpowers/feasibility/m1-r1/G1-E2.3-authority-graph.json')
with zipfile.ZipFile(ASSET) as z:
    result=inspect(z.read('archive/data.pkl'))
graph=authority_graph(result)
graph['asset_member']='archive/data.pkl'
graph['physical_values']='PHYSICAL_OPAQUE'
OUT.write_text(json.dumps(graph,sort_keys=True,indent=2)+'\n')
