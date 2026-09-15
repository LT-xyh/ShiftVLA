import json, zipfile
from scripts.m1_e2_symbolic import inspect, authority_graph
from scripts.m1_e2_metadata import extract_structural_metadata, FIXED_ASSET_SHA256

ASSET='external/hf-libero/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init'

def fixed_bytes():
    with zipfile.ZipFile(ASSET) as z: return z.read('archive/data.pkl')

def test_fixed_role_bound_metadata_and_physical_opaque_payload():
    data=fixed_bytes(); g=authority_graph(inspect(data))
    r=extract_structural_metadata(data, asset_sha256=FIXED_ASSET_SHA256, authority_graph=g)
    assert r['dtype']['descriptor']['value']=='f8'
    assert r['dtype']['descriptor']['classification']=='STRUCTURAL_METADATA'
    assert r['dtype']['hasobject'] is False
    assert r['dtype']['nested_object_fields'] is False
    assert r['dtype']['align']['value'] is False
    assert r['dtype']['copy']['value'] is True
    assert r['dtype_build']['arity']==8
    assert [x['value'] for x in r['dtype_build']['fields']] == [3, '<', None, None, None, -1, -1, 0]
    assert r['ndarray_build']['arity']==5
    assert r['ndarray_build']['data']['classification']=='PHYSICAL_OPAQUE'
    assert r['ndarray_build']['fields'][1]['value']['arity'] == 2
    assert r['ndarray_build']['fields'][2]['classification'] == 'AUTHORITY_REFERENCE'
    assert r['ndarray_build']['fields'][3]['value'] is False
    assert r['fields'][-2]['classification'] == 'PHYSICAL_OPAQUE'
    # ``inspect`` intentionally returns Sym objects; repr is used only to
    # serialize the redacted view for this regression assertion.
    assert 'f8' not in json.dumps(inspect(data), default=repr)
    assert 'f8' not in json.dumps(g)
    physical_marker = data[188:204].hex()
    assert physical_marker not in json.dumps(r)
    assert all(x['classification'] != 'PHYSICAL_OPAQUE' or x['value'] is None for x in r['fields'])

def test_hash_and_symbolic_path_drift_disable_decoding():
    data=fixed_bytes(); g=authority_graph(inspect(data))
    try:
        extract_structural_metadata(data, asset_sha256='0'*64, authority_graph=g)
    except ValueError: pass
    else: raise AssertionError('hash drift must fail closed')
    bad=json.loads(json.dumps(g)); bad['operations'][2]['offset']=153
    r=extract_structural_metadata(data, asset_sha256=FIXED_ASSET_SHA256, authority_graph=bad)
    assert r['status']=='UNRESOLVED'
    assert all(x['classification'] != 'STRUCTURAL_METADATA' for x in r['fields'])

def test_unapproved_same_opcode_remains_opaque():
    data=fixed_bytes(); g=authority_graph(inspect(data))
    r=extract_structural_metadata(data, asset_sha256=FIXED_ASSET_SHA256, authority_graph=g, allow_paths=())
    assert r['status']=='UNRESOLVED'
    assert all(x['classification'] != 'STRUCTURAL_METADATA' for x in r['fields'])

def test_extra_authority_operation_fails_closed():
    data=fixed_bytes(); g=authority_graph(inspect(data))
    bad=json.loads(json.dumps(g))
    bad['operations'].append({'offset': 999, 'opcode': 'REDUCE', 'callable': {'ref': 'GLOBAL#1'}, 'args': {'ref': 'TUPLE#2'}, 'result': {'ref': 'REDUCE_RESULT#5'}})
    r=extract_structural_metadata(data, asset_sha256=FIXED_ASSET_SHA256, authority_graph=bad)
    assert r['status']=='UNRESOLVED'
