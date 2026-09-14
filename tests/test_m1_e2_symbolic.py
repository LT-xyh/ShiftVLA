import pickle
import zipfile
from scripts.m1_e2_symbolic import inspect, authority_graph

def test_symbolic_reduce_build_and_memo_without_execution():
    class P:
        def __reduce__(self): return (str, ('opaque',))
    data=pickle.dumps(P(), protocol=2)
    r=inspect(data)
    assert r['reducers'] and not r['unsupported']

def test_rejects_dynamic_and_persistent_authority_opcodes():
    assert inspect(b'\x80\x02cfoo\nbar\n.')['globals']
    assert inspect(b'Px\n.')['unsupported'][0][1] == 'PERSID'

def test_scalar_mark_tuple_and_memo_identity():
    r=inspect(b'\x80\x02K\x01K\x02\x86.')
    assert r['final_stack']==('TUPLE',)

def test_empty_containers_are_not_silently_ignored():
    for data,name in [(b'\x80\x02]\x94.', 'LIST'),(b'\x80\x02}\x94.', 'DICT'),(b'\x80\x02)\x94.', 'TUPLE')]:
        r=inspect(data); assert not r['unsupported'], (name,r['unsupported'])
    r=inspect(b'\x80\x02](K\x01K\x02e.'); assert r['final_stack']==('LIST',)
    r=inspect(b'\x80\x02}(K\x01K\x02u.'); assert r['final_stack']==('DICT',)

def test_memo_put_get_and_missing_reference_are_explicit():
    assert inspect(b'\x80\x02K\x01q\x00h\x00\x86.')['memo_size']==1
    assert inspect(b'\x80\x02h\x00.')['final_stack']==('MISSING',)

def test_nested_global_tuple_reduce_and_build_identity():
    data=b'\x80\x02cmod\nfn\n)\x85R.'
    r=inspect(data); assert len(r['reducers'])==1
    pos,fn,args,out=r['reducers'][0]
    assert fn.value=='mod fn' and args.kind=='TUPLE'

def test_authority_opcodes_are_not_silently_accepted():
    r=inspect(b'\x80\x02cmod\nfn\n)\x81.')
    assert r['unsupported'] and r['unsupported'][0][1]=='NEWOBJ'

def test_inspect_never_retains_secret_literal_recursively():
    secret=b'UNIQUE_PHYSICAL_SECRET_9f4c'
    data=b'\x80\x02X'+len(secret).to_bytes(4,'little')+secret+b'0.'
    r=inspect(data)
    def walk(x):
        if isinstance(x,bytes): assert secret not in x
        elif isinstance(x,str): assert secret.decode() not in x
        elif isinstance(x,dict):
            for k,v in x.items(): walk(k); walk(v)
        elif isinstance(x,(list,tuple)):
            for v in x: walk(v)
    walk(r)

def test_authority_graph_has_stable_operation_records():
    r=inspect(b'\x80\x02cmod\nfn\n)\x85R.')
    g=authority_graph(r)
    assert g['operations'][0]['opcode']=='REDUCE'
    assert g['operations'][0]['callable']['ref'].startswith('GLOBAL#')

def test_fixed_authority_graph_callable_offsets_are_canonical():
    p='external/hf-libero/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init'
    with zipfile.ZipFile(p) as z: g=authority_graph(inspect(z.read('archive/data.pkl')))
    nodes={n['id']:n for n in g['nodes']}
    ops={x['offset']:x for x in g['operations'] if x['opcode']=='REDUCE'}
    assert set(ops)=={104,110,152,43851}
    assert nodes[ops[104]['callable']['ref']]['authority']=='_codecs encode'
    assert nodes[ops[110]['callable']['ref']]['authority']=='numpy.core.multiarray _reconstruct'
    assert nodes[ops[152]['callable']['ref']]['authority']=='numpy dtype'
    assert nodes[ops[43851]['callable']['ref']]['authority']=='_codecs encode'
