import pickle
from scripts.m1_e2_symbolic import inspect

def test_symbolic_reduce_build_and_memo_without_execution():
    class P:
        def __reduce__(self): return (str, ('opaque',))
    data=pickle.dumps(P(), protocol=2)
    r=inspect(data)
    assert r['reducers'] and not r['unsupported']

def test_rejects_dynamic_and_persistent_authority_opcodes():
    assert inspect(b'\x80\x02cfoo\nbar\n.')['globals']
    assert inspect(b'Px\n.')['unsupported'][0][1] == 'PERSID'
