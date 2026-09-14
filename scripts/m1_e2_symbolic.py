"""Non-executing pickle VM inspection for the fixed LIBERO init asset."""
from __future__ import annotations
import pickletools
from dataclasses import dataclass

@dataclass(frozen=True)
class Sym:
    kind: str
    value: object = None

def _opaque(op, arg):
    return Sym(op, '<opaque>' if isinstance(arg, (bytes, str, int, float)) else None)

def inspect(data: bytes) -> dict:
    stack: list[Sym] = []; memo: dict[int, Sym] = {}; marks=[]; events=[]; globals=[]; reducers=[]; builds=[]; unsupported=[]
    for op,arg,pos in pickletools.genops(data):
        before=tuple(x.kind for x in stack); reads=[]; writes=[]
        n=op.name
        if n=='MARK': marks.append(len(stack))
        elif n=='POP':
            if stack: stack.pop()
        elif n=='POP_MARK':
            if marks: del stack[marks.pop():]
        elif n in ('NONE','NEWTRUE','NEWFALSE','BININT','BININT1','BININT2','LONG','BINFLOAT','BINUNICODE','SHORT_BINUNICODE','BINBYTES','SHORT_BINBYTES'):
            stack.append(Sym(n, arg if n not in ('BINBYTES','SHORT_BINBYTES') else '<opaque>'))
        elif n in ('EMPTY_TUPLE','EMPTY_LIST','EMPTY_DICT','EMPTY_SET'):
            stack.append(Sym(n[6:].upper() if n!='EMPTY_TUPLE' else 'TUPLE', ()))
        elif n in ('GLOBAL','STACK_GLOBAL'):
            if n=='GLOBAL':
                target=str(arg); stack.append(Sym('GLOBAL',target)); globals.append((pos,n,target))
            else:
                name=stack.pop() if stack else Sym('MISSING'); mod=stack.pop() if stack else Sym('MISSING')
                target=(mod.value,name.value); stack.append(Sym('GLOBAL',target)); globals.append((pos,n,target))
        elif n in ('BINPUT','LONG_BINPUT','PUT'):
            idx=int(arg); memo[idx]=stack[-1] if stack else Sym('MISSING'); writes.append(idx)
        elif n=='MEMOIZE':
            idx=len(memo); memo[idx]=stack[-1] if stack else Sym('MISSING'); writes.append(idx)
        elif n in ('BINGET','LONG_BINGET','GET'):
            idx=int(arg); stack.append(memo.get(idx,Sym('MISSING'))); reads.append(idx)
        elif n in ('TUPLE','TUPLE1','TUPLE2','TUPLE3'):
            if n=='TUPLE': start=marks.pop() if marks else 0; vals=stack[start:]; del stack[start:]
            else: count=int(n[-1]); vals=stack[-count:]; del stack[-count:]
            stack.append(Sym('TUPLE',tuple(vals)))
        elif n in ('DUP',):
            if stack: stack.append(stack[-1])
            else: unsupported.append((pos,n,arg))
        elif n in ('LIST','DICT'):
            start=marks.pop() if marks else 0; vals=stack[start:]; del stack[start:]
            stack.append(Sym('LIST' if n=='LIST' else 'DICT', tuple(vals)))
        elif n in ('APPENDS','SETITEMS','ADDITEMS'):
            start=marks.pop() if marks else len(stack); vals=stack[start:]; del stack[start:]
            if not stack: unsupported.append((pos,n,arg))
            else:
                target=stack.pop(); stack.append(Sym(target.kind, target.value + tuple(vals)))
        elif n in ('REDUCE',):
            args=stack.pop() if stack else Sym('MISSING'); fn=stack.pop() if stack else Sym('MISSING'); out=Sym('REDUCE_RESULT',(fn,args)); stack.append(out); reducers.append((pos,fn,args,out))
        elif n=='BUILD':
            state=stack.pop() if stack else Sym('MISSING'); target=stack[-1] if stack else Sym('MISSING'); builds.append((pos,target,state)); stack[-1]=Sym('BUILT',(target,state)) if stack else stack.append(Sym('MISSING'))
        elif n in ('NEWOBJ','NEWOBJ_EX','INST','OBJ','EXT1','EXT2','EXT4','PERSID','BINPERSID'):
            unsupported.append((pos,n,arg))
        elif n=='STOP':
            pass
        else:
            # Preserve authority while avoiding value interpretation.
            if n not in {'PROTO','FRAME','MARK','POP','POP_MARK','DUP','NONE','NEWTRUE','NEWFALSE','BININT','BININT1','BININT2','LONG','BINFLOAT','BINUNICODE','SHORT_BINUNICODE','BINBYTES','SHORT_BINBYTES','EMPTY_TUPLE','TUPLE','TUPLE1','TUPLE2','TUPLE3','EMPTY_LIST','APPENDS','LIST','EMPTY_DICT','DICT','SETITEMS','EMPTY_SET','ADDITEMS','TUPLE1','MEMOIZE'}:
                unsupported.append((pos,n,arg))
        events.append({'offset':pos,'opcode':n,'before':before,'after':tuple(x.kind for x in stack),'memo_reads':reads,'memo_writes':writes})
    return {'events':events,'globals':globals,'reducers':reducers,'builds':builds,'unsupported':unsupported,'final_stack':tuple(x.kind for x in stack),'memo_size':len(memo)}

if __name__ == '__main__':
    import pathlib, zipfile, json
    p=pathlib.Path('external/hf-libero/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init')
    with zipfile.ZipFile(p) as z: result=inspect(z.read('archive/data.pkl'))
    # Never serialize symbolic ``value`` fields: fixed assets contain physical
    # simulator state in BINUNICODE/BINBYTES payloads.
    safe = {'globals': result['globals'],
            'reducers': [(p, fn.kind, args.kind, out.kind) for p,fn,args,out in result['reducers']],
            'builds': [(p, target.kind, state.kind) for p,target,state in result['builds']],
            'unsupported': [(p,n) for p,n,_ in result['unsupported']],
            'final_stack': result['final_stack'], 'memo_size': result['memo_size']}
    print(json.dumps(safe, default=str))
