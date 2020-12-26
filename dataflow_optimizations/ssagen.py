#!/usr/bin/env python3

import tac
from cfg import infer, linearize
import cse
import copy
import cfg as tac_cfg
import re, random, os

# ------------------------------------------------------------------------------
# liveness

_arg1_use = re.compile(r'add|sub|mul|div|mod|neg|and|or|xor|not|shl|shr|copy|ret|jz|jnz|jl|jle')
_arg2_use = re.compile(r'add|sub|mul|div|mod|and|or|xor|shl|shr|param')
_dest_def = re.compile(r'add|sub|mul|div|mod|neg|and|or|xor|not|shl|shr|const|copy|phi|call')

def use_set(instr):
    s = set()
    if _arg1_use.fullmatch(instr.opcode): s.add(instr.arg1)
    if _arg2_use.fullmatch(instr.opcode): s.add(instr.arg2)
    if instr.opcode == 'phi': s.update(instr.arg1.values())
    return s

def rewrite_use_temps_nonphi(instr, fn):
    if _arg1_use.fullmatch(instr.opcode):
        instr.arg1 = fn(instr.arg1)
    if _arg2_use.fullmatch(instr.opcode):
        instr.arg2 = fn(instr.arg2)

def def_set(instr):
    s = set()
    if _dest_def.fullmatch(instr.opcode): s.add(instr.dest)
    return s

def rewrite_temps(instr, fn):
    if _arg1_use.fullmatch(instr.opcode):
        instr.arg1 = fn(instr.arg1)
    if _arg2_use.fullmatch(instr.opcode):
        instr.arg2 = fn(instr.arg2)
    if instr.opcode == 'phi':
        for l, t in instr.arg1.items():
            instr.arg1[l] = fn(t)
    if _dest_def.fullmatch(instr.opcode):
        instr.dest = fn(instr.dest)

# ------------------------------------------------------------------------------
# crude SSA gen

def tmp_root(tmp):
    try: return tmp[:tmp.rindex('.')]
    except ValueError: return tmp

def tmp_version(tmp):
    try: return tmp[tmp.rindex('.')+1:]
    except ValueError: return ''

def crude_ssagen(tlv, cfg):
    livein, liveout = dict(), dict()
    tac_cfg.recompute_liveness(cfg, livein, liveout)
    for bl in cfg.nodes():
        prev_labs = list(cfg.predecessors(bl.label))
        ts = livein[bl.first_instr()]
        if len(prev_labs) == 0: prev_labs = [cfg.proc_name]
        bl.body[:0] = [tac.Instr(t, 'phi', {l: t for l in prev_labs}, None) \
                       for t in ts]
    versions = tac_cfg.counter(transfn=lambda x: f'.{x}')
    for i in cfg.instrs():
        if i.dest: i.dest = i.dest + next(versions)
    ver_maps = {cfg.proc_name: {t: t for t in tlv.t_args}}
    for bl in cfg.nodes():
        ver_map = dict()
        for instr in bl.instrs():
            rewrite_use_temps_nonphi(instr, lambda t: ver_map.get(t, t))
            if instr.dest:
                ver_map[tmp_root(instr.dest)] = instr.dest
        ver_maps[bl.label] = ver_map
    for bl in cfg.nodes():
        for instr in bl.instrs():
            if instr.opcode != 'phi': continue
            for lab_prev, root in instr.arg1.items():
                instr.arg1[lab_prev] = ver_maps[lab_prev].get(root, root)

class ufset:
    def __init__(self):
        self._parents = dict()

    def __call__(self, x):
        if x not in self._parents: return x
        p = self(self._parents[x])
        self._parents[x] = p    # path compression
        return p

    def union(self, x, y):
        px = self(x)
        py = self(y)
        self._parents[px] = py

def ssa_minimize(tlv, cfg):
    teq = ufset()
    dirty = True
    change = False
    while dirty:
        dirty = False
        for instr in cfg.instrs():
            if instr.opcode != 'phi': continue
            lhs = teq(instr.dest)
            rhs = set(teq(t) for t in instr.arg1.values())
            # null choice
            if len(rhs) == 1 and lhs in rhs:
                dirty = True
                change = True
                instr.dest, instr.arg1 = None, None
                instr.opcode = 'nop'
            # rename
            if len(rhs.union({lhs})) == 2:
                dirty = True
                change = True
                if lhs in rhs: rhs.remove(lhs)
                teq.union(lhs, rhs.pop())
    tlv.t_args = tuple(teq(t) for t in tlv.t_args)
    for bl in cfg.nodes():
        bl.body = list(filter(lambda instr: instr.opcode != 'nop', bl.body))
    for instr in cfg.instrs(): rewrite_temps(instr, teq)
    return change

# ------------------------------------------------------------------------------

class IG:
    def __init__(self):
        self._nxt = dict()

    def add_edge(self, f, t):
        if f == t: return
        self._nxt.setdefault(f, set()).add(t)
        self._nxt.setdefault(t, set()).add(f)

    def remove_edge(self, f, t):
        if f == t: return
        self._nxt[f].discard(t)
        self._nxt[t].discard(f)

    def remove_node(self, n):
        for t in self._nxt[n]: self._nxt[t].discard(n)
        del self._nxt[n]

    def mcs(self):
        waiting = set(self._nxt.keys())
        wt = {n: 0 for n in waiting}
        order = []
        while len(waiting) > 0:
            v = max(waiting, key=lambda n: wt[n])
            order.append(v)
            for w in self._nxt[v]:
                if w not in waiting: continue
                wt[w] += 1
            waiting.remove(v)
        return order

    def color(self, *, pre=None, order=None):
        col = pre.copy() if pre else dict()
        order = order or self.mcs()
        for n in self._nxt: col.setdefault(n, 0)
        for v in order:
            if col[v] != 0: continue
            ncols = {col[w] for w in self._nxt[v]}
            k = 1
            while True:
                if k not in ncols: break
                k += 1
            col[v] = k
        for u, vs in self._nxt.items():
            for v in vs:
                assert col[u] != col[v]
        return col

    def write_dot(self, name, filename, col=None):
        with open(filename, 'w') as f:
            print(f'graph {name} {{\ngraph[overlap=false];', file=f)
            ids = {u: k for k, u in enumerate(self._nxt)}
            for u in self._nxt:
                k = col[u] if col else '?'
                print(f'{ids[u]} [label="{u}/{k}",fontname=monospace,fontsize=8];', file=f)
            for u, vs in self._nxt.items():
                for v in vs:
                    if v < u: continue
                    print(f'{ids[u]} -- {ids[v]};', file=f)
            print('}', file=f)
        os.system(f'neato -Tpdf -O {filename}')

# ------------------------------------------------------------------------------

def get_temps(cfg):
    Val,Ev = {},{}
    for block in cfg._blockmap.values():
        if block.label not in Ev:
            Ev[block.label]=False
        for instr in block.body:
            for v in [instr.dest,instr.arg1,instr.arg2]:
                if (type(v)==str and v[0]=="%") and v not in Val:
                    Val[v]="Bot"
    return Val,Ev

def sccp(cfg): #Cases are numbered according to the pdf of the project
    c_jumps = ['jz', 'jnz', 'jl', 'jle']
    Val,Ev = get_temps(cfg)
    #For every temp in input, we set Val(v) to top
    for temp in cfg.t_args:
        Val[temp]="Top"

    #For the initial block, we set Ev(B)=True
    Ev[list(cfg._blockmap.keys())[0]]=True

    #print(Ev)
    #Visiting blocks and updating Ev
    order = list(cfg._blockmap.keys())
    random.shuffle(order)
    for b in order:
        definite = False #Presence of a definite jump ()
        if Ev[b]==True:
            for instr in cfg._blockmap[b].body:
                if instr.opcode in c_jumps:
                    jmp_dst = instr.arg2 #Check if it is indeed arg2
                    used = use_set(instr)
                    if len(used)==1:
                        if Val[used]=="Top":
                            Ev[jmp_dst]=True
                        elif Val[used]=="Bot":
                        #stop further updates based on this and later jumps in B
                            break
                        else:
                        #check conditions, is the jump definite only when condition is satisfied?
                            x = Val[used]
                            if instr.opcode=="jz":
                                if x==0: 
                                    Ev[jmp_dst],definite=True,True
                            if instr.opcode == "jnz":
                                if x!=0: Ev[jmp_dst],definite=True,True
                            if instr.opcode == "jl":
                                if x<0: Ev[jmp_dst],definite=True,True
                            if instr.opcode == "jle":
                                if x<=0: Ev[jmp_dst],definite=True,True


                elif instr.opcode=="jmp":
                    if definite==False:
                        Ev[instr.arg1]=True
    
                else:
                    used = use_set(instr)
                    if instr.opcode=="phi":
                        used_temps = list(instr.arg1.values())
                        block_labels = list(instr.arg1.keys())

                        vals = [Val[i] for i in used_temps]
                        setv = set(vals)

                        #Case 5
                        if len(setv)==1 and list(setv)[0] not in ["Top","Bot"]: 
                            Val[instr.dest] = val
                        
                        #Case 3
                        elif len(setv)==2 and "Bot" in setv:
                            cop = list(setv)
                            cop.remove("Bot")
                            if cop[0] not in ["Top","Bot"]:
                                Val[instr.dest] = val
                                break
                        
                        else:     
                            #Keep constant value
                            constants = []

                            for temp in used_temps:
                                val = Val[temp]

                                if val not in ["Top","Bot"]:
                                    
                                    cst.append(temp)

                                    #Case 1
                                    if len(set(cst))==2:
                                        Val[instr.dest] = "Top"
                                        break

                                #Case 2
                                if val=="Top":
                                    #We go fetch the block from where it comes from
                                    origin_block = block_labels[used_temps.index(temp)]
                                    if Ev[origin_block]==True: 
                                        Val[instr.dest] = "Top"
                                        break
                            
                            #Case 4
                            for c_temp in cst:
                                others = used_temps.remove(c_temp)
                                n = len(others)
                                k = 0
                                for oth in others:
                                    origin_block = block_labels[used_temps.index(oth)]
                                    if Ev[origin_block]==True: break
                                    else: k+=1

                                if k==n:
                                    Val[instr.dest] = Val[c_temp]
                        
                    else:
                        if len(used)==2:
                            x = used.pop()
                            y = used.pop()
                            if Val[x]=="Top" or Val[y]=="Top":
                                for temp in def_set(instr): 
                                    Val[temp]="Top"
                            elif (Val[x] not in ["Top","Bot"]) and (Val[y] not in ["Top","Bot"]):
                                dst = def_set(instr)
                                if instr.opcode=="add": Val[dst]= Val[x]+Val[y]
                                elif instr.opcode=="sub": Val[dst]= Val[x]-Val[y]
                                elif instr.opcode=="mul": Val[dst]= Val[x]*Val[y]
                                elif instr.opcode=="div": Val[dst]= Val[x]/Val[y]
                                elif instr.opcode=="mod": Val[dst]= Val[x]%Val[y]
                                elif instr.opcode=="shl": Val[dst]= Val[x]<<Val[y]
                                elif instr.opcode=="shr": Val[dst]= Val[x]>>Val[y]
                                elif instr.opcode=="and": Val[dst]= Val[x]&Val[y]
                                elif instr.opcode=="or": Val[dst]= Val[x]|Val[y]
                                elif instr.opcode=="xor": Val[dst]= Val[x]^Val[y]

# ------------------------------------------------------------------------------

def ssagen_and_optimise(loaded_tac):
    for tlv in loaded_tac:
        if isinstance(tlv, tac.Proc):
            cfg = tac_cfg.infer(tlv)
            crude_ssagen(tlv, cfg)
            dirty = True
            while dirty: 
                dirty = False
                if ssa_minimize(tlv, cfg): dirty = True # rename and nce
                if cse.run_cse(cfg): dirty = True       # common sub-expression elimination
                if cfg.copy_propagate(): dirty = True   # copy propagation
                sccp(cfg)                               # sccp
            livein, liveout = dict(), dict()
            tac_cfg.recompute_liveness(cfg, livein, liveout)
            tac_cfg.linearize(tlv, cfg)
    return loaded_tac

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    ap = ArgumentParser(description='TAC library, parser, and interpreter')
    ap.add_argument('file', metavar='FILE', type=str, nargs=1, help='A TAC file')
    ap.add_argument('-v', dest='verbosity', default=0, action='count',
                    help='increase verbosity')
    args = ap.parse_args()
    gvars, procs = dict(), dict()
    for tlv in tac.load_tac(args.file[0]):
        if isinstance(tlv, tac.Proc):
            cfg = tac_cfg.infer(tlv)
            crude_ssagen(tlv, cfg)
            ssa_minimize(tlv, cfg)
            livein, liveout = dict(), dict()
            tac_cfg.recompute_liveness(cfg, livein, liveout)
            tac_cfg.linearize(tlv, cfg)
            if args.verbosity >= 2:
                print(tlv)