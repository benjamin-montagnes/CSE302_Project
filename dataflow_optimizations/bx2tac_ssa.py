#!/usr/bin/env python3

import bx2ast as ast
import tac
from tac import Instr, Gvar, Proc, execute
from io import StringIO
from cfg import infer, linearize
import cse
import copy
import argparse
import random

binop_opcode_map = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod',
                    '&': 'and', '|': 'or', '^': 'xor', '<<': 'shl', '>>': 'shr'}
unop_opcode_map = {'u-': 'neg', '~': 'not'}
relop_map = {
    '==': (lambda l, r: ('jz', l, r)),
    '!=': (lambda l, r: ('jnz', l, r)),
    '<':  (lambda l, r: ('jl', l, r)),
    '<=': (lambda l, r: ('jle', l, r)),
    '>':  (lambda l, r: ('jl', r, l)),
    '>=': (lambda l, r: ('jle', r, l)),
}
bool_ops = tuple(relop_map.keys()) + ('!', '&&', '||')

# ------------------------------------------------------------------------------

class ProcMunch:
    """Process a single procedure"""

    def __init__(self, pdecl: ast.Procdecl):
        self.name = pdecl.name
        proc_scope = {arg : f'%{arg}' for arg, _ in pdecl.args}
        self.t_args = tuple(proc_scope[arg] for arg, _ in pdecl.args)
        self.scopes = [proc_scope]
        # break/continue stacks
        self.break_stack = []
        self.continue_stack = []
        # returning
        self.lab_entry = f'.Lstart'
        self.lab_exit = f'.Lend'
        self.t_result = (tac.dummy_temp if pdecl.retty == ast.VOID \
                         else self.fresh_temp())
        # main processing loop
        self.body = [Instr(None, 'label', self.lab_entry, None)]
        if pdecl.body: self.munch_stmt(pdecl.body)

    def fresh_label(self):
        self.last_label = getattr(self, 'last_label', -1) + 1
        return f'.L{self.last_label}'

    def fresh_temp(self):
        self.last_anontemp = getattr(self, 'last_anontemp', -1) + 1
        return f'%{self.last_anontemp}'

    def __setitem__(self, var, tmp):
        self.scopes[-1][var] = tmp

    def __getitem__(self, var):
        for scope in reversed(self.scopes):
            if var in scope: return scope[var]
        # global variable
        return '@' + var

    def emit(self, instr):
        """Emit a given TAC instruction. May adjust the output slightly
        to make basic blocks inference easier later."""
        if instr.opcode == 'label':
            if self.body[-1].opcode not in ('jmp', 'ret', 'label'):
                self.body.append(Instr(None, 'jmp', instr.arg1, None))
        else:
            if self.body[-1].opcode in ('jmp', 'ret'):
                self.body.append(Instr(None, 'label', self.fresh_label(), None))
        self.body.append(instr)

    def to_tac_proc(self):
        self.emit(Instr(None, 'label', self.lab_exit, None))
        self.emit(Instr(None, 'ret', self.t_result, None))
        def cleanup_labels():
            lcount = 0
            canon_labs = dict()
            instrs = self.body
            self.body = []
            cur = 0
            while cur < len(instrs):
                instr = instrs[cur]
                if instr.opcode == 'label':
                    lab = f'.L{lcount}'
                    lcount += 1
                    self.body.append(Instr(None, 'label', lab, None))
                    while cur < len(instrs):
                        next_instr = instrs[cur]
                        if next_instr.opcode != 'label': break
                        canon_labs[next_instr.arg1] = lab
                        cur += 1
                else:
                    self.body.append(instr)
                    cur += 1
            for instr in self.body:
                if instr.opcode == 'jmp':
                    instr.arg1 = canon_labs[instr.arg1]
                elif instr.opcode in ('jz', 'jnz', 'jl', 'jle'):
                    instr.arg2 = canon_labs[instr.arg2]
        cleanup_labels()
        return tac.Proc(f'@{self.name}', self.t_args, self.body)

    def munch_stmt(self, stmt):
        """Munch a single BX2 statement"""
        if isinstance(stmt, ast.LocalVardecl):
            t_init = self.munch_int_expr(stmt.init)
            t_var = self.fresh_temp()
            self.emit(Instr(t_var, 'copy', t_init, None))
            self[stmt.var] = t_var
        elif isinstance(stmt, ast.Assign):
            t_rhs = self.munch_int_expr(stmt.value)
            self.emit(Instr(self[stmt.var.name], 'copy', t_rhs, None))
        elif isinstance(stmt, ast.Eval):
            if stmt.arg.ty == ast.BOOL:
                # jump to the same place regardless of true or false
                l_same = self.fresh_label()
                self.munch_bool_expr(stmt.arg, l_same, l_same)
                self.emit(Instr(None, 'label', l_same, None))
            else:
                self.munch_int_expr(stmt.arg) # ignore result temp
        elif isinstance(stmt, ast.Block):
            self.scopes.append(dict())
            for inner_stmt in stmt.body: self.munch_stmt(inner_stmt)
            self.scopes.pop()
        elif isinstance(stmt, ast.IfElse):
            l_true = self.fresh_label()
            l_false = self.fresh_label()
            l_end = self.fresh_label()
            self.munch_bool_expr(stmt.cond, l_true, l_false)
            self.emit(Instr(None, 'label', l_false, None))
            self.munch_stmt(stmt.els)
            self.emit(Instr(None, 'jmp', l_end, None))
            self.emit(Instr(None, 'label', l_true, None))
            self.munch_stmt(stmt.thn)
            self.emit(Instr(None, 'label', l_end, None))
        elif isinstance(stmt, ast.While):
            l_header = self.fresh_label()
            l_body = self.fresh_label()
            l_end = self.fresh_label()
            self.emit(Instr(None, 'label', l_header, None))
            self.munch_bool_expr(stmt.cond, l_body, l_end)
            self.emit(Instr(None, 'label', l_body, None))
            self.break_stack.append(l_end)
            self.continue_stack.append(l_header)
            self.munch_stmt(stmt.body)
            self.continue_stack.pop()
            self.break_stack.pop()
            self.emit(Instr(None, 'jmp', l_header, None))
            self.emit(Instr(None, 'label', l_end, None))
        elif isinstance(stmt, ast.Break):
            if len(self.break_stack) == 0:
                raise RuntimeError(f'Cannot break here; not in a loop')
            self.emit(Instr(None, 'jmp', self.break_stack[-1], None))
        elif isinstance(stmt, ast.Continue):
            if len(self.continue_stack) == 0:
                raise RuntimeError(f'Cannot continue here; not in a loop')
            self.emit(Instr(None, 'jmp', self.continue_stack[-1], None))
        elif isinstance(stmt, ast.Return):
            if stmt.arg != None:
                t_arg = self.munch_int_expr(stmt.arg)
                self.emit(Instr(self.t_result, 'copy', t_arg, None))
            self.emit(Instr(None, 'jmp', self.lab_exit, None))
        else:
            raise RuntimeError(f'munch_stmt() cannot handle: {stmt.__class__}')

    def munch_bool_expr(self, expr, l_true, l_false):
        if expr.ty is not ast.BOOL:
            raise RuntimeError(f'munch_bool_expr(): expecting {ast.BOOL}, got {expr.ty}')
        if isinstance(expr, ast.Boolean):
            self.emit(Instr(None, 'jmp', l_true if expr.value else l_false, None))
        elif isinstance(expr, ast.Appl):
            if expr.func in relop_map:
                (opcode, left, right) = relop_map[expr.func](expr.args[0], expr.args[1])
                t_left = self.munch_int_expr(left)
                t_right = self.munch_int_expr(right)
                t_result = self.fresh_temp()
                self.emit(Instr(t_result, 'sub', t_left, t_right))
                self.emit(Instr(None, opcode, t_result, l_true))
                self.emit(Instr(None, 'jmp', l_false, None))
            elif expr.func == '!':
                self.munch_bool_expr(expr.args[0], l_false, l_true)
            elif expr.func == '&&':
                li = self.fresh_label()
                self.munch_bool_expr(expr.args[0], li, l_false)
                self.emit(Instr(None, 'label', li, None))
                self.munch_bool_expr(expr.args[1], l_true, l_false)
            elif expr.func == '||':
                li = self.fresh_label()
                self.munch_bool_expr(expr.args[0], l_true, li)
                self.emit(Instr(None, 'label', li, None))
                self.munch_bool_expr(expr.args[1], l_true, l_false)
            else:
                raise RuntimeError(f'munch_bool_expr(): unknown operator or function {expr.func}')
        else:
            # assume this is returning an int
            t = self.munch_int_expr(expr)
            self.emit(Instr(None, 'jz', t, l_false))
            self.emit(Instr(None, 'jmp', l_true, None))

    def munch_int_expr(self, expr):
        if isinstance(expr, ast.Variable):
            return self[expr.name]
        if isinstance(expr, ast.Number):
            t_result = self.fresh_temp()
            self.emit(Instr(t_result, 'const', expr.value, None))
            return t_result
        if isinstance(expr, ast.Appl):
            if expr.func in binop_opcode_map:
                t_left = self.munch_int_expr(expr.args[0])
                t_right = self.munch_int_expr(expr.args[1])
                t_result = self.fresh_temp()
                self.emit(Instr(t_result, binop_opcode_map[expr.func], t_left, t_right))
                return t_result
            elif expr.func in unop_opcode_map:
                t_left = self.munch_int_expr(expr.args[0])
                t_result = self.fresh_temp()
                self.emit(Instr(t_result, unop_opcode_map[expr.func], t_left, None))
                return t_result
            elif expr.func not in bool_ops:
                # here we assume that we are making a function call
                for i in range(min(len(expr.args), 6)):
                    arg = expr.args[i]
                    t_arg = self.munch_int_expr(arg)
                    self.emit(Instr(None, 'param', i + 1, t_arg))
                remaining_args = []
                for i in range(6, len(expr.args)):
                    arg = expr.args[i]
                    t_arg = self.munch_int_expr(arg)
                    remaining_args.append((i, t_arg))
                if len(remaining_args) % 2 != 0:
                    remaining_args.append((len(expr.args), tac.dummy_temp))
                for i, t in reversed(remaining_args):
                    self.emit(Instr(None, 'param', i + 1, t))
                t_result = tac.dummy_temp if expr.ty == ast.VOID else self.fresh_temp()
                self.emit(Instr(t_result, 'call', '@' + expr.func, len(expr.args)))
                return t_result
        if expr.ty == ast.BOOL:
            t_bool = self.fresh_temp()
            self.emit(Instr(t_bool, 'const', 0, None))
            l_true = self.fresh_label()
            l_false = self.fresh_label()
            self.munch_bool_expr(expr, l_true, l_false)
            self.emit(Instr(None, 'label', l_true, None))
            self.emit(Instr(t_bool, 'const', 1, None))
            self.emit(Instr(None, 'label', l_false, None))
            return t_bool
        raise RuntimeError(f'Unknown expr kind: {expr.__class__}')

# ------------------------------------------------------------------------------

class ssafile:
    def use_set(self, instr):
        # to do:
        # what to do with opcode phi
        if instr.opcode in ['jz', 'jnz', 'jl', 'jle', 'neg', 'not', 'copy','ret']:
            return set([instr.arg1]).difference({"%_"})
        elif instr.opcode in ['param']:
            return set([instr.arg2]).difference({"%_"})
        elif instr.opcode in ['add','sub','mul','div','mod','and','or','xor','shl','shr']:
            return set([instr.arg1,instr.arg2]).difference({"%_"})
        else: return set()
        # label, jmp, nop, const, call -> nothing used

    def def_set(self, instr):
        if instr.opcode in ['neg', 'not', 'copy','add','phi',
                            'sub','mul','div','mod','and','or',
                            'xor','shl','shr','call','const']:
            return set([instr.dest])
        else: return set()
        # param, label, jmp, nop, 'jz', 'jnz', 'jl', 'jle','ret', -> nothing defined

    def live_in(self, cfg):
        livein = dict()
        for instr in cfg.instrs():
            livein[instr] = self.use_set(instr)
        # run the livein set update loop until there are no more changes
        dirty = True
        while dirty:
            dirty = False
            for (i1, i2) in cfg.instr_pairs():
                t_live = livein[i2].difference(self.def_set(i1))
                if not t_live.issubset(livein[i1]):
                    dirty = True # there was a change, so run it again
                    livein[i1] = livein[i1].union(t_live)
        return livein

    def live_out(self, cfg,livein):
            liveout = dict()
            for block in cfg._blockmap.values():
                for i,instr in enumerate(block.instrs()):
                    block_instrs =list(block.instrs())
                    for j in range(i+1,len(block_instrs)):
                        if instr in liveout.keys(): liveout[instr] = (livein[block_instrs[j]]).union(liveout[instr])
                        else: liveout[instr] = livein[block_instrs[j]]
                    for succ_block in [cfg[label] for label in cfg.successors(block.label)]:
                        for succ_instr in succ_block.instrs():
                            if instr in liveout.keys(): liveout[instr] = (livein[succ_instr]).union(liveout[instr])
                            else: liveout[instr] = livein[succ_instr]
            return liveout

    def crude_ssa(self, cfg):
        livein = self.live_in(cfg)
        ### 1.  Add φ-function definitions for all temporaries that are live-in at the start of each block.
        for block in cfg._blockmap.values():
            instrs = block.body+block.jumps
            if instrs != []: live_vars = livein[instrs[0]]  # get livein of first instruction in block
            for live_var in live_vars:                      # loop of variables in this livein
                phi_instr = Instr(live_var, 'phi', dict(), None)    # add a phony "phi" def: mapping will be implemented later
                block.body.insert(0,phi_instr)                      # add to block
        ### 2.  Uniquely version every temporary that is def’d by any instruction in the entire CFG.
        i = 0
        for instr in cfg.instrs():
            if instr.opcode in ['neg', 'not', 'copy','add','phi',   # these opcodes mean we are defining something
                                'sub','mul','div','mod','and','or',
                                'xor','shl','shr','call','const']:
                if instr.dest != "%_": instr.dest +='.'+str(i)
                i += 1
        ### 3.  Update the uses of each temporary within the same block to their most recent versions.
        for block in cfg._blockmap.values():
            instrs = block.body+block.jumps
            for i,instr in enumerate(instrs):
                for var in self.use_set(instr):  # for each var used in the instruction
                    for instr1 in instrs[:i]:
                        if self.def_set(instr1) != set():
                            defed = self.def_set(instr1).pop() # we only ever have one variable defined anyways
                            if defed.split('.')[0] == var:
                                if instr.arg1 == var: instr.arg1 = defed
                                elif instr.arg2 == var: instr.arg2 = defed
        ### 4.  For every edge in the CFG (use the .edges() iterator) fill in the arguments of the φ functions.
                # i.e. of the form L1:%n for every temporary %n that comes to .L2 from .L1.
        # use predecessor blocks to fill args of phi func
        for L1,L2 in cfg.edges():
            for curr_instr in cfg._blockmap[L2].instrs():
                if not curr_instr.opcode == 'phi': continue  # don't do anything if not phi instruction
                # set instr.arg1['.Label1'] = last definition of temp with the same root 
                for prev_instr in cfg._blockmap[L1].instrs():
                    if self.def_set(prev_instr): 
                        defed = self.def_set(prev_instr).pop() # we only ever have one variable defined anyways
                        if defed.split('.')[0] == curr_instr.dest.split('.')[0]:
                            curr_instr.arg1[L1] = defed
        # in entry block we also get the args of the phi func from the args of the proc
        for instr in cfg._blockmap[cfg.lab_entry].instrs():
            if not instr.opcode == 'phi': continue
            instr.arg1[cfg.proc_name] = instr.dest.split('.')[0]
        return cfg

    def nce(self, cfg):
        '''Null choice elimination'''
        # change = 0
        for block in cfg._blockmap.values():
            for instr in block.body:
                if instr.opcode == "phi":
                    args = set(instr.arg1.values())
                    if len(args)==1 and args.pop()==instr.dest:
                        block.body.remove(instr)
                        # change = 1

    def rename_elim(self, cfg):
        change = 0
        for block in cfg._blockmap.values():
            for instr in block.body:
                if instr.opcode == "phi":
                    args = set(instr.arg1.values())
                    old = instr.dest
                    if old in args: args.remove(old)
                    if len(args)==1:
                        renamed = args.pop()
                        if old.split('.')[0] == renamed.split('.')[0]: #Check if same root
                            #Replace all instructions in block
                            for blk in cfg._blockmap.values():
                                for ins in blk.body+block.jumps:
                                    if ins.dest == old: 
                                        ins.dest = renamed
                                        change = 1
                                    if isinstance(ins.arg1,dict):
                                        for k in ins.arg1:
                                            if ins.arg1[k] ==old: 
                                                ins.arg1[k]=renamed
                                                change = 1
                                    else:
                                        if ins.arg1==old: 
                                            ins.arg1=renamed
                                            change = 1
                                    if ins.arg2 == old: 
                                        ins.arg2 = renamed
                                        change = 1
        return change

    def dead_copy_elim(self,cfg):
        change = 0
        for b in cfg._blockmap.values():
            
            to_delete = []          
            for instr in b.body:
                # dead store
                livein = self.live_in(cfg)
                liveout = self.live_out(cfg,livein)

                if not self.def_set(instr).issubset(liveout[instr]):
                    # call instr
                    if instr.opcode == 'call':
                        if instr.dest != '%_':
                            change = 1
                            instr.dest = '%_'
                    # arithmetic operation
                    elif instr.opcode in ['add', 'sub', 'mul', 'div', 'mod', 'neg', 'and', 'or',
                                            'xor', 'not', 'shl', 'shr', 'const', 'copy']:
                        to_delete.append(instr)
                        change = 1

            
            for instr in reversed(to_delete):
                b.body.remove(instr)
        
            
        return change
            
        

    def get_temps(self,cfg):
        #Function that returns the initial Val and Ev mappings
        Val,Ev = {},{}
        for block in cfg._blockmap.values():
            if block.label not in Ev:
                Ev[block.label]=False
            for instr in block.body:
                for v in [instr.dest,instr.arg1,instr.arg2]:
                    if (type(v)==str and v[0]=="%") and v not in Val:
                        Val[v]="Bot"
        return Val,Ev

    def sccp(self, cfg): 
        #We fill Ev and Var with the initial block and arguments

        c_jumps = ['jz', 'jnz', 'jl', 'jle']
        Val,Ev = self.get_temps(cfg)
        #For every temp in input, we set Val(v) to top
        for temp in cfg.t_args:
            Val[temp]="Top"

        #For the initial block, we set Ev(B)=True
        Ev[list(cfg._blockmap.keys())[0]]=True

        #Visiting blocks and updating Ev
        order = list(cfg._blockmap.keys())
        random.shuffle(order)
        
        #We then repeat the below process until there is no change in Ev or Var
        changed = True
        while changed:
            print("Round")
            changed = False

            for b in order:
                definite = False #Presence of a definite jump ()
                if Ev[b]==True:
                    for instr in cfg._blockmap[b].body:
                        #Case of conditional jumps
                        if instr.opcode in c_jumps:
                            jmp_dst = instr.arg2 #Check if it is indeed arg2
                            used = self.use_set(instr)
                            if len(used)==1:
                                if Val[used]=="Top":
                                    Ev[jmp_dst]=True
                                    changed = True
                                elif Val[used]=="Bot":
                                #stop further updates based on this and later jumps in B
                                    break
                                else:
                                    x = Val[used]
                                    if instr.opcode=="jz":
                                        if x==0: 
                                            Ev[jmp_dst],definite=True,True
                                            changed = True
                                    if instr.opcode == "jnz":
                                        if x!=0: 
                                            Ev[jmp_dst],definite=True,True
                                            changed = True
                                    if instr.opcode == "jl":
                                        if x<0: 
                                            Ev[jmp_dst],definite=True,True
                                            changed = True
                                    if instr.opcode == "jle":
                                        if x<=0: 
                                            Ev[jmp_dst],definite=True,True
                                            changed = True

                        #Case of jump
                        elif instr.opcode=="jmp":
                            if definite==False:
                                Ev[instr.arg1]=True
                                changed = True
            
                        else:
                            used = self.use_set(instr)
                            #Cases of phi instruction (numbered as in the pdf of the project task)
                            if instr.opcode=="phi":
                                used_temps = list(instr.arg1.values())
                                block_labels = list(instr.arg1.keys())

                                vals = [Val[i] for i in used_temps]
                                setv = set(vals)

                                #Case 5
                                if len(setv)==1 and list(setv)[0] not in ["Top","Bot"]: 
                                    Val[instr.dest] = val
                                    changed = True
                                
                                #Case 3
                                elif len(setv)==2 and "Bot" in setv:
                                    cop = list(setv)
                                    cop.remove("Bot")
                                    if cop[0] not in ["Top","Bot"]:
                                        Val[instr.dest] = val
                                        changed = True
                                        break
                                
                                else:     
                                    #Keep constant value
                                    cst = []

                                    for temp in used_temps:
                                        val = Val[temp]

                                        if val not in ["Top","Bot"]:
                                            
                                            cst.append(temp)

                                            #Case 1
                                            if len(set(cst))==2:
                                                Val[instr.dest] = "Top"
                                                changed = True
                                                break

                                        #Case 2
                                        if val=="Top":
                                            #We go fetch the block from where it comes from
                                            origin_block = block_labels[used_temps.index(temp)]
                                            if Ev[origin_block]==True: 
                                                Val[instr.dest] = "Top"
                                                changed = True
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
                                            changed = True
                                
                            else:
                                if len(used)==2:
                                    x = used.pop()
                                    y = used.pop()
                                    if Val[x]=="Top" or Val[y]=="Top":
                                        for temp in self.def_set(instr): 
                                            Val[temp]="Top"
                                            changed = True
                                    elif (Val[x] not in ["Top","Bot"]) and (Val[y] not in ["Top","Bot"]):
                                        changed = True
                                        dst = self.def_set(instr)
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

            #print(Ev,Val)
            
            #If the above code was working, we would then delete some instructions
            # according to Ev and Val.

                                

                            

# ------------------------------------------------------------------------------

def process(bx2_prog):
    tac_prog = []
    for tlv in bx2_prog:
        if isinstance(tlv, ast.GlobalVardecl):
            if tlv.ty == ast.INT: value = tlv.init.value
            elif tlv.ty == ast.BOOL: value = (1 if tlv.init.value else 0)
            else: raise RuntimeError(f'{tlv.loc}'
                                     f'COMPILER BUG: process(): '
                                     f'unknown type {tlv.ty}')
            tac_prog.append(Gvar('@' + tlv.var, value))
        elif isinstance(tlv, ast.Procdecl):
            pm = ProcMunch(tlv)
            tac_prog.append(pm.to_tac_proc())
        else:
            raise RuntimeError(f'{tlv.loc}'
                               f'COMPILER BUG: process(): '
                               f'unknown toplevel {tlv.__class__.__name__}')
    return tac_prog

# ------------------------------------------------------------------------------

def ssagen(tacfile):
    # list of gvars and procs
    ssag = ssafile()
    loaded_tac = tacfile
    procs = [tac_proc for tac_proc in loaded_tac if isinstance(tac_proc,Proc)]
    # list: cfg for each proc
    cfgs = [infer(proc) for proc in procs]
    # generate crude ssa
    ssa = [ssag.crude_ssa(cfg) for cfg in cfgs]
    # minimisation steps
    optimise(ssag,ssa)
    # linearise back to TAC
    for i,proc in enumerate(procs):
        linearize(proc,ssa[i])
    output_tac = [gvar for gvar in loaded_tac if isinstance(gvar,Gvar)]+procs
    return output_tac

def optimise(ssag,ssa):
    for cfg in ssa:
        start=True
        #ssag.sccp(cfg) <- The function has problems 
        while start:
            start = False
            if ssag.nce(cfg)==1:            start = True
            if ssag.rename_elim(cfg)==1:    start = True
            if cse.run_cse(cfg)==1:         start = True
            if cfg.copy_propagate()==1:     start = True
            if ssag.dead_copy_elim(cfg)==1: start = True

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    import sys, argparse, time, random, bx2
    from pathlib import Path
    ap = argparse.ArgumentParser(description='BX2 to TAC compiler')
    ap.add_argument('bx_src', metavar='FILE', nargs=1, type=str, help='BX2 program')
    ap.add_argument('-v', dest='verbosity', default=0, action='count', help='increase verbosity')
    ap.add_argument('--interpret', dest='interpret', default=False, action='store_true', help='Run the TAC interpreter instead of outputing a .tac')
    args = ap.parse_args()
    bx_src = Path(args.bx_src[0])
    if not bx_src.suffix == '.bx':
        print(f'File name {bx_src} does not end in ".bx"')
        exit(1)
    bx2.lexer.load_source(bx_src)
    bx2_prog = bx2.parser.parse(lexer=bx2.lexer)
    cx = ast.analyze_program(bx2_prog)
    for tlv in bx2_prog: tlv.type_check(cx)
    tac_prog_bef = process(bx2_prog)
    # print('or',tac_prog_bef)
    tac_prog = ssagen(tac_prog_bef)
    
    
    """     if args.interpret:
        # to do: this isn't working
        # Error: 'no value for args in function call'
        execute(tac_prog, '@main', (), show_proc=(args.verbosity>0), show_instr=(args.verbosity>1), only_decimal=(args.verbosity<=2))
    else: """

    tac_file = bx_src.with_suffix('.tac')
    with open(tac_file, 'w') as f:
        for tlv in tac_prog:
            print(tlv, file=f)
    print(f'{bx_src} -> {tac_file} done') 
