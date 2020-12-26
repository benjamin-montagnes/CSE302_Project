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
