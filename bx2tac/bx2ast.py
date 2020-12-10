#!/usr/bin/env python3

"""
BX1 Abstract Syntax Tree (with optional types)
"""

from io import StringIO
from ply_util import Locatable
from util import indent, Brak

# ------------------------------------------------------------------------------
# Types of BX1

class Type:
    """Parent class of all types"""
    def __repr__(self):
        return f'Type({repr(str(self))})'

class _Basic(Type):
    """Basic types"""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.name == other.name

INT  = _Basic('int')
BOOL = _Basic('bool')
VOID = _Basic('void')

class FUNC(Type):
    """Function types"""
    def __init__(self, result, *args):
        self.result = result or VOID
        self.args = args

    def __str__(self):
        if not hasattr(self, '_str'):
            self._str = f'({", ".join(str(ty) for ty in self.args)}) -> {self.result!s}'
        return self._str

    def __hash__(self):
        return hash((self.result, *self.args))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.result == other.result \
               and self.args == other.args

# ------------------------------------------------------------------------------
# Contexts, aka a hierarchy of symbol tables

class Context:
    """Symbol management"""
    def _make_builtins():
        cx = dict()
        ty = FUNC(INT, INT, INT)
        for op in ('+', '-', '*', '/', '%',
                   '&', '|', '^', '<<', '>>'):
            cx[op] = ty
        ty = FUNC(INT, INT)
        for op in ('u-', '~'):
            cx[op] = ty
        ty = FUNC(BOOL, INT, INT)
        for op in ('==', '!=', '<', '<=', '>', '>='):
            cx[op] = ty
        ty = FUNC(BOOL, BOOL, BOOL)
        for op in ('&&', '||'):
            cx[op] = ty
        cx['!'] = FUNC(BOOL, BOOL)
        cx['__bx_print_int'] = FUNC(VOID, INT)
        cx['__bx_print_bool'] = FUNC(VOID, BOOL)
        return cx
    _builtins = _make_builtins()

    def __init__(self):
        self.global_defs = self._builtins.copy()
        self.local_defs = []

    @property
    def current(self):
        """Return the current scope"""
        if len(self.local_defs) == 0:
            return self.global_defs
        else:
            return self.local_defs[-1]

    def enter(self):
        """Enter a new (local) scope"""
        self.local_defs.append(dict())

    def leave(self):
        """Leave a local scope"""
        self.local_defs.pop()

    def __contains__(self, name):
        for scope in reversed(self.local_defs):
            if name in scope: return True
        return name in self.global_defs

    def __getitem__(self, name):
        """Lookup the name in the context."""
        for scope in reversed(self.local_defs):
            if name in scope: return scope[name]
        return self.global_defs[name]

    def __setitem__(self, name, ty):
        """Add the name in the current scope"""
        current = self.current
        if name in current:
            raise KeyError(f'Redeclaration of {name} in same scope')
        current[name] = ty

    def __str__(self):
        return str((self.global_defs, self.local_defs))

# ------------------------------------------------------------------------------
# Expression

class Expr(Locatable):
    """Parent class of all expressions"""
    __slots__ = ('_ty')

    @property
    def ty(self):
        """Read the type of the expression"""
        return self._ty

    def __repr__(self):
        return f'Expr({repr(str(self))})'

    def __str__(self):
        return str(self._brak())

class Number(Expr):
    """64-bit signed integers (2's complement)"""
    def __init__(self, value):
        if not isinstance(value, int):
            raise ValueError(f'Invald Boolean: {val}')
        self.value = value
        self._ty = INT

    def _brak(self):
        return Brak.atom(str(self.value))

    def type_check(self, context):
        pass                    # nothing to do

class Boolean(Expr):
    """Booleans"""
    def __init__(self, value):
        if not isinstance(value, bool):
            raise ValueError(f'Invald Boolean: {val}')
        self.value = value
        self._ty = BOOL

    def _brak(self):
        return Brak.atom('true' if self.value else 'false')

    def type_check(self, context):
        pass                    # nothing to do

class Variable(Expr):
    """Variables"""
    def __init__(self, name):
        self.name = name

    def _brak(self):
        return Brak.atom(self.name)

    def type_check(self, context):
        if self.name not in context:
            raise TypeError(f'{self.loc}'
                            f'Cannot determine type: '
                            f'variable {self.name} not in scope')
        self._ty = context[self.name]

class Appl(Expr):
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    _builtin_map = {
        '||' : (lambda args: Brak.appl(' || ', 3,  'infixl', *args)),
        '&&' : (lambda args: Brak.appl(' && ', 6,  'infixl', *args)),
        '|'  : (lambda args: Brak.appl(' | ',  10, 'infixl', *args)),
        '^'  : (lambda args: Brak.appl(' ^ ',  20, 'infixl', *args)),
        '&'  : (lambda args: Brak.appl(' & ',  30, 'infixl', *args)),
        '==' : (lambda args: Brak.appl(' == ', 33, 'infix',  *args)),
        '!=' : (lambda args: Brak.appl(' != ', 33, 'infix',  *args)),
        '<'  : (lambda args: Brak.appl(' < ',  36, 'infix',  *args)),
        '<=' : (lambda args: Brak.appl(' <= ', 36, 'infix',  *args)),
        '>'  : (lambda args: Brak.appl(' > ',  36, 'infix',  *args)),
        '>=' : (lambda args: Brak.appl(' >= ', 36, 'infix',  *args)),
        '<<' : (lambda args: Brak.appl(' << ', 40, 'infixl', *args)),
        '>>' : (lambda args: Brak.appl(' >> ', 40, 'infixl', *args)),
        '+'  : (lambda args: Brak.appl(' + ',  50, 'infixl', *args)),
        '-'  : (lambda args: Brak.appl(' - ',  50, 'infixl', *args)),
        '*'  : (lambda args: Brak.appl(' * ',  60, 'infixl', *args)),
        '/'  : (lambda args: Brak.appl(' / ',  60, 'infixl', *args)),
        '%'  : (lambda args: Brak.appl(' % ',  60, 'infixl', *args)),
        'u-' : (lambda args: Brak.appl('-',    70, 'prefix', *args)),
        '!'  : (lambda args: Brak.appl('!',    70, 'prefix', *args)),
        '~'  : (lambda args: Brak.appl('~',    80, 'prefix', *args)),
    }

    def _brak(self):
        if self.func in self._builtin_map:
            args = (arg._brak() for arg in self.args)
            return self._builtin_map[self.func](args)
        argstr = ', '.join(str(arg) for arg in self.args)
        return Brak.atom(f'{self.func}({argstr})')

    def type_check(self, context):
        if self.func == 'print':
            if len(self.args) != 1:
                raise TypeError(f'{self.loc}'
                                f'Arity mismatch for {self.func}: '
                                f'expected 1, got {len(self.args)}')
            self.args[0].type_check(context)
            if self.args[0].ty is INT:
                self.func = '__bx_print_int'
            elif self.args[0].ty is BOOL:
                self.func = '__bx_print_bool'
            else:
                raise TypeError(f'{self.args[0].loc}'
                                f'Type mismatch: '
                                f'expected int or bool, got {self.args[0].ty}')
            self._ty = VOID
        try:
            func_ty = context[self.func]
            if len(func_ty.args) != len(self.args):
                raise TypeError(f'{self.loc}'
                                f'Arity mismatch for {self.func}: '
                                f'expected {len(func_ty.args)}, got {len(self.args)}')
            for i, arg_ty in enumerate(func_ty.args):
                self.args[i].type_check(context)
                if self.args[i].ty != arg_ty:
                    raise TypeError(f'{self.args[i].loc}'
                                    f'Type mismatch for {self.func}, argument #{i + 1}: '
                                    f'expected {arg_ty}, got {self.args[i].ty}')
            self._ty = func_ty.result
        except KeyError:
            raise TypeError(f'{self.loc}'
                            f'Unknown procedure or builtin {self.func}')

# ------------------------------------------------------------------------------
# Statements

class Stmt(Locatable):
    """Parent class of all statements"""
    def __repr__(self):
        return f'Stmt({repr(str(self))})'

class Assign(Stmt):
    """Assignment statements"""
    def __init__(self, var, value):
        if not all([isinstance(var, Variable), isinstance(value, Expr)]):
            raise ValueError(f'Invalid assignment: {var} = {value}')
        self.var = var
        self.value = value

    def type_check(self, context):
        self.var.type_check(context)
        self.value.type_check(context)
        if self.var.ty != self.value.ty:
            raise TypeError(f'{self.value.loc}'
                            f'Type mismatch in assignment: '
                            f'lhs is {self.var.ty}, rhs is {self.value.ty}')

    def __str__(self):
        tycom = f' // {self.value.ty}' if hasattr(self.value, '_ty') else ''
        return f'{self.var} = {self.value};{tycom}'

class Eval(Stmt):
    """Eval statements"""
    def __init__(self, arg):
        if not isinstance(arg, Expr):
            raise ValueError(f'Invalid eval: {arg}')
        self.arg = arg

    def type_check(self, context):
        self.arg.type_check(context)

    def __str__(self):
        tycom = f' // {self.arg.ty}' if hasattr(self.arg, '_ty') else ''
        return f'{self.arg};{tycom}'

class Block(Stmt):
    """Block statements"""
    def __init__(self, *stmts):
        for i, stmt in enumerate(stmts):
            if not isinstance(stmt, Stmt):
                raise ValueError(f'Unexpected object (position {i + 1}) in block: '
                                 f'{stmt}')
        self.body = stmts

    def type_check(self, context):
        try:
            context.enter()
            for stmt in self.body:
                stmt.type_check(context)
        finally:
            context.leave()

    def __len__(self):
        return len(self.body)

    def __str__(self):
        res = StringIO()
        res.write('{\n')
        for stmt in self.body:
            res.write(indent(str(stmt), 2))
            res.write('\n')
        res.write('}')
        return res.getvalue()

class IfElse(Stmt):
    """Conditional statements"""
    def __init__(self, cond, thn, els):
        if not all([isinstance(cond, Expr),
                    isinstance(thn, Block),
                    isinstance(els, Block)]):
            raise ValueError(f'Invalid IfElse: if ({cond}) {thn} else {els}')
        self.cond = cond
        self.thn = thn
        self.els = els

    def type_check(self, context):
        self.cond.type_check(context)
        if self.cond.ty is not BOOL:
            raise TypeError(f'{self.cond.loc}'
                            f'Type mismatch in condition: '
                            f'expected {BOOL}, got {self.cond.ty}')
        self.thn.type_check(context)
        self.els.type_check(context)

    def __str__(self):
        res = StringIO()
        res.write(f'if ({self.cond}) ')
        res.write(str(self.thn))
        if len(self.els) > 0:
            res.write(' else ')
            if len(self.els) == 1 and isinstance(self.els.body[0], IfElse):
                res.write(str(self.els.body[0]))
            else:
                res.write(str(self.els))
        return res.getvalue()

class While(Stmt):
    """While loops"""
    def __init__(self, cond, body):
        if not all([isinstance(cond, Expr),
                    isinstance(body, Block)]):
            raise ValueError(f'Invalid While: while ({cond}) {body}')
        self.cond = cond
        self.body = body

    def type_check(self, context):
        self.cond.type_check(context)
        if self.cond.ty is not BOOL:
            raise TypeError(f'{self.cond.loc}'
                            f'Type mismatch in condition: '
                            f'expected {BOOL}, got {self.cond.ty}')
        self.body.type_check(context)

    def __str__(self):
        res = StringIO()
        res.write(f'while ({self.cond}) ')
        res.write(str(self.body))
        return res.getvalue()

class Break(Stmt):
    """Structured jump -- break"""
    def type_check(self, context):
        pass

    def __str__(self):
        return 'break;'

class Continue(Stmt):
    """Structured jump -- continue"""
    def type_check(self, context):
        pass

    def __str__(self):
        return 'continue;'

class Return(Stmt):
    """Return from a function"""
    def __init__(self, arg):
        self.arg = arg

    def type_check(self, context):
        if not hasattr(context, 'return_ty'):
            raise RuntimeError(f'{self.loc}'
                               f'COMPILER BUG: return encountered outside a procedure')
        if self.arg is None:
            if context.return_ty != VOID:
                raise TypeError(f'{self.loc}'
                                f'Missing return argument: '
                                f'expecting a {self._retty} expression')
        else:
            self.arg.type_check(context)
            if self.arg.ty != context.return_ty:
                raise TypeError(f'{self.arg.loc}'
                                f'Type mismatch in return: '
                                f'expected {context.return_ty}, got {self.arg.ty}')

    def __str__(self):
        if self.arg is None: return 'return;'
        tycom = f' // {self.arg.ty}' if hasattr(self.arg, '_ty') else ''
        return f'return {self.arg};{tycom}'

class LocalVardecl(Stmt):
    """Declaration of a local variable"""
    def __init__(self, var, init, ty):
        self.var = var
        self.init = init
        self.ty = ty

    def type_check(self, context):
        self.init.type_check(context)
        if self.init.ty != self.ty:
            raise TypeError(f'{self.init.loc}'
                            f'Type mismatch in variable declaration: '
                            f'expected {self.ty}, got {self.init.ty}')
        try:
            context[self.var] = self.ty
        except KeyError:
            raise TypeError(f'{self.loc}'
                            f'Redeclaration of {self.var} in same scope')

    def __str__(self):
        return f'var {self.var} = {self.init} : {self.ty};'

# ------------------------------------------------------------------------------
# top-level declarations and programs

class Toplevel(Locatable):
    """Parent class of all toplevel declarations"""
    def __repr__(self):
        return f'Toplevel({repr(str(self))})'

class GlobalVardecl(Toplevel):
    def __init__(self, var, init, ty):
        self.var = var
        self.init = init
        self.ty = ty

    def type_check(self, context):
        assert len(context.local_defs) == 0
        self.init.type_check(context)
        if self.init.ty != self.ty:
            raise TypeError(f'{self.init.loc}'
                            f'Type mismatch in variable declaration: '
                            f'expected {self.ty}, got {self.init.ty}')
        if self.init.ty == INT and not isinstance(self.init, Number):
            raise TypeError(f'{self.init.loc}'
                            f'Initializer for {self.var} must be a number')
        if self.init.ty == BOOL and not isinstance(self.init, Boolean):
            raise TypeError(f'{self.init.loc}'
                            f'Initializer for {self.var} must be either true or false')

    def __str__(self):
        return f'var {self.var} = {self.init} : {self.ty};'

class Procdecl(Toplevel):
    def __init__(self, name, args, retty, body):
        self.name = name
        self.args = args
        self.retty = retty
        self.body = body
        self.ty = FUNC(self.retty, *(arg[1] for arg in args))

    def type_check(self, context):
        if self.retty != VOID and not always_returns(self.body, nomsg=True):
            raise TypeError(f'{self.loc}'
                            f'Missing return on some code paths for function')
        assert len(context.local_defs) == 0
        context.enter()
        for var, ty in self.args:
            try: context[var] = ty
            except KeyError:
                raise TypeError(f'{self.loc}'
                                f'Multiple arguments with same name: {var}')
        context.return_ty = self.retty
        self.body.type_check(context)
        context.leave()
        del context.return_ty

    def __str__(self):
        res = StringIO()
        args_str = ', '.join(f'{var} : {ty}' for (var, ty) in self.args)
        retty_str = '' if self.retty == VOID else f' : {self.retty}'
        res.write(f'proc {self.name}({args_str}){retty_str} ')
        res.write(str(self.body))
        return res.getvalue()

# --------------------------------------------------------------------------------
# AST syntactic analysis and simplification

def always_returns(stmt, file=None, nomsg=False):
    if isinstance(stmt, Return):
        return True
    elif isinstance(stmt, Block):
        for i, inner_stmt in enumerate(stmt.body):
            if always_returns(inner_stmt, file=file):
                # remainder of body is actually dead code, so print warning
                if i + 1 < len(stmt.body):
                    dead_stmt = stmt.body[i + 1]
                    until = '' if i + 2 == len(stmt.body) else f' (until line {stmt.body[-1]._loc.line})'
                    if not nomsg:
                        print(f'Warning: {dead_stmt.loc}'
                              f'Start of dead code (after return) on this codepath{until}.',
                              file=file)
                return True
        return False
    elif isinstance(stmt, IfElse):
        return always_returns(stmt.thn, file=file) and always_returns(stmt.els, file=file)
    else: # Assign, Eval, Break, Continue, LocalVardecl
        return False

def analyze_program(prog, file=None):
    """Perform syntactic analysis on `prog', which is a list of decls.
    Returns an initial Context object."""
    cx = Context()
    for decl in prog:
        if isinstance(decl, GlobalVardecl):
            if decl.var == 'main':
                raise SyntaxError(f'{decl.loc}'
                                  f'Illegal variable name: main')
            if decl.var in cx:
                raise SyntaxError(f'{decl.loc}'
                                  f'Multiple declarations of {decl.var} '
                                  f'in global scope')
            cx[decl.var] = decl.ty
        elif isinstance(decl, Procdecl):
            if decl.name in cx:
                raise SyntaxError(f'{decl.loc}'
                                  f'Multiple declarations of {decl.name} '
                                  f'in global scope')
            if decl.retty != VOID and not always_returns(decl.body, file=file):
                raise SyntaxError(f'{decl.loc}'
                                  f'Missing return on some code paths for function')
            if decl.name == 'main':
                if decl.args != []:
                    raise TypeError(f'{decl.loc}'
                                    f'Invalid number of parameters for main(): '
                                    f'expected 0, got {len(decl.args)}')
                if decl.retty != VOID:
                    raise TypeError(f'{decl.loc}'
                                    f'Invalid return type for main(): '
                                    f'expected {VOID}, got {decl.retty}')
            cx[decl.name] = decl.ty
        else:
            raise RuntimeError(f'{decl.loc}'
                               f'COMPILER BUG: analyze_program(): unknown toplevel declaration')
    if 'main' not in cx:
        raise SyntaxError(f'Invalid BX2 program: missing main() subroutine')
    return cx

def extrude_local_vardecls(proc):
    lvds = []
    def uniqify(var):
        x = getattr(uniqify, '__last', 0)
        setattr(uniqify, '__last', x + 1)
        return f'{var}${x}'
    calling_scope = {var : uniqify(var) for var, _ in proc.args}
    proc.args = tuple((calling_scope[var], ty) for var, ty in proc.args)
    scopes = [calling_scope]
    def emit(var, ty):
        nonlocal lvds
        init = Number(0) if ty == INT else Boolean(False)
        lvds.append(LocalVardecl(var, init, ty))
    def traverse_stmt(stmt):
        nonlocal scopes
        if isinstance(stmt, LocalVardecl):
            traverse_expr(stmt.init)
            vuniq = uniqify(stmt.var)
            scopes[-1][stmt.var] = vuniq
            emit(vuniq, stmt.ty)
            var = Variable(vuniq)
            var._ty = stmt.ty
            stmt = Assign(var, stmt.init)
        elif isinstance(stmt, Assign):
            traverse_expr(stmt.value)
        elif isinstance(stmt, Eval):
            traverse_expr(stmt.arg)
        elif isinstance(stmt, Block):
            scopes.append(dict())
            stmt.body = tuple(traverse_stmt(inner) for inner in stmt.body)
            scopes.pop()
        elif isinstance(stmt, IfElse):
            traverse_expr(stmt.cond)
            stmt.thn = traverse_stmt(stmt.thn)
            stmt.els = traverse_stmt(stmt.els)
        elif isinstance(stmt, While):
            traverse_expr(stmt.cond)
            stmt.body = traverse_stmt(stmt.body)
        elif isinstance(stmt, Break) or isinstance(stmt, Continue):
            pass
        elif isinstance(stmt, Return):
            if stmt.arg is not None:
                traverse_expr(stmt.arg)
        else:
            raise RuntimeError(f'{stmt.loc}'
                               f'COMPILER BUG: extrude_local_vardecls: '
                               f'unknown <stmt> form')
        return stmt
    def traverse_expr(expr):
        nonlocal scopes
        if isinstance(expr, Number) or isinstance(expr, Boolean):
            pass
        elif isinstance(expr, Variable):
            for scope in reversed(scopes):
                if expr.name in scope:
                    expr.name = scope[expr.name]
                    break
        elif isinstance(expr, Appl):
            for arg in expr.args: traverse_expr(arg)
    proc.body = traverse_stmt(proc.body)
    proc.body.body = tuple(lvds) + proc.body.body
