#!/usr/bin/env python3

"""
Three Address Code (TAC) intermediate representation
"""

from ply import lex, yacc
from ply_util import *
from io import StringIO

# ------------------------------------------------------------------------------

opcode_kinds = {
    'nop': 'NNN',
    'jmp': 'NLN',
    'jz': 'NVL', 'jnz': 'NVL', 'jl': 'NVL', 'jle': 'NVL',
    'add': 'VVV', 'sub': 'VVV', 'mul': 'VVV', 'div': 'VVV',
    'mod': 'VVV', 'neg': 'VVN', 'and': 'VVV', 'or': 'VVV',
    'xor': 'VVV', 'not': 'VVN', 'shl': 'VVV', 'shr': 'VVV',
    'const': 'VIN', 'copy': 'VVN',
    'label': 'NLN',
    'param': 'NIV', 'call': 'VGI', 'ret': 'NVN',
    'phi': 'VFN',
}
opcodes = frozenset(opcode_kinds.keys())

class Instr:
    __slots__ = ('iid', 'dest', 'opcode', 'arg1', 'arg2')
    def __init__(self, dest, opcode, arg1, arg2):
        """Create a new TAC instruction with given `opcode' (must be non-None).
        The other three arguments, `dest', 'arg1', and 'arg2' depend on what
        the opcode is.

        Raises ValueError if attempting to create an invalid Instr."""
        self.dest = dest
        self.opcode = opcode
        self.arg1 = arg1
        self.arg2 = arg2
        self._check()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    @classmethod
    def _isvar(cls, thing):
        return isinstance(thing, str) and \
               len(thing) > 0 and \
               (thing[0] == '%' or thing[0] == '@')

    @classmethod
    def _isint(cls, thing):
        return isinstance(thing, int)

    @classmethod
    def _islabel(cls, thing):
        return isinstance(thing, str) and \
               thing.startswith('.L')

    @classmethod
    def _isglobal(cls, thing):
        return isinstance(thing, str) and \
               not thing.startswith('.L')

    @classmethod
    def _isphiargs(cls, thing):
        return (isinstance(thing, dict) and \
                all(cls._isvar(x) for x in thing.values()))

    @classmethod
    def _isvalid(cls, thing, k):
        if k == 'N': return thing == None
        if k == 'I': return cls._isint(thing)
        if k == 'V': return cls._isvar(thing)
        if k == 'L': return cls._islabel(thing)
        if k == 'G': return cls._isglobal(thing)
        if k == 'F': return cls._isphiargs(thing)
        return ValueError(f'Unknown argument kind: {k}')

    def _check(self):
        """Perform a well-formedness check on this instruction"""
        if self.opcode not in opcodes:
            raise ValueError(f'bad tac.Instr opcode: {self.opcode}')
        kind = opcode_kinds[self.opcode]
        if not self._isvalid(self.dest, kind[0]):
            raise ValueError(f'bad tac.Instr/{self.opcode} destination: {self.dest}')
        if not self._isvalid(self.arg1, kind[1]):
            raise ValueError(f'bad tac.Instr/{self.opcode} arg1: {self.arg1}')
        if not self._isvalid(self.arg2, kind[2]):
            raise ValueError(f'bad tac.Instr/{self.opcode} arg2: {self.arg2}')

    def __repr__(self):
        result = StringIO()
        if self.dest != None:
            result.write(f'(set {self.dest} ')
        result.write('('); result.write(self.opcode)
        if self.arg1 != None:
            result.write(f' {self.arg1}')
        if self.arg2 != None:
            result.write(f' {self.arg2}')
        result.write(')')
        if self.dest != None:
            result.write(')')
        return result.getvalue()

    def __str__(self):
        result = StringIO()
        if self.opcode == 'label':
            result.write(f'{self.arg1}:')
        elif self.opcode == 'phi':
            result.write(f'  {self.dest} = phi(')
            result.write(', '.join(f'{lab}:{tmp}' for lab, tmp in self.arg1.items()))
            result.write(');')
        else:
            result.write('  ')
            if self.dest != None:
                result.write(f'{self.dest} = ')
            result.write(f'{self.opcode}')
            if self.arg1 != None:
                result.write(f' {self.arg1}')
                if self.arg2 != None:
                    result.write(f', {self.arg2}')
            result.write(';')
        return result.getvalue()

    @classmethod
    def _istemp(cls, thing):
        return (isinstance(thing, str) and \
                len(thing) > 1 and \
                thing[0] == '%' and \
                thing[1] != '_')

    def defs(self):
        """Returns an iterator over the temporaries defined in this instruction"""
        if self._istemp(self.dest): yield self.dest

    def uses(self):
        """Returns an iterator over the temporaries used in this instruction.
        Each item of the terator is either a temporary by itself or a
        2-tuple of the form (label, temporary) that corresponds to an
        argument of a phi-function."""
        if self._istemp(self.arg1): yield self.arg1
        if self._istemp(self.arg2): yield self.arg2
        if self.opcode == 'phi':
            for l, t in self.arg1.items():
                if self._istemp(t): yield (l, t)

    def rewrite_temps(self, rew):
        """Apply `rew' to rewrite the temps in this instruction. If `rew' is a
        dictionary, then it will be used to look up the new mapping of
        a temporary (if one exists), defaulting to no change if there
        is no mapping. If `rew' is a function, then every temporary t will
        be mapped to `rew(t)'."""
        if isinstance(rew, dict): lookup = lambda t: rew.get(t, t)
        else: lookup = rew
        if self._istemp(self.dest): self.dest = lookup(self.dest)
        if self._istemp(self.arg1): self.arg1 = lookup(self.arg1)
        if self._istemp(self.arg2): self.arg2 = lookup(self.arg2)
        if self.opcode == 'phi':
            for l, t in self.arg1.items(): self.arg1[l] = lookup(t)

class Proc:
    def __init__(self, name, t_args, body):
        self.name = name
        self.body = body or []
        self.t_args = tuple(t_args)

    def __str__(self):
        result = StringIO()
        result.write(f'proc {self.name}({", ".join(self.t_args)}):\n')
        for instr in self.body:
            print(instr, file=result)
        return result.getvalue()

class Gvar:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return f'var {self.name} = {self.value};\n'

# ------------------------------------------------------------------------------

tokens = (
    'TEMP',                     # *local* identifier
    'NUM64',                    # immediate integer
    'OPCODE',                   # instruction opcode
    'LABEL',                    # label
    'GLABEL',                   # global label
    'EQ', 'COMMA', 'SEMICOLON', 'COLON', 'LPAREN', 'RPAREN',
    'VAR', 'PROC', 'PHI',
)

dummy_temp = '%_'

def __create_lexer():
    """Create and return a lexer for a TAC"""

    t_ignore = ' \t\f\v\r'

    t_TEMP = r'%(_|0|[1-9][0-9]*|[A-Za-z][A-Za-z0-9_]*)(\.[0-9]+)?'
    t_EQ = r'='
    t_COMMA = r','
    t_SEMICOLON = r';'
    t_COLON = r':'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'

    def t_newline(t):
        r'\n|//[^\n]*\n?'
        t.lexer.lineno += 1

    t_LABEL = r'\.L(?:0|[1-9][0-9]*|[A-Za-z0-9_]+)'
    t_GLABEL = r'@[A-Za-z_][A-Za-z0-9_]*'

    def t_OPCODE(t):
        r'[A-Za-z_][A-Za-z0-9_]*'
        if t.value == 'var': t.type = 'VAR'
        elif t.value == 'proc': t.type = 'PROC'
        elif t.value == 'phi': t.type = 'PHI'
        elif t.value not in opcodes:
            print_at(t, f'Error: unknown opcode {t.value}')
            raise SyntaxError(f'badop')
        return t

    def t_NUM64(t):
        r'0|-?[1-9][0-9]*'
        t.value = int(t.value)
        if t.value & 0xffffffffffffffff != t.value:
            print_at(t, f'Error: numerical literal {t.value} not in [{-1<<63}, {1<<63})')
            raise SyntaxError('immint')
        return t

    from sys import stderr
    def t_error(t):
        print_at(t, f'Warning: skipping illegal character: {t.value[0]}')
        t.lexer.skip(1)

    return extend_lexer(lex.lex())

# ------------------------------------------------------------------------------

def __create_parser():
    def p_program(p):
        '''program : program gvar
                   | program proc
                   | '''
        if len(p) == 1:
            p[0] = []
        else:
            p[0] = p[1]
            p[0].append(p[2])

    def p_gvar(p):
        '''gvar : VAR GLABEL EQ NUM64 SEMICOLON'''
        p[0] = Gvar(p[2], p[4])

    def p_proc(p):
        '''proc : PROC GLABEL LPAREN argtemps RPAREN COLON instrs'''
        p[0] = Proc(p[2], p[4], p[7])

    def p_argtemps(p):
        '''argtemps : argtemps1
                    | '''
        p[0] = p[1] if len(p) == 2 else []

    def p_argtemps1(p):
        '''argtemps1 : TEMP
                     | argtemps1 COMMA TEMP'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]
            p[0].append(p[3])

    def p_instrs(p):
        '''instrs : instrs instr
                  | '''
        if len(p) == 1:
            p[0] = []
        else:
            p[0] = p[1]
            p[0].append(p[2])

    def p_instr(p):
        '''instr : lhs OPCODE args SEMICOLON'''
        lhs = p[1]
        opcode = p[2]
        arg1, arg2 = p[3]
        p[0] = Instr(lhs, opcode, arg1, arg2)

    def p_phidef(p):
        '''instr : TEMP EQ PHI LPAREN phiargs RPAREN SEMICOLON'''
        phiargs = {l: t for l, t in p[5]}
        p[0] = Instr(p[1], 'phi', phiargs, None)

    def p_phiargs(p):
        '''phiargs : phiargs1
                   | '''
        p[0] = [] if len(p) == 1 else p[1]

    def p_phiargs1(p):
        '''phiargs1 : phiargs1 COMMA phiarg
                    | phiarg'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]
            p[1].append(p[3])

    def p_phiarg(p):
        '''phiarg : LABEL COLON TEMP
                  | GLABEL COLON TEMP'''
        p[0] = (p[1], p[3])

    def p_label(p):
        '''instr : LABEL COLON'''
        p[0] = Instr(None, 'label', p[1], None)

    def p_lhs(p):
        '''lhs : TEMP EQ
               | GLABEL EQ
               | '''
        p[0] = None if len(p) == 1 else p[1]

    def p_args(p):
        '''args : arg COMMA arg
                | arg
                | '''
        arg1 = None if len(p) < 2 else p[1]
        arg2 = None if len(p) != 4 else p[3]
        p[0] = (arg1, arg2)

    def p_arg(p):
        '''arg : TEMP
               | NUM64
               | LABEL
               | GLABEL'''
        p[0] = p[1]

    def p_error(p):
        if p:
            p.lexer.lexpos -= len(str(p.value))
            print_at(p, f'Error: syntax error at token {p.type}')
        raise RuntimeError('parsing')

    return yacc.yacc(start='program')

# ------------------------------------------------------------------------------

word_bytes = 8
word_bits = 8 * word_bytes
sign_mask = 1 << (word_bits - 1)
full_mask = (1 << word_bits) - 1
def untwoc(x):
    """Convert a 64-bit word in two's complement representation
    to a Python int"""
    return x - full_mask - 1 if x & sign_mask else x
def twoc(x):
    """Convert a Python int in range to a 64-bit word in two's
    complement representation"""
    return x & full_mask

binops = {
    'add' : (lambda u, v: twoc(untwoc(u) + untwoc(v))),
    'sub' : (lambda u, v: twoc(untwoc(u) - untwoc(v))),
    'mul' : (lambda u, v: twoc(untwoc(u) * untwoc(v))),
    'div' : (lambda u, v: twoc(int(untwoc(u) / untwoc(v)))),
    'mod' : (lambda u, v: twoc(untwoc(u) - untwoc(v) * int(untwoc(u) / untwoc(v)))),
    'and' : (lambda u, v: twoc(untwoc(u) & untwoc(v))),
    'or'  : (lambda u, v: twoc(untwoc(u) | untwoc(v))),
    'xor' : (lambda u, v: twoc(untwoc(u) ^ untwoc(v))),
    'shl' : (lambda u, v: twoc(untwoc(u) << untwoc(v))),
    'shr' : (lambda u, v: twoc(untwoc(u) >> untwoc(v))),
}
unops = {
    'neg' : (lambda u: twoc(-untwoc(u))),
    'not' : (lambda u: twoc(~untwoc(u))),
}
jumps = {
    'jz':  (lambda k: k == 0),
    'jnz': (lambda k: k != 0),
    'jl':  (lambda k: untwoc(k) < 0),
    'jle': (lambda k: untwoc(k) <= 0),
}

class TempMap(dict):
    """Mapping temporaries to values"""

    def __init__(self, gvars):
        super().__init__()
        self.gvars = gvars

    def _valid_temp(self, tmp):
        return isinstance(tmp, str) and \
               (tmp.startswith('%') or tmp.startswith('@'))

    def _valid_value(self, val):
        return isinstance(val, int) and \
               0 <= val <= 0xffffffffffffffff

    def __getitem__(self, tmp):
        if tmp == dummy_temp:
            # raise ValueError(f'Cannot read value of {dummy_temp}')
            return 0
        if tmp.startswith('@'):
            return self.gvars[tmp].value
        return super().__getitem__(tmp)

    def copy(self):
        tm = TempMap(self.gvars.copy())
        for k, v in super().items():
            tm[k] = v
        return tm

    def __setitem__(self, tmp, val):
        assert self._valid_temp(tmp)
        if tmp != dummy_temp:
            if not self._valid_value(val):
                raise RuntimeError(f'Illegal value: {val}: '
                                   f'{-0x8000000000000000 <= val} '
                                   f'{val < 0x8000000000000000}')
            if tmp.startswith('@'):
                self.gvars[tmp].value = val
            else:
                super().__setitem__(tmp, val)

def execute(tac_prog, proc_name, args, **kwargs):
    gvars, procs = tac_prog
    show_proc = kwargs.get('show_proc', False)
    show_instr = kwargs.get('show_instr', False)
    only_decimal = kwargs.get('only_decimal', True)
    depth = kwargs.get('depth', 0)
    indent = '  ' * depth

    values = TempMap(gvars)
    proc = procs[proc_name]

    for i in range(len(proc.t_args)):
        values[proc.t_args[i]] = args[i]

    oldvalues = values.copy()

    proc_desc = f'{proc_name}({",".join(k + "=" + str(v) for k, v in values.items())})'
    if show_proc: print(f'// {indent}entering {proc_desc}')

    labels = dict()
    for i, instr in enumerate(proc.body):
        if instr.opcode != 'label': continue
        if instr.arg1 in labels:
            raise RuntimeError(f'Reused label {instr.arg1}')
        ni = i + 1 # next instruction index
        while ni < len(proc.body):
            if proc.body[ni].opcode != 'label': break
            ni += 1
        labels[instr.arg1] = ni

    lab_prev, lab_cur = None, proc_name
    pc = 0
    params = []
    while pc in range(len(proc.body)):
        instr = proc.body[pc]
        pc += 1

        if show_instr: print(f'// {indent}[{pc+1: 4d}] {instr}')
        if instr.opcode == 'nop':
            pass
        elif instr.opcode == 'label':
            lab_prev, lab_cur = lab_cur, instr.arg1
        elif instr.opcode == 'phi':
            for lab, tmp in instr.arg1.items():
                if lab == lab_prev:
                    values[instr.dest] = oldvalues[tmp]
                    break
            else:
                raise RuntimeError(f'cannot resolve phi: '
                                   f'came from {lab_prev}, '
                                   f'can only handle [{",".join(instr.arg1.keys())}]')
        elif instr.opcode == 'jmp':
            if instr.arg1 not in labels:
                raise RuntimeError(f'Unknown jump destination {instr.arg1}')
            lab_prev, lab_cur = lab_cur, instr.arg1
            oldvalues = values.copy()
            pc = labels[lab_cur]
        elif instr.opcode in jumps:
            k = values[instr.arg1]
            if instr.arg2 not in labels:
                raise RuntimeError(f'Unknown jump destination {instr.arg2}')
            if jumps[instr.opcode](k):
                lab_prev, lab_cur = lab_cur, instr.arg2
                oldvalues = values.copy()
                pc = labels[lab_cur]
        elif instr.opcode == 'const':
            if not isinstance(instr.arg1, int):
                print(f'Missing or bad argument: {instr.arg1}')
                raise RuntimeError
            values[instr.dest] = twoc(instr.arg1)
        elif instr.opcode == 'copy':
            values[instr.dest] = values[instr.arg1]
        elif instr.opcode == 'param':
            if not isinstance(instr.arg1, int) or instr.arg1 < 1:
                print(f'Bad argument to param: '
                      f'expecting int >= 1, got {instr.arg1}')
            # make params big enough to hold instr.arg1 items
            for _ in range(instr.arg1 - len(params)):
                params.append(None)
            params[instr.arg1 - 1] = values[instr.arg2]
        elif instr.opcode == 'call':
            if instr.arg1.startswith('@__bx_print'):
                if len(params) != 1:
                    raise RuntimeError(f'Bad number of arguments to print(): '
                                       f'expected 1, got {len(params)}')
                if instr.arg1 == '@__bx_print_int':
                    u = params[0]
                    if only_decimal: print(str(untwoc(u)))
                    else: print(f'{untwoc(u): 20d}  0x{u:016x}  0b{u:064b}')
                elif instr.arg1 == '@__bx_print_bool':
                    print('false' if params[0] == 0 else 'true')
                else:
                    raise RuntimeError(f'Unknown print() specialization: {instr.arg1}')
            else:
                if len(params) < instr.arg2:
                    raise RuntimeError(f'Bad number of arguments to {instr.arg1}(): '
                                       f'expected {instr.arg2}, got {len(params)}')
                kwargs['depth'] = depth + 1
                values[instr.dest] = execute(tac_prog, instr.arg1, params, **kwargs)
            params = []
        elif instr.opcode == 'ret':
            retval = None if instr.arg1 == dummy_temp else values[instr.arg1]
            if show_proc:
                print(f'// {indent}{proc_desc} --> {retval}')
            return retval
        elif instr.opcode in binops:
            u = values[instr.arg1]
            v = values[instr.arg2]
            values[instr.dest] = binops[instr.opcode](u, v)
        elif instr.opcode in unops:
            u = values[instr.arg1]
            if instr.arg2 != None:
                print(f'Unary operator {self.opcode} has two arguments!')
                raise RuntimeError
            values[instr.dest] = unops[instr.opcode](u)
        else:
            print(f'Unknown opcode {instr.opcode}')
            raise RuntimeError
    print(f'// {indent}{proc_desc} --> NONE')

# --------------------------------------------------------------------------------

lexer = None
parser = None

if '__file__' in globals():
    lexer  = __create_lexer()
    parser = __create_parser()

def load_tac(tac_file):
    """Load the TAC instructions from the given `tac_file'"""
    lexer.load_source(tac_file)
    return parser.parse(lexer=lexer)

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='TAC library, parser, and interpreter')
    ap.add_argument('files', metavar='FILE', type=str, nargs='*', help='A TAC file')
    ap.add_argument('-v', dest='verbosity', default=0, action='count',
                    help='increase verbosity')
    ap.add_argument('--no-exec', dest='execute', action='store_false',
                    default=True, help='Do not run the interpreter')
    args = ap.parse_args()
    kwargs = dict(show_proc = args.verbosity > 0,
                  show_instr = args.verbosity > 1,
                  only_decimal = args.verbosity <= 2)
    for srcfile in args.files:
        # lexer.load_source(srcfile)
        # print(*lexer, sep='\n')
        gvars, procs = dict(), dict()
        seen = set()
        for tlv in load_tac(srcfile):
            if tlv.name in seen:
                raise RuntimeError(f'Repeated definition of {tlv.name}')
            seen.add(tlv.name)
            if isinstance(tlv, Proc): procs[tlv.name] = tlv
            else: gvars[tlv.name] = tlv
        tac_prog = (gvars, procs)
        if args.execute:
            execute(tac_prog, '@main', (), **kwargs)
        elif args.verbosity > 0:
            for gvar in gvars.values(): print(gvar)
            for proc in procs.values(): print(proc)
