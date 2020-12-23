#!/usr/bin/env python3

"""
Some general utilities for the BX compilers
"""

def indent(s, n):
    """Indent all the lines of `s' by `n' spaces"""
    ind = ' '*n
    return '\n'.join(ind + line for line in s.split('\n'))

import math

class Brak:
    """Bracketing expressions"""

    class _Expr:
        def __init__(self, kind, prec, **kwargs):
            self.kind = kind
            self.prec = prec
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __str__(self):
            def _bracket_if(s, *conds):
                return f'({s})' if any(conds) else s
            if self.kind == 'atom':
                return self.value
            elif self.kind == 'bracket':
                return f'{self.left}{self.contents}{self.right}'
            elif self.kind == 'apply':
                if self.fixity == 'prefix':
                    arg_str = _bracket_if(str(self.args[0]),
                                          self.prec > self.args[0].prec,
                                          self.prec == self.args[0].prec and \
                                          self.args[0].kind != 'prefix')
                    return f'{self.op}{arg_str}'
                elif self.fixity == 'postfix':
                    arg_str = _bracket_if(str(self.args[0]),
                                          self.prec > self.args[0].prec,
                                          self.prec == self.args[0].prec and \
                                          self.args[0].kind != 'postfix')
                    return f'{arg_str}{self.op}'
                elif self.fixity in ('infix', 'infixl', 'infixr'):
                    larg_str = _bracket_if(str(self.args[0]),
                                           self.prec > self.args[0].prec,
                                           self.prec == self.args[0].prec and \
                                           (self.fixity != 'infixl' or \
                                            (self.args[0].kind == 'apply' and
                                             self.args[0].fixity != 'infixl')))
                    rarg_str = _bracket_if(str(self.args[1]),
                                           self.prec > self.args[1].prec,
                                           self.prec == self.args[1].prec and \
                                           (self.fixity != 'infixr' or \
                                            (self.args[1].kind == 'apply' and
                                             self.args[1].fixity != 'infixr')))
                    return f'{larg_str}{self.op}{rarg_str}'
                else:
                    raise RuntimeError(f'Cannot handle apply/{self.fixity}')
            else:
                raise RuntimeError(f'Cannot handle {self.kind}')

    @classmethod
    def atom(cls, s):
        return cls._Expr('atom', math.inf, value=s)

    @classmethod
    def bracket(cls, expr, left='(', right=')'):
        return cls._Expr('bracket', math.inf, contents=expr, left=left, right=right)

    @classmethod
    def appl(cls, op, prec, fixity, *args):
        return cls._Expr('apply', prec, op=op, fixity=fixity, args=args)
