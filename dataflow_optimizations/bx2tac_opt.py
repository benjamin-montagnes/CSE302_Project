#!/usr/bin/env python3

import bx2ast as ast
import tac
from tac import Instr, Gvar, Proc, execute, load_tac
from bx2tac_ssa import process,ssagen
from io import StringIO
from cfg import infer, linearize
import copy
import argparse

    # tac_file to ssa form
    # run optimisations
    #   not sure of whole order but I would put CSE then DCE straight after it 
    #   (they work well together)

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
    # print('af',tac_prog)
    if args.interpret:
        # to do: this isn't working
        # Error: 'no value for args in function call'
        execute(tac_prog, '@main', (), show_proc=(args.verbosity>0), show_instr=(args.verbosity>1), only_decimal=(args.verbosity<=2))
    else:
        tac_file = bx_src.with_suffix('.tac')
        with open(tac_file, 'w') as f:
            for tlv in tac_prog:
                print(tlv, file=f)
        print(f'{bx_src} -> {tac_file} done')