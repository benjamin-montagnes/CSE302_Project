#!/usr/bin/env python3

import bx2ast as ast
from tac import load_tac
from bx2tac import process
import argparse
from ssagen import ssagen_and_optimise

if __name__ == '__main__':
    import sys, argparse, time, random, bx2
    from pathlib import Path
    ap = argparse.ArgumentParser(description='BX2 to TAC compiler')
    ap.add_argument('bx_src', metavar='FILE', nargs=1, type=str, help='BX2 program')
    ap.add_argument('-v', dest='verbosity', default=0, action='count', help='increase verbosity')
    args = ap.parse_args()
    bx_src = Path(args.bx_src[0])
    if not bx_src.suffix == '.bx':
        print(f'File name {bx_src} does not end in ".bx"')
        exit(1)

    ## BX2 -> TAC
    bx2.lexer.load_source(bx_src)
    bx2_prog = bx2.parser.parse(lexer=bx2.lexer)
    cx = ast.analyze_program(bx2_prog)
    for tlv in bx2_prog: tlv.type_check(cx)
    tac_prog_bef = process(bx2_prog)

    ## OPTIMISE TAC
    tac_prog = ssagen_and_optimise(tac_prog_bef)
    
    ## WRITE TO A FILE
    tac_file = bx_src.with_suffix('.tac')
    with open(tac_file, 'w') as f:
        for tlv in tac_prog:
            print(tlv, file=f)
    print(f'{bx_src} -> {tac_file} done')