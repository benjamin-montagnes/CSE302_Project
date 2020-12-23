from tac import load_tac, Proc, Gvar, Instr
from cfg import infer, linearize
import argparse
import copy

# ----------------------------------------------------------------------------------

def use_set(instr):
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

def def_set(instr):
    if instr.opcode in ['neg', 'not', 'copy','add','phi',
                        'sub','mul','div','mod','and','or',
                        'xor','shl','shr','call','const']:
        return set([instr.dest])
    else: return set()
     # param, label, jmp, nop, 'jz', 'jnz', 'jl', 'jle','ret', -> nothing defined

def live_in(cfg):
    livein = dict()
    for instr in cfg.instrs():
        livein[instr] = use_set(instr)
    # run the livein set update loop until there are no more changes
    dirty = True
    while dirty:
        dirty = False
        for (i1, i2) in cfg.instr_pairs():
            t_live = livein[i2].difference(def_set(i1))
            if not t_live.issubset(livein[i1]):
                dirty = True # there was a change, so run it again
                livein[i1] = livein[i1].union(t_live)
    return livein

def live_out(cfg,livein):
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

# ----------------------------------------------------------------------------------

def crude_ssa(cfg):
    livein = live_in(cfg)
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
            for var in use_set(instr):  # for each var used in the instruction
                for instr1 in instrs[:i]:
                    if def_set(instr1) != set():
                        defed = def_set(instr1).pop() # we only ever have one variable defined anyways
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
                if def_set(prev_instr): 
                    defed = def_set(prev_instr).pop() # we only ever have one variable defined anyways
                    if defed.split('.')[0] == curr_instr.dest.split('.')[0]:
                        curr_instr.arg1[L1] = defed
    # in entry block we also get the args of the phi func from the args of the proc
    for instr in cfg._blockmap[cfg.lab_entry].instrs():
        if not instr.opcode == 'phi': continue
        instr.arg1[cfg.proc_name] = instr.dest.split('.')[0]
    return cfg

def nce(cfg):
    '''Null choice elimination'''
    change = 0
    for block in cfg._blockmap.values():
        for instr in block.body:
            if instr.opcode == "phi":
                args = set(instr.arg1.values())
                if len(args)==1 and args.pop()==instr.dest:
                    block.body.remove(instr)
                    change = 1

def rename_elim(cfg):
    change = 0
    for block in cfg._blockmap.values():
        for instr in block.body:
            if instr.opcode == "phi":
                args = set(instr.arg1.values())
                old = instr.dest
                if old in args:
                    args.remove(old)
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

# ----------------------------------------------------------------------------------

# 1. Reads in a TAC file, computes its CFG, and then performs liveness analysis
# 2. Uses the live sets information to do crude SSA generation
# 3. Performs all the SSA minimization steps until no further minimizations can be done
# 4. Then linearizes the SSA CFG back to TAC, and then outputs it to a file with the extension .ssa.tac.
#    That is, an input file prog.tac should be converted to prog.ssa.tac.

def ssagen(tacfile,ln):
    """ TAC file -> CFG -> liveness analysis -> crude SSA generation 
        --> minimisation steps -> linearises back to TAC 
        outputs a file with extension .ssa.tac """
    # list of gvars and procs
    loaded_tac = load_tac(tacfile)
    procs = [tac_proc for tac_proc in loaded_tac if isinstance(tac_proc,Proc)]
    # list: cfg for each proc
    cfgs = [infer(proc) for proc in procs]
    # generate crude ssa
    ssa = [crude_ssa(cfg) for cfg in cfgs]
    # minimisation steps
    for cfg in ssa:
        start=True
        while start:
            start = False
            if nce(cfg)==1: start= True
            if rename_elim(cfg)==1: start=True
    if ln==True:
        # linearise back to TAC
        for i,proc in enumerate(procs):
            linearize(proc,ssa[i])
        output_tac = [gvar for gvar in loaded_tac if isinstance(gvar,Gvar)]+procs
        # write this TAC to a file
        tac_filename = tacfile.split(".")[0]+".ssa.tac"
        with open(tac_filename,'w') as f:
            for instr in output_tac:
                f.write(str(instr)+'\n') 
    return ssa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",type=str, help="")
    args = parser.parse_args()
    ssagen(args.filename,True)

