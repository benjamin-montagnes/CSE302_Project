import cfg
import copy 
from tac import Instr

expr_opcodes = {
    'add', 'sub', 'mul', 'div',
    'mod', 'neg', 'and', 'or',
    'xor', 'not', 'shl', 'shr',
}

# ------------------------------------------------------------------------------

def dominators(cfg):
    """ Compute the dominator tree of the labels in a CFG """
    # initialise dominators
    labels = set([lab for lab in cfg._blockmap.keys()])
    e = cfg.lab_entry
    dom = {e:{e}}
    for lab in labels-{e}: dom[lab] = labels
    # iteration: find the actual dominators
    different = True
    while different:
        dom1 = copy.deepcopy(dom)
        for lab in labels.difference({e}):
            prev_doms = labels
            for prev in cfg.predecessors(lab):
                prev_doms = prev_doms.intersection(dom1[prev])
            dom[lab] = {lab}.union(prev_doms)
        if dom1 == dom: different = False
    return dom

def available_exprs(cfg, dom):
    """ Given a dominator tree of labels in a CFG and the CFG,
        compute the available expressions for each instruction.
        Note: 
        1. only instructions with opcodes in expr_opcodes (above)
           are considered as these are the only ones necessary 
           for CSE 
        2. this is stored as a dictionary:
           { str(instr) : { str(available expressions/instructions) } } 
    """
    available_exprs = dict()
    for lab,bl in cfg._blockmap.items():
        instrs = list(bl.instrs())
        for i in range(len(instrs)):
            avails = set()
            if i == 0:
            # add instructions from dominating blocks
                for dom_lab in dom[lab] - {lab}:    
                    for dom_instr in cfg._blockmap[dom_lab].instrs():
                        if dom_instr.opcode in expr_opcodes: avails |= {str(dom_instr)}
            else:
            # add instructions from dominating blocks and prev instructions in same block
                # as we are in the same block the dominating instrs of the previous instr
                # still dominate the current one (no need to recompute)
                # we just add the previous instruction
                if instrs[i-1].opcode in expr_opcodes: 
                    avails = available_exprs[str(instrs[i-1])].union({str(instrs[i-1])})
                else: avails = available_exprs[str(instrs[i-1])]
            available_exprs[str(instrs[i])] = avails
    # print("Avail exprs:")
    # for x,y in available_exprs.items():
    #     print(x,":")
    #     for z in y:
    #         print("\t",z)
    # print()
    return available_exprs

# ------------------------------------------------------------------------------
            
def run_cse(cfg):
    change_cfg = 0
    dom = dominators(cfg)
    avail_exprs = available_exprs(cfg, dom)
    for label,block in cfg._blockmap.items():
        instrs = list(block.instrs())
        for i in range(len(instrs)):
            # can only do CSE on these types of expressions
            if instrs[i].opcode in expr_opcodes:
                ending = str(instrs[i]).split("=")[1]
                for avail in avail_exprs[str(instrs[i])]:
                    a_start, a_end = str(avail).split("=")[0],str(avail).split("=")[1]
                    if ending == a_end:
                        new_instr = Instr(instrs[i].dest,'copy',a_start.strip(),None)
                        cfg[label].body[i] = new_instr
                        change_cfg = 1  # we have altered the CFG

    return change_cfg 