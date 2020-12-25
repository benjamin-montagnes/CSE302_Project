import cfg
import copy 

def run_cse(cfg):
    dom = dominators(cfg)
    return 0 # to do: change - only returns 0 when old and new cfgs are different

# ------------------------------------------------------------------------------

# Algorithm: compute dominators

def dominators(cfg):
    ### first we make a dominator graph of labels
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
    print("\nlabel doms")
    for x,y in dom.items():
        print(x,"\t",y)
    ### now we turn this label dominator -> instr dominator
    # the dominating instructions are the instructions 
    # - in the dominating blocks
    # - in the same block before the instruction

# ------------------------------------------------------------------------------

# CSE: common subexpression elimination


            
