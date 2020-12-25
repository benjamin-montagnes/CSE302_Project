# CSE302_Project

Anya Fries, Antonin Wattel, Amine Abdeljaoued, Benjamin Montagnes

Project number 3: Dataflow Optimizations

The program `bx2tac_opt.py` takes a BX2 program, say prog.bx, and generates optimized TAC
in SSA form, `prog.tac`. 

It runs for example with `python bx2tac_opt.py ../tests/if.bx` and produces the file `../tests/if.tac` in optimised tac form.

## The components of the project
### 1. Implement constant folding at the typed AST level (Ben): 
**explain**
### 2. Produce TAC and compute a compact SSA form (effectively combining lab4 and lab5, week1) (Ben)
### 3. Optimisations at the SSA level
The optimisations are repeated until no further changes are found/implemented, and then the tac is file is created. The optimisations are called in the file `bx2tac_ssa.py` from the function `ssagen` which calls `optimise` which in turn calls these optimisations. 
#### a) Global Copy Propagation (GCP) (Antonin): 
**explain**
#### b) Dead Store Elimination (DSE) (Antonin): 
**explain**
#### c) Sparse Conditional Constant Propagation (SCCP) (Amine): 
**explain**
#### d) Dominator tree of the CFG and Common Subexpression Elimination (CSE) (Anya): 
We first create a dominator tree of the labels of the CFG. This is the simplest dominator tree to compute and is all that is necessary. This follows 'Algorithm: Compute Dominators' in the project handout. 

From this we compute the available expressions for each instruction. This is stored as a dictionary: `{ str(instr) : set of str(available expressions/instructions) }`. In order for this to be slightly more efficient we have made the following observations: 
* We only need the instructions with opcodes ('add', 'sub', 'mul', 'div', 'mod', 'neg', 'and', 'or', 'xor', 'not', 'shl', 'shr') when performing CSE so we only store the instructions in the available expressions that have these opcodes. This means less memory is used and we have faster lookup later.
* In a single block, if we have two instructions `instr1` and `instr2` that follow each other directly, then the available expressions for `instr2` will be those of `instr1` and `instr1` itselfs. This avoids a lot of recomputation.



