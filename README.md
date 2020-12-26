# CSE302_Project

Anya Fries, Antonin Wattel, Amine Abdeljaoued, Benjamin Montagnes

Project number 3: Dataflow Optimizations

The program `bx2tac_opt.py` takes a BX2 program, say prog.bx, and generates optimized TAC
in SSA form, `prog.tac`. 

It runs for example with `python bx2tac_opt.py ../tests/if.bx` and produces the file `../tests/if.tac` in optimised tac form.

## A quick summary of the files

In the folder `dataflow_optimisations`:
* `bx2tac_opt.py` is as described above, and calls the other files. 
* `tester.py` checks the correctness of the `.tac` files produced after the optimisations. To use: run `python tester.py`.
* `ssagen.py` builds from the solutions, has the code for SCCP and has a function `ssagen_and_optimise` which goes from TAC to SSA to TAC, and performs the big optimising loop, by calling all the other optimisations. 
* `cse.py` computes the dominator tree of a CFG and does common sub-expression elimination
* `cfg.py` is based on the solutions. Copy propagation has been added.
* `bx2tac.py`, `ply`, `bx2.py`, `bx2ast.py`, `ply_util.py`,`tac.py`, `util.py` are as from solutions.

The folder `tests` simply contains some test files (which are called by `tester.py`, but can also be called individually).

## The components of the project

### 1. Implement constant folding at the typed AST level (Ben): 
**explain**

### 2. Produce TAC and compute a compact SSA form (effectively combining lab4 and lab5, week1) (Ben)

### 3. Optimisations at the SSA level
The optimisations are repeated until no further changes are found/implemented, and then the tac is file is created. The optimisations are called in the file `bx2tac_ssa.py` from the function `ssagen` which calls `optimise` which in turn calls these optimisations. 

#### a) Global Copy and Constant Propagation (GCP) (Antonin): 
The code for this is found in `cfg.py`. Copy progagation is implemented as before (within each block). With NCE and rename elimination (lab5), this becomes global copy propagation. 

**constant propagation still to be implemented**

#### b) Dead Store Elimination (DSE) (Antonin): 
**explain**

#### c) Sparse Conditional Constant Propagation (SCCP) (Amine): 
**explain**

#### d) Dominator tree of the CFG and Common Subexpression Elimination (CSE) (Anya): 
This code is found in `cse.py`. We first create a dominator tree of the labels of the CFG. This is the simplest dominator tree to compute and is all that is necessary. This follows 'Algorithm: Compute Dominators' in the project handout. 

From this we compute the available expressions for each instruction. This is stored as a dictionary: `{ str(instr) : set of str(available expressions/instructions) }`. In order for this to be slightly more efficient we have made the following observations: 
* We only need the instructions with opcodes ('add', 'sub', 'mul', 'div', 'mod', 'neg', 'and', 'or', 'xor', 'not', 'shl', 'shr') when performing CSE so we only store the instructions in the available expressions that have these opcodes. This means less memory is used and we have faster lookup later.
* In a single block, if we have two instructions `instr1` and `instr2` that follow each other directly, then the available expressions for `instr2` will be those of `instr1` and `instr1` itselfs. This avoids a lot of recomputation.

Once the available expressions are computed it is a simple task to implement CSE. We iterate over the instructions. If they have an appropriate opcode, using string comparisons, we check if the expression has already been computed in a dominating instruction. If so, we replace the current expression by a copy of the previous destination.


