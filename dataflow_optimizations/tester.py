import os
import subprocess

def assemble_and_link(asm_file, runtime_file, debug=False):
    """Run gcc on `asm_file' and `runtime_file' to create a suitable
    executable file. If `debug' is True, then also send the `­g' option
    to gcc that will store debugging information useful for gdb."""
    cmd = ["gcc"]
    if debug: cmd.append("-g")
    assert asm_file.endswith('.s')
    exe_file = asm_file[:-1] + 'exe'
    cmd.extend(["-o", exe_file, asm_file, runtime_file])
    result = subprocess.run(cmd)
    return result.returncode # 0 on success, non­zero on failure

def main():
        print("* Testing files")
        print("* Please note that no output after 'Testing: filename' = success")
        # Files to run tests on
        tests = ["add","bitop","bool_simple","copy","div",
                 "if","jumps","minmax","mod","mul_for_cse","neg","relop","sub","while"] 
            # fizzbuzz excluded bc it has a second proc
            # bool_complex excluded bc it takes long
        for file in tests:
            print("Testing: ",file)
            cmd0 = "python bx2tac_opt.py ../tests/"+file+".bx"
            cmd1 = "python tac.py ../tests/"+file+".tac > a.txt"
            cmd2 = "diff a.txt ../tests/"+file+".expected"
            cmd3 = "rm a.txt"
            os.system(cmd0)
            os.system(cmd1)
            os.system(cmd2)
            os.system(cmd3)

if __name__ == '__main__':
        main()