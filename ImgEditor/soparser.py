import os
import subprocess
import pandas as pd
import logging
import cxxfilt
import re
from typing import List, Set

LOGGER = logging.getLogger(__name__)

def nm(so_fname:str)->pd.DataFrame:
    cmd_str = f'nm {so_fname}'
    result = subprocess.run(["nm", so_fname], capture_output=True)

    out_str = result.stdout.decode('utf-8')
    lines= out_str.split('\n')
    data = {
        'address':[],
        'type':[],
        'name':[]
    }
    for line in lines:
        if len(line)<20:
            continue
        address = line[:16]
        stype = line[17]
        name = line[19:]
        data['address'].append(address)
        data['type'].append(stype)
        data['name'].append(name)

    df = pd.DataFrame(data)
    return df

def test_nm():
    so_fname = os.path.join(os.path.dirname(__file__),'..','src','app')
    result = nm(so_fname)
    pass


def ldd(so_fname:str)->list:
    import re
    result = subprocess.run(["ldd", so_fname], capture_output=True)
    out_str = result.stdout.decode('utf-8')
    lines = out_str.split('\n')

    fnames = []
    for line in lines:
        result = re.match('\t([\S]+) => ([\S]+)', line)
        if result:
            if result[2] == 'not':
                # file not found
                LOGGER.warning(line)
                continue
            fname = result[2]
            fnames.append(fname)
            pass
    return fnames

def test_ldd():
    so_fname = os.path.join(os.path.dirname(__file__),'..','src','app')
    result = ldd(so_fname)
    print(result)
    pass

def get_callee_function_names(so_fname:str, func_name:str, func_info:dict={})->Set[str]:
    if func_name in func_info:
        return func_info[func_name]
    result = subprocess.run(["objdump","-j",".text",so_fname,f"--disassemble={func_name}"], capture_output=True)
    out_str = result.stdout.decode('utf-8')
    lines = out_str.split("\n")
    called_function_names = set()
    for line in lines:
        if 'callq' in line:
            r = re.findall('<(\S+)>', line)
            if r:
                called_function_name = r[0]
                called_function_name = called_function_name.replace('@plt','')
                called_function_names.add(called_function_name)

    return called_function_names

def test_get_callee_function_names():
    so_fname = os.path.join(os.path.dirname(__file__),'..','src','app')
    function_name = 'main'
    result = get_callee_function_names(so_fname, function_name)
    print([cxxfilt.demangle(r) for r in result])
    pass


def get_all_functions(so_fname:str):
    result = subprocess.run(["objdump","-t",so_fname], capture_output=True)
    out_str = result.stdout.decode('utf-8')
    lines = out_str.split('\n')
    result = {
        'local':[],
        'plt':[]
    }

    local_funcs = []
    plt_funcs = []
    for line in lines:
        a = re.match('([\da-fA-F]{16}) ([\S ])([\S ])   ([\S ])([\S ]) ([\S*]+)\t([\da-fA-F]{16})\s+(\S+)', line)
        if a:
            section = a[6]
            if section == '.text':
                local_funcs.append(a[8])
            elif section == '*UND*':
                plt_funcs.append(a[8])
            else:
                LOGGER.info(f'unsupported section: {line}')
                pass
        else:
            LOGGER.warning(f'parser failed: {line}')
            pass
    result['local'] = local_funcs
    result['plt'] = plt_funcs
    return result

def test_get_all_functions():
    so_fname = os.path.join(os.path.dirname(__file__),'..','src','app')
    result = get_all_functions(so_fname)
    print(result)
    pass

class FunctionInfo:
    def __init__(self, func_name:str, so_fname:str=None, callee_funcs:list=[], caller_funcs:list=[]) -> None:
        self.func_name = func_name
        self.so_fname = so_fname
        self.callee_funcs = callee_funcs
        self.caller_funcs = caller_funcs
        pass


class LibInfo:
    def __init__(self, lib_fname:str) -> None:
        self.root_lib_fname = lib_fname
        self.callee_libs = ldd(self.root_lib_fname)
        result = get_all_functions(self.root_lib_fname)
        self.local_functions = result['local']
        self.plt_functions = result['plt']
        self.func_db = {}
        for func in self.local_functions:
            self.func_db[func] = self.root_lib_fname
        pass

    def get_callee_libs(self):
        callee_libs = ldd(self.root_lib_fname)
        return callee_libs
    
    def get_functions(self):
        pass

    def update_db(self):
        lib_fnames = [self.root_lib_fname]
        for lib_fname in self.callee_libs:
            result = get_all_functions(lib_fname)
            local_funcs = result['local']
            for local_func in local_funcs:
                if local_func in self.func_db:
                    pass
                else:
                    self.func_db[local_func] = lib_fname
            
        pass

    def get_all_callee_functions(self, so_fname:str, func_name:str):
        callee_functions = get_callee_function_names(so_fname, func_name)
        for callee_function in callee_functions:
            if callee_function.endswith('@plt'):
                # 3rd party
                callee_func_name = callee_function[:-4]

                pass
            else:
                pass

def prepare_func_lib_map(root_lib_fname:str, db={}):
    callee_libs = ldd(root_lib_fname)

    result = get_all_functions(root_lib_fname)
    for local_func in result['local']:
        if local_func in db:
            continue
        db[local_func] = root_lib_fname
    
    known_lnames = set(db.values())
    for lname in callee_libs:
        if lname in known_lnames:
            continue
        db = prepare_func_lib_map(lname, db)

    # check plt funcs
    for plt_func in result['plt']:
        if plt_func not in db:
            LOGGER.warning(f'{plt_func} is not found.')
    return db

def test_prepare_func_lib_map():
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] += ":" + os.path.join(os.path.dirname(__file__),'..','src')
    else:
        os.environ['LD_LIBRARY_PATH'] = os.path.join(os.path.dirname(__file__),'..','src')
    so_fname = os.path.join(os.path.dirname(__file__),'..','src','app')
    db = prepare_func_lib_map(so_fname)
    print(len(db), set(db.values()))
    print([cxxfilt.demangle(f) for f in db.keys()])
    for f,l in db.items():
        print(f'{cxxfilt.demangle(f):25s}:{f:25s}:{l}')
    pass

def get_all_callee_funcs(func_name, db:dict={}):
    func_info = {}
    known_funcs = set()
    new_funcs = set([func_name])
    while len(new_funcs)>0:
        unknown_funcs = set()
        for f in new_funcs:
            if f in func_info:
                known_funcs.add(f)
            elif f in db:
                so_fname = db[f]
                callee_functions = get_callee_function_names(so_fname, f, func_info)
                func_info[f] = callee_functions
                known_funcs.add(f)
                unknown_funcs = unknown_funcs.union(callee_functions) - known_funcs
            else:
                LOGGER.warning(f'{f} : lib  file is not known.')
                known_funcs.add(f)
        new_funcs = new_funcs.union(unknown_funcs) - known_funcs
        pass
    return known_funcs

def test_get_all_callee_funcs():
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] += ":" + os.path.join(os.path.dirname(__file__),'..','src')
    else:
        os.environ['LD_LIBRARY_PATH'] = os.path.join(os.path.dirname(__file__),'..','src')
    so_fname = os.path.join(os.path.dirname(__file__),'..','src','app')
    db = prepare_func_lib_map(so_fname)
    

    func_name = 'main'
    result = get_all_callee_funcs(func_name, db)
    print([cxxfilt.demangle(f) for f in result])
    pass    

if __name__ == '__main__':
    # test_nm()
    # test_ldd()
    # test_read_function()
    # test_get_all_functions()
    test_prepare_func_lib_map()

    # test_get_all_callee_funcs()
    pass