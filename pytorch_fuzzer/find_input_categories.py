
import sqlite3
import os
import subprocess
from collections import defaultdict
import random
import shutil

from pytorch_fuzzingbook_invariant_utils import INVARIANT_PROPERTIES_TYPES

INV_EXTRACTION = 'pytorch_fuzzingbook_invariant_utils.get_invariants_hold(pytorch_fuzzingbook_invariant_utils.INVARIANT_PROPERTIES,'

rootdir = os.getcwd()

def read_typedb_cached_file():
    result = {}

    for f in os.listdir(rootdir):
        if '.typedb' not in f:
            continue
        prefix = f.split('.typedb')[0]
        if not prefix.startswith('root_'):
            print('skipping since file does not match root typedb', f)

        matched_type = prefix.split('__')[1]
        ways = []
    
        # read the file
        with open( f) as infile:
            way = ''
            
            invariants = []
            way_id = None
            func_name_to_num_call_to_arg_name = {}

            for line in infile:
                line = line.strip()
                if not line.startswith('==='):
                    if line.startswith('id:'):
                        way_id = line.split('id:')[1]
                        way_id = way_id.split('#')[0]

                    elif line.startswith('!invariants:'):
                        invariant = line.split('!invariants: ')[1]
                        readable_invariant, invariant_type, invariant_value = eval(invariant)
                        invariants.append((readable_invariant, invariant_type, invariant_value))
                    elif 'c 2' in line and line.strip().startswith('#'):
                        try:
                            function_name = line.split('c 2 ')[1].split()[0]
                            num_call = line.split('c 2 ')[1].split()[1]
                            arg_name = line.split('c 2 ')[1].split()[2]
                        except Exception as e:
                            # print(line)
                            raise e
                        if function_name not in func_name_to_num_call_to_arg_name:
                            func_name_to_num_call_to_arg_name[function_name] = {}

                        func_name_to_num_call_to_arg_name[function_name][num_call] = arg_name
                        continue

                    elif INV_EXTRACTION in line:
                        pass

                    else:
                        if not line.startswith('#'):
                            way += line + '\n'

                elif line.startswith('==='):

                    if way_id is not  None:

                        if len(way.strip()) > 0:
                            # if the `way` does not include any function call or attribtue access and not a primitive type, it's likely to be bogus
                            if '(' in way or '.' in way or matched_type in ['float', 'bytes', 'int', 'uint8', 'uint32', 'uint64', 'set', 'tuple', 'int32', 'int8', 'float64', 'float32', 'bool', 'bool_']:

                                # the last '')'' can be removed since it matches INV_EXTRACTION's opening bracket
                                if way.strip().endswith(')'):
                                    way = way[:-1]
                                ways.append((way_id, way, invariants, func_name_to_num_call_to_arg_name))
                        print('appending way of id=', way_id)
                    way = ''
                    way_id = None
                    invariants = []
                    func_name_to_num_call_to_arg_name = {}

        result[matched_type ] = ways

    return result



types_to_ways_invs = read_typedb_cached_file()

invariant_sets = defaultdict(list)
all_func_seeds = {}
all_func_seeds_invs = {}
for typ, ways_invs_func_seeds in types_to_ways_invs.items():
    for way_id, way, invariants, func_name_to_num_call_to_arg_name in ways_invs_func_seeds:
        invariants = sorted(invariants)
        invariant_tuple = tuple(invariants)

        invariant_sets[invariant_tuple].append(way_id)
        print('add to invariant_sets, wayid=', way_id)

        for func_name, num_call_to_arg_name in func_name_to_num_call_to_arg_name.items():
            if func_name not in all_func_seeds:
                all_func_seeds[func_name] = {}
                all_func_seeds_invs[func_name] = {}

            for num_call, arg_name in num_call_to_arg_name.items():
                if num_call not in all_func_seeds[func_name]:
                    all_func_seeds[func_name][num_call] = {}
                    all_func_seeds_invs[func_name][num_call] = {}

                all_func_seeds[func_name][num_call][arg_name] = way_id
                all_func_seeds_invs[func_name][num_call][arg_name] = invariant_tuple


with open('invariant_sets.txt', 'w+') as outfile:
    for one_set, ways in invariant_sets.items():
        outfile.write('[' + ','.join([str(item) for item in one_set]) + ']' + ' !!! ')
        outfile.write('[' + ','.join(way for way in ways) + ']')
        outfile.write('\n')

with open('func_seeds.txt', 'w+') as outfile:
    for func_name, num_call_to_argname_to_invs in all_func_seeds_invs.items():
        for num_call, argname_to_invs in num_call_to_argname_to_invs.items():
            for argname, invs in argname_to_invs.items():
                way_id = all_func_seeds[func_name][num_call][argname]
                outfile.write(func_name)
                outfile.write("...")
                outfile.write(num_call)
                outfile.write(';;')
                outfile.write(argname)
                outfile.write(':')
                outfile.write('[' + ','.join([str(item) for item in invs]) + ']' + ' !!! ')
                outfile.write(way_id)
                outfile.write('\n')
