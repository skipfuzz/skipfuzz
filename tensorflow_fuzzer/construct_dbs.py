import os
import subprocess
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count


def how_many_needed(target_type):
    if target_type[0].islower():
        return 100000 # primitive, should be fine to have many of them
    if 'Tensor' in target_type:
        return 100000
    else:
        return 3000



INV_EXTRACTION = 'fuzzingbook_invariant_utils.get_invariants_hold(fuzzingbook_invariant_utils.INVARIANT_PROPERTIES,'

def read_typedb_file(target_type):
    result = []

    has_seen = set()
    for root, subdirs, files in os.walk(rootdir):
        for f in files:
            if not f.endswith('.typedb'):
                continue

            prefix = f.split('.typedb')[0]
            matched_type = prefix.split('__')[1]
            if matched_type == target_type:
                # read the file
                with open(root + '/' + f) as infile:
                    way = ''
                    func_name_to_num_call_to_arg_name = {}

                    has_seen_first_non_import_line = False
                    
                    for line in infile:
                        line = line.strip()
                        if not line.startswith('==='):

                            if 'import ' not in line and '#' not in line:
                                if not has_seen_first_non_import_line:
                                    way += INV_EXTRACTION + ' \n'
                                    # way += ''
                                has_seen_first_non_import_line = True
                            elif 'c 2' in line and line.strip().startswith('#'):
                                try:
                                    function_name = line.split('c 2 ')[1].split()[0]
                                    num_call = line.split('c 2 ')[1].split()[1]
                                    arg_name = line.split('c 2 ')[1].split()[2]
                                except Exception as e :
                                    # print(line)
                                    raise e
                                if function_name not in func_name_to_num_call_to_arg_name:
                                    func_name_to_num_call_to_arg_name[function_name] = {}

                                func_name_to_num_call_to_arg_name[function_name][num_call] = arg_name

                                continue # don't append to way, otherwise we cannot detect deuplicate ways
                            way += line + '\n'

                        else:
                            if way not in has_seen:
                                has_seen.add(way)

                                way = '# ' + f + '\n' + 'import tensorflow as tf\n' + 'import fuzzingbook_invariant_utils\n' + way + ')'

                                result.append((way, func_name_to_num_call_to_arg_name))
                                has_seen_first_non_import_line = False
                                
                            way = ''
                            func_name_to_num_call_to_arg_name = {}
                            
    return result

def write_invs(outfile, invs):
    for line in invs:
        outfile.write(line + '\n')

def read_invs(lines):
    result = []
    for line in lines:
        if '!invariants:' in line:
            result.append(line)
    return result

def write_and_run_way(way_and_identifier_and_funcnamemap):
    way, identifier, func_name_to_num_call_to_arg_name = way_and_identifier_and_funcnamemap
    with open('/tmp/tmp__test_' + identifier + '.py', 'w+') as outfile:
        
        # outfile.write('fuzzingbook_invariant_utils.get_invariants_hold(fuzzingbook_invariant_utils.INVARIANT_PROPERTIES, ' + way + ' )')
        outfile.write(way)

    proc = subprocess.Popen(['timeout', '180', "python", '/tmp/tmp__test_' + identifier + '.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    retcode = proc.returncode

    if retcode == 0:
        invs = read_invs(stdout.decode('utf-8').split('\n'))
        # read_and_write_invs(outfile, stdout.decode('utf-8').split('\n'))
        return True, invs, way, func_name_to_num_call_to_arg_name

    # otherwise, try to repair
    # TODO
    if 'NameError' in stderr.decode('utf-8'):
        try:
            missing_thing = stderr.decode('utf-8').split('NameError')[1].split("name '")[1].split("'")[0]
            # print(way)
            if missing_thing in all_imported and len(all_imported[missing_thing]) > 1:
                way = all_imported[missing_thing][0] + '\n' + way
        except:
            pass

    else:
        return False, [], way, func_name_to_num_call_to_arg_name

    with open('/tmp/tmp__test_' + identifier + '.py', 'w+') as outfile:
        outfile.write(way)

    proc = subprocess.Popen(['timeout', '180', "python", '/tmp/tmp__test_' + identifier + '.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    retcode = proc.returncode
    
    if retcode == 0:
        invs = read_invs(stdout.decode('utf-8').split('\n'))
        # read_and_write_invs(outfile, stdout.decode('utf-8').split('\n'))
        # if int(identifier) % 10 != 0:

        os.remove('/tmp/tmp__test_' + identifier + '.py')
        return True, invs, way, func_name_to_num_call_to_arg_name
    else :
        return False, [], way, func_name_to_num_call_to_arg_name




def check_runnable(ways, target_type):
    filtered = []

    ways_count = 0

    if how_many_needed(target_type) > 0:
        with Pool(cpu_count()//2 + 1) as pool:
        # with Pool(2) as pool:
            indexed_ways = []
            for way, func_name_to_num_call_to_arg_name in ways[:how_many_needed(target_type)]:
                indexed_ways.append((way, str(ways_count), func_name_to_num_call_to_arg_name))
                ways_count += 1

            run_results = pool.map(write_and_run_way, indexed_ways)

            pool.close()
            pool.join()

            # print(run_results)
            for result, invariants, way, func_name_to_num_call_to_arg_name in run_results:
                if result:
                    filtered.append((way, invariants, func_name_to_num_call_to_arg_name))

    return filtered

def has_cache(type):
    if os.path.exists('root__' + type + '.typedb'):
        return True
    return False

def cached_typedb_spells(type, ways_and_invs):

    os.chdir(rootdir)
    if len(ways_and_invs) == 0:
        return
    with open('root__' + type + '.typedb', 'w+') as outfile:
        for way_id, way, invs, func_name_to_num_call_to_arg_name in ways_and_invs:


            outfile.write('id:' + str(way_id) + '\n')
            outfile.write(way + '\n')
            write_invs(outfile, invs)
            for func_name, num_call_to_arg_name in func_name_to_num_call_to_arg_name.items():
                for num_call, arg_name in num_call_to_arg_name.items():
                    outfile.write('# c 2 ' + func_name + ' ' + num_call + ' ' + arg_name + '\n')
            outfile.write('======\n')

    print('wrote to ', 'root__' + type + '.typedb')



def obtain_all_imports(ways):
    # go through all ways
    # extract all import statements
    imports = defaultdict(set)
    
    for way, func_name_to_num_call_to_arg_name in ways:
        for line in way.split('\n'):
            if line.startswith('import '):
                imported_thing = line.split('import ')[1] if ' as ' not in line else line.split(' as ')[1]
                imports[imported_thing].add('import tensorflow as tf\n')
                imports[imported_thing].add('from numpy import inf')
                imports[imported_thing].add(line)

    return imports


def find_all_known_types():
    result = set()

    has_seen = set()
    for root, subdirs, files in os.walk(rootdir):
        for f in files:
            if not f.endswith('.typedb'):
                continue
            prefix = f.split('.typedb')[0]
            matched_type = prefix.split('__')[1]
            result.add(matched_type)

    return result

rootdir = os.getcwd()


all_types = find_all_known_types()
# all_types = ['Eager
# Tensor', 'str', 'list']

print('copied fuzzingbook_invariant_utils.py to tmp folder')
shutil.copyfile('fuzzingbook_invariant_utils.py', '/tmp/fuzzingbook_invariant_utils.py')

type_count = 0
way_count = 0
for t in all_types:
    if t.endswith('Test'):
        continue
    if has_cache(t):
        print('skip ', t, ' since cache already found')
        continue
    ways = read_typedb_file(t)
    print('found', len(ways), 'ways of building', t)

    all_imported = obtain_all_imports(ways)

    # trim ways based on runnability(?)
    # try some way to automatically resolve the missing idents
    runnable_ways_and_invariants_and_func_seeds = check_runnable(ways, t)
    print('trimmed:', 'left with ', len(runnable_ways_and_invariants_and_func_seeds), ' of building', t)

    id_runnable_ways_and_invariants = []
    for way, invs, func_name_to_num_call_to_arg_name  in runnable_ways_and_invariants_and_func_seeds:
        id_runnable_ways_and_invariants.append((way_count, way, invs, func_name_to_num_call_to_arg_name))
        way_count += 1

    cached_typedb_spells(t, id_runnable_ways_and_invariants)

    type_count += 1
    


