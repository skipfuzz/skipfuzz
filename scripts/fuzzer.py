import os
import subprocess
from collections import defaultdict
import random
import shutil
import socket
import string
import sys

from fuzzingbook_invariant_utils import INVARIANT_PROPERTIES_TYPES, INVARIANT_PROPERTIES
import sqlite3
import time

from multiprocessing import Pool, cpu_count
import string

INV_EXTRACTION = 'fuzzingbook_invariant_utils.get_invariants_hold(fuzzingbook_invariant_utils.INVARIANT_PROPERTIES,'




num_crashes = 0
last_crash_index = -1
NUM_TESTS_PER_FUNC = 1000

epsilon = float(sys.argv[1])
try:
    rand_ident = sys.argv[2]
except:
    rand_ident = ''.join(random.choice(string.ascii_lowercase) for i in range(5))

try:
    target_api_file = sys.argv[3]
except:
    target_api_file = 'target_api.txt'


global_all_tests_outfile_imports = set()

def create_one_call(API_name, functions_to_sigs,
                    target_invariant_set_for_arg, invariant_sets, values_store_by_id, values_store_by_type,
                    required_imports,
                    restrictions,
                    test_num):
    num_arguments = len(functions_to_sigs[API_name].keys())
    argument_keys = functions_to_sigs[API_name].keys()

    argument_values = []
    selected_way_ids = []
    arg_id_to_constraints = {}
    for i in range(num_arguments):
        argument_values.append('')
        selected_way_ids.append(-1)
        arg_id_to_constraints[i] = []

    if len(arg_id_to_constraints) > 0:

        for arg_id, constraints in arg_id_to_constraints.items():
            
            if arg_id in restrictions[API_name] and len(restrictions[API_name][arg_id]) > 0:
                print('using restricted values for arg', arg_id, restrictions[API_name][arg_id])
                selected_restricted_id = random.choice(list(restrictions[API_name][arg_id]))

                selected_way_ids[arg_id] = selected_restricted_id

                argument_values[arg_id] = values_store_by_id[str(selected_way_ids[arg_id])]

                restrictions[API_name][arg_id] = set()
                continue


            if arg_id not in target_invariant_set_for_arg or target_invariant_set_for_arg[arg_id] == -1:
                selected_way_ids[arg_id] = random.sample(list(values_store_by_id.keys()), 1)[0]
                argument_values[arg_id] = values_store_by_id[selected_way_ids[arg_id]]

                continue

            matching_ways_id = invariant_sets[target_invariant_set_for_arg[arg_id]]

            if len(matching_ways_id) == 0:
                
                selected_way_ids[arg_id] = random.sample(list(values_store_by_id.keys()), 1)[0]  
                argument_values[arg_id] = values_store_by_id[selected_way_ids[arg_id]]
                continue

            # for each combination of input categories, we only select at most one input from a category.
            sampled_matching_way = random.sample(matching_ways_id, 1)[0]

            selected_way_ids[arg_id] = sampled_matching_way
            argument_values[arg_id] = values_store_by_id[str(sampled_matching_way)]

    test_file_name = 'synthesized_cov_test_' + rand_ident + '_' + API_name + '_' + str(test_num) + '.py'

    full_code = ''
    full_imports = ''
    with open(test_file_name, 'w+') as outfile:

        imports = []
        # gather imports
        for argument_value in argument_values:
            for line in argument_value.split('\n'):
                if 'import ' in line:
                    imports.append(line)

        imports.extend(required_imports)

        prev_import = None
        for one_import in sorted(imports):
            if one_import != prev_import:
                outfile.write(one_import + '\n')
                if one_import not in global_all_tests_outfile_imports:
                    full_imports += one_import + '\n'
                    global_all_tests_outfile_imports.add(one_import)
            prev_import = one_import

        outfile.write('\n')
        outfile.write('\n')

        full_code += 'try:\n'
        full_code += '\n'
        full_code += '\n'

        for i, argument_value in enumerate(argument_values):
            written_first_line = False
            for line in argument_value.split('\n'):
                if 'import ' in line:
                    continue
                if len(line.strip()) == 0:
                    continue
                if not written_first_line:
                    outfile.write('v_' + str(i) + ' = ' + line + '\n')

                    full_code += '\t' + 'v_' + str(i) + ' = ' + line + '\n'
                    written_first_line = True
                else:
                    outfile.write(line + '\n')
                    full_code += line + '\n'

        outfile.write(API_name + ','.join([argument_key + '=v_' + str(i) for i, (argument_key, _) in
                                           enumerate(zip(argument_keys, argument_values))]) + ')\n')
        full_code += '\t' + API_name + ','.join([argument_key + '=v_' + str(i) for i, (argument_key, _) in
                                           enumerate(zip(argument_keys, argument_values))]) + ')\n'
        full_code += 'except:\n'
        full_code += '\tpass\n'
    return True, selected_way_ids, test_file_name, full_code, full_imports


rootdir = os.getcwd()


def read_API_function_sigs():
    functions_to_sigs = {}
    functions_to_imports = {}
    function_arg_to_index = {}

    for root, subdirs, files in os.walk(rootdir):
        for f in files:
            if not f.endswith('.typesig'):
                continue
            API_function = f.split('.typesig')[0].split('--')[1]

            if API_function not in functions_to_sigs:
                functions_to_sigs[API_function] = {}
                function_arg_to_index[API_function] = {}

            imports = []
            with open(root + '/' + f) as infile:
                for line in infile:
                    line = line.strip()

                    if 'import' in line:
                        imports.append(line)
                        continue

                    arg_name = line.split(':')[0]
                    arg_types = line.split(':')[1]

                    function_arg_to_index[API_function][arg_name] = len(function_arg_to_index[API_function].keys())

                    if arg_name not in functions_to_sigs[API_function]:
                        functions_to_sigs[API_function][arg_name] = set()

                    for arg_type in arg_types.split(','):
                        functions_to_sigs[API_function][arg_name].add(arg_type)

            functions_to_imports[API_function] = imports

    return functions_to_sigs, functions_to_imports, function_arg_to_index


def get_invariant_sets():
    invariants_sets = defaultdict(list)
    with open('undominated_invariant_sets.txt') as infile:
        for line in infile:
            invs_part = line.split(' !!! ')[0]
            ways_part = line.split(' !!! ')[1]
            invariants_set = eval(invs_part)
            ways = eval(ways_part)

            invariants_sets[tuple(invariants_set)] = ways

    invariants_sets['misc'] = []
    return invariants_sets


MAX_LOOP = 5

def get_rules(target_function, functions_to_sigs,
              invariant_sets, func_to_already_checked_invs, seeds, ):

    function_sig = functions_to_sigs[target_function]

    if target_function in func_to_already_checked_invs:
        already_checked_invs = func_to_already_checked_invs[target_function]
    else:
        already_checked_invs = set()
        func_to_already_checked_invs[target_function] = already_checked_invs

    # pick something from invariant_sets
    target_invariant_set_for_arg = {}
    restrictions = {}

    pick_random = random.uniform(0, 1) < epsilon

    if pick_random or target_function not in seeds or len(seeds[target_function]) == 0:  # random or no seeds
        target_args = range(len(function_sig.items()))
        restrictions[target_function] = {}
    else:
        selected_seed_mapping = random.choice(seeds[target_function])
        restrictions[target_function] = {}
        for seed_mapping_i, seed_mapping_value in enumerate(selected_seed_mapping):
            restrictions[target_function][seed_mapping_i] = [seed_mapping_value]

        # pick random arg to transform
        if len(selected_seed_mapping) >0:
            target_args = [random.randint(0, len(selected_seed_mapping) - 1)]
            restrictions[target_function][target_args[0]] = set()

            if not isinstance(target_args, list):
                raise Exception('wrong type')
        else:
            target_args = range(len(function_sig.items()))

    loop_i = 0
    while len(target_invariant_set_for_arg) == 0 or tuple(target_invariant_set_for_arg.items()) in already_checked_invs:
        for i in target_args:
            selected_invariant_sets = invariant_sets.keys()

            selectables = tuple(selected_invariant_sets)
            target_invariant_set_index = random.randint(-1, len(selectables) - 1)
            target_invariant_set = selectables[target_invariant_set_index] if target_invariant_set_index != -1 else -1
            
            target_invariant_set_for_arg[i] = target_invariant_set

        loop_i += 1
        if loop_i >= MAX_LOOP:
            for i in target_args:
                target_invariant_set_for_arg[i] = -1
            break

    if all([v != -1 for v in target_invariant_set_for_arg.values()]):
        already_checked_invs.add(tuple(target_invariant_set_for_arg.items()))
    return target_invariant_set_for_arg, restrictions


def read_typedb_cached_file_with_id():
    values_store_by_id = {}
    values_store_by_type = {}

    for f in os.listdir(rootdir):
        if '.typedb' not in f:
            continue
        prefix = f.split('.typedb')[0]
        if not prefix.startswith('root_'):
            print('skipping since file does not match root typedb', f)

        matched_type = prefix.split('__')[1]
        ways = []

        # read the file
        with open(f) as infile:
            way = ''

            invariants = []
            way_id = None
            for line in infile:
                line = line.strip()
                if not line.startswith('===') and not line.startswith('#'):
                    if line.startswith('id:'):
                        if way_id is not None:
                            raise Exception('overwriting way id', 'f', f, 'line is ', line)

                        way_id = line.split('id:')[1]
                        way_id = way_id.split('#')[0]

                    elif line.startswith('!invariants:'):
                        invariant = line.split('!invariants: ')[1]
                        readable_invariant, invariant_type, invariant_value = eval(invariant)
                        invariants.append((readable_invariant, invariant_type, invariant_value))


                    elif INV_EXTRACTION in line:
                        pass

                    else:
                        way += line + '\n'

                elif line.startswith('==='):

                    if way_id is not None:

                        if len(way.strip()) > 0:
                            # if the `way` does not include any function call or attribtue access and not a primitive type, it's likely to be bogus
                            if '(' in way or '.' in way or matched_type in ['float', 'bytes', 'int', 'uint8', 'uint32',
                                                                            'uint64', 'set', 'tuple', 'int32', 'int8',
                                                                            'float64', 'float32', 'bool', 'bool_']:

                                # the last '')'' can be removed since it matches INV_EXTRACTION's opening bracket
                                if way.strip().endswith(')'):
                                    way = way.strip()[:-1]

                                ways.append((way_id, way, invariants))
                                values_store_by_id[way_id] = way

                    way = ''
                    way_id = None
                    invariants = []

        values_store_by_type[matched_type] = ways

    return values_store_by_id, values_store_by_type



def find_all_known_functions():
    functions_to_sigs = {}
    functions_to_imports = {}
    crawled_function_arg_to_index = {}
    # with open('all_functions.txt') as infile:
    with open('all_functions.txt') as infile:
        way = ''
        imports = []
        function_name = None
        params = {}

        for line in infile:
            line = line.strip()
            if line.startswith('==='):
                if function_name is None:
                    raise Exception('unknown func name')

                functions_to_sigs[function_name] = params
                functions_to_imports[function_name] = imports

                crawled_function_arg_to_index[function_name] = {}
                for param_i, param in enumerate(params):
                    crawled_function_arg_to_index[function_name][param] = param_i

                function_name = None
                params = {}
                imports = []
            else:
                if 'import ' in line:
                    imports.append(line)
                elif ':' in line:
                    param_name = line.split(':')[0]
                    param_type = line.split(':')[1]
                    params[param_name] = param_type
                else:
                    function_name = line.strip() + '('
    # print(functions_to_sigs)
    return functions_to_sigs, functions_to_imports, crawled_function_arg_to_index


rootdir = os.getcwd()


def run_script_server(prev_p = None, delay=15, func=None, port = 65433):
    def wait_for_proc_end(prev_p,delay_bef, delay_aft, func):
        poll = prev_p.poll()
        start_time = time.time()
        while poll is None:
            time.sleep(1)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            if time_elapsed > delay_bef:
                break
            poll = prev_p.poll()
            print('run_script_server: poll again. time passed=', time_elapsed)
        if poll is None:
            print('poll is None -> prev_p is still running')
            if func is not None:
                print('running func')
                func()
            time.sleep(delay_aft)
        else:
            print('poll is not None -> prev_p is not running')

    time.sleep(1)
    if prev_p is not None:
        print('wait for/kill previous instance first')

        wait_for_proc_end(prev_p, 30, 5, func=lambda: prev_p.terminate())
        wait_for_proc_end(prev_p, 20, 3, func=lambda: prev_p.kill())

    print('starting server on port', port)


    p1 = subprocess.Popen(['coverage', 'run', '--source=/usr/local/lib/python3.8/dist-packages/tensorflow/', '--append', '--data-file=' + rand_ident + '_fuzz.coverage', '-m', '5a_test_executor', str(port)])

    print('waiting for', delay, 'seconds')
    time.sleep(delay)

    return p1


def add_to_seeds_if_improve(prev_coverage, current_coverage, current_choices, seeds, target_function):
    if current_coverage > prev_coverage:
        print('adding to seeds', current_choices, 'of', target_function)
        seeds.append(current_choices)
        # experimental: keep only the best N seeds
        del seeds[0:-5]


def create_one_test_and_run(target_function, all_functions,
                            all_function_sigs, usable_types, invariant_sets, func_to_already_checked_invs,
                            seeds, function_arg_to_index,
                            values_store_by_id, values_store_by_type, functions_to_imports,
                            invariant_type_comparison_direction,
                            s,
                            prev_coverage_num,
                            all_outfile,
                            test_i):
    global num_crashes

    is_no_problemo_runs = False
    is_problem = False
    is_py_error = False
    is_crash = False

    target_invariant_set_for_arg, restrictions = get_rules(target_function, all_function_sigs,
                                                                            invariant_sets,
                                                                            func_to_already_checked_invs, seeds,)
    succeeded, selected_way_ids, test_file_name, full_code, full_imports = create_one_call(
        target_function, all_function_sigs,
        target_invariant_set_for_arg, invariant_sets,
        values_store_by_id,
        values_store_by_type,
        functions_to_imports[target_function],
        restrictions, test_i)
    if not succeeded:
        all_functions.remove(target_function)
        return False

    if target_function not in func_to_already_checked_invs:
        func_to_already_checked_invs[target_function] = set()

    data = ''

    all_outfile.write(full_imports)

    try:
        print('[test builder] sending ', test_file_name)
        s.sendall(test_file_name.encode('utf-8'))
        s.sendall('=='.encode('utf-8'))


        start_time = time.time()
        while data is None or not data.endswith('rann'):
            s.settimeout(30)
            data_raw = s.recv(1024)
            try:
                data += data_raw.decode('utf-8')
            except Exception as e:
                print('exception caught while decoding data from the script-server', e)
                pass


            end_time = time.time()
            time_elapsed = (end_time - start_time)

            if time_elapsed > 30 and len(data_raw) == 0:
                print('too much time elasped', time_elapsed)
                data = 'Server down!'
                break


    except ConnectionRefusedError as e:
        print('run-script server down!')
        print(e)
        data = 'Server down!'
        raise e
    except ConnectionResetError as e:
        data = 'Server down!'
        print('run-script server is down. Exception:')
        print(e)
    except socket.timeout as e:
        detected_crash(test_file_name, test_i)
        print('timeout detected?')
        print('exception = ', e)
        data = 'Server down!'
        raise e
    except BrokenPipeError as e:

        detected_crash(test_file_name, test_i)
        print('crash detected?')
        print('exception = ', e)
        data = 'Server down!'
        raise e

    all_outfile.write(full_code)

    data = data.split('rann')[0]
    coverage_num = prev_coverage_num
    outcome = ''
    if len(data) > 0:
        try:
            coverage_num = int(data.split('===')[1].strip())
            outcome = data.split('===')[0].strip()

            print('current cov', coverage_num)
            add_to_seeds_if_improve(prev_coverage_num, coverage_num, selected_way_ids, seeds[target_function], target_function)
        except Exception as e:
            print('caught e while trying to parse coverage', e)
            print('assuming crashed? Is it true?')
    if data != 0 and 'NameError' in data:
        copied_fn = 'problem_test_' + str(test_i) + '.py'
        print('copying to ', copied_fn)
        shutil.copy(test_file_name, copied_fn)
        is_problem = True


    elif data != 0 and data == 'Server down!':
        detected_crash(test_file_name, test_i)
        print('num crashes', num_crashes)
        if num_crashes >= 5000:
            return None
        else:
            pass
        is_crash = True

    else:
        if data == 0:
            print('successful run')
            is_no_problemo_runs = True
        else:
            if data != 0 and 'Error:' in data:
                is_py_error = True

            else:
                print('unknown', 'data=', data[:200])

    outcome = outcome.split('Error')[0].split()[-1] if 'Error' in outcome else outcome[:30]

    if is_crash:
        shutil.copy(test_file_name,
                    'ran_tests/' + 'test_' + target_function + '_' + str(test_i) + '_' + outcome[:30] + '.py')

    os.remove(test_file_name)

    return is_crash, is_no_problemo_runs, is_problem, is_py_error, is_crash, outcome, coverage_num

def detected_crash(test_file_name, test_i):
    global num_crashes
    global last_crash_index
    print('!!!!!!!detected crash!!!!!!')
    copied_fn = 'crashed_test_' + test_file_name + '.py'
    print('copying to ', copied_fn)
    shutil.copy(test_file_name, copied_fn)
    if test_i != 0:
        num_crashes += 1
        print('num crashes', num_crashes)
    else:
        print('not counting crash as it is the first run')
    last_crash_index = test_i
    print('setting last_crash_index to', last_crash_index)


def main():
    all_function_sigs, functions_to_imports, function_arg_to_index = read_API_function_sigs()

    crawled_function_sigs, crawled_functions_to_imports, crawled_function_arg_to_index = find_all_known_functions()
    all_function_sigs.update(crawled_function_sigs)
    functions_to_imports.update(crawled_functions_to_imports)
    function_arg_to_index.update(crawled_function_arg_to_index)
    all_functions = set(crawled_function_sigs.keys())  # TODO

    invariant_sets = get_invariant_sets()

    values_store_by_id, values_store_by_type = read_typedb_cached_file_with_id()


    invariant_type_comparison_direction = {}
    for prop_bef, _ in INVARIANT_PROPERTIES.items():
        if '>' in prop_bef and '<' not in prop_bef:
            invariant_type_index = INVARIANT_PROPERTIES_TYPES.index(prop_bef)
            invariant_type_comparison_direction[invariant_type_index] = False
        elif '<' in prop_bef and '>' not in prop_bef:
            invariant_type_index = INVARIANT_PROPERTIES_TYPES.index(prop_bef)
            invariant_type_comparison_direction[invariant_type_index] = True

    run_outcomes = defaultdict(list)

    # functions that do not have arguments do not need to be tested
    all_functions = set([f for f in all_functions if len(all_function_sigs[f].keys()) > 0])

    print('total of ', len(all_functions), 'functions')
    try:
        func_to_already_checked_invs = {}

        start_time = time.time()

        if not os.path.exists('ran_tests'):
            os.mkdir('ran_tests')


        if os.path.exists(target_api_file):
            target_functions = []
            with open(target_api_file) as infile:
                for line in infile:
                    function_name = line.strip() + '('
                    target_functions.append(function_name)

        else:
            target_functions = all_functions
        seeds = defaultdict(list)

        with Pool(1) as pool:
            # with Pool(1) as pool:


            args_to_run = []
            selected_target_functions = target_functions

            args_to_run.append((all_function_sigs, all_functions, func_to_already_checked_invs,
                                function_arg_to_index, functions_to_imports, invariant_sets,
                                invariant_type_comparison_direction, seeds,
                                selected_target_functions, set(), values_store_by_id,
                                values_store_by_type))

            pool.map(create_tests_for_one_target_function, args_to_run)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print('caught exception', e)
        raise e

    print('run metadata')
    print('elapsed time --- %s seconds ---' % (time.time() - start_time))



def create_tests_for_one_target_function(arguments):
    all_function_sigs, all_functions, func_to_already_checked_invs, \
    function_arg_to_index, functions_to_imports, invariant_sets, \
    invariant_type_comparison_direction, seeds, \
    target_functions, usable_types, values_store_by_id, values_store_by_type = arguments

    HOST = "127.0.0.1"
    PORT1 = 65433
    PORT2 = 65435

    conn_retry_count = 0
    script_server_proc = None
    run_outcomes = {}
    for target_function_i, target_function in enumerate(target_functions):
        all_test_file_name = get_test_file_record_filename(target_function)
        if os.path.exists(all_test_file_name):
            print('skip target_function', target_function, ' #', target_function_i)
            continue
        print('starting server for target_function', target_function, ' #', target_function_i)
        port = PORT1 if target_function_i % 2 == 0 else PORT2
        script_server_proc = run_script_server(script_server_proc, delay=5, port=port)
        run_outcomes[target_function] = []
        crashes_for_target_func = 0
        while len(run_outcomes[target_function]) < NUM_TESTS_PER_FUNC:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.connect((HOST, port))
                    except (ConnectionRefusedError, socket.timeout):
                        # just retry a few times
                        conn_retry_count += 1
                        if conn_retry_count > 20:
                            raise Exception('script server issue: could not connect')
                        print('starting server again for target_function', target_function)
                        script_server_proc = run_script_server(script_server_proc, delay=max(10*conn_retry_count, 10*3), port = port)


                    create_tests_for_one_target_function_1(all_function_sigs, all_functions, func_to_already_checked_invs,
                                                           function_arg_to_index, functions_to_imports, invariant_sets,
                                                           invariant_type_comparison_direction, seeds, run_outcomes,
                                                           target_function, usable_types, values_store_by_id, values_store_by_type, s)
                    s.sendall('done=='.encode('utf-8'))
                    break
            except (BrokenPipeError, socket.timeout):
                if last_crash_index != 0:  # if it's zero, we're dealing with some error starting the script-server
                    crashes_for_target_func += 1
                if crashes_for_target_func > 4:
                    print('stop testing', target_function, 'since crashed', crashes_for_target_func, 'times')
                    try:
                        s.sendall('done=='.encode('utf-8'))
                        print('send done! from error path')
                    except:
                        print('tried sending done, but failed. from error path')

                    break
                print('detected server down')
                print('total ran', len(run_outcomes[target_function]))
                # print('===stat===')
                print_outcomes(run_outcomes[target_function])

                script_server_proc = run_script_server(script_server_proc, delay=2, port = port)
                print('should have restart server from the broken pipe error')

        print('===stat===')

        print_outcomes(run_outcomes[target_function], target_function, 'outcomes' + rand_ident + '.txt')


def create_tests_for_one_target_function_1(all_function_sigs, all_functions, func_to_already_checked_invs,
                                           function_arg_to_index, functions_to_imports, invariant_sets,
                                           invariant_type_comparison_direction, seeds, run_outcomes,
                                           target_function, usable_types, values_store_by_id, values_store_by_type, s):
    prev_coverage_num = 0  # doesn't matter, we always add the first randomly generated test

    global_all_tests_outfile_imports.clear()

    all_test_file_name = get_test_file_record_filename(target_function)
    with open(all_test_file_name, 'w+') as all_outfile:
        for i in range(NUM_TESTS_PER_FUNC):
            print('target', target_function, 'i', i)

            is_crash, is_no_problemo_runs, is_problem, is_py_error, is_crash, outcome, coverage_num = create_one_test_and_run(
                target_function, all_functions,
                all_function_sigs, usable_types, invariant_sets, func_to_already_checked_invs,
                seeds, function_arg_to_index,
                values_store_by_id, values_store_by_type, functions_to_imports, invariant_type_comparison_direction,
                s, prev_coverage_num,
                all_outfile,
                i)
            print('outcome from the outside:', outcome, '\t for function', target_function, 'i=', i)

            prev_coverage_num = coverage_num

            run_outcomes[target_function].append(outcome)

            if len(run_outcomes[target_function]) > NUM_TESTS_PER_FUNC:
                print('early breaking since len > NUM_TESTS_PER_FUNC', NUM_TESTS_PER_FUNC)
                break


def get_test_file_record_filename(target_function):
    all_test_file_name = 'synthesized_all_cov_test_' + rand_ident + '_'+ target_function + '.py'
    return all_test_file_name

def print_outcomes(outcomes, log_name=None, filename = None):
    result = defaultdict(int)
    for outcome in outcomes:
        result[outcome] += 1

    if filename is None:
        print('outcomes')
        for k, v in result.items():
            print(k, ':', v)
    else:
        with open(filename, 'a') as outfile:
            outfile.write('outcomes' + '\n')
            if log_name is not None:
                outfile.write(log_name + '\n')
            for k, v in result.items():
                outfile.write(k + ':' + str(v) + '\n')



main()

