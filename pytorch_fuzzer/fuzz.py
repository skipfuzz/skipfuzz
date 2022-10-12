import multiprocessing
import os
import queue
import subprocess
import traceback
from collections import defaultdict
import random
import shutil
import socket
import sys

import time

from multiprocessing import Pool
from multiprocessing import Manager

import string

INV_EXTRACTION = 'pytorch_fuzzingbook_invariant_utils.get_invariants_hold(pytorch_fuzzingbook_invariant_utils.INVARIANT_PROPERTIES,'

num_crashes = 0
last_crash_index = -1
NUM_TESTS_PER_FUNC = 1000


try:
    rand_ident = sys.argv[1]
except:
    rand_ident = ''.join(random.choice(string.ascii_lowercase) for i in range(6))

print('experiments rand_ident is', rand_ident)

try:
    start_from = int(sys.argv[2])

    rand_ident += '_' + sys.argv[2]
except:
    start_from = int(0)

try:
    end_at = int(sys.argv[3])
    rand_ident += '_' + sys.argv[3]
except:
    end_at = int(0)

target_api_file = 'target_api.txt'


global_all_tests_outfile_imports = defaultdict(set)
global_known_targets = defaultdict(lambda: defaultdict(set))
global_found_target_at = defaultdict(lambda: defaultdict(int))
global_explained_level = defaultdict(lambda: defaultdict(int))

global_running_mode = defaultdict(lambda: defaultdict(int))
global_name_to_invs = {}
global_running_solvers = []



def write_clingo(API_name, target_arg, history):
    print('[write_clingo]', API_name, target_arg, 'history len', len(history))
    program = ''
    # history
    valid_counts = defaultdict(int)
    for invariant_set, status in history:
        program += 'seen(' + str(invariant_set) + ').\n'
        if status == "crash":
            program += 'crash(' + str(invariant_set) + ').\n'
        elif status == 'valid':
            # program += 'valid(' + str(invariant_set) + ').\n'
            valid_counts[str(invariant_set)] += 1
        elif status == 'invalid':
            program += 'invalid(' + str(invariant_set) + ').\n'
        else:
            raise Exception("unhandled status when writing history")

    for valid_invset, count in valid_counts.items():
        # only if we see `valid` more than once, then we consider it as valid
        # one invalid input is all we need to know that smoething is invalid,
        # but to determine that something is valid, we need stronger evidence than just a single successful input
        if count > 1:
            program += 'valid(' + str(valid_invset) + ').\n'

    program += '\n'
    with open('lp_' + rand_ident + '_' + API_name + '_' + str(target_arg) + '.lp', 'w+') as outfile:
        outfile.write(program)
    return 'lp_' + rand_ident + '_' + API_name + '_' + str(target_arg) + '.lp'

def execute_clingo(lp_filename, time, history, inferred_queue, solution_queue, target_arg, target_function, ):

    result = subprocess.Popen(['timeout', '-k', str(time + 10) + 's', str(time) + 's', 'clingo', 'lp_invset_stronger.lp', 'lp_invset.lp', 'lp_rules.lp', lp_filename], stdout=subprocess.PIPE)

    answers_used = 0
    selecteds = []
    to_runs = []
    targets = []
    total_explained = -1
    total_invalid_matched = -1
    for line in result.stdout:
        line = line.decode('utf-8')
        print('[clingo_raw]', 'from' , lp_filename, ' : ', line)
        if line.startswith('Answer'):
            answers_used += 1
            #
            # if answers_used % 10 == 9  and solution_queue.qsize() < 5: # if we are running out of things to test, add stuff early
            #     print('execute_clingo, add to queue for every 5th answer', lp_filename)
            #     add_to_queues(history, inferred_queue, solution_queue, target_arg, target_function, targets, to_run,
            #                   total_explained)

            selected = []
            selecteds.append(selected)

            to_run = []
            to_runs.append(to_run)

            # take the best 2
            selecteds = selecteds[-2:]
            to_runs = to_runs[-2:]

            targets.clear()
            total_explained = -1
            total_invalid_matched = -1

        if 'select' in line:
            for part in line.split():
                if 'select' in part:
                    invset_id = part.split('select(')[1].split(')')[0]
                    selected.append(invset_id)
        if 'to_run_stronger' in line:
            for part in line.split():
                if 'to_run_stronger' in part:
                    invset_id = part.split('to_run_stronger(')[1].split(')')[0]
                    to_run.append(invset_id)
        if 'to_run_weaker' in line:
            for part in line.split():
                if 'to_run_weaker' in part:
                    invset_id = part.split('to_run_weaker(')[1].split(')')[0]
                    to_run.append(invset_id)
                    # print('execute_clingo (while running) output:', invset_id)
        if 'target' in line:
            for part in line.split():
                if 'target' in part:
                    target_id = part.split('target(')[1].split(')')[0]
                    targets.append(target_id)

        if 'total_actual_explained' in line:
            for part in line.split():
                if 'total_actual_explained' in part:
                    total_explained = int(part.split('total_actual_explained(')[1].split(')')[0])

        if 'total_invalid_unexplained' in line:
            for part in line.split():
                if 'total_invalid_unexplained' in part:
                    total_invalid_matched = int(part.split('total_invalid_unexplained(')[1].split(')')[0])

    final_to_run = []
    for to_run in to_runs:
        for item in to_run:
            if item not in final_to_run:
                final_to_run.append(item)

    print('execute_clingo from', lp_filename, 'output:', final_to_run[:30], 'target', targets[:5])

    add_to_queues(history, inferred_queue, solution_queue, target_arg, target_function, targets, final_to_run,
                  total_explained, total_invalid_matched)
    return final_to_run, targets, total_explained, total_invalid_matched


def create_one_call(API_name, functions_to_sigs,
                    target_invariant_set_for_arg, invariant_sets, values_store_by_id,
                    required_imports,
                    valid_mapping,
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
            if arg_id not in target_invariant_set_for_arg or target_invariant_set_for_arg[arg_id] == -1:
                if arg_id in valid_mapping:
                    if valid_mapping[arg_id] == -10:
                        argument_key = list(argument_keys)[arg_id]
                        default_val = functions_to_sigs[API_name][argument_key]
                        selected_way_ids[arg_id] = -10
                        argument_values[arg_id] = default_val
                    else:
                        previously_valid_invariant_set = invariant_sets[valid_mapping[arg_id]]
                        selected_way_ids[arg_id] = select_way_from_invset(API_name, arg_id, previously_valid_invariant_set)
                        argument_values[arg_id] = values_store_by_id[str(selected_way_ids[arg_id])]
                else:
                    # pick random
                    rand_invariant_set = random.choice(invariant_sets)
                    selected_way_ids[arg_id] = select_way_from_invset(API_name, arg_id,rand_invariant_set)
                    argument_values[arg_id] = values_store_by_id[str(selected_way_ids[arg_id])]
                    # print('choosing a random input (due to uninitialized valid_mapping) arg_id=', arg_id)
                # selected_way_ids[arg_id] = random.sample(list(values_store_by_id.keys()), 1)[0]
                # argument_values[arg_id] = values_store_by_id[selected_way_ids[arg_id]]
                # print('choosing random input (due to target set = -1) arg_id=', arg_id, 'way=', selected_way_ids[arg_id])
                # raise Exception("we don't support this anymore")
                continue

            if target_invariant_set_for_arg[arg_id] == -10:
                # sometimes
                # use default value
                if random.choice([True, False]):
                    argument_key = list(argument_keys)[arg_id]
                    default_val = functions_to_sigs[API_name][argument_key]
                    selected_way_ids[arg_id] = -10
                    argument_values[arg_id] = default_val
                # sometimes
                # random
                else:
                    rand_invariant_set = random.choice(invariant_sets)
                    selected_way_ids[arg_id] = select_way_from_invset(API_name, arg_id, rand_invariant_set)
                    argument_values[arg_id] = values_store_by_id[str(selected_way_ids[arg_id])]
                continue

            matching_ways_id = invariant_sets[target_invariant_set_for_arg[arg_id]][1]

            if len(matching_ways_id) == 0:
                rand_invariant_set = random.choice([inv_set_i for inv_set_i in  range(len(invariant_sets))])
                selected_way_ids[arg_id] = select_way_from_invset(API_name, arg_id,invariant_sets[rand_invariant_set])
                argument_values[arg_id] = values_store_by_id[str(selected_way_ids[arg_id])]

                # print('choosing a random input (due to inability to find matching value) arg_id=', arg_id, 'val=', argument_values[arg_id])
                continue
                # return False

            sampled_matching_way = random.choice(matching_ways_id)
            # print('[trying to pick way from the chosen targets] selected way:', sampled_matching_way, ' for arg_id', arg_id)


            try:
                selected_way_ids[arg_id] = sampled_matching_way
                argument_values[arg_id] = values_store_by_id[str(sampled_matching_way)]
            except Exception as e:
                print('some error when selecting ways', e)
                # pick random
                rand_invariant_set = random.choice(invariant_sets)
                selected_way_ids[arg_id] = select_way_from_invset(API_name, arg_id, rand_invariant_set)
                argument_values[arg_id] = values_store_by_id[str(selected_way_ids[arg_id])]
                print('choosing a random input (due to uninitialized valid_mapping) arg_id=', arg_id)

    print('[after picking ways] selected_way_ids', selected_way_ids)
    test_file_name = 'synthesized_lp_test_' + rand_ident + '_' + API_name + '_' + str(test_num) + '.py'

    full_code = ''
    full_imports = ''
    with open(test_file_name, 'w+') as outfile:

        imports = []
        imports.append('from torch import *\n')
        # gather imports
        for argument_value in argument_values:
            for line in argument_value.split('\n'):
                if 'import ' in line:
                    if 'tensorflow.compiler.test' not in line and 'fuzzingbook_invariant' not in line and '_internal' not in line and '._' not in line:
                        imports.append(line)


        imports.extend(required_imports)

        prev_import = None
        for one_import in sorted(imports):
            if one_import != prev_import:
                outfile.write(one_import + '\n')
                if one_import not in global_all_tests_outfile_imports[API_name]:
                    full_imports += one_import + '\n'
                    global_all_tests_outfile_imports[API_name].add(one_import)
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

                    if i in target_invariant_set_for_arg:
                        outfile.write('# inv id=' + str(target_invariant_set_for_arg[i]) + '\n')
                        full_code += '# inv id=' + str(target_invariant_set_for_arg[i]) + '\n'
                    elif i in valid_mapping:
                        outfile.write('# inv id= valid mapping' + str(valid_mapping[i]) + ' \n')
                        full_code += '# inv id= valid mapping' + str(valid_mapping[i]) + ' \n'
                    else:
                        outfile.write('# inv id= -1 \n')
                        full_code += '# inv id= -1 \n'

                    outfile.write('v_' + str(i) + ' = ' + line + '\n')

                    full_code += '\t' + 'v_' + str(i) + ' = ' + line + '\n'
                    # print('writing: ', 'v_' + str(i) + ' = ' + line + '\n')
                    written_first_line = True
                else:
                    # print('writing:', line + '\n')
                    outfile.write(line + '\n')
                    full_code += line + '\n'

        API_name_to_write = API_name if '__init__' not in API_name else API_name.split('.__init__')[0] + '('
        # outfile.write(API_name_to_write + ','.join(['v_' + str(i) for i, (argument_key, _) in
        #                                             enumerate(zip(argument_keys, argument_values))]) + ')\n')
        # full_code += '\t' + API_name_to_write + ','.join(['v_' + str(i) for i, (argument_key, _) in
        #                                                   enumerate(zip(argument_keys, argument_values))]) + ')\n'
        invocation_str = API_name_to_write

        index_of_first_position_arg = -1
        for i, argument_key in enumerate(argument_keys):
            if argument_key.startswith('arg'):
                index_of_first_position_arg = i
                break

        for i, (argument_key, _) in enumerate(zip(argument_keys, argument_values)):
            if i < index_of_first_position_arg or index_of_first_position_arg == -1:
                param_str =  'v_' + str(i)
            else:
                param_str = ((argument_key + '=') if not argument_key.startswith('arg') else '') + 'v_' + str(i)
            invocation_str += param_str + ','
        invocation_str += ')\n'

        outfile.write(invocation_str)
        full_code += '\t' + invocation_str

        full_code += 'except:\n'
        full_code += '\tpass\n'

        print('=======================')
        print('[created] ', full_code)
        print('=====[end created]=====')
    return True, selected_way_ids, test_file_name, full_code, full_imports


def select_way_from_invset(API_name, arg_id, invset):
    # if global_running_mode[API_name][arg_id] == 1:
    # print(API_name, arg_id, 'returning random way of invset')
    return random.choice(list(invset[1]))
    # else:
    #     print(API_name, arg_id, 'returning fixed way of invset')
    #     return list(invset[1])[0]


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
    invariants_sets = []
    all_invs = set()
    with open('invariant_sets.txt') as infile:
        for line in infile:
            invs_part = line.split(' !!! ')[0]
            ways_part = line.split(' !!! ')[1]
            invariants_set = eval(invs_part)
            ways = eval(ways_part)

            invariants_sets.append((tuple(invariants_set), ways))

            for _, inv, _ in invariants_set:
                all_invs.add(inv)

    return invariants_sets, all_invs

def read_stronger_invsets_of():
    stronger_invsets_of = defaultdict(list)
    with open('lp_invset_stronger.lp') as infile:
        for line in infile:
            if 'stronger' not in line:
                continue

            invset_a_str, invset_b_str = line.split('stronger(')[1].split(').')[0].split(',')
            invset_a, invset_b = int(invset_a_str), int(invset_b_str)

            stronger_invsets_of[invset_b].append(invset_a)

    return stronger_invsets_of

def get_invariant_sets_with_only_uniq_ways(invariants_sets, stronger_invsets_of):
    queue = []

    startings = [i for i in range(len(invariants_sets))]
    for key, stronger_invsets in stronger_invsets_of.items():
        for stronger_invset in stronger_invsets:
            if stronger_invset in startings:
                startings.remove(stronger_invset)

    for starting in startings:
        queue.append((starting, []))

    belongs_to = {}
    while len(queue) > 0:
        invset, acc_ways = queue.pop()
        ways_of_invset = invariants_sets[invset][1]

        next_acc_ways = acc_ways.copy()
        next_acc_ways.extend(ways_of_invset)

        for next_invset in stronger_invsets_of[invset]:
            queue.append((next_invset, next_acc_ways))

        # update ways of invariant_sets
        for way in next_acc_ways:
            belongs_to[way] = invset

    original_ways_of_invsets = defaultdict(list)
    # nuke all ways in invariants_sets
    for invset in range(len(invariants_sets)):
        original_ways_of_invsets[invset] = invariants_sets[invset][1]
        invariants_sets[invset] = (invariants_sets[invset][0], [])

    # add only uniq ways
    for way in belongs_to:
        invset = belongs_to[way]
        invariants_sets[invset][1].append(way)

    # now see how many invariant sets became empty and restore the original ways if they are empty
    for invset in range(len(invariants_sets)):
        if len(invariants_sets[invset][1]) == 0:
            invariants_sets[invset] = (invariants_sets[invset][0], original_ways_of_invsets[invset])


def get_undominated_invariant_set_indexes(invariant_sets):
    indexes = []
    # with open('undominated_invariant_sets.txt') as infile:
    #     for line in infile:
    #         line = line.strip()
    #         indexes.append(int(line))

    for i in range(0, len(invariant_sets)):
        indexes.append(i)
    return indexes


def write_and_run_clingo(target_function, history, target_arg, already_started, solution_queue, inferred_queue, manager_history_cache):
    print('writing clingo', target_function, target_arg)
    already_started.put(True)

    lp_filename = write_clingo(target_function, target_arg, history[target_arg])

    cache_key ='.'.join([str(current_choices) + '_' + outcome for current_choices, outcome in sorted(history[target_arg])])
    if cache_key not in manager_history_cache:

        print('execute clingo', target_function, target_arg, min(180, len(history[target_arg]) + 60))
        to_run, targets, total_explained, total_invalid_matched = execute_clingo(lp_filename, min(180, len(history[target_arg]) + 60),
                                                          history, inferred_queue, solution_queue, target_arg, target_function, )

        manager_history_cache[cache_key] = (to_run, targets, total_explained, total_invalid_matched)
    else:

        to_run, targets, total_explained, total_invalid_matched = manager_history_cache[cache_key]
        print('skipping clingo', target_function, target_arg, 'history len =', len(history), 'answer len is ', len(to_run), 'total explained = ', total_explained)

        add_to_queues(history, inferred_queue, solution_queue, target_arg, target_function, targets, to_run,
                      total_explained, total_invalid_matched)

    already_started.get()

    print('ending write_and_run_clingo',target_function, target_arg )


def add_to_queues(history, inferred_queue, solution_queue, target_arg, target_function, targets, to_run,
                  total_explained, total_invalid_matched):
    # print('[clingo output] to add to qeueu', target_function, target_arg, 'queue len:', solution_queue.qsize())
    for inv_set in to_run:
        # print('putting solution queue', time.time())
        try:
            solution_queue.put(inv_set, block=False)
        except queue.Full:
            print('failed to write to solution queue')

        # print('after putting solution queue', time.time())
    for target in targets:
        # print('putting inferred_queue', time.time())
        try:
            inferred_queue.put((target, (total_explained,
                                     total_invalid_matched,
                                     len(set([current_choices for current_choices, outcome in history[target_arg]])),
                                     len(set([current_choices for current_choices, outcome in history[target_arg] if
                                          outcome == 'valid'])))), block=False)
        except queue.Full:
            print('failed to write to solution queue')
        # print('after putting inferred_queue', time.time())
    # print('[clingo output] added to qeueu', target_function, target_arg, 'queue len:', solution_queue.qsize())
    # print('[clingo output] added to targets', target_function, target_arg, 'inferred_queue len:',
    #       inferred_queue.qsize())


def get_rules(target_function, functions_to_sigs, target_arg,
              solution_queue,inferred_queue,  already_started,
              seeds,
              undominated_invariant_sets,
              history, random_or_solved_choices, generate_random, manager_history_cache, already_ran):

    # pick something from invariant_sets
    target_invariant_set_for_arg = {}

    if target_arg == -1:
        for arg_num in range(len(functions_to_sigs[target_function].keys()) - 1):
            run_clingo_solver_if_running_low(already_started, history, inferred_queue, solution_queue, arg_num, target_function, manager_history_cache)
    else:
        run_clingo_solver_if_running_low(already_started, history, inferred_queue, solution_queue, target_arg, target_function, manager_history_cache)

    # bookkeeping, if there's anything in inferred_queue[target_function], print it
    if inferred_queue[target_arg] is not None and inferred_queue[target_arg].qsize() > 0:
        global_known_targets[target_function][target_arg].clear()
    while inferred_queue[target_arg] is not None and inferred_queue[target_arg].qsize() > 0:
        try:
            found_target, (total_explained, total_invalid_matched, total_inputs1, total_valid_inputs) = inferred_queue[
                target_arg].get(block=False)
        except Exception as e:
            print('unexpected e', e)
            pass

        if found_target not in global_known_targets[target_function][target_arg]:
            global_known_targets[target_function][target_arg].add(found_target)
            print('target spec found for ', target_function, target_arg, found_target, 'size is ',len(global_known_targets[target_function][target_arg]))
            if target_arg not in global_found_target_at[target_function]:
                global_found_target_at[target_function][target_arg] = len(random_or_solved_choices[target_arg])

        global_explained_level[target_function][target_arg] = total_explained/ float(total_valid_inputs) if total_valid_inputs > 0 else 0

        print('computing total explained', target_function, target_arg, 'total_explained=', total_explained, 
            'total_invalid_matched', total_invalid_matched, 'total inputs', total_inputs1, 'total valid inputs', total_valid_inputs)


        matching_precision = float(total_explained) / (total_explained  + float(total_invalid_matched))
        matching_recall = global_explained_level[target_function][target_arg]
        matching_F1 = 2 * matching_precision * matching_recall / (matching_recall + matching_precision)

        print('current metrics', target_function, target_arg, 'matching_F1', matching_F1, 'recall', matching_recall , 'matching_precision', matching_precision, float(total_valid_inputs),  float(total_inputs1), )

        if matching_F1 >= 0.25 and float(total_valid_inputs) >= 1 and float(total_inputs1) >= 10 and matching_recall >= 0.15 and matching_precision >= 0.2:
            print('setting to valid mode', target_function, target_arg, 'matching_F1', matching_F1, 'recall', matching_recall , 'matching_precision', matching_precision, float(total_valid_inputs),  float(total_inputs1), )
            global_running_mode[target_function][target_arg] = 1
        else:
            print('still in search mode', target_function, target_arg, 'matching_F1', matching_F1, 'recall',
                  matching_recall, 'matching_precision', matching_precision, float(total_valid_inputs),
                  float(total_inputs1), )

    # while we say `generate random` in the code, we are not actually generating random inputs,
    # instead, we are guided by coverage
    # it is `random` in that we do not guide it through our input enumeration technique
    # and we allow repeated inputs from the same `category`
    if global_running_mode[target_function][target_arg] == 0 and \
            generate_random:
        num_arguments = len(functions_to_sigs[target_function].keys())

        if target_function in seeds:
            print('[get_rules]' , target_function, ' random from seed', 'generate_random', generate_random)
            if solution_queue[target_arg] is not None:
                print('[get_rules]', target_function, ' random from seed', 'solution_queue[target_function]', solution_queue[target_arg].qsize())
            # pick random seed
            selected_seed = random.choice(seeds[target_function])
            target_invariant_set_for_arg = selected_seed.copy()

            if num_arguments - 1 > 1:
                num_changes = random.randint(1, num_arguments - 1)
                random_args = random.sample([arg_i for arg_i in range(num_arguments)], num_changes)
            else:
                random_args = [0]

            for random_arg in random_args:
                target_invariant_set_for_arg[random_arg] = random.choice(undominated_invariant_sets)
                # target_invariant_set_for_arg[random_arg] = random.choice( invar )
            print('[get_rules]', target_function,target_arg, ' random from seed. choices are ', target_invariant_set_for_arg)

            random_or_solved_choices[target_arg].append('seed')
        else:
            # print('[get_rules]', target_function, ' initial sweep')
            for i, arg_name in enumerate(functions_to_sigs[target_function].keys()):
                # leave blank if there was a default value that we are already aware of
                if 'Any' not in functions_to_sigs[target_function][arg_name]:
                    print('has default value!, ', target_function, arg_name)
                    target_invariant_set_for_arg[i] = -10
                    continue

                if random.choice([True, False]):
                    target_invariant_set_for_arg[i] =  random.choice(undominated_invariant_sets)
                else:
                    # arg_name = list(functions_to_sigs[target_function].keys())[i]
                    # print('checking arg name ', arg_name)
                    if arg_name in global_name_to_invs:
                        # print('found arg name var')
                        target_invariant_set_for_arg[i] = random.choice(global_name_to_invs[arg_name])
                    else:
                        target_invariant_set_for_arg[i] = random.choice(undominated_invariant_sets)

            print('[get_rules]', target_function, target_arg, ' initial sweep. choices are ', target_invariant_set_for_arg)

            random_or_solved_choices[target_arg].append('random')
        return target_invariant_set_for_arg
    elif global_running_mode[target_function][target_arg] == 1: # valid mode
        print('[get_rules]' ,target_function, target_arg, ' pick from valid targets')
        targets = global_known_targets[target_function][target_arg]
        rand_target = random.choice(tuple(targets))
        random_or_solved_choices[target_arg].append('target')
        target_invariant_set_for_arg[target_arg] = int(rand_target)

        return target_invariant_set_for_arg
    else:
        if solution_queue[target_arg].qsize() is None or solution_queue[target_arg].qsize() == 0:
            print('[get_rules]', target_function, target_arg, 'mode:', global_running_mode[target_function][target_arg], ' perturb as [clingo] solution queue size is 0')
            selected_invset = random.choice(undominated_invariant_sets)
            target_invariant_set_for_arg[target_arg] = selected_invset

            random_or_solved_choices[target_arg].append('perturb')
            return target_invariant_set_for_arg
        else:
            print('[get_rules]', target_function, target_arg, 'mode:', global_running_mode[target_function][target_arg], ' pick from [clingo] solution queue. qsize=', solution_queue[target_arg].qsize())
            try:
                to_run = solution_queue[target_arg].get(block=False)
            except Exception as e:
                print('unexpected e', e)
                to_run = str(random.choice(undominated_invariant_sets))
            while to_run in already_ran[target_function][target_arg]:
                if solution_queue[target_arg].qsize() > 0:
                    try:
                        to_run = solution_queue[target_arg].get()
                    except Exception as e:
                        print('unexpected e', e)
                        to_run = str(random.choice(undominated_invariant_sets))
                else:
                    print('[get_rules]', target_function, target_arg, ' but picked random since [clingo] output was already ran before')
                    to_run = str(random.choice(undominated_invariant_sets))

            print('to_run = ', to_run)
            for one_invset in to_run.split('__'):
                for token in one_invset.split('_'):
                    invset = token

                target_invariant_set_for_arg[target_arg] = int(invset)
            random_or_solved_choices[target_arg].append('solution')

            return target_invariant_set_for_arg


def run_clingo_solver_if_running_low(already_started, history, inferred_queue, solution_queue, target_arg, target_function, manager_history_cache):
    if solution_queue[target_arg] is None or solution_queue[target_arg].qsize() < 5:
        if already_started[target_arg].empty():
            print(target_function, target_arg, 'to start clingo solving for the next set of things to run')
            if solution_queue[target_arg] is None:
                queue = multiprocessing.Queue()
                one_inferred_queue = multiprocessing.Queue()
            else:
                queue = solution_queue[target_arg]
                one_inferred_queue = inferred_queue[target_arg]

            p = multiprocessing.Process(target=write_and_run_clingo,
                                        args=(target_function, history, target_arg, already_started[target_arg], queue,
                                              one_inferred_queue, manager_history_cache))
            p.start()

            global_running_solvers.append(p)

            solution_queue[target_arg] = queue
            inferred_queue[target_arg] = one_inferred_queue


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


def find_all_known_types():
    result = set()

    has_seen = set()
    for f in os.listdir(rootdir):
        if not f.endswith('.typedb'):
            continue
        prefix = f.split('.typedb')[0]
        matched_type = prefix.split('__')[1]
        result.add(matched_type)

    return result


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

        for line_i, line in enumerate(infile):
            line = line.strip()
            if line.startswith('==='):
                if function_name is None:
                    print('params', params)
                    print('skipping')
                    continue

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
                    if '.Tensor.' in function_name:
                        params['arg_self'] = 'Any'
    # print(functions_to_sigs)
    return functions_to_sigs, functions_to_imports, crawled_function_arg_to_index


rootdir = os.getcwd()


def get_error_description(stderr_string):
    after_traceback = stderr_string.split('Traceback ')[1]
    description = []
    for line in after_traceback.split('  File '):
        if ' line ' not in line:
            continue
        line_num = line.split(' line ')[1].split(',')[0]
        function_name = line.split(', in  ')[1].split()[0]
        description.append((line_num, function_name))

    error_type = stderr_string.split('Error:')[0].split()[-1]
    related_args = []
    if '`' in stderr_string.split('Error:')[1]:
        for item in stderr_string.split('Error:')[1].split('`')[1::2]:
            related_args.append(item)
    if "'" in stderr_string.split('Error:')[1]:
        for item in stderr_string.split('Error:')[1].split("'")[1::2]:
            related_args.append(item)
    description.append((error_type, tuple(related_args)))

    return tuple(description)

def run_script_server(prev_p = None, delay=1, func=None, port = 65433):
    def wait_for_proc_end(prev_p,delay_bef, delay_aft, func):
        poll = prev_p.poll()
        start_time = time.time()

        time_printed = None
        while poll is None:
            # time.sleep(1)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            if time_elapsed > delay_bef:
                break
            poll = prev_p.poll()
            if time_printed is None or time_elapsed - time_printed > 1:
                print('run_script_server: poll again. time passed=', time_elapsed)
                time_printed = time_elapsed
        if poll is None:
            print('poll is None -> prev_p is still running')
            if func is not None:
                print('running func')
                func()
            time.sleep(delay_aft)
        else:
            print('poll is not None -> prev_p is not running')

    # time.sleep(1)
    if prev_p is not None:
        print('wait for/kill previous instance first')

        wait_for_proc_end(prev_p, 1, 1, func=lambda: prev_p.terminate())
        wait_for_proc_end(prev_p, 1, 1, func=lambda: prev_p.kill())

    print('starting server on port', port)


    # p1 = subprocess.Popen(['coverage', 'run', '--source=/usr/local/lib/python3.8/dist-packages/tensorflow/', '--append', '--data-file=' + rand_ident + '_fuzz.coverage', '-m', 'test_executor', str(port)])

    p1 = subprocess.Popen(['python3', '5a_test_executor.py', str(port)])
    # print('waiting for', delay, 'seconds')
    # time.sleep(delay)

    return p1


def add_to_history(current_choices, history, target_function, outcome):
    if current_choices != "":
        inconsistent_results = False
        for previous_choice, previous_outcome in history:
            if current_choices == previous_choice:
                if previous_outcome != outcome:
                    inconsistent_results = True
        if inconsistent_results:
            history.append((current_choices, 'invalid'))
        else:
            history.append((current_choices, outcome))


def create_one_test_and_run(target_function, all_functions,
                            all_function_sigs, invariant_sets, undominated_invariant_sets,
                            func_to_already_checked_invs, previous_coverage,
                            target_arg, solution_queue,inferred_queue, already_started,
                            history,
                            random_or_solved_choices,
                            seeds,
                            values_store_by_id, functions_to_imports,
                            s, valid_mapping,
                            manager_history_cache,
                            all_outfile,
                            already_ran,
                            test_i):

    global num_crashes

    is_no_problemo_runs = False
    is_problem = False
    is_py_error = False
    is_crash = False

    generate_random = target_arg == -1

    target_invariant_set_for_arg = get_rules(target_function, all_function_sigs, target_arg,
                                             solution_queue[target_function], inferred_queue[target_function],
                                             already_started[target_function],
                                             seeds,
                                             undominated_invariant_sets,
                                             history[target_function], random_or_solved_choices[target_function],
                                             generate_random, manager_history_cache, already_ran)
    succeeded, selected_way_ids, test_file_name, full_code, full_imports = create_one_call(
        target_function, all_function_sigs,
        target_invariant_set_for_arg, invariant_sets,
        values_store_by_id,
        functions_to_imports[target_function],
        valid_mapping,
        test_i)
    if not succeeded:
        print('removing', target_function)
        all_functions.remove(target_function)
        print(len(all_functions), 'APIs remain')
        return False

    # if target_function not in func_to_already_checked_invs:
    #     func_to_already_checked_invs[target_function] = set()

    data = ''

    all_outfile.write(full_imports)

    try:
        print('[test builder] sending ', test_file_name)
        s.sendall(test_file_name.encode('utf-8'))
        s.sendall('=='.encode('utf-8'))
        start_time = time.time()
        while data is None or not data.endswith('rann'):
            s.settimeout(10)
            data_raw = s.recv(1024)
            try:
                data += data_raw.decode('utf-8')
            except Exception as e:
                print('exception caught while decoding data from the script-server', e)
                pass
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            # if len(data) < 200:
            #     print('[test builder] after receiving, current message:', data, 'recv len', len(data_raw))
            # else:
            #     print('[test builder] after receiving, current message too long,  recv len', len(data_raw))
            # print('randstr', ''.join(random.choices(string.ascii_uppercase + string.digits, k=3)), time_elapsed)

            if time_elapsed > 10 and len(data_raw) == 0:
                print('too much time elasped', time_elapsed)
                data = 'Server down!'
                break
    except ConnectionRefusedError as e:
        print('run-script server down!')
        print(e)
        raise e
    except ConnectionResetError as e:
        data = 'Server down!'
        print('run-script server is down. Exception:')
        print(e)
    except socket.timeout as e:
        detected_crash(test_file_name, test_i)
        print('timeout detected?')
        print('exception = ', e)
        raise e
    except BrokenPipeError as e:

        detected_crash(test_file_name, test_i)
        print('crash detected?')
        print('exception = ', e)
        raise e

    # print("[test builder] Received", data)

    all_outfile.write(full_code)

    data = data.split('rann')[0]

    if len(data) > 0:
        try:
            coverage_num = int(data.split('===')[1].strip())
            data = data.split('===')[0].strip()
            try:
                data =int(data)
            except:
                pass

            print('current cov', coverage_num)

            all_outfile.write('\n\n')
            all_outfile.write("# data=" + str(data).replace('\n', '\n#') + '\n')
            all_outfile.write('# coverage: ' + str(coverage_num) + '\n')
            all_outfile.write('# made by: ' + str(random_or_solved_choices[target_function][target_arg][-1]) + '\n')
            all_outfile.write('\n\n')

            if coverage_num > previous_coverage[target_function]:
                previous_coverage[target_function] = coverage_num
                seeds[target_function].append(target_invariant_set_for_arg.copy())
                seeds[target_function] = seeds[target_function][-5:]

        except Exception as e:
            print('caught e while trying to parse coverage', e)
            print('assuming crashed? Is it true?')

    if data != 0 and 'NameError' in data:
        # copied_fn = 'problem_test_' + str(test_i) + '.py'
        # print('copying to ', copied_fn)
        # shutil.copy(test_file_name, copied_fn)
        #
        # with open(copied_fn, 'a') as outfile:
        #     outfile.write('# problem was ' + str(data))
        is_problem = True

    elif data != 0 and data == 'Server down!':
        detected_crash(test_file_name, test_i)
        print('num crashes', num_crashes)
        if num_crashes >= 50000:
            return None
        else:
            pass
            # run_script_server()
        is_crash = True

    else:
        if data == 0:
            print('successful run')
            is_no_problemo_runs = True
        else:
            if data != 0 and 'Error:' in data:
                is_py_error = True
                # try:
                #     error_description = get_error_description(data)
                #     error_outcome_identifier = ','.join(error_description)
                # except Exception as e:
                #     print('found err', e, 'from', data)
                #     # raise Exception('err on ' + data)

            else:
                print('unknown', 'data=', data[:200])

    if is_no_problemo_runs:
        outcome = 'valid'

        if len(valid_mapping) == 0:
            print('setting valid mapping', target_function)
            for key, val in target_invariant_set_for_arg.items():
                valid_mapping[key] = val

        if target_arg == -1:
            for key in target_invariant_set_for_arg.keys():
                # seed history with the valid call.
                print('[calling add to history] ', 'target_function', target_function , 'key(for target arg=-1)', key, 'target inv set',
                      target_invariant_set_for_arg[key], 'outcome', outcome)
                add_to_history(target_invariant_set_for_arg[key], history[target_function][key],
                               target_function, outcome)
                add_to_history(target_invariant_set_for_arg[key], history[target_function][key],
                               target_function, outcome)

    elif is_crash:
        outcome = 'crash'
    else:
        outcome = 'invalid'

    if target_arg != -1:
        print('[calling add to history] ','target_function', target_function, 'target arg', target_arg, 'target inv set', target_invariant_set_for_arg[target_arg], 'outcome', outcome)
        add_to_history(target_invariant_set_for_arg[target_arg], history[target_function][target_arg], target_function, outcome)

        random_or_solved_choices[target_function][target_arg][-1] = random_or_solved_choices[target_function][target_arg][-1] + "_" + outcome

        # debugging

        if random_or_solved_choices[target_function][target_arg][-1] == 'target_invalid':
            shutil.copy(test_file_name, 'debug_' + rand_ident + '_invalid_' + target_function + '_' + str(target_arg) + '.py')
        elif random_or_solved_choices[target_function][target_arg][-1] == 'target_valid':
            shutil.copy(test_file_name, 'debug_' + rand_ident + '_valid_' + target_function + '_' + str(target_arg) + '.py')

    if is_crash:
        shutil.copy(test_file_name,
                    'ran_tests/' + 'test_' + target_function + '_' + str(test_i) + '_' + outcome[:30] + '.py')

    os.remove(test_file_name)

    return is_crash, is_no_problemo_runs, is_problem, is_py_error, is_crash, outcome

def detected_crash(test_file_name, test_i):
    global num_crashes
    global last_crash_index
    print('!!!!!!!detected crash!!!!!!')

    if test_i > 0:
        crash_ident = ''.join(random.choice(string.ascii_lowercase) for i in range(3))

        copied_fn = 'crashed_test_' + test_file_name + '_' + crash_ident +'.py'
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
    global global_name_to_invs
    all_function_sigs, functions_to_imports, function_arg_to_index = read_API_function_sigs()

    crawled_function_sigs, crawled_functions_to_imports, crawled_function_arg_to_index = find_all_known_functions()
    all_function_sigs.update(crawled_function_sigs)
    functions_to_imports.update(crawled_functions_to_imports)
    function_arg_to_index.update(crawled_function_arg_to_index)
    all_functions = set(crawled_function_sigs.keys())  # TODO

    print('reading invsets')
    invariant_sets, all_invs = get_invariant_sets()
    undominated_invariant_sets = get_undominated_invariant_set_indexes(invariant_sets)

    print('preprocess invsets')
    stronger_invsets_of = read_stronger_invsets_of()
    # clean up the ways in invariant set to keep only uniq ways
    get_invariant_sets_with_only_uniq_ways(invariant_sets, stronger_invsets_of)


    global_name_to_invs = find_function_seeds(invariant_sets)

    print('reading usable values')
    values_store_by_id, values_store_by_type = read_typedb_cached_file_with_id()
    # values_store = read_typedb_cached_file()
    # values_store['NoneType'] = ['None']


    all_types = find_all_known_types()
    usable_types = set()
    for one_type in all_types:
        if one_type in values_store_by_type and len(values_store_by_type[one_type]) > 0:
            usable_types.add(one_type)

    # functions that do not have arguments do not need to be tested
    all_functions = set([f for f in all_functions if len(all_function_sigs[f].keys()) > 0])
    all_functions = sorted(list(all_functions))

    print('total of ', len(all_functions), 'functions')
    try:
        func_to_already_checked_invs = {}

        start_time = time.time()

        if not os.path.exists('ran_tests'):
            os.mkdir('ran_tests')

        # target_functions = random.sample(all_functions, 1000)
        # target_functions.insert(0, 'tf.math.bincount(')

        target_functions = all_functions
        if os.path.exists(target_api_file):
            target_functions = []
            with open(target_api_file) as infile:
                for line in infile:
                    function_name = line.strip() + '('
                    target_functions.append(function_name)
            print('[found target_api_file] targeting a total of ', len(target_functions), 'functions')

        print('taking ', start_from, 'to ', end_at)
        target_functions = target_functions[start_from:end_at]


        # filter things with `out`
        filtered_target_functions = []
        for target_function in target_functions:
            # if 'out' in all_function_sigs[target_function].keys():
            #     print('filtering out ', target_function)
            #     continue
            filtered_target_functions.append(target_function)


        target_functions = filtered_target_functions

        # with Pool(1) as pool:
        #     # with Pool(1) as pool:
        #     args_to_run = []
        #     selected_target_functions = target_functions
        #     args_to_run.append((all_function_sigs, all_functions, func_to_already_checked_invs,
        #                         function_arg_to_index, functions_to_imports, invariant_sets,
        #                         selected_target_functions, usable_types, values_store_by_id,
        #                         values_store_by_type))
        #
        #     pool.map(create_tests_for_the_target_functions, args_to_run)

        selected_target_functions = target_functions
        print('len of selected_target_functions', len(selected_target_functions))
        create_tests_for_the_target_functions((all_function_sigs, all_functions, func_to_already_checked_invs,
                                 function_arg_to_index, functions_to_imports, invariant_sets, undominated_invariant_sets,
                                 selected_target_functions, usable_types, values_store_by_id,
                                 values_store_by_type))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print('caught exception', e)
        # print('should be flushed')
        raise e

    print('run metadata')
    print('elapsed time --- %s seconds ---' % (time.time() - start_time))



def create_tests_for_the_target_functions(arguments):
    all_function_sigs, all_functions, func_to_already_checked_invs, \
    function_arg_to_index, functions_to_imports, invariant_sets, undominated_invariant_sets, \
    target_functions, usable_types, values_store_by_id, values_store_by_type = arguments

    HOST = "127.0.0.1"
    PORT1 = 65433
    all_ports = [PORT1,]
    for port_i in range(65435, 65473, 2):
        all_ports.append(port_i)

    # prepare the script-execution server
    print('starting servers')
    script_server_proc = None


    script_server_procs = {}
    for port in all_ports:
        script_server_proc = run_script_server(None, delay=1, port=port)
        script_server_procs[port] = script_server_proc

    conn_retry_count = defaultdict(int)

    run_outcomes = {}
    valid_mappings = {}
    crashes_for_target_func = {}
    solution_queue = {}
    inferred_queue = defaultdict(list)
    already_started = {}
    previous_coverage = defaultdict(int)
    history = defaultdict(lambda: defaultdict(list))
    random_or_solved_choices = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(list)

    already_ran = defaultdict(lambda: defaultdict(set))

    # for stats
    found_valid_at = {}

    remaining_function_indexes = [target_function_i for target_function_i, func1 in enumerate(target_functions) if func1 in all_function_sigs]

    # clean generated file from previous runs
    for target_function_i, target_function in enumerate(target_functions):
        name = get_test_file_record_filename(target_function)
        if os.path.exists(name):
            print('removing generated file from previous run: ', name)
            os.remove(name)

    manager = Manager()
    manager_history_cache = manager.dict()


    port = PORT1
    while len(remaining_function_indexes) > 0:
        # pick N target functions
        # if len(manager_history_cache.keys()) < 3: # fresh run
        N = 4
        # else:
        #     N = 8 # go faster if there is cached history
        if len(remaining_function_indexes) >= N:
            selected_functions_indexes = random.sample(remaining_function_indexes, N)
        else:
            selected_functions_indexes = remaining_function_indexes[:]

        # update remaining_function_indexes
        for selected_function_index in selected_functions_indexes:
            selected_function = target_functions[selected_function_index]
            remaining_function_indexes.remove(selected_function_index)

            # and setup bookkeeping files
            run_outcomes[selected_function] = []
            valid_mappings[selected_function] = {}
            print('setup bookkeeping (e.g. valid_mappings)', selected_function)
            crashes_for_target_func[selected_function] = 0
            solution_queue[selected_function] = defaultdict(lambda: None)
            inferred_queue[selected_function] = defaultdict(lambda: None)
            already_started[selected_function] = defaultdict(lambda: multiprocessing.Queue()) #

            found_valid_at[selected_function] = -1


        # construct the sequence of target_functions
        target_functions_sequence = []
        for seq_item_i in range(NUM_TESTS_PER_FUNC):
            for selected_function_index in selected_functions_indexes:
                selected_function = target_functions[selected_function_index]
                target_functions_sequence.append(selected_function)

        to_skip = set() # when we reach a certain level of confidence of an API, we can skip it

        is_done = False
        while not is_done:
            target_function_i = 0
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.connect((HOST, port))
                    except (ConnectionRefusedError, socket.timeout) as e:
                        print('socket connection failed:')
                        print(e)
                        # just retry a few times
                        conn_retry_count[target_function] += 1
                        if conn_retry_count[target_function] > 20:
                            raise Exception('script server issue: could not connect')
                        print('starting server again for target_function', target_function)
                        script_server_procs[port] = run_script_server(script_server_procs[port],
                                                                      delay=max(10 * conn_retry_count[target_function],
                                                                                10 * 3), port=port)

                    while target_function_i < len(target_functions_sequence):
                        target_function = target_functions_sequence[target_function_i]
                        print('=================')
                        print('[seq info] testing ', target_function, 'out of', len(target_functions_sequence),'target_function_i', target_function_i, )
                        # try:
                        #     print('[seq info] after this is ', target_functions_sequence[target_function_i + 1])
                        # except:
                        #     pass

                        name = get_test_file_record_filename(target_function)
                        num_args = len(all_function_sigs[target_function].items()) if target_function in all_function_sigs else 0
                        if num_args== 0:
                            print('[seq info] num args is 0, no need to test?', target_function)
                            target_function_i += 1
                            continue

                        valid_mapping = valid_mappings[target_function]
                        with open(name, 'a') as all_outfile:

                            test_i = len(run_outcomes[target_function])

                            if len(valid_mapping) == 0:
                                target_arg = -1
                            else:
                                # target_arg = random.randint(0, num_args - 1)
                                # favour args that are not 'target' yet, and have lots of things in the solution_queue
                                weights = []
                                for arg_i in range(0, num_args):
                                    # if list(all_function_sigs[target_function].keys())[arg_i] == 'out':
                                    #     weights.append(0)
                                    #     continue
                                    if global_running_mode[selected_function][arg_i] != 1:
                                        if solution_queue[selected_function][arg_i] is not None and solution_queue[selected_function][arg_i].qsize()  > 0:
                                            weights.append(
                                                min(
                                                solution_queue[selected_function][arg_i].qsize(),
                                                    50
                                                ))
                                        else:
                                            weights.append(2)
                                    else:
                                        weights.append(0)

                                min_weight = min(weights)

                                for arg_i in range(0, num_args):
                                    if global_running_mode[selected_function][arg_i] == 1:
                                        weights[arg_i] = min_weight // 2


                                args = [arg_i for arg_i in range(0, num_args)]
                                target_arg = random.choices( args, weights, k= 1 )[0]

                            print('target', target_function, 'target_arg', target_arg, 'out of num_args=', num_args, 'valid mapping size=', len(valid_mapping))

                            is_crash, is_no_problemo_runs, is_problem, is_py_error, is_crash, outcome = create_one_test_and_run(
                                target_function, all_functions,
                                all_function_sigs, invariant_sets, undominated_invariant_sets,
                                func_to_already_checked_invs, previous_coverage,
                                target_arg, solution_queue, inferred_queue, already_started,
                                history,
                                random_or_solved_choices,
                                seeds,
                                values_store_by_id, functions_to_imports,
                                s, valid_mapping,
                                manager_history_cache,
                                all_outfile,
                                already_ran,
                                test_i)
                            print('outcome from the outside:', outcome, '\t for function', target_function, 'i=', test_i)
                            run_outcomes[target_function].append(outcome)

                            # stats
                            if len(valid_mapping) > 0 and found_valid_at[target_function] == -1:
                                print('found sett valid mapping. update has valid?', valid_mapping)
                                found_valid_at[target_function] = test_i

                            print('valid_mapping', target_function, ':', valid_mapping)


                        target_function_i += 1

                    s.sendall('done=='.encode('utf-8'))
                    if target_function_i >= len(target_functions_sequence):
                        print('set is done to True', 'target_function_i', target_function_i, 'len(target_functions_sequence)', len(target_functions_sequence))
                        is_done = True
            except (BrokenPipeError, socket.timeout):
                if last_crash_index != 0:  # if it's zero, we're dealing with some error starting the script-server
                    crashes_for_target_func[target_function] += 1
                if crashes_for_target_func[target_function] >= 1:
                    print('stop testing', target_function, 'since crashed', crashes_for_target_func[target_function], 'times')
                    # try:
                    #     # s.sendall('done=='.encode('utf-8'))
                    #     print('send done! from error path')
                    # except:
                    #     print('tried sending done, but failed. from error path')

                    # break
                    indexes_to_clear = []
                    print('[seq info] clearing unwanted function', target_function)
                    for to_clear_target_function_i in range(target_function_i, len(target_functions_sequence)):
                        # target_functions_sequence[target_function_i]
                        if target_functions_sequence[to_clear_target_function_i]  == target_function:
                            indexes_to_clear.append(to_clear_target_function_i)
                    for index_to_clear in sorted(indexes_to_clear, reverse=True):
                        del target_functions_sequence[index_to_clear]


                print('detected server down')
                print('total tests ran for func=', target_function, " times=", len(run_outcomes[target_function]))
                # print('===stat===')
                print_outcomes(run_outcomes[target_function])

                script_server_procs[port] = run_script_server(script_server_procs[port], delay=2, port = port)

                # use the other port while the server on this port is restarting
                port = all_ports[(all_ports.index(port) + 1) % len(all_ports)]

                print('should have restarted server from the broken pipe error')
            except Exception as e:
                print('crazy, unexpected exception ):', e)
                traceback.print_exc()
                print('==============!!!!!!!!!!!!!!!================')

                target_function_i += 1

            print_outcomes(run_outcomes[target_function], target_function,
                           'outcomes_' + rand_ident + '_' + target_function + '.txt')
            print('===stat===')

            # empty data structures that are not used further
            for selected_function_index in selected_functions_indexes:
                selected_function = target_functions[selected_function_index]
                global_all_tests_outfile_imports[selected_function].clear()



    print_output_stats(found_valid_at, invariant_sets, random_or_solved_choices)


def print_output_stats(found_valid_at, invariant_sets, random_or_solved_choices):
    with open('valid_reached_' + rand_ident + '.txt', 'w+') as outfile:
        print('has valid (by chance)?')
        for key, reached_valid_at in found_valid_at.items():
            print('\t', '[has valid?]', rand_ident, 'key', key, reached_valid_at)
            outfile.write(key + ': ' + str(reached_valid_at) + '\n')

    with open('random_or_solved_choices_' + rand_ident + '.txt', 'w+') as outfile:
        print('random_or_solved_choices (seq)?')
        for key, arg_to_seq in random_or_solved_choices.items():
            for arg, seq in arg_to_seq.items():
                print('\t', '[total choices]', len(seq))
                outfile.write(key + ' : ' + str(arg) + ' : ' + ','.join(item for item in seq) + '\n')
            outfile.write('\n\n')

    # for key, arg_to_set in global_known_targets.items():
    #     with open('known_targets_' + rand_ident + '_' + key + '.txt', 'w+') as outfile:
    #         for arg, one_set in arg_to_set.items():
    #             outfile.write('=================\n')
    #             outfile.write('arg' + str(arg) + ":")
    #             for target_inv in one_set:
    #                 # print('for tar
    #                 # t_sets[int(target_inv)][0])
    #                 outfile.write( str(target_inv) + ' : ' + ', '.join(
    #                     [str_version for str_version, int_version, _ in invariant_sets[int(target_inv)][0]]) + '\n')
    #             outfile.write('\n')
    #             outfile.write('\n')

    # for key, arg_to_bool in global_running_mode.items():
    #     with open('running_mode_' + rand_ident + '_' + key + '.txt', 'w+') as outfile:
    #         for arg, one_boolean in arg_to_bool.items():
    #             outfile.write('=================\n')
    #             outfile.write('arg' + str(arg) + ":")
    #             outfile.write(str(one_boolean) + '\n')
    #             outfile.write('\n')
    #             outfile.write('\n')

    # for key, arg_to_int in global_found_target_at.items():
    #     with open('found_target_' + rand_ident + '_' + key + '.txt', 'w+') as outfile:
    #         for arg, one_int in arg_to_int.items():
    #             outfile.write('=================\n')
    #             outfile.write('arg' + str(arg) + ":")
    #             outfile.write(str(one_int) + '\n')
    #             outfile.write('\n')
    #             outfile.write('\n')


def find_function_seeds(invariant_sets):
    argname_to_invsets = defaultdict(list)

    invariant_map = {inv_set[0] : inv_set_index  for inv_set_index, inv_set in enumerate(invariant_sets)}
    with open('func_seeds.txt') as infile:
        for line in infile:
            line = line.strip()


            argname = line.split(':[')[0].split(';;')[1]
            invs = '[' + line.split(':[')[1].split('!!!')[0]
            way_id = line.split('!!!')[1]

            try:
                argname_to_invsets[argname].append(invariant_map[tuple(eval(invs))])
            except Exception as e:
                print(invs)
                print(line)
                raise e


    return argname_to_invsets

def get_test_file_record_filename(target_function):
    all_test_file_name = 'synthesized_all_lp_test_' + rand_ident + '_'+ target_function + '.py'
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

# clean
print('killing stray processes')
print('!!!!')
for grs in global_running_solvers:
    grs.join(timeout=1.0)
    grs.kill()
print('end')
