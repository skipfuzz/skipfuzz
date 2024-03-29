import sqlite3
import os
import subprocess
from collections import defaultdict
import random
import shutil

from pytorch_fuzzingbook_invariant_utils import INVARIANT_PROPERTIES, INVARIANT_PROPERTIES_TYPES

INV_EXTRACTION = 'pytorch_fuzzingbook_invariant_utils.get_invariants_hold(pytorch_fuzzingbook_invariant_utils.INVARIANT_PROPERTIES,'

rootdir = os.getcwd()

def get_all_invariant_sets():
    result = []
    invariants_sets = []
    with open('invariant_sets.txt') as infile:
        for i, line in enumerate(infile):
            if len(line.strip()) == 0:
                print('skipping line', i)
                continue
            invs_part = line.split(' !!! ')[0]
            ways_part = line.split(' !!! ')[1]
            invariants_set = eval(invs_part)
            ways = eval(ways_part)

            invariants_sets.append((tuple(invariants_set), ways))
    return invariants_sets


invariant_sets = get_all_invariant_sets()


invariant_type_comparison_direction = {}
for prop_bef, _ in INVARIANT_PROPERTIES.items():
    if '>' in prop_bef and '<' not in prop_bef:
        invariant_type_index = INVARIANT_PROPERTIES_TYPES.index(prop_bef)
        invariant_type_comparison_direction[invariant_type_index] = False
    elif '<' in prop_bef and '>' not in prop_bef:
        invariant_type_index = INVARIANT_PROPERTIES_TYPES.index(prop_bef)
        invariant_type_comparison_direction[invariant_type_index] = True


def is_dominated(invs1, invs2, target_invariant_type, invariant_type_comparison_direction):
    # is invs2 dominated by invs1?

    mapping1 = {}
    for inv1 in invs1:
        readable_invariant1, invariant_type1, invariant_value1 = inv1
        mapping1[invariant_type1] = invariant_value1

    for inv2 in invs2:
        try:
            readable_invariant2, invariant_type2, invariant_value2 = inv2
        except Exception as ve:
            print(inv2)
            print('of', invs2)
            continue
        if target_invariant_type != invariant_type2:
            continue

        if invariant_type2 not in mapping1:
            return False  # invs2 can't be dominated since it has some invariant that invs1 does not

        if mapping1[invariant_type2] is None and invariant_value2 is None:
            continue

        if invariant_type2 not in invariant_type_comparison_direction:
            continue # no evidence that invs2 is or is not dominated. If this branch is reached, inv1 and inv2 share this invariant

        reverse_direction_check = not invariant_type_comparison_direction[invariant_type2]
        try:
            if ((mapping1[invariant_type2] > invariant_value2) and not reverse_direction_check) or \
                    ((mapping1[invariant_type2] < invariant_value2) and reverse_direction_check):
                # no evidence to suggest not dominated
                pass
            else:
                print('inv2 dominated by inv1 as', readable_invariant2, 'has a value dominated by', mapping1[invariant_type2])
                return False
        except TypeError as e:
            return False

    return True




nondominated_invariant_sets = {}

for invariant_type_id, _ in enumerate(INVARIANT_PROPERTIES_TYPES):
    print('checking' , invariant_type_id)
    nondominated_invariant_sets[invariant_type_id] = defaultdict(list)
    for invset_index, (invariant_set, ways) in enumerate(invariant_sets):
        is_dom = False
        for one_undominated_set in nondominated_invariant_sets[invariant_type_id]:
            is_dom |= is_dominated(invariant_sets[one_undominated_set][0],
                                   invariant_set, invariant_type_id, invariant_type_comparison_direction)
            if is_dom:
                break

        if not is_dom:
            print('adding for ', invariant_type_id)
            nondominated_invariant_sets[invariant_type_id][invset_index] = ways

    to_remove = set()
    for i, invset_index1 in enumerate(nondominated_invariant_sets[invariant_type_id].keys()):
        for j, invset_index2 in enumerate(nondominated_invariant_sets[invariant_type_id].keys()):
            if i == j:
                continue
            one_set = invariant_sets[invset_index1][0]
            another_set = invariant_sets[invset_index2][0]
            if is_dominated(another_set, one_set, invariant_type_id, invariant_type_comparison_direction):
                to_remove.add(invset_index1)

    for item_to_remove in to_remove:
        nondominated_invariant_sets[invariant_type_id].pop(item_to_remove, None)



with open('undominated_invariant_sets.txt', 'w+') as outfile:
    written = set()
    for invset_index_to_ways in nondominated_invariant_sets.values():
        for key in invset_index_to_ways.keys():
            if key not in written:
                outfile.write(str(key) + '\n')
                written.add(key)