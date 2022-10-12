import itertools
import os
import subprocess
from collections import defaultdict
import random
import shutil
import socket
import string
import sys

from pytorch_fuzzingbook_invariant_utils import INVARIANT_PROPERTIES_TYPES, INVARIANT_PROPERTIES
import sqlite3
import time

from multiprocessing import Pool, cpu_count
import string
from operator import itemgetter


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



def main(invariant_sets):
    invset_contains = defaultdict(set)
    for index, invariant_set_one_arg in enumerate(invariant_sets):

        for thing in invariant_set_one_arg[0]:
            invset_contains[index].add(thing[1])

    invset_stronger_than = defaultdict(set)

    for invset_a in invset_contains.keys():
        for invset_b in invset_contains.keys():
            if all([(a in invset_contains[invset_b]) for a in invset_contains[invset_a]]):
                invset_stronger_than[invset_b].add(invset_a)

    distance = defaultdict(lambda : defaultdict(int))
    for invset_a in invset_contains.keys():
        for invset_b in invset_contains.keys():
            dist =len(invset_contains[invset_a] - invset_contains[invset_b]) + len(invset_contains[invset_b] - invset_contains[invset_a])
            distance[invset_a][invset_b] = dist
            distance[invset_b][invset_a] = dist

    with open('lp_invset.lp', 'w+') as outfile:
        for invset_a  in invset_contains.keys():
            outfile.write('invset(' + str(invset_a) + ').\n')

    # with open('invset_stronger.txt', 'w+') as outfile1,\
    #     open('lp_invset_stronger.lp', 'w+') as outfile2:
    #
    #     weakest = defaultdict(list)
    #     weakest_distance = defaultdict(lambda: 1000)
    #     for invset_a, stronger_than in invset_stronger_than.items():
    #         strongest = []
    #         strongest_distance = 1000
    #         for invset_b in stronger_than:
    #             if invset_a == invset_b:
    #                 continue
    #             if distance[invset_a][invset_b] < strongest_distance:
    #                 strongest_distance = distance[invset_a][invset_b]
    #                 strongest.clear()
    #                 strongest.append(invset_b)
    #             elif distance[invset_a][invset_b] == strongest_distance:
    #                 strongest.append(invset_b)
    #
    #             if distance[invset_a][invset_b] < weakest_distance[invset_b]:
    #                 weakest_distance[invset_b] = distance[invset_a][invset_b]
    #                 weakest[invset_b].clear()
    #                 weakest[invset_b].append(invset_a)
    #             elif distance[invset_a][invset_b] == weakest_distance[invset_b]:
    #                 weakest[invset_b].append(invset_a)
    #
    #         for invset_b in strongest:
    #             outfile1.write(str(invset_a) + ' > ' + str(invset_b) + '\n')
    #             outfile2.write('stronger(' + str(invset_a) + ',' + str(invset_b) + ').\n')
    #
    #     for invset_b, invset_as in weakest.items():
    #         for invset_a in invset_as:
    #             outfile1.write(str(invset_a) + ' > ' + str(invset_b) + '\n')
    #             outfile2.write('stronger(' + str(invset_a) + ',' + str(invset_b) + ').\n')



    with open('invset_distance.txt', 'w+') as outfile1,\
            open('lp_invset_distance.lp', 'w+') as outfile2:
        for invset_a, invsets_b in distance.items():
            for invset_b in invsets_b:
                outfile1.write(str(invset_a) + ' ' + str(invset_b) +  ':' + str(distance[invset_a][invset_b]) + '\n')
                outfile2.write('distance(' + str(invset_a) + ',' + str(invset_b) +  ',' + str(distance[invset_a][invset_b]) + ').\n')

    # solve for:
    # 1. we want a fully connected graph
    #   all invsets to be connected
    # 4. minimize the total distance on the edges on the links

    lp = """
    0 < { stronger(A, B) : invset(A), invset(B) } < 5000.
    
    :- stronger(A, A).
        
    connected(A) :- stronger(0, A).
    connected(A) :- stronger(A, 0).
    connected(B) :- connected(A), stronger(A, B).
    connected(B) :- connected(A), stronger(B, A).

    all_connected :- connected(A) : invset(A).

connected(0).

not_connected(A) :- not connected(A), invset(A).

    % :- not all_connected.
#maximize { A@2 : connected(A) }.

    :- stronger(A, B), not stronger_than(A, B, _).

    #minimize { D@1 : stronger(A,B),stronger_than(A,B, D) }.

    #show stronger/2.
#show not_connected/1.
    """

    for invset_a, stronger_than in invset_stronger_than.items():
        for invset_b in stronger_than:
            if invset_a == invset_b:
                continue
            lp += 'stronger_than(' + str(invset_a) + ',' + str(invset_b) + ',' + str(distance[invset_a][invset_b]) + ').\n'

    with open('query_for_connected_graph.lp', 'w+') as outfile:
        outfile.write(lp)

    result = subprocess.Popen(
        ['timeout', '6h', 'clingo', 'lp_invset.lp',
         'query_for_connected_graph.lp'], stdout=subprocess.PIPE)

    strongers = []
    for line in result.stdout:
        line = line.decode('utf-8')

        if line.startswith('Answer'):
            # print('one answer', 'prev answer was of length', len(strongers))
            with open('lp_invset_stronger_tmp.lp', 'w+') as outfile:
                for stronger in strongers:
                    outfile.write(stronger + '.\n')

            # reset
            strongers = []

        if 'stronger' in line:
            for part in line.split():
                strongers.append(part)

    with open('lp_invset_stronger.lp', 'w+') as outfile:
        for stronger in strongers:
            outfile.write(stronger + '.\n')


invariant_sets, all_invs = get_invariant_sets()
main(invariant_sets)

print('done 4a_preprocess.py')
