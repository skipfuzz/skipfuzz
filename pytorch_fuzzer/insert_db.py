# parse the typedb file and insert into sqlite

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
        with open(f) as infile:
            way = ''

            invariants = []
            way_id = None
            for line in infile:
                line = line.strip()
                if not line.startswith('===') and not line.startswith('#'):
                    if line.startswith('id:'):
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
                                    way = way[:-1]
                                ways.append((way_id, way, invariants))

                    way = ''
                    way_id = None
                    invariants = []

        result[matched_type] = ways

    return result



types_to_ways_invs = read_typedb_cached_file()


con = sqlite3.connect('ways.db')
cur = con.cursor()



cur.execute('''CREATE TABLE IF NOT EXISTS ways_invs 
               (way_id int, invariant_type int,  invariant_value text) ''')
con.commit()


types_with_no_values = set()
for way_type, id_ways_invs in types_to_ways_invs.items():
    for (way_id, way, invs) in id_ways_invs:
        for readable_invariant, invariant_type, invariant_value in invs:

            if invariant_value is None:
                types_with_no_values.add(invariant_type)
                cur.execute("INSERT INTO ways_invs VALUES(" + str(way_id) + "," + str(invariant_type) + "," + "NULL )")
            else:
                print("INSERT INTO ways_inv", way_id,  str(invariant_type),  str(invariant_value))
                cur.execute("INSERT INTO ways_invs VALUES(?, ?, ?)", (way_id,  str(invariant_type),  str(invariant_value)))

    con.commit()
con.close()

print('type_with_no_values:')
for type_with_no_values in types_with_no_values:
    print('\t', type_with_no_values, INVARIANT_PROPERTIES_TYPES[type_with_no_values])



