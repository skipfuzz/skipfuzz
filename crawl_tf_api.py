
# https://www.tensorflow.org/api_docs/python/tf/all_symbols


page = 'https://www.tensorflow.org/api_docs/python/tf/all_symbols'

import requests
import subprocess

requested = requests.get(page)

from bs4 import BeautifulSoup
soup = BeautifulSoup(requested.text)

symbols = []
for code in soup.find_all('code'):
    symbols.append(code.text)

import sys

def find_known_functions(existing_functions_file):
    # with open('all_functions.txt') as infile:
    with open(existing_functions_file) as infile:
        function_names = set()

        function_name = None

        for line in infile:
            line = line.strip()
            if line.startswith('==='):
                if function_name is None:
                    raise Exception('unknown func name')

                function_names.add(function_name)
            else:
                if 'import ' in line:
                    pass
                elif ':' in line:
                    pass
                else:
                    function_name = line.strip()

    return function_names

try:
    known_functions = find_known_functions(sys.argv[2])
    print('read from', sys.argv[2], ' ', len(known_functions), 'functions')
    print('')
except Exception as e:
    known_functions = set()
    pass

functions = []
for one_symbol in symbols:
    if one_symbol in known_functions:
        print('skip', one_symbol)
        continue
    with open('dummy_for_filtering_functions.py', 'w+') as outfile:
        outfile.write('import tensorflow as tf \n')
        outfile.write('print(type(' + one_symbol + ').__name__)\n')

        outfile.write('import inspect\n')
        outfile.write('for key in inspect.signature(' + one_symbol+ ').parameters.keys():\n')
        outfile.write('    print("!param!", key)')

    proc = subprocess.Popen(["python", 'dummy_for_filtering_functions.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    retcode = proc.returncode

    is_func_or_class = False
    for line in stdout.decode('utf-8').split('\n'):
        if 'function' in line or 'class' in line or 'type' in line:
            is_func_or_class = True

    if is_func_or_class:
        functions.append(one_symbol)
        print('added', one_symbol)
    else:
        continue

    params = []
    for line in stdout.decode('utf-8').split('\n'):
        if 'param' in line:
            name = line.split('!param!')[1].strip()
            params.append(name)

    functions[-1] = (functions[-1], params)


with open(sys.argv[1], 'w+') as outfile:
    for one_function, params in functions:
        outfile.write('import tensorflow as tf\n')
        outfile.write(one_function + '\n')
        for param in params:
            outfile.write(param + ':Any\n')
        outfile.write('=========\n')

