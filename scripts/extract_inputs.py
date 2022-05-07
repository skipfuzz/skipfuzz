import os
import subprocess
from multiprocessing import Pool, cpu_count

prefix_template = """
import sys
import atexit
from collections import defaultdict
import numpy as np
import traceback
import os

import pathlib
import random

current_path = __file__# pathlib.Path(__file__).parent.resolve()

typed_to_objects = defaultdict(set)
id_to_object = {}  # store things like literals
object_graph = {}

def exit_handler():
    sys.settrace(None)
    def construct_sequence_for_object_id(object_id, indent_level=0):
        if indent_level > 10:
            # we are likely in trouble
            raise Exception('too deep')
        result = ''
#        print('querying', object_id)
        item = object_graph[object_id]
        function_call = item[1]
        objects_args = item[2]

        is_func = function_call.endswith('(')
        result = '  ' * (indent_level ) +  function_call +  '\\n'
        for key, val in sorted(objects_args.items()):
            if key == 'self':
                if 'NumpyProxy' in type(val[0]).__name__:
                    result += ' ' * (indent_level + 1)  + 'NumpyProxy(np),'
                    continue
                if type(val[0]).__name__.endswith('Test'):
                    result += ' ' * (indent_level + 1) + 'self,'
                    continue
            elif val[1] in object_graph and None is not val[0]:
                # print('abstract val', key, '', val[1])
                result += ' ' * (indent_level + 1) + (key + '=' if not key.startswith('.') else '')
                result += construct_sequence_for_object_id(val[1], indent_level + 1)
            else:
                # print('concrete val', key, 'type', type(val[0]), ' concrete val', val[0], ' id of obj', id(val[0]))
                if type(val[0]).__name__ == 'str':
                    str_ed_val = "'" + val[0] + "'"
                elif type(val[0]).__name__ == 'type':
                    str_ed_val = val[0].__name__
                elif type(val[0]).__name__ == 'DType':
                    str_ed_val = 'dtypes.' + val[0].name
                else:
                    str_ed_val = str(val[0])
                result += ' ' * (indent_level + 1) + (key + '=' if not key.startswith('.') else '')  + str_ed_val + '\\n'
            result += ','

        result += ' ' * (indent_level ) + (')\\n' if is_func else '\\n')
        if object_id in arg_to_function_name_and_count and indent_level==0:
            for func_calls_and_cnt in arg_to_function_name_and_count[object_id]:
                function_call, fn_call_count, param_name = func_calls_and_cnt
                result += '# c 2 ' + function_call + ' ' + str(fn_call_count) + ' ' + param_name + '\\n'
        return result 


    def print_sample(target_type='SparseTensor'):
        sampled = random.sample(list(typed_to_objects[target_type]), 1)[0]
        # print('construct a ', target_type)
        a = construct_sequence_for_object_id(sampled)
        # print(a)
        
    for type_key, ways in typed_to_objects.items():
        if type_key.startswith('_'):
            continue
        if type_key in ['function']:
            continue
        # print('constructing ways of type=', type_key)
        # delete file if exists
        if os.path.exists('(!filename!)__' + type_key + '.typedb'):
            print('removing', '(!filename!)__' + type_key + '.typedb')
            os.remove('(!filename!)__' + type_key + '.typedb')
        with open('(!filename!)__' + type_key + '.typedb', 'w+') as outfile:
            for way in ways:
                try:
(!imports!)
                    outfile.write(construct_sequence_for_object_id(way))
                    outfile.write('===\\n')
                except Exception:
                    print('failed to construct one way.. ignoring')
                    continue
    obj_id_type_to_objects = defaultdict(list)

    for obj_id, concrete_object in id_to_object.items():
        if obj_id in object_graph:
            continue
        obj_type = type(concrete_object)
        obj_id_type_to_objects[obj_type].append(concrete_object)

    for obj_type, concrete_objects in obj_id_type_to_objects.items():
        with open('(!filename!)__' + obj_type.__name__ + '.typedb', 'a') as outfile:
            for concrete_object in concrete_objects:
                if obj_type.__name__ == 'str':
                    outfile.write("'" + concrete_object + "'")
                else:
                    outfile.write(str(concrete_object))

                outfile.write('\\n')
                object_id = id(concrete_object)
                if object_id in arg_to_function_name_and_count:
                    for func_calls_and_cnt in arg_to_function_name_and_count[object_id]:
                        function_call, fn_call_count, param_name = func_calls_and_cnt
                        outfile.write('# c 2 ' + function_call + ' ' + str(fn_call_count) + ' ' + param_name + '\\n')
                outfile.write('===\\n')

    for function_call, arg_types in function_arg_types.items():
        with open('(!filename!)--' + function_call + '.typesig', 'w+') as outfile:
            for dummy_i in range(1):
                for dummy_ii in range(1):
(!imports!)
                    for arg_name, arg_type_set in arg_types.items():
                        outfile.write(arg_name + ':' + ','.join(list(dict.fromkeys(arg_type_set))) + '\\n')



atexit.register(exit_handler)

class NumpyProxy(object):
    def __init__(self, np):
        self.np = np

    def func_decorator(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            
            code_str = 'np.' + func.__name__ + '(' + ','.join(args_repr) + ',' + ','.join(kwargs_repr)  + ')'

            arg_map = {}
            for arg_count, arg_val in enumerate(args):
                arg_map['.' + str(arg_count)] = (arg_val, id(arg_val))

            for k, v in kwargs.items():
                arg_map[k] = (v, id(v))
            # print('numpy thing', code_str)
            # print('numpy handler argmap', arg_map)
           # print('id=', id(result))

            if type(self.np).__name__ == 'module':
                name = 'np.' + func.__name__ + '('
            else:
                # some other numpy object
                name = type(self.np).__name__ + '.' + func.__name__ + '('
            # print('numpy function call?', name)
            object_graph[id(result)] = (code_str, name, arg_map)
            
            for k, (v, v_id) in arg_map.items():
                arg_to_function_name_and_count[v_id].add((name, function_call_count[name], k))
            function_call_count[name] += 1

            typed_to_objects[type(result).__name__].add(id(result))
            #print('returning', result, 'type', type(result))
            return result
        return wrapper
   
   
    def __getattr__(self, name):
        try:
            ret = getattr(self.np, name)
            if type(ret).__name__ == 'builtin_function_or_method':
#                print('wrapping',type(ret), name)
                return self.func_decorator(ret)
            if type(ret).__name__ == 'type':
                result = ret
                # print('numpy type access', result)
                object_graph[id(result)] = (name, 'np.' + name, dict())
                function_call_count['np.' + name] += 1
                typed_to_objects[type(result).__name__].add(id(result))

                return result
                
            else:
                # print('NumpyProxy not wrapping', type(ret), name )
                return ret

        except Exception as e:
            traceback.print_exc()
#            print('??')

            raise e
np = NumpyProxy(np)

file_cache = {}
arg_maps = {}

function_call_count = defaultdict(int)
arg_to_function_name_and_count = defaultdict(set)

function_arg_types = {}
def string_match_function_receiver(line, function_name):
    # not sure if we can do better than this...
    if function_name + '(' not in line:
        return None
    prefix = line.split(function_name + '(')[0]
    if not prefix.endswith('.'):
        return None
    prefix = prefix.split('.')[-2].split()[-1].split('(')[-1]
    return prefix

def string_match_is_function(line, function_name):
    if function_name +'(' not in line:
        return False
   # suffix = line.split(function_name +'(')[1]

    return True 

def is_right_frame(frame):
    # first, look for a frame with the current path
    calling_frame = None
    current_frame = frame
    while calling_frame is None and current_frame is not None:
        if current_frame.f_code.co_filename == current_path:
            calling_frame = current_frame
        current_frame = current_frame.f_back

    # next, "guess" the likely function call
    if calling_frame is None:
        return False, None
    line_with_call = line_content_from_filename_lineno(calling_frame.f_code.co_filename, calling_frame.f_lineno - 1).strip()
    guessed_calls = [ token.split('.')[-1] for token in  line_with_call.split('(')]

    # print('guessed call', guessed_call, ' from', line_with_call, current_frame.f_code)
    expected_call = frame.f_code.co_name if '_init_' not in frame.f_code.co_name else type(frame.f_locals['self']).__name__ 
    if any([guessed_call == expected_call for guessed_call in guessed_calls]):
        return True, calling_frame
    else:
        return False, None

def line_content_from_filename_lineno(filename, lineno):
  if filename in file_cache:
    line_content = file_cache[filename][lineno]
  else:
        
    calling_filename = filename
    with open(calling_filename) as infile:
        content = infile.readlines()
        line_content = content[lineno ]

    file_cache[calling_filename] = content
  return line_content

def traceit(frame, event, arg):
  is_matching_frame, back_frame=is_right_frame(frame)

  if frame.f_back is None or back_frame is None  or str(current_path) not in back_frame.f_code.co_filename or not is_matching_frame:
    return traceit

  line_content = line_content_from_filename_lineno(back_frame.f_code.co_filename, back_frame.f_lineno -1)

  is_func = True
  if not string_match_is_function(line_content, frame.f_code.co_name) and '_init' not in frame.f_code.co_name and '__getattr_' not in frame.f_code.co_name:
      is_func = False

  if 'self' in frame.f_locals:
    function_name = type(frame.f_locals['self']).__name__ + '.' + frame.f_code.co_name
  else:
    function_name = frame.f_code.co_name
    possible_prefix = string_match_function_receiver(line_content, function_name) 
    if possible_prefix is not None:
        function_name = possible_prefix + '.' + function_name
  if is_func:
      function_name+= '('

  arg_map = {}
  if event == 'call':
    arg_types = {}
    for i in range(frame.f_code.co_argcount):
        name = frame.f_code.co_varnames[i]
        if name in frame.f_locals and '._' in type(frame.f_locals[name]).__name__ or type(frame.f_locals[name]).__name__.startswith('_'):
            # if private type, can't construct it...
            #print('skipping arg of type,' , type(frame.f_locals[name]).__name__, 'for key=', name)
            continue
        arg_map[name] = (frame.f_locals[name], id(frame.f_locals[name])) if name in frame.f_locals else None

        if name not in arg_types:
            arg_types[name] = set()
        if name in frame.f_locals:
            arg_types[name].add(type(frame.f_locals[name]).__name__)

            id_to_object[id(frame.f_locals[name])] = frame.f_locals[name]

    arg_maps[function_name] = arg_map

    function_arg_types[function_name]  = arg_types
  elif event =='return':
    if '<' in function_name and '>' in function_name:
        return traceit

    if '_init_' in function_name:
#        print('is init')
        retval = frame.f_locals['self']
    else:
        retval = arg

    if type(retval).__name__ == 'function':
        return traceit
    arg_map = arg_maps[function_name].copy()
    # arg_maps[function_name].clear()

    # print('id', id(retval), 'of type=', type(retval))
    # print('from file', back_frame.f_code.co_filename)
    # print(' line=',line_content)
    # print('all locals', frame.f_locals)
    # print('how many?', len(typed_to_objects[str(type(retval).__name__)]))
    #print('frame function name', frame.f_code.co_name)
    # print('saveable function name', function_name, arg_map)
    
    #for line in traceback.format_stack():
    #    print(line.strip())

    if not function_name.endswith('('):
        # print('rejecting', function_name)
        pass
    elif 'with ' in line_content:
        # print('rejecting', function_name, 'since it contains with')
        pass
    else:
        typed_to_objects[str(type(retval).__name__)].add(id(retval))
        object_graph[id(retval)] = (line_content, function_name, arg_map )
        for k, (v, v_id) in arg_map.items():
            arg_to_function_name_and_count[v_id].add((function_name, function_call_count[function_name],k))
            # print('putting into arg_to_function_name_and_count', k, v_id, v, function_name)

        function_call_count[function_name] += 1

    # print('==============================\\n')
  return traceit
"""

import ast
import astunparse

def create_mod_test(original_filename):
    original_code = None
    with open(original_filename) as infile:
        original_code = infile.read()
    original_tree = ast.parse(original_code)


    imports = [] # keep track of the original imports for recording and replaying each object construction
    prettified_code_ = astunparse.unparse(original_tree)
    prettified_code = ""
    for line in prettified_code_.split('\n'):
        if 'import' in line and 'numpy' in line and 'as np' in line:
            imports.append(line.strip())
            continue
        else:
            if 'import ' in line :
                imports.append(line.strip())
            prettified_code += line + '\n'


    prefix = prefix_template.replace("(!filename!)", original_filename.split('.py')[0])

    prefix=prefix.replace("(!imports!)", '\n'.join([ "                    outfile.write('" + one_import + "\\n')" for one_import in imports]))

    modified_file = 'mod_' + original_filename
    with open(modified_file, 'w+') as outfile:
        outfile.write(prefix)

        # assuming all files end with "if __name__ == ".. .
        outfile.write(prettified_code.replace("(__name__ == '__main__'):", "(__name__ == '__main__'):\n    sys.settrace(traceit)"))
        outfile.write("    " )

    print('created', modified_file)
    return modified_file


rootdir = os.getcwd()
def collect_initialization_graphs():

    for root, subdirs, files in os.walk(rootdir):
        os.chdir(root)
        with Pool(max(cpu_count() // 2 + 1, 1)) as pool:
            modified_filenames = []
            for f in files:
                if not f.endswith('_test.py') or f.startswith('mod_'):
                    continue
                print('collect_initialization_graphs:: ', root, f)

                modified_filename = create_mod_test(f)
                modified_filenames.append(modified_filename)

            pool.map(run, modified_filenames)

            pool.close()
            pool.join()

        os.chdir(rootdir)
        print(root)

# collect database of tensors, etc..


def run(modified_filename):
    proc = subprocess.Popen(['timeout','--signal=SIGKILL', '500', "python", modified_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    retcode = proc.returncode
    print('running', modified_filename)
    print(stdout)
    print('....')
    print('err', stderr)

    return retcode == 0

  


def read_typedb_file(target_type):
    result = []
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
                    way += '# ' + f + '\n'
                    for line in infile:
                        line = line.strip()
                        if not line.startswith('==='):
                            way += line + '\n'
                        else:
                            result.append(way)
                            way = ''
                            way += '# ' + f + '\n'
    return result

def cached_typedb_spells(type, ways):
    os.chdir(rootdir)
    with open('root__' + type + '.typedb', 'w+') as outfile:
        for way  in ways:
            outfile.write(way + '\n')
            outfile.write('======\n')

    print('wrote to ', 'root__' + type + '.typedb')


collect_initialization_graphs()

