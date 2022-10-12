# https://www.fuzzingbook.org/html/DynamicInvariants.html

import itertools
import sys 
import ast
import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.framework import sparse_tensor

INVARIANT_PROPERTIES = {
    "float(X) < !C!": [],
    "float(X) > !C!": [],
    "float(X) == !C!": [],
}

INVARIANT_PROPERTIES.update({
    "float(X) == !C!": [],
    "float(X) > !C!": [],
    "float(X) < !C!": [],
    "0 <= float(X) <= 1": [],
})

INVARIANT_PROPERTIES.update({
    "isinstance(X, bool)": [],
    "isinstance(X, int)": [],
    "isinstance(X, float)": [],
    "isinstance(X, list)": [],
    "isinstance(X, dict)": [],
    "isinstance(X, str)": [],
    "isinstance(X, tuple)": [],
    "isinstance(X, numpy.ndarray)": [],
    "type(X).__name__ == 'Tensor'": [],
    "type(X).__name__ == 'EagerTensor'": [],
    "type(X).__name__ == 'RaggedTensor'": [],
    "type(X).__name__ == 'SparseTensor'": [],
})


INVARIANT_PROPERTIES.update({
    "len(X) == !C!": [],
    "len(X) < !C!": [],
    "len(X) > !C!": [],
    
})

INVARIANT_PROPERTIES.update({
    "X.dtype == tf.float64": [],
    "X.dtype == tf.float32": [],
    "X.dtype == tf.float16": [],
    "X.dtype == tf.int64": [],
    "X.dtype == tf.int32": [],
    "X.dtype == tf.int8": [],
    "X.dtype == tf.int16": [],
    "X.dtype == tf.uint16": [],
    "X.dtype == tf.uint8": [],
    "X.dtype == tf.string": [],
    "X.dtype == tf.bool": [],
    "X.dtype == tf.complex64": [],
    "X.dtype == tf.complex128": [],
    "X.dtype == tf.qint8": [],
    "X.dtype == tf.quint8": [],
    "X.dtype == tf.qint16": [],
    "X.dtype == tf.quint16": [],
    "X.dtype == tf.qint32": [],
    "X.dtype == tf.bfloat16": [],
    "X.dtype == tf.resource": [],
    "X.dtype == tf.half": [],
    "X.dtype == tf.variant": [],
})

INVARIANT_PROPERTIES.update({
    # "X.shape.rank == Y.shape.rank": {},
    # "X.shape.rank == int(Y)": {},
    "X.shape.rank == 0": [],
    "X.shape.rank > !C!": [],
    "X.shape.rank < !C!": [],

})

INVARIANT_PROPERTIES.update({
    "X.shape[0] == int(!C!)": [],
    "X.shape[1] == int(!C!)": [],
    "X.shape[2] == int(!C!)": [],
    "X.shape[0] > int(!C!)": [],
    "X.shape[1] > int(!C!)": [],
    "X.shape[2] > int(!C!)": [],
    "X.shape[0] < int(!C!)": [],
    "X.shape[1] < int(!C!)": [],
    "X.shape[2] < int(!C!)": [],

})

INVARIANT_PROPERTIES.update({
    # for all element in tensor, element > Y 
    "tf.experimental.numpy.all(X > !C! ) .numpy()": [],
    "tf.experimental.numpy.all(X < !C! ) .numpy()": [],
    "tf.experimental.numpy.any(X == !C!) .numpy()": [],
    "tf.experimental.numpy.all(X == !C!) .numpy()": []
})


INVARIANT_PROPERTIES.update({
    "isinstance(X, sparse_tensor.SparseTensor)": [],
    "isinstance(X, ragged_tensor.RaggedTensor)": [],
})


INVARIANT_PROPERTIES.update({
    "all(i > !C! for i in X)": [],
    "all(i == !C! for i in X)": [],
    "all(i < !C! for i in X)": [],
    "all(i != 0 for i in X)": [],
    "all(i is not None for i in X)": [],
})

INVARIANT_PROPERTIES.update({
    # for all element in the shape of X
    "all(i > !C! for i in X.shape)": [],
    "all(i == !C! for i in X.shape)": [],
    "all(i < !C! for i in X.shape)": [],
})

INVARIANT_PROPERTIES.update({
    "all(type(X) == !T!)": [],
    "any(type(X) == !T!)": [],
    "all([type(x) == !T! for x in X])": [],
    "all([type(x) == !T! for x in X.values()])": [],
    "any([type(x) == !T! for x in X])": [],
    "any([type(x) == !T! for x in X.values()])": [],
    "all([x.dtype == !T! for x in X])": [],
    "any([x.dtype == !T! for x in X])": [],
})

# misc
INVARIANT_PROPERTIES.update({
    "X.isupper()": [],
    "X[0].isupper()": [],
    "X.islower()": [],
    "X[0].islower()": [],
    "X[0] == !C!": [],
    "X[1] == !C!": [],
    "X[-1] == !C!": [],
    "X[-2] == !C!": [],
    "all([x.dtype == !T! for x in X])": [],
    "any([x.dtype == !T! for x in X])": [],
})

INVARIANT_PROPERTIES.update({
    "X is None": [],
    "X is not None": [],
})


NEW_INVARIANT_PROPERTIES = {}
for key, value in INVARIANT_PROPERTIES.items():

    if '!T!' in key:
        for t in ['int', 'long', 'float', 'bool', "tf.float64", "tf.float32", "tf.float16",
                  "tf.int64", "tf.int32", "tf.int8", "tf.int16", "tf.uint16", "tf.uint8",
                  "tf.string", "tf.bool", "tf.complex64", "tf.complex128",
                  'tf.qint8', 'tf.quint8', 'tf.qint32', 'tf.quint32', 'tf.qint16', 'tf.quint16']:
            new_key = key.replace('!T!', t)
            NEW_INVARIANT_PROPERTIES[new_key] = {}
    elif '!C!' in key:
        for c in range(-10, 11):
            new_key = key.replace('!C!', str(c))
            NEW_INVARIANT_PROPERTIES[new_key] = {}
        for c in [2 ** 60 - 1, sys.maxsize, -sys.maxsize]:  # boundary values
            new_key = key.replace('!C!', str(c))
            NEW_INVARIANT_PROPERTIES[new_key] = {}
    else:
        NEW_INVARIANT_PROPERTIES[key] = value

INVARIANT_PROPERTIES = NEW_INVARIANT_PROPERTIES

INVARIANT_PROPERTIES_TYPES = sorted(INVARIANT_PROPERTIES.keys())



def metavars(prop):
    metavar_list = []

    class ArgVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id.isupper():
                metavar_list.append(node.id)

    ArgVisitor().visit(ast.parse(prop))
    return metavar_list

def instantiate_prop_ast(prop, var_names):
    class NameTransformer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in mapping:
                return node
            return ast.Name(id=mapping[node.id], ctx=ast.Load())

    meta_variables = metavars(prop)
    assert len(meta_variables) == len(var_names)

    mapping = {}
    for i in range(0, len(meta_variables)):
        mapping[meta_variables[i]] = var_names[i]

    prop_ast = ast.parse(prop, mode='eval')
    new_ast = NameTransformer().visit(prop_ast)

    return new_ast

def prop_function_text(prop):
    return "lambda " + ", ".join(metavars(prop)) + ": " + prop

def prop_function(prop):
    function_text = prop_function_text(prop)
    return eval(function_text)

def true_property_instantiations(prop, variable_value, invariant_type_index, invariant_placeholder_value, log=False):
    instantiations = set()
    invariant_type_and_value = set()

    p = prop_function(prop)

    args = [variable_value]

    try:
        result = p(*args)
    except Exception as e:
        result = None

    if log:
        print(' prop result:', prop, args, result)
    try:
        if result:
            instantiations.add((prop, invariant_type_index, invariant_placeholder_value) )
    except:
        try:
            if all(result):
                instantiations.add((prop, invariant_type_index, invariant_placeholder_value))
        except:
            pass

    return instantiations

def get_arguments(frame):
    """Return call arguments in the given frame"""
    # When called, all arguments are local variables
    local_variables = dict(frame.f_locals)  # explicit copy
    arguments = [(var, frame.f_locals[var]) for var in local_variables]
    arguments.reverse()  # Want same order as call
    return arguments

def simple_call_string(function_name, argument_list, return_value=None):
    """Return function_name(arg[0], arg[1], ...) as a string"""
    call = function_name + "(" + \
        ", ".join([var + "=" + repr(value)
                   for (var, value) in argument_list]) + ")"

    if return_value is not None:
        call += " = " + repr(return_value)

    return call


def is_functionlevel_invariant(inv):
    if isinstance(inv, str):
        pass




def get_invariants_hold(props, variable_value):
    """
    get invariants (from props) that hold
    """

    s = set()

    # invariant_type_and_values = set()

    for prop_bef, prop_value in props.items():
        invariant_placeholder_value = None
        invariant_type_index = INVARIANT_PROPERTIES_TYPES.index(prop_bef)

        if len(prop_value) > 0:
            constant_extractor = prop_function(prop_value[0])
            try:
                invariant_placeholder_value = constant_extractor(variable_value)
            except:
                continue
            prop_bef = prop_bef.replace('!C!', str(invariant_placeholder_value))
            print('new prop (filled in with extracted constant)', prop_bef)

        true_props = true_property_instantiations(prop_bef, variable_value, invariant_type_index, invariant_placeholder_value)

        s |= true_props


    invariants = s

    for inv in invariants:
        print('!invariants:', inv)

    # return invariants, invariant_type_and_value

def save_invariants_hold(props, ident, **data):
    """
    save invariants (from props) that hold
    """


    # invariant_type_and_values = set()
    for key_i, key in enumerate(data.keys()):

        s = set()

        variable_value = data[key]
        for prop_bef, prop_value in props.items():
            invariant_placeholder_value = None
            invariant_type_index = INVARIANT_PROPERTIES_TYPES.index(prop_bef)

            if len(prop_value) > 0:
                constant_extractor = prop_function(prop_value[0])
                try:
                    invariant_placeholder_value = constant_extractor(variable_value)
                except:
                    continue
                prop_bef = prop_bef.replace('!C!', str(invariant_placeholder_value))
                print('new prop (filled in with extracted constant)', prop_bef)

            true_props = true_property_instantiations(prop_bef, variable_value, invariant_type_index, invariant_placeholder_value)
            s |= true_props

        invariants = s

        with open('inv_coverage_' + ident +'_' + str(key_i) + '.txt', 'a') as outfile:
            for inv in invariants:
                outfile.write(str(inv[1]) + '\n')

    return data



def save_invariants_hold_position_ident(props, ident, variable_value):
    """
    save invariants (from props) that hold
    """
    s = set()

    # invariant_type_and_values = set()

    for prop_bef, prop_value in props.items():
        invariant_placeholder_value = None
        invariant_type_index = INVARIANT_PROPERTIES_TYPES.index(prop_bef)

        if len(prop_value) > 0:
            constant_extractor = prop_function(prop_value[0])
            try:
                invariant_placeholder_value = constant_extractor(variable_value)
            except:
                continue
            prop_bef = prop_bef.replace('!C!', str(invariant_placeholder_value))
            print('new prop (filled in with extracted constant)', prop_bef)

        true_props = true_property_instantiations(prop_bef, variable_value, invariant_type_index,
                                                  invariant_placeholder_value)

        s |= true_props

    invariants = s

    with open('inv_coverage_' + str(ident) + '.txt', 'a') as outfile:
        for inv in invariants:
            outfile.write(str(inv[1]) + '\n')

    return variable_value
