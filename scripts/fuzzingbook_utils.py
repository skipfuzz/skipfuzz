# https://www.fuzzingbook.org/html/DynamicInvariants.html

import itertools
import sys 
import ast


class Tracker:
    def __init__(self, log=False):
        self._log = log
        self.reset()

    def reset(self):
        self._calls = {}
        self._stack = []

    def traceit(self):
        """Placeholder to be overloaded in subclasses"""
        pass

    # Start of `with` block
    def __enter__(self):
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.traceit)
        return self

    # End of `with` block
    def __exit__(self, exc_type, exc_value, tb):
        sys.settrace(self.original_trace_function)

class CallTracker(Tracker):
    def traceit(self, frame, event, arg):
        """Tracking function: Record all calls and all args"""
        if event == "call":
            self.trace_call(frame, event, arg)
        elif event == "return":
            self.trace_return(frame, event, arg)
        return self.traceit

    def trace_call(self, frame, event, arg):
        """Save current function name and args on the stack"""
        code = frame.f_code
        function_name = code.co_name

        self.add_to_tracker_stack(function_name, frame)

    def add_to_tracker_stack(self,function_name, frame):
        arguments = get_arguments(frame)
        self._stack.append((function_name, arguments))

        if self._log:
            print(simple_call_string(function_name, arguments))


    def trace_return(self, frame, event, arg):
        """Get return value and store complete call with arguments and return value"""
        code = frame.f_code
        function_name = code.co_name

        return_value = arg
        # TODO: Could call get_arguments() here to also retrieve _final_ values of argument variables
        self.do_add_call(function_name, return_value)
        

    def do_add_call(self,function_name, return_value):
        try:
            called_function_name, called_arguments = self._stack.pop()
            if function_name == called_function_name:

                if self._log:
                    print(simple_call_string(function_name, called_arguments), "returns", return_value)

                self.add_call(function_name, called_arguments, return_value)
        except:
            pass

    def add_call(self, function_name, arguments, return_value=None):
        """Add given call to list of calls"""
        if function_name not in self._calls:
            self._calls[function_name] = []
        self._calls[function_name].append((arguments, return_value))

    def calls(self, function_name=None):
        """Return list of calls for function_name, 
        or a mapping function_name -> calls for all functions tracked"""
        if function_name is None:
            return self._calls

        return self._calls[function_name]



from fuzzingbook_invariant_utils import INVARIANT_PROPERTIES, INVARIANT_PROPERTIES_TYPES

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
    return eval(prop_function_text(prop))

def true_property_instantiations(prop, vars_and_values, log=False):
    instantiations = set()

    interesting_lengths = [1,3]


    p = prop_function(prop)

    len_metavars = len(metavars(prop))
    interestings = ''
    for combination in itertools.permutations(vars_and_values, len_metavars):
        args = [value for var_name, value in combination]
        var_names = [var_name for var_name, value in combination]

        try:
            result = p(*args)
        except Exception as e:
            result = None

        if log:
            print(' prop result:', prop, combination, result)
        interestings = ''
        if result:
            if any(['_interesting_values' in var_name for var_name in var_names]):
                interestings =  ','.join([var_name + ': ' +  str(value) for var_name, value in combination if '_interesting_values' in var_name])
            else:
                instantiations.add((prop, tuple(var_names)))

    return instantiations, interestings

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

class InvariantTracker(CallTracker):
    def __init__(self, props=None, **kwargs):
        if props is None:
            props = INVARIANT_PROPERTIES

        self.props = props

        self.valid_values_set = {}
        super().__init__(**kwargs)

    def invariants(self, function_name=None):
        if function_name is None:
            return {function_name: self.invariants(function_name) for function_name in self.calls()}

        invariants = None
        for variables, return_value in self.calls(function_name):
            vars_and_values = variables

            # do some meta meta variable handling
            if function_name not in self.valid_values_set:
                self.valid_values_set[function_name] = {}
            valid_values = self.valid_values_set[function_name]

            for variable_name, variable_value in variables.copy():
                
                if variable_name not in valid_values:
                    valid_values[variable_name] = set()
                    valid_values['len(' + variable_name + ')'] = set()
                    valid_values['shape(' + variable_name + ')'] = set()
                    valid_values['shape_0(' + variable_name + ')'] = set()
                    valid_values['shape_1(' + variable_name + ')'] = set()
                    valid_values['dtype(' + variable_name + ')'] = set()
                    valid_values['rank(' + variable_name + ')'] = set()

                if isinstance(variable_value, int) or isinstance(variable_value, str):
                    valid_values[variable_name].add(variable_value)
                elif isinstance(variable_value, list):
                    valid_values['len(' + variable_name + ')'].add(len(variable_value))
                elif 'Tensor' in type(variable_value).__name__:
                    try:
                        valid_values['shape(' + variable_name + ')'].add(tuple(variable_value.shape.as_list()))

                        if len(variable_value.shape) > 0:
                            valid_values['shape_0(' + variable_name + ')'].add(variable_value.shape[0])
                        if len(variable_value.shape) > 1:
                            valid_values['shape_1(' + variable_name + ')'].add(variable_value.shape[1])
                        valid_values['dtype(' + variable_name + ')'].add(variable_value.dtype)
                        valid_values['rank(' + variable_name + ')'].add(variable_value.shape.rank)
                    except:
                        pass


                for named_interesting_value in [variable_name, 'len(' + variable_name + ')', 'shape(' + variable_name + ')', 'shape_0(' + variable_name + ')', 
                                                'shape_1(' + variable_name + ')', 'dtype(' + variable_name + ')', 'rank(' + variable_name + ')']:
                    if named_interesting_value in valid_values:
                        vars_and_values += [(named_interesting_value, valid_values[named_interesting_value]) ]

            s = set()
            acc_interestings = []
            for prop_bef, _ in self.props.items():
                true_props, interestings = true_property_instantiations(prop_bef, vars_and_values, self._log)

                s |= true_props

                if len(interestings) > 0:
                    acc_interestings.append(interestings)
            if invariants is None:
                invariants = s
            else:
                invariants &= s

        return invariants, acc_interestings

