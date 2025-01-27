import numpy as np
from graphviz import Digraph
import inspect
from functools import partial
import types
from shapflow.flow import CausalLinks, build_feature_graph
# from shapflow.flow import Node, CreditFlow, Graph, get_source_nodes, flatten_graph, eval_graph, boundary_graph, single_source_graph, viz_graph, save_graph, hcluster_graph
# from shapflow.flow import ParallelCreditFlow, GraphExplainer, translator
# from shapflow.flow import group_nodes, build_feature_graph
# from shapflow.flow import CausalLinks, create_xgboost_f, create_linear_f
# from shapflow.flow import edge_credits2edge_credit
# from shapflow.on_manifold import OnManifoldExplainer, IndExplainer, FeatureAttribution


class Value:
    """
    Stores a single scalar value and its gradient.
    Assumes each instance of Value has a unique 'label'
    """
    # values_used_so_far is a class variable that keeps track of the instances of Value already created.
    # We store instances of Value by their unique 'label', which maps to its respective Value object in a Dict
    values_used_so_far = {}

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self.label = label
        self.score = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

        # Check if there is already an instance of Value with the same label
        if self.label in Value.values_used_so_far:
            raise Exception(f"Error: label '{self.label}' already in use")
        else:
            # add this new instance of Value to dict of all Values created thus far
            Value.values_used_so_far[self.label] = self

        # print(f"Value {self.label} init")
        # pprint(Value.values_used_so_far)

    def __add__(self, other):
        if not isinstance(other, Value):
            # Check if there is already another instance of 'other'
            if str(other) in Value.values_used_so_far:
                # Make 'other' point to the already existing Instance with its label
                other = Value.values_used_so_far[str(other)]
            else:
                other = Value(other, label=str(other))

        # If 'other' has label that starts with minus sign ('-'), then the label of 'out' should not use a '+' sign
        # e.g., 'a + -b' should be written as 'a - b'.
        if other.label[0] == '-':
            new_label = '(' + self.label + other.label + ')'
        else:
            new_label = '(' + self.label + '+' + other.label + ')'

        if new_label in Value.values_used_so_far:
            out = Value.values_used_so_far[new_label]
        else:
            out = Value(self.data + other.data, (self, other), '+', new_label)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        if isinstance(other, Value):
            # Ensure when multiplied by -1 that it is rendered correctly (i.e. we don't want a * -1; rather, we want -a)
            if other.data == -1:
                new_label = '(-' + self.label + ')'
            else:
                new_label = '(' + other.label + '*' + self.label + ')'
        else:
            if str(other) in Value.values_used_so_far:
                other = Value.values_used_so_far[str(other)]
            else:
                other = Value(other, label=str(other))

            # If multiplying by negative -1, we just prepend '-' to self.label.
            if other.data == -1:
                new_label = '(-' + self.label + ')'
            else:
                new_label = '(' + other.label + '*' + self.label + ')'

        if new_label in Value.values_used_so_far:
            out = Value.values_used_so_far[new_label]
        else:
            out = Value(self.data * other.data, (self, other), '*', new_label)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        new_label = '(' + self.label + '**' + str(other) + ')'
        if new_label in Value.values_used_so_far:
            out = Value.values_used_so_far[new_label]
        else:
            out = Value(self.data**other, (self,), f'**{other}', '(' + self.label + '**' + str(other) + ')')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        new_label = 'e^(' + self.label + ')'
        if new_label in Value.values_used_so_far:
            out = Value.values_used_so_far[new_label]
        else:
            out = Value(np.exp(self.data), (self,), 'exp', 'e^(' + self.label + ')')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def __neg__(self): # -self
        # First, check if -self already exists
        if '-' + self.label in Value.values_used_so_far:
            return Value.values_used_so_far['-' + self.label]

        # Second, check if '-1' already exists
        if '-1' in Value.values_used_so_far:
            negative_1 = Value.values_used_so_far['-1']
        else:
            negative_1 = Value(-1, label='-1')

        # We create a new Value for '-self'
        return self * negative_1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, label={self.label})"

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    @classmethod
    def reset(cls):
        cls.values_used_so_far = {}


def zero_grad(nodes):
    """
    Zero out of every gradient of each node in nodes
    """
    for n in nodes:
        n.grad = 0


def topo_sort(node):
    ''''
    Topological order all of the children of 'node' in the graph
    '''

    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(node)
    return topo


def get_function_arguments(func):
    # Get the signature of the function
    signature = inspect.signature(func)
    # Extract the parameters and convert them to a list of strings
    return [param.name for param in signature.parameters.values()]


def populate_dataframe(df, AI_DARWIN_function):
    """Populates a dataframe with every intermediate term in an AI-Darwin equation"""
    input_var_names = get_function_arguments(AI_DARWIN_function)
    print(input_var_names)
    # Iterate through each row in df
    for index, row in df.iterrows():
        input_vars = list(row[input_var_names])
        input_kwargs = {name: var for name, var in zip(input_var_names, input_vars)}

        y = AI_DARWIN_function(**input_kwargs)

        # Iterate through nodes topological sorted ordering
        for node in topo_sort(y):
            # Fill out each intermediate term (node) in the dataframe
            df.at[index, node.label] = node.data

def op_to_function(op):
    """Takes an operation and turns it into its respective expression/function.

    NOTE: This function should be updated to include more operations as needed.
    """
    def add(a, b):
        return a + b

    def mul(a, b):
        return a * b

    def exp(a):
        return np.exp(a)

    def pow(a, exponent):
        return a ** exponent

    if op == '+':
        return add
    elif op == '*':
        return mul
    elif op.startswith("**"):
        exponent = int(op[2:])
        return partial(pow, exponent=exponent)
    elif op == 'exp':
        return exp
    else:
        raise NotImplementedError(f"operation {op} is not implemented.")


def build_causal_graph(root, df, target_name, debug=False):
    """
    Insert description here
    """
    causal_links = CausalLinks()
    nodes, edges = trace(root)

    for n in nodes:
        # if has an '_op' field, then n has a parent node, which means there is a causal link that needs to be added
        if n._op:
            parents = list(n._prev)
            parent_labels = [parent.label for parent in parents]
            f = op_to_function(n._op)

            # Case where binary operation x+x or x*x is performed. Since both sides of the operation are equal, we need to handle this case differently to ensure that the the correct number of causes and effects are in the causal graph.
            if len(parent_labels) < 2 and type(f) == types.FunctionType:
                if f.__name__ == 'mul':
                    f = lambda a: a*a
                elif f.__name__ == "add":
                    f = lambda a: a+a

            causal_links.add_causes_effects(parent_labels, n.label, f)
            if debug:
                print(parent_labels)
                print(parent_labels, n.label)
                print(f)
                print('\n')
    print(causal_links.items)
    causal_graph = build_feature_graph(df.fillna(df.mean()), causal_links, target_name=target_name)
    return causal_graph


def trace(root):
    """
    Creates set of nodes and edges given some root node as input
    using DFS and backtracing through each node's children
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_computational_graph(root, format='png', rankdir='LR', disable_data_field=False, disable_grad_field=False):
    """
    Creates a computational digraph

    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        if disable_data_field and disable_grad_field:
            dot.node(name=str(id(n)), label = n.label, shape='record')
        elif disable_data_field:
            dot.node(name=str(id(n)), label = "{ %s | grad %.4f }" % (n.label, n.grad), shape='record')
        elif disable_grad_field:
            dot.node(name=str(id(n)), label = "{ %s | data %.4f }" % (n.label, n.data), shape='record')
        else:
            dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    # For each pair of nodes that share an edge, add an edge from the parent node to the _op node
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def draw_computational_graph_with_score(root, format='png', rankdir='LR'):
    """
    Creates a computational digraph with each term given a score for it's
    average influence on the final output (gradient scoring)

    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """

    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | score %.4f }" % (n.label, n.score), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot