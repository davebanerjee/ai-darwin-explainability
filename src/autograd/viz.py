from graphviz import Digraph

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