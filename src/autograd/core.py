import numpy as np

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