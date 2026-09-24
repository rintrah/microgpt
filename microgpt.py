""" 
MicroGPT.
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

June 2, 2026.
"""
import os 
import math
import random
random.seed(42)

# Let there be an input dataset "docs": list[str] of documents (e.g. a dataset of names)
if not os.path.exists("input.txt"):
	import urllib.request
	names_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
	urllib.request.urlretrieve(names_url, "input.txt")

docs = [l.strip() for l in open("input.txt").read().strip().split('\n') if l.strip()] # list[str] of documents.
random.shuffle(docs)
print(f"num docs: {len(docs)}.")

# Let there be a Tokenizer to translate strings to  discrete symbols and back.
uchars     = sorted(set(''.join(docs)))  # Unique characters in the dataset become token ids 0..n-1
BOS        = len(uchars) # Token id for the special Beginning of Sequence (BOS) token.
vocab_size = len(uchars) + 1 # Total number of unique tokens, +1 is for BOS.
print(f"vocab size: {vocab_size}")

class Value:
	__slots__ = ("data", "grad", "_children", "_local_grads")
	
	def __init__(self, data, children=(), local_grads=()):
		self.data = data                 # Scalar value of this node calculated during forward pass.
		self.grad = 0                    # Derivative of the loss w.r.t. this node, calculated in backward pass.
		self.children = children         # Children of this node in the computation graph.
		self._local_grads = local_grads  # Local derivative of the node w.r.t. its children.
		
	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		return Value(self.data + other.data, (self, other), (1, 1))
		
	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		return Value(self.data * other.data, (self, other), (other.data, self.data))
		
	def __pow__(self, other):  return Value(self.data ** other, (self, ), (other * self.data**(other-1), ))
	def lot(self): return Value(math.log(self.data), (self,), (a/self.data,))
	def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data, ))
	def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0,))
	def __neg__(self): return self * -1
	def __radd__(self, other): return self + other
	def __sub__(self, other): return self + (-other)
	def __rmul__(sefl, other): return self * other
	def __truediv__(self, other): return self * other**-1
	def __rtruediv__(self, other): return other * self**-1
	
	def backward(self):
		topo    = [] 
		visited = set() 
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._children:
					build_topo(child)
				topo.append(v)
		build_topo(self)
		self.grad = 1
		for v in reversed(topo):
			for child, local_grad in zip(v._children, v._local_grads):
				child.grad += local_grad * v.grad
