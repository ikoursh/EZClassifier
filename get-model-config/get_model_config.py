#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import tensorflow.compat.v1 as tf
from tkinter import Tk
from tkinter import filedialog, messagebox
from typing import Iterable

# In[14]:

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
file = filedialog.askopenfile(title="Select the converted pb model", filetypes=[("pb", "*.pb")]).name

if (file == None):
    messagebox.showerror("Error", "model cannot be empty")
    exit()
print(file)

# In[13]:


graph_def = None
graph = None

print('Loading graph definition ...', file=sys.stderr)
try:
    with tf.gfile.GFile(file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
except BaseException as e:
    print("Error")
    exit()

print('Importing graph ...', file=sys.stderr)
try:
    assert graph_def is not None
    with tf.Graph().as_default() as graph:  # type: tf.Graph
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='',
            op_dict=None,
            producer_op_list=None
        )
except BaseException as e:
    parser.exit(2, 'Error importing the graph: {}'.format(str(e)))

assert graph is not None
ops = graph.get_operations()  # type: Iterable[tf.Operation]
print("--------------------")
for op in ops:
    if op.type == "Placeholder":
        print("input operation: " + op.name)
        break
print("output operation: " + ops[len(ops) - 1].name)
print("--------------------")

# In[3]:
