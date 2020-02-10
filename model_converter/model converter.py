#!/usr/bin/env python
# coding: utf-8

# In[1]:


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


# In[2]:


from tkinter import Tk
from tkinter import filedialog, messagebox

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
h5 = filedialog.askopenfile(title="Select the h5 model you downloaded", filetypes=[("h5", "*.h5")])

if (h5 == None):
    messagebox.showerror("Error", "H5 model cannot be empty")
    exit()

# In[3]:


from tensorflow.keras.models import load_model

model = load_model(h5.name)  # C:\Users\Student\Desktop\converted_keras-1\keras_model.h5
model.summary()

# In[ ]:


print("loading keras...")
from keras import backend as K
import tensorflow as tf

print("freezing graph...")

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

# In[ ]:


print("geting output dir...")
out = filedialog.asksaveasfilename(title="Save converted model", filetypes=[("pb", "*.pb")])
print(out)

# In[ ]:


# tf.train.write_graph(frozen_graph, input("enter dir: "), "my_model.pb", as_text=False)
print("saving graph...")
path = out.rsplit('/', 1)[0] + "\\"
filename = out.rsplit('/', 1)[1] + ".pb"
tf.train.write_graph(frozen_graph, path, filename, as_text=False)

# In[ ]:
