from graph_utils import load_graph
import tensorflow as tf
from tensorflow.python.platform import gfile

sess, _ = load_graph('./runs/optimized_graph.pb')
#sess, _ = load_graph('./runs/frozen_graph.pb')
graph = sess.graph
adam_angst = graph.get_tensor_by_name('adam_logit:0')

print('hallo')
print(adam_angst)
