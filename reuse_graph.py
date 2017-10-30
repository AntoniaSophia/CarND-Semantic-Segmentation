from graph_utils import load_graph
import tensorflow as tf
from tensorflow.python.platform import gfile

sess, _ = load_graph('./runs/optimized_graph.pb')
#sess, _ = load_graph('./runs/frozen_graph.pb')
graph = sess.graph
adam_angst = graph.get_tensor_by_name('adam_logit:0')
image_input = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

print('hallo')
print('Adam Angst = ',adam_angst)

# this only works after training......
#probs = sess.run(adam_angst, {image_input: img, keep_prob: 1.0})
