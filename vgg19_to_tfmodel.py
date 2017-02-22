import vgg19
import tensorflow as tf

images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
vgg = vgg19.Vgg19()
vgg.build(images)

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
print "graph_def byte size", graph_def.ByteSize()
graph_def_s = graph_def.SerializeToString()

save_path = "vgg19.tfmodel"
with open(save_path, "wb") as f:
    f.write(graph_def_s)

print "saved model to %s" % save_path
