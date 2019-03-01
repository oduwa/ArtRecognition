from __future__ import print_function
import numpy as np
from tensorflow.python.tools.freeze_graph import freeze_graph
import tfcoreml
import tensorflow as tf
from skimage import io, transform, util
import skimage
import PIL
from PIL import Image

# Provide these to run freeze_graph:
# Graph definition file, stored as protobuf TEXT
graph_def_file = 'Serial/PopCNNX.pbtxt'
# Trained model's checkpoint name
checkpoint_file = 'Serial/PopCNNX.ckpt'
# Frozen model's output name
frozen_model_file = './PopCNN2.pb'
# Output nodes. If there're multiple output ops, use comma separated string, e.g. "out1,out2".
output_node_names = 'Softmax'

test_img_path = "/Users/Odie/Documents/Xcode Projects/CoreMLTest/CoreMLTest/Resources/Images/papa1.JPG"

def parse_tf_graph_and_print(pb_path):
    with open(pb_path, 'rb') as f:
        serialized = f.read()
    tf.reset_default_graph()
    original_gdef = tf.GraphDef()
    original_gdef.ParseFromString(serialized)

    # Lets get some details about a few ops in the beginning and the end of the graph
    with tf.Graph().as_default() as g:
        tf.import_graph_def(original_gdef, name='')
        ops = g.get_operations()
        N = len(ops)
        for i in [0,1,2,N-3,N-2,N-1]:
            print('\n\nop id {} : op type: "{}"'.format(str(i), ops[i].type));
            print('input(s):'),
            for x in ops[i].inputs:
                print("name = {}, shape: {}, ".format(x.name, x.get_shape())),
            print('\noutput(s):'),
            for x in ops[i].outputs:
                print("name = {}, shape: {},".format(x.name, x.get_shape())),

def test_tf_model(pb_path):
    with open(pb_path, 'rb') as f:
        serialized = f.read()
    tf.reset_default_graph()
    original_gdef = tf.GraphDef()
    original_gdef.ParseFromString(serialized)

    # Lets get some details about a few ops in the beginning and the end of the graph
    with tf.Graph().as_default() as g:
        tf.import_graph_def(original_gdef, name='')
        ops = g.get_operations()

    #img = io.imread(test_img_path)
    img = Image.open(test_img_path)
    img_resized = img.resize([96,96], PIL.Image.ANTIALIAS)
    #img_resized = transform.resize(img, (96,96,3))
    img_np = np.array(img_resized).astype(np.float32)
    print( 'image shape:', img_np.shape)
    img_tf = np.expand_dims(img_np, axis = 0) #now shape is [1,224,224,3] as required by TF

    # Evaluate TF and get the highest label
    tf_input_name = 'Placeholder:0'
    tf_output_name = 'Softmax:0'
    img_tf = (1.0/255.0) * img_tf
    with tf.Session(graph = g) as sess:
        tf_out = sess.run(tf_output_name,
                          feed_dict={tf_input_name: img_tf})
    tf_out = tf_out.flatten()
    idx = np.argmax(tf_out)
    label_file = 'dataset/PopCNN2_labels.txt'
    with open(label_file) as f:
        labels = f.readlines()

    #print predictions
    print('\n')
    print("TF prediction class = {}, probability = {}".format(labels[idx],
                                                str(tf_out[idx])))



freeze_graph(input_graph=graph_def_file,
             input_saver="",
             input_binary=False,
             input_checkpoint=checkpoint_file,
             output_node_names=output_node_names,
             restore_op_name="save/restore_all",
             filename_tensor_name="save/Const:0",
             output_graph=frozen_model_file,
             clear_devices=True,
             initializer_nodes="")

#parse_tf_graph_and_print(frozen_model_file)

# convert to coreML

# Provide these inputs in addition to inputs in Step 1
# A dictionary of input tensors' name and shape (with batch)
input_tensor_shapes = {"Placeholder:0":[1, 96, 96, 3]} # batch size is 1
# Output CoreML model path
coreml_model_file = './PopCNN2.mlmodel'
output_tensor_names = ['Softmax:0']


coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names='Placeholder:0',
        class_labels = 'dataset/PopCNN2_labels.txt',
                     image_scale = 1.0/255.0)

# Test the converted model by getting the coreML prediction
# Provide CoreML model with a dictionary as input. Change ':0' to '__0'
# as Swift / Objective-C code generation do not allow colons in variable names
#img = io.imread("/Users/Odie/Documents/Xcode Projects/CoreMLTest/CoreMLTest/Resources/Images/x.JPG")
img = Image.open(test_img_path)
img = img.resize([96,96], PIL.Image.ANTIALIAS)
#img_resized = transform.resize(img, (3,96,96))
#input = np.array(img, dtype="float") / 255.0
# # inp = np.zeros((1,1,96,96,3))
# # inp[0][0] = input
# # print(inp.shape)
coreml_inputs = {'Placeholder__0': img} # (sequence_length=1,batch=1,channels=784)
coreml_output = coreml_model.predict(coreml_inputs, useCPUOnly=False)
print(coreml_output)

# Test the original TF model
test_tf_model(frozen_model_file)
