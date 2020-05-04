# Mark Schutera (mark.schutera@mailbox.org) subject [leafpathology]
# running under env tfp3.6


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from numba import jit

#import tensorflow.compat.v1 as tf
import tensorflow as tf
#tf.disable_eager_execution()

import argparse
import sys
import time
import os
import csv
import glob
import random
import re


import numpy as np



class Leafcracker():

  def __init__(self):

      # Load configuration arguments
      self.model_file =  "../tmp/intermediate_graph/intermediate_490000.pb" #"../tmp/output_graph.pb"
      self.label_file = "../tmp/output_labels.txt"
      self.input_height = 299
      self.input_width = 299
      self.input_mean = 128
      self.input_std = 128
      input_layer = "Placeholder"
      output_layer = "final_result"

      parser = argparse.ArgumentParser()
      parser.add_argument("--graph", help="graph/model to be executed")
      parser.add_argument("--labels", help="name of file containing labels")
      parser.add_argument("--input_height", type=int, help="input height")
      parser.add_argument("--input_width", type=int, help="input width")
      parser.add_argument("--input_mean", type=int, help="input mean")
      parser.add_argument("--input_std", type=int, help="input std")
      parser.add_argument("--input_layer", help="name of input layer")
      parser.add_argument("--output_layer", help="name of output layer")
      args = parser.parse_args()

      if args.graph:
        self.model_file = args.graph
      if args.labels:
        self.label_file = args.labels
      if args.input_height:
        self.input_height = args.input_height
      if args.input_width:
        self.input_width = args.input_width
      if args.input_mean:
        self.input_mean = args.input_mean
      if args.input_std:
        self.input_std = args.input_std
      if args.input_layer:
        input_layer = args.input_layer
      if args.output_layer:
        output_layer = args.output_layer

      # load labels
      self.labels = self.load_labels()
      # load graph
      self.graph = self.load_graph()
      # print(self.graph.get_operations())

      self.input_operation = self.graph.get_operation_by_name("import/" + input_layer);
      self.output_operation = self.graph.get_operation_by_name("import/" + output_layer);

      self.sess = tf.Session(graph=self.graph)
      self.sess2 = tf.Session()


  def load_graph(self):
      graph = tf.Graph()
      graph_def = tf.GraphDef()

      with open(self.model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
      with graph.as_default():
        tf.import_graph_def(graph_def)

      return graph


  def read_tensor_from_image_file(self, file_name):
      input_name = "file_reader"
      output_name = "normalized"
      file_reader = tf.read_file(file_name, input_name)

      if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name='png_reader')
      elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
      elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
      else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')

      float_caster = tf.cast(image_reader, tf.float32)
      dims_expander = tf.expand_dims(float_caster, 0);
      resized = tf.image.resize_bilinear(dims_expander,
  [self.input_height, self.input_width])
      normalized = tf.divide(tf.subtract(resized, [self.input_mean]),
  [self.input_std])
      #sess = tf.Session()
      result = self.sess2.run(normalized)

      return result


  def load_labels(self):
      label = []
      proto_as_ascii_lines = tf.gfile.GFile(self.label_file).readlines()
      for l in proto_as_ascii_lines:
        label.append(l.rstrip())
      return label

  def write_to_csv(self, key, img_name, pred):
      # Write predictions to csv
      with open(key, mode='a', newline='') as f:
          writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          writer.writerow([img_name, "{:.2f}".format(pred[0]), "{:.2f}".format(pred[1]), "{:.2f}".format(pred[2]), "{:.2f}".format(pred[3])])

  def init_csv(self, key):
      # Write predictions to csv
      with open(key, mode='w', newline='') as f:
          writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          writer.writerow(['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
          
  # ===============================================================================================================
  def soft_prob(self, results):
      healthy = results[0]
      rust = results[1]
      scab = results[2]


      multiple_diseases = scab*rust*(1-healthy)
      scab = scab*(1-healthy)*(1-multiple_diseases)
      rust = rust*(1-healthy)*(1-multiple_diseases)

      results = np.array([healthy, multiple_diseases, rust, scab])
      return results
  # ===============================================================================================================


  def whattheleaf(self, file_name, key):
      print(file_name)
      #start = time.time()
      t = self.read_tensor_from_image_file(file_name)
      results = self.sess.run(self.output_operation.outputs[0],
                             {self.input_operation.outputs[0]: t})
      #end = time.time()

      results = np.squeeze(results)
      results = self.soft_prob(results)
      #print(results)
      self.write_to_csv(key, file_name[14:-4], results)
      self.write_to_csv('onehot_' + key, file_name[14:-4], np.where(results == results.max(), 1, 0))

      # top_k = results.argsort()[-5:][::-1]

      # print('\nEvaluation time (1-image): {:.3f}s\n'.format(end - start))
      # template = "{} (score={:0.5f})"
      # for i in top_k:
        # print(template.format(self.labels[i], results[i]))
      # return end-start

  #@jit(nopython=True)
  def looping(self, images, submission_key):
    for file in images:
        self.whattheleaf(file, submission_key)


if __name__ == "__main__":
    leafcracker = Leafcracker()

    img_path = './test_images/'

    # load images
    images = [file for file in glob.glob(img_path + '*.jpg')]
    # sorts images in human order
    images.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)])

    # generate random key to store submission
    num = random.randrange(1, 10 ** 4)
    submission_key = str(num).zfill(5) + '_submission.csv'
    leafcracker.init_csv(submission_key)
    leafcracker.init_csv('onehot_' + submission_key)

    leafcracker.looping(images, submission_key)














