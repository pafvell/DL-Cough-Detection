#!/usr/bin/python

import tensorflow as tf
import importlib
import json
import os
from argparse import ArgumentParser




def build_parser():

	parser = ArgumentParser(description = "args for saving graphs")

	parser.add_argument("--subdir", "-s",
						dest="subdir",
						metavar="SUBDIR")

	return parser


def main():

	parser = build_parser()
	options = parser.parse_args()
	subdir = options.subdir

	ckpt_dir = "checkpoints"

	model_ckpt_dir = os.path.join(ckpt_dir, subdir, 'cv2')

	if os.path.isdir(model_ckpt_dir):

		print(f"saving model {subdir} as graph...")
		
		with open(os.path.join(ckpt_dir, subdir, 'config.json')) as json_data_file:
			config = json.load(json_data_file)
			control_config = config["controller"]  # reads the config for the controller file
			config_db = config["dataset"]
			config_train = control_config["training_parameter"]

		model_name = control_config["model"]
		bands = config_db["BAND"]
		size_cub = control_config["spec_size"]
		num_estimator = config_train["num_estimator"]
		num_filter = config_train["num_filter"]

		sess_config = tf.ConfigProto(
				log_device_placement=False,
				allow_soft_placement=True
			)

		with tf.Session(config=sess_config) as sess:

			latest_ckpt = tf.train.latest_checkpoint(model_ckpt_dir)
			model = importlib.import_module(model_name)

			input_tensor = tf.placeholder(tf.float32, shape=[bands, size_cub], name='Input')
			x = tf.expand_dims(input_tensor, 0)
			_, output_tensor = model.build_model(x, [1], num_estimator=num_estimator, num_filter=num_filter, is_training=False)

			saver = tf.train.Saver()
			saver.restore(sess, latest_ckpt)

			#save the model
			if not os.path.exists(os.path.join("tmp", subdir)):
				os.makedirs("tmp/"+ subdir)

			saver.save(sess, "tmp/" + subdir +"/" + subdir + ".ckpt")

			#save the graph
			tf.train.write_graph(sess.graph_def, 'graphs', subdir + '.pbtxt')


if __name__ == "__main__":
	main()