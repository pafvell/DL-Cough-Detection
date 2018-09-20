export PATH="$PATH:$HOME/bin"

TF_PATH=/Users/maurice/projects/machine_learning/tensorflow
COUGH_PATH=/Users/maurice/projects/eth/deepLearning/DL-Cough-Detection/cough_detect4

myExit() {
  echo -en "\n*** Exiting ***\n\n"
  pkill python
  exit $?
}
 
trap myExit SIGINT

# get shape and output layer name from user
echo "model name: "
read model_name
echo "output_layer: "
read output_layer
echo "last checkpoint: "
read last_checkpoint

input_layer_shape="16,128"

# make sure frozen_graphs dir exists
if [ ! -d "$COUGH_PATH/frozen_graphs" ]; then
	mkdir $COUGH_PATH/frozen_graphs
fi

# make sure we're in tensorflow dir
cd $TF_PATH


echo -en "\n\n============================== $model_name ==============================\n"
echo -en "\nfreezing model...\n"
bazel-bin/tensorflow/python/tools/freeze_graph \
	--input_graph="$COUGH_PATH/graphs/$model_name.pbtxt" \
	--input_checkpoint="$COUGH_PATH/checkpoints/$model_name/cv1/$last_checkpoint" \
	--output_graph="$COUGH_PATH/frozen_graphs/$model_name.pb" \
	--output_node_names=$output_layer \
	--input_binary=false

echo -en "\nbenchmarking model...\n"
bazel-bin/tensorflow/tools/benchmark/benchmark_model \
	--graph="$COUGH_PATH/frozen_graphs/$model_name.pb"  \
	--input_layer="Input" \
	--input_layer_shape=$input_layer_shape \
	--input_layer_type="float" \
	--output_layer=$output_layer \
	--show_run_order=false \
	--show_time=false \
	--show_memory=true \
	--show_summary=true \
	--show_flops=true

echo -en "\n\n summary \n"
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
	--in_graph="$COUGH_PATH/frozen_graphs/$model_name.pb"
	
echo -en "\n============================== end model $model_name ==============================\n\n"

## this should remove train ops from the graph
# echo -en "\ntransform_graph...\n"
# bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
# 	--in_graph=$COUGH_PATH/frozen_graphs/$model_name.pb \
# 	--out_graph=$COUGH_PATH/optimized_graphs/optimized_$model_name.pb \
# 	--inputs='Input' \
# 	--outputs=$output_layer \
# 	--transforms='
# 		strip_unused_nodes(type=float, shape="$input_layer_shape")
# 		fold_constants(ignore_errors=true)
# 		fold_batch_norms'