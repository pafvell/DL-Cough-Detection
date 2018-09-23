FOLDER=./checkpoints
SPLIT=''

 
myExit() {
  echo -en "\n*** Exiting ***\n\n"
  pkill python
  exit $?
}
 
trap myExit SIGINT


for d in ${FOLDER}/*/; do
	echo $d
	python3 evaluate.py --config $d/config.json --ckpt_dir $d > $d/results.txt
	echo 
	echo '==========================================================================' 
	echo 
done

