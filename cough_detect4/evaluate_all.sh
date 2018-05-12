FOLDER=./checkpoints/new
SPLIT=''

 
myExit() {
  echo -en "\n*** Exiting ***\n\n"
  pkill python
  exit $?
}
 
trap myExit SIGINT


for d in ${FOLDER}/*/; do
	echo $d
	python3 evaluate.py --config $d/config.json --ckpt_dir $d > $d/report$SPLIT.txt
	echo 
	echo '==========================================================================' 
	echo 
done

