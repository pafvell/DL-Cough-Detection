FOLDER=./checkpoints/2048

for d in ${FOLDER}/*/; do
	echo $d
	python3 evaluate.py --checkpoint_dir $d > $d/report.txt
	echo '====================================================================================' 
done

