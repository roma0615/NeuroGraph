batch_size="16" # old: 16
# model="GCNConv"
hidden="64"
main="main.py"
epochs=10 # old: 100
# for dataset in "HCPAge" "DynHCPAge"; do
	# for model in "MBPGNN" "GCNConv"; do
for dataset in "HCPAge"; do
	for model in "MBPGNN"; do
		echo $dataset $model
		python -m pdb -c continue $main --dataset $dataset --model $model --device 'cuda' --batch_size $batch_size --runs 1 --epochs $epochs
	done
done
