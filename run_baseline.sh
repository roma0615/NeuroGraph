batch_size="4" # old: 16
# model="GCNConv"
hidden="64"
main="main.py"
epochs=100 # old: 100
# for dataset in "HCPAge" "HCPTask"; do
for dataset in "HCPTask" "HCPAge"; do
	# for model in "MBPGNN"; do
	for model in "MBPGNN" "GCNConv" "GraphConv"; do
		echo $dataset $model
		python $main --dataset $dataset --model $model --device 'cuda' --batch_size $batch_size --runs 1 --epochs $epochs
	done
done
