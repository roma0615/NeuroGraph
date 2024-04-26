batch_size="16" # old: 16
model="GCNConv"
hidden="64"
main="main.py"
for dataset in "DynHCPGender" "DynHCPAge"; do
	for model in "GCNConv" "DGM_Model"; do
		echo $dataset $model
		python $main --dataset $dataset --model $model --device 'cuda' --batch_size $batch_size --runs 1
	done
done
