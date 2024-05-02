batch_size="4" # old: 16
# model="GCNConv"
hidden="64"
main="main.py"
epochs=100 # old: 100

label="novecs_w_history"
# label="randn_vecs"

# for dataset in "HCPGender"; do
for dataset in "HCPAge" "HCPGender" "HCPTask"; do
	for model in "MBPGNN"; do
	# for model in "MBPGNN" "GCNConv" "GraphConv" "GeneralConv"; do
	# for model in "GCNConv" "GraphConv" "GeneralConv"; do
		echo $dataset $model
		# -m pdb -c continue 
		python $main --dataset $dataset --model $model --label $label --device 'cuda' --batch_size $batch_size --runs 1 --epochs $epochs
	done
done
