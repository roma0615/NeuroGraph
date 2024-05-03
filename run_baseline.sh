batch_size="4" # old: 16
# model="GCNConv"
hidden="64"
main="main.py"
epochs=100 # old: 100

# label="novecs_w_history"
# label="randn_vecs"

# for dataset in "HCPGender"; do
for dataset in "HCPAge" "HCPGender" "HCPTask"; do
	for model in "MaceGNN" "GCNConv" "GraphConv" "GeneralConv"; do
	# for model in "MaceGNN"; do
		for num_layers in 3; do
		# for model in "GCNConv" "GraphConv" "GeneralConv"; do
			echo $dataset $model $num_layers
			# -m pdb -c continue 
			python $main --dataset $dataset --model $model --label "${num_layers}_layers" --device 'cuda' --batch_size $batch_size --runs 1 --epochs $epochs --num_layers $num_layers
		done
	done
done
