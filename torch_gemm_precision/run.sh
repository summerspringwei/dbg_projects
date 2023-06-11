CMD="/home/xiachunwei/Software/anaconda3/bin/python3 verify_torch_precision.py"
sudo -E /usr/local/cuda-11.7/bin/ncu --set full -f --target-processes all -o precision ${CMD}
