module load gcc/8.2.0
module load python_gpu/3.8.5
bsub -o output -n 4 -R "rusage[mem=40000, ngpus_excl_p=1]" "python mu_net_composite_wgan.py"