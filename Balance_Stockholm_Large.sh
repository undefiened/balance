#!/bin/sh
#SBATCH -t 24:00:00
#SBATCH -J Balance_Stockholm_Large
#SBATCH --mem=300000
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --mail-user=leonid.sedov@liu.se
#SBATCH -o results/Balance_Stockholm_Large_$1.out
#SBATCH -e results/Balance_Stockholm_Large_$1.err
#SBATCH --mail-type=ALL

#sleep 100
module load gurobi/9.0.0-nsc1
module load Python/3.10.4-bare-hpc1-gcc-2022a-eb
source venv/bin/activate

mkdir results/stockholm_large_const_uncert_"$1"
mkdir results/stockholm_large_const_uncert_"$1/models"
touch results/stockholm_large_const_uncert_"$1/logs.txt"
touch results/stockholm_large_const_uncert_"$1/doc.txt"

python3 main.py --graph_name=stockholm_large --random_intents --run_name=const_uncert_"$1" --num_intents="$1" > results/stockholm_large_const_uncert_"$1"/logs.txt
