#!/bin/bash
#SBATCH --job-name="RP-Train"
#SBATCH --mail-user=arjun.sivasankar@mailbox.tu-dresden.de
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=00-4:00:00
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

python preprocess_reduction.py --strat=random --drop=10
python INN_train.py --drop=10
python preprocess_reduction.py --strat=random --drop=10
python INN_train.py --drop=10
python preprocess_reduction.py --strat=random --drop=10
python INN_train.py --drop=10
python preprocess_reduction.py --strat=random --drop=10
python INN_train.py --drop=10
python preprocess_reduction.py --strat=random --drop=10
python INN_train.py --drop=10


python preprocess_reduction.py --strat=random --drop=20
python INN_train.py --drop=20
python preprocess_reduction.py --strat=random --drop=20
python INN_train.py --drop=20
python preprocess_reduction.py --strat=random --drop=20
python INN_train.py --drop=20
python preprocess_reduction.py --strat=random --drop=20
python INN_train.py --drop=20
python preprocess_reduction.py --strat=random --drop=20
python INN_train.py --drop=20

python preprocess_reduction.py --strat=random --drop=30
python INN_train.py --drop=30
python preprocess_reduction.py --strat=random --drop=30
python INN_train.py --drop=30
python preprocess_reduction.py --strat=random --drop=30
python INN_train.py --drop=30
python preprocess_reduction.py --strat=random --drop=30
python INN_train.py --drop=30
python preprocess_reduction.py --strat=random --drop=30
python INN_train.py --drop=30

python preprocess_reduction.py --strat=random --drop=40
pythonINN_train.py --drop=40
python preprocess_reduction.py --strat=random --drop=40
python INN_train.py --drop=40
python preprocess_reduction.py --strat=random --drop=40
python INN_train.py --drop=40
python preprocess_reduction.py --strat=random --drop=40
python INN_train.py --drop=40
python preprocess_reduction.py --strat=random --drop=40
python INN_train.py --drop=40

python preprocess_reduction.py --strat=random --drop=50
python INN_train.py --drop=50
python preprocess_reduction.py --strat=random --drop=50
python INN_train.py --drop=50
python preprocess_reduction.py --strat=random --drop=50
python INN_train.py --drop=50
python preprocess_reduction.py --strat=random --drop=50
python INN_train.py --drop=50
python preprocess_reduction.py --strat=random --drop=50
python INN_train.py --drop=50

python preprocess_reduction.py --strat=random --drop=60
python INN_train.py --drop=60
python preprocess_reduction.py --strat=random --drop=60
python INN_train.py --drop=60
python preprocess_reduction.py --strat=random --drop=60
python INN_train.py --drop=60
python preprocess_reduction.py --strat=random --drop=60
python INN_train.py --drop=60
python preprocess_reduction.py --strat=random --drop=60
python INN_train.py --drop=60

python preprocess_reduction.py --strat=random --drop=70
pythonINN_train.py --drop=70
python preprocess_reduction.py --strat=random --drop=70
python INN_train.py --drop=70
python preprocess_reduction.py --strat=random --drop=70
python INN_train.py --drop=70
python preprocess_reduction.py --strat=random --drop=70
python INN_train.py --drop=70
python preprocess_reduction.py --strat=random --drop=70
python INN_train.py --drop=70

python preprocess_reduction.py --strat=random --drop=80
python INN_train.py --drop=80
python preprocess_reduction.py --strat=random --drop=80
python INN_train.py --drop=80
python preprocess_reduction.py --strat=random --drop=80
python INN_train.py --drop=80
python preprocess_reduction.py --strat=random --drop=80
python INN_train.py --drop=80
python preprocess_reduction.py --strat=random --drop=80
python INN_train.py --drop=80

python preprocess_reduction.py --strat=random --drop=90
python INN_train.py --drop=90
python preprocess_reduction.py --strat=random --drop=90
python INN_train.py --drop=90
python preprocess_reduction.py --strat=random --drop=90
python INN_train.py --drop=90
python preprocess_reduction.py --strat=random --drop=90
python INN_train.py --drop=90
python preprocess_reduction.py --strat=random --drop=90
python INN_train.py --drop=90

echo 'Done...'