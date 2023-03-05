#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
# if you install anaconda in a different directory, try the following command
# source path_to_anaconda3/anaconda3/etc/profile.d/conda.sh

conda activate /dropbox/21-22/572/hw10/code/env/

cd .

#python main.py --num_epochs 6 --data_dir /dropbox/21-22/572/hw10/code/data/

# q3
python main.py --num_epochs 6 --data_dir /dropbox/21-22/572/hw10/code/data/ > q3.out

# q4
python main.py --num_epochs 6 --data_dir /dropbox/21-22/572/hw10/code/data/ --L2 > q4.out

# q5
python main.py --num_epochs 12 --data_dir /dropbox/21-22/572/hw10/code/data/ --patience 3 --L2 > q5.out

