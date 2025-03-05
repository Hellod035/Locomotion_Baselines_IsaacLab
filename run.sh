### Locomotion H1 ###

conda activate unitree_lab
export WANDB_USERNAME='xxxxx'
python scripts/rsl_rl/train.py --task H1-Flat --headless

conda activate unitree_lab
python scripts/rsl_rl/play.py --task H1-Flat-Play

conda activate unitree_lab
export WANDB_USERNAME='xxxxx'
python scripts/rsl_rl/train.py --task H1-Rough --headless

conda activate unitree_lab
python scripts/rsl_rl/play.py --task H1-Rough-Play


### Locomotion G1 ###

conda activate unitree_lab
export WANDB_USERNAME='xxxxx'
python scripts/rsl_rl/train.py --task G1-Flat --headless

conda activate unitree_lab
python scripts/rsl_rl/play.py --task G1-Flat-Play

conda activate unitree_lab
export WANDB_USERNAME='xxxxx'
python scripts/rsl_rl/train.py --task G1-Rough --headless

conda activate unitree_lab
python scripts/rsl_rl/play.py --task G1-Rough-Play

