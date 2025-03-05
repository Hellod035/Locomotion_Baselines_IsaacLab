conda activate unitree_lab
export WANDB_USERNAME='xxxxx'

### Locomotion H1 ###
python scripts/rsl_rl/train.py --task H1-Flat --headless
python scripts/rsl_rl/play.py --task H1-Flat-Play

python scripts/rsl_rl/train.py --task H1-Rough --headless
python scripts/rsl_rl/play.py --task H1-Rough-Play


### Locomotion G1 ###

python scripts/rsl_rl/train.py --task G1-Flat --headless
python scripts/rsl_rl/play.py --task G1-Flat-Play

python scripts/rsl_rl/train.py --task G1-Rough --headless
python scripts/rsl_rl/play.py --task G1-Rough-Play

