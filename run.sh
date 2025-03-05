### Locomotion H1 ###

conda activate unitree_lab
cd ~/projects/IsaacLabExtensionTemplate
export WANDB_USERNAME='hellod035-mit'
python scripts/rsl_rl/train.py --task H1-Flat --headless

conda activate unitreelab
cd ~/projects/IsaacLabExtensionTemplate
python scripts/rsl_rl/play.py --task H1-Flat-Play

conda activate unitree_lab
cd ~/projects/IsaacLabExtensionTemplate
export WANDB_USERNAME='hellod035-mit'
python scripts/rsl_rl/train.py --task H1-Rough --headless

conda activate unitreelab
cd ~/projects/IsaacLabExtensionTemplate
python scripts/rsl_rl/play.py --task H1-Rough-Play


### Locomotion G1 ###

conda activate unitree_lab
cd ~/projects/IsaacLabExtensionTemplate
export WANDB_USERNAME='hellod035-mit'
python scripts/rsl_rl/train.py --task G1-Flat --headless

conda activate unitreelab
cd ~/projects/IsaacLabExtensionTemplate
python scripts/rsl_rl/play.py --task G1-Flat-Play

conda activate unitree_lab
cd ~/projects/IsaacLabExtensionTemplate
export WANDB_USERNAME='hellod035-mit'
python scripts/rsl_rl/train.py --task G1-Rough --headless

conda activate unitreelab
cd ~/projects/IsaacLabExtensionTemplate
python scripts/rsl_rl/play.py --task G1-Rough-Play

