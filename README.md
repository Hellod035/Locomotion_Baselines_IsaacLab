# Isaac Lab Locomotion Baselines

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
git clone https://github.com/Hellod035/Locomotion_Baselines_IsaacLab
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/unitree_lab
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task G1-Flat --headless
```

- Check run_bash.sh to see the available environment
