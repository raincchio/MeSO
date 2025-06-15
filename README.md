# Universal Stabilization for Maximum Entropy Optimization in Reinforcement Learning
Our code is based on the implementation provided in the paper "Better Exploration with Optimistic Actor Critic."[https://github.com/microsoft/oac-explore] 

## install some dependencies.

```bash
sudo apt-get install libglew-dev patchelf
```

## install mujuco

Download MuJoCo[https://github.com/deepmind/mujoco/releases]

For example, we can downloand mujoco210-linux-x86_64.tar.gz for linux

```bash
tar -xvf  mujoco200-linux-x86_64.tar.gz
cd  mujoco200-linux-x86_64
mkdir ~/.mujoco
mv mujoco200 ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco200/bin
```


## create virtual env using Miniconda
```bash
conda create -n meso python=3.7
conda active meso
conda install torch==1.12 # you can find the detail command on the pytroch website
pip install -r requirements.txt # Packages are listed in environment_config/requirements.txt file
```

## run experiments
For example, train MeSO for HalfCheetah-v2
```bash
python main.py --algo=meso --domain=halfcheetah --seed=1
```


