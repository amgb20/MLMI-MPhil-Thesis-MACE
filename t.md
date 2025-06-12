
## How to run Jupyter Notebook with HPC interactive CPU/GPU node

This guide assumes you're using a GPU partition and want to launch Jupyter inside a Conda environment with CUDA support.

### ⚙️ 1. Start an Interactive GPU Session

```
sintr -A MLMI-ab3149-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 1:00:00 --qos=INTR
```

### 🐍 2. Activate Your Conda Environment
```
conda activate <your_env>
```

### 📓 3. Launch Jupyter Notebook on the GPU Node
```
jupyter notebook --no-browser --ip=0.0.0.0 --port=8081
```

### 🔁 4. Open Tunnel from Your Local Machine
In a terminal on your laptop, forward the port:
```
ssh -L 8081:gpu-q-8:8081 <your-crsid>@login-e-4.hpc.cam.ac.uk
```
- Replace gpu-q-8 with your node name (from squeue)

- Replace login-e-4 with the login node you used

### 🌐 5. Access Jupyter Notebook in Browser
```
jupyter notebook list
```

And open the link or copy paste the token in the link in step 3