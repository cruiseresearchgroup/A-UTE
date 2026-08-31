# A-UTE: Advection Informed, Uncertainty Aware Temperature Emulator

![architecture](images/a-ute.png)

# Abstract:
Physics-based Earth system models (ESMs) are essential for attributing climate change and generating scenario projections, yet their reliance on high-resolution numerical integration makes multi-decadal experiments expensive. In parallel, deep learning has delivered strong gains in short-range weather forecasting; however, auto-regressive roll-outs can accumulate error and become unstable when extended to decade-scale climate emulation. We introduce A-UTE: Advection Informed, Uncertainty Aware Temperature Emulator, aimed at stable multi-year emulation across heterogeneous climate models and grid resolutions. A-UTE is trained on various physics-based models at varying spatial resolutions to emulate temperature fields over a 10-year horizon. A-UTE formulates climate emulation as a forward-time stochastic dynamical system. We propose an auto-regressive ODE–SDE surrogate in which transport dynamics are constrained by an advection-consistent ODE component, while a learned neural SDE term models coarse-grained variability and cross-model discrepancy at monthly resolution. We train A-UTE under negative log-likelihood objective for principled uncertainty estimates and probabilistic evaluation. Experiments across 20 climate models show that A-UTE improves long roll-out stability and accuracy relative to relevant baselines, advancing data-driven climate emulation with explicit physical structure and uncertainty-aware predictions.

# Requirements:
* python=3.10
* torch=2.2.2
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
* xarray
* netcdf4=1.7.1
* pytorch-lightning=2.2.3
* wandb=0.17.5
* huggingface-hub=0.24.3
* xarray
* [ClimateSet](https://github.com/RolnickLab/ClimateSet) [1] (The code follows the directory structure of ClimateSet)

# Data Preparation
Download ClimateSet data for following 15 climate models from HuggingFace.
```python
python scripts/download_climateset_huggingface.py
```
1. AWI-CM-1-1-MR
2. BCC-CSM2-MR
3. CESM2-WACCM
4. CAS-ESM2-0
5. CESM2
6. CMCC-CM2-SR5
7. CMCC-ESM2
8. CNRM-CM6-1-HR
9. EC-Earth3
10. EC-Earth3-Veg
11. EC-Earth3-Veg-LR
12. FGOALS-f3-L
13. GFDL-ESM4
14. INM-CM4-8
15. INM-CM5-0
16. MPI-ESM1-2-HR
17. MRI-ESM2-0
18. NorESM2-MM
19. TaiESM1

# Setup
1. Create virtual environment
  ```python
  python -m venv env_ute
  source env_ute/bin/activate
  pip install -r requirements.txt
  cd emulator 
  pip install -e .
  ```
2. Set Python Path
   ```python
   export PYTHONPATH=/home/user/UTE:$PYTHONPATH"
   ```
4. Set Hydra for Debugging
   ```python
   export HYDRA_FULL_ERROR=1
   ```

# Training Single Emulator
The main model path is /emulator/src/core/models/ute_net.py
```python
python emulator/run.py experiment=single_emulator/ute_net/single_ute.yaml 
```


```
[1] Kaltenborn, J., Lange, C., Ramesh, V., Brouillard, P., Gurwicz, Y., Nagda, C., ... & Rolnick, D. (2023). Climateset: A large-scale climate model dataset for machine learning. Advances in Neural Information Processing Systems, 36, 21757-21792.
```
Note: We are also constantly updating the repo to make it accustomed to more Climate models and different resolution, and removing bugs and modifying certain parts.
