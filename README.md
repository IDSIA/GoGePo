Install required python packages: 
```bash
pip install -r requirements.txt
```

To reproduce the main results, run GoGePo in different environments:
```bash
python3 gogepo.py --env_name Swimmer-v3 --use_gpu 1
python3 gogepo.py --env_name Hopper-v3 --use_gpu 1
python3 gogepo.py --env_name InvertedPendulum-v2 --use_gpu 1
python3 gogepo.py --env_name MountainCarContinuous-v0 --use_gpu 1
```