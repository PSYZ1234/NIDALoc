# NIDALoc
NIDALoc: Neurobiologically Inspired Deep LiDAR Localization

## Environment

- python 3.7

- pytorch 1.9.1


## Data

We support the Oxford Radar RobotCar and NCLT datasets right now.
```
Oxford data_root
├── 2019-01-11-14-02-26-radar-oxford-10k
│   ├── velodyne_left
│       ├── xxx.bin
├── pose_stats.txt
├── pose_max_min.txt
├── train_split.txt
├── test_split.txt
```

## Run
### Oxford

- train  -- 2 GPUs
```
python train.py
```

### NCLT

- train  -- 2 GPUs
```
python train.py
```

## Citation

```
@ARTICLE{10296854,
  author={Yu, Shangshu and Sun, Xiaotian and Li, Wen and Wen, Chenglu and Yang, Yunuo and Si, Bailu and Hu, Guosheng and Wang, Cheng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={NIDALoc: Neurobiologically Inspired Deep LiDAR Localization}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TITS.2023.3324700}}
```
