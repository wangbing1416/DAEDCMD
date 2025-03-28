# DAEPCMD

This repo is the released code of our work 
**Remember Past, Anticipate Future: Learning Continual Multimodal Misinformation Detectors**

## Requirements


```
torch==1.13.1
cudatoolkit==11.8.0
transformers==4.25.1
diffuser==0.31.0.dev0
```

Partial required packages are listed in `./requirement.txt`

## Train

1. Prepare the datasets Weibo, Gossip and Twitter. The datasets can be downloaded or required from https://github.com/yaqingwang/EANN-KDD18 and https://github.com/shiivangii/SpotFakePlus,
and you should put them in `./Data`

2. Run `event_clustering.py` to split the datasets into K subsets using single-pass clustering

3. Run the python file

```shell
cd src
python ./run_ours.py
```
where `run.py` is the file for baseline models, and `run_ours.py` is the file for our framework.

4. Check log files in `./log`, you can run `auto_logging.py` to automatically read logs.


## Citation
```

```