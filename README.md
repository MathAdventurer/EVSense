# EVSense

The repository of EVSense: a robust and scalable approach to non-intrusive EV charging detection.

> Authors: Xudong Wang, Guoming Tang

## What is EVSense

 A DNN model designed for household EV charging detection only using the aggregated load curve, so called *non-intrusive EV charging detection*.

## What dose this project contain

  - EVSense, a DNN model implemented in Pytorch, Python3
  - Data loader, data processing, data query, etc. for Pecan Street dataset
  - Model pruning and compression, transfer learning for the proposed EVsense, which have been validation on both sever, PC and the end device: Raspberry Pi 3B+
  - Some revised version loss function like Dice-loss, Forcal loss, etc.
  - Implemented benchmarks, codes and datasets
  - EV charging session analysis and visualization

## Datasets

 Since the Pecan Street datasets are not open source, due to the term of usage, this Github repository provides cleaned data of only two residents in pickle formula. You can buy or apply the Pecan Street datasets from [this link](https://www.pecanstreet.org/dataport/).
 
 Once having right to access the Pecan Street Dataport, you can easily download the dataset. This Github project also provides a complete solution to various data manipulations, including data query from SQLlite3 file, data cleaning and saving, and so on.

## Requirements

  - Python 3.5+
  - Pytorch 1.4+ (Pytorch 1.8 was used on Raspberry Pi 4B. You can refer to [this link](https://sites.google.com/view/steam-for-vision/raspberry-pi) to compile the Pytorch, torchvision version on Raspberry Pi.)
  - *hmmlearn* package for the learning-based (FHMM) benchmark.
  - MATLAB code (with version after 2009) for the rule-based benchmark. Refer to [this link](https://www.mathworks.com/matlabcentral/fileexchange/47474-energy-disaggregation-algorithm-for-electric-vehicle-charging-load).
  - If you have no GPUs, it may takes a long time to directly train the uncompressed model.

## Installation and Usage

  Using `git clone` command in your terminal.

## License

  MIT License.

---

#### Contact: 
- Xudong Wang (xudongwang@link.cuhk.edu.cn)
- Guoming Tang (tangguo1999@gmail.com)
