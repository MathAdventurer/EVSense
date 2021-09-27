# EVSense

The repository of *EVSense: A Robust and Scalable Approach to Non-Intrusive EV Charging Detection*

> Authors: Xudong Wang, Guoming Tang, TBD

## What is EVSense

 A DNN model designed for household EV charging detection only using the aggregated load curve (so called "non-intrusive EV charging detection").

## What dose this project contain

  - EVSense, a DNN model implemented in Pytorch, Python3
  - Data loader, data precessing, data query, etc. for Pecan Street dataset
  - Model pruning and compression, transfer learning for the proposed EVsense, which have been validation on both sever, PC and the end device: Raspberry Pi 3B+
  - Some revised version loss funciton like Dice-loss, Forcal loss, etc.
  - Implemented benchmarks, codes and datasets
  - EV charging session analysis and visualization

## Datasets

 Since the Pecan Street datasets are not open sourse, due to the term of usesage, this github only provides cleaned data of two residents' in pickle formula. You can buy or apply the Pecan Street datasets from here: (https://www.pecanstreet.org/dataport/).
 
 Once you have access to the Pecan Street Dataport, you can easy obtain the data. This Github project has provided a complete solution, including data query from SQLlite3 file, data clean and saving, and so on.

## Requirements

  - Python 3.5+
  - Pytorch 1.4+ (Here the pytorch 1.8 use on Raspberry Pi 4B, you can following this link to compile the Pytorch, torchvision version you wanna use on Raspberry Pi:   (https://sites.google.com/view/steam-for-vision/raspberry-pi))
  - hmmlearn pakage if you wanna run the implemented FHMM
  - MATLAB after version 2009 for the IECON benchmark: (https://www.mathworks.com/matlabcentral/fileexchange/47474-energy-disaggregation-algorithm-for-electric-vehicle-charging-load).
  - If you have no GPUs, it may takes a long time to directly train the uncompressed model.

## Installation and Usage

  Using the git clone command in your termainal.

## License

  MIT license.

---

#### Contact:
- Emails: [Xudong Wang] (xudongwang@link.cuhk.edu.cn), [Guoming Tang] (tanggm@pcl.ac.cn)
