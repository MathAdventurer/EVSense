# EVsense
The repository of $EVsense: A Non-Intrusive Approach to Robust and Scalable EV Charging Detection$

> The author's GitHub Homepage: (https://github.com/MathAdventurer)

## What is EVsense

 A DNN model which can detection the household EV charging event from  aggregation power load. It's belongs to the single source blind disaggregation and the NILM(No-intrusive Load Monitoring.)

## What dose this project contain

  - EVsense, a DNN model implemented in Pytorch, Python3
  - Data Loader, Data precessing, Data Query, etc. for the Pecanstreet Datasets
  - Model pruning and compression, transfer learning for the proposed EVsense, which have been validation on both sever, PC and the end device: Raspberry Pi 3B+
  - Some revised version loss funciton like Dice-loss, Forcal loss, etc.
  - Benchmark model implemented like FHMM for time series event dection, Codes for test and benchmark the dataset for MATLAB code
  - EV chargin session analysis and visualization

## Datasets

 Since the Pecanstreet is not the open sourse datasets, due to the term of usesage, this github only provide two residents cleaned datasets in pickle formula, which is frendly for u to try these code. And if you have an interest for Pecanstreet data, you can buy or apply from here: (https://www.pecanstreet.org/dataport/).
 
 Once you have a access for the Pecanstreet Dataport, you can easy for using these data sinc this Github project have done an complete approach for Data query from SQLlite3 file, Data Clean and saving, and so on.

## Requirements

  - Python 3.5+
  - Pytorch 1.4+ (Here the pytorch 1.8 use on Raspberry Pi 3B+)
  - hmmlearn pakage if you wanna run the implemented FHMM
  - MATLAB after version 2009 for the IECON benchmark: (https://www.mathworks.com/matlabcentral/fileexchange/47474-energy-disaggregation-algorithm-for-electric-vehicle-charging-load).

## Installation and Usage

  Using the git clone command in your termainal.

## License

  MIT license.

---

#### Contact me：
- Email: [Xudong Wang] (xudongwang@link.cuhk.edu.cn)
