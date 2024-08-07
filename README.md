# EVSense

The repository of ACM e-Enery 2022 Paper: EVSense: A Robust and Scalable Approach to Non-Intrusive EV Charging Detection.

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2022evsense,
  title={Evsense: A robust and scalable approach to non-intrusive ev charging detection},
  author={Wang, Xudong and Tang, Guoming and Wang, Yi and Keshav, Srinivasan and Zhang, Yu},
  booktitle={Proceedings of the Thirteenth ACM International Conference on Future Energy Systems},
  pages={307--319},
  year={2022}
}
```

## What is EVSense

 A DNN model designed for household EV charging detection only using the aggregated load curve, namely *non-intrusive EV charging detection*.

![ ](https://github.com/MathAdventurer/EVSense/blob/main/image/EVSense_model.png)

**Structure:**
- 1D-Conv Layers

      - input length = 20:
  
        (conv1): in_channels=1,  out_channels=30, kernel_size=3, stride=1
    
        (conv2): in_channels=30, out_channels=30, kernel_size=5, stride=1
    
        (conv3): in_channels=30, out_channels=40, kernel_size=6, stride=1
    
        (conv4): in_channels=40, out_channels=50, kernel_size=5, stride=1
    
        (conv5): in_channels=50, out_channels=50, kernel_size=5, stride=1
  
      - input length = 10:
  
        (conv1): in_channels=1,  out_channels=30, kernel_size=2, stride=1
    
        (conv2): in_channels=30, out_channels=30, kernel_size=3, stride=1
    
        (conv3): in_channels=30, out_channels=40, kernel_size=3, stride=1
    
        (conv4): in_channels=40, out_channels=50, kernel_size=4, stride=1
    
        (conv5): in_channels=50, out_channels=50, kernel_size=2, stride=1

- Bi-LSTM Layers:

        input_size=50, hidden_size=50, num_layers=2
  
- Fully connection layers:

        (fc1): in_features=100, out_features=1024
        
        (fc2): in_features=1024, out_features=1

## What does this project contain

  - EVSense, a DNN model implemented in Pytorch, Python3
  - Data loader, data processing, data query, etc. for Pecan Street dataset
  - Model pruning and compression, transfer learning for the proposed EVsense, which have been validation on both sever, PC and the end device: Raspberry Pi 4B
  - Some revised version loss function like Dice-loss, Focal loss, etc.
  - Implemented benchmarks, codes and datasets
  - EV charging session analysis and visualization

## Datasets

 Since the Pecan Street datasets are not open source, due to the term of usage, this Github repository provides cleaned and partial data of only two residents in pickle formula. You can buy or apply the Pecan Street datasets from [this link](https://www.pecanstreet.org/dataport/).
 
 Once having right to access the Pecan Street Dataport, you can easily download the dataset. This Github project also provides a complete solution to various data manipulations, including data query from SQLlite3 file, data cleaning and saving, and so on.

## Requirements

  - Python 3.5+ (pickle data protocol is 5, you can install the pakage "pickle5" to read it if your python version < 3.8)
  - Pytorch 1.4+ (Pytorch 1.8 was used on Raspberry Pi 4B. You can refer to [this link](https://sites.google.com/view/steam-for-vision/raspberry-pi) to compile the Pytorch, torchvision version on Raspberry Pi.)
  - *hmmlearn* package for the learning-based (FHMM) benchmark.
  - MATLAB code (with version after 2009) for the rule-based benchmark. Refer to [this link](https://www.mathworks.com/matlabcentral/fileexchange/47474-energy-disaggregation-algorithm-for-electric-vehicle-charging-load).
  - If you have no GPUs, it may takes a long time to directly train the uncompressed model.

## Installation and Usage

  Using `git clone` command in your terminal.
  
  
## Project File Structure
```bash
  EVSense/
  ├── LICENSE
  ├── README.md
  ├── benchmark
  │   ├── 2014Zhang_benchmark_transfer2MATLAB.py
  │   └── FHMM_benchmark.py
  ├── checkpoint
  ├── data_processing.py
  ├── session_analysis.py
  ├── datasets
  │   ├── austin
  │   ├── california
  │   ├── newyork
  │   └── synthesis_data 
  │   ├── metadata
  │   │   ├── California_EV_real_exist_data_info.csv
  │   │   ├── California_NoEV_real_exist_data_info.csv
  │   │   ├── California_total_real_exist_data_info.csv
  │   │   ├── Newyork_EV_real_exist_data_info.csv
  │   │   ├── Newyork_NoEV_real_exist_data_info.csv
  │   │   ├── Newyork_total_real_exist_data_info.csv
  │   │   ├── Texas_EV_real_exist_data_info.csv
  │   │   ├── Texas_NoEV_real_exist_data_info.csv
  │   │   ├── Texas_total_real_exist_data_info.csv
  │   │   ├── real_resident_list_1min.pkl
  │   │   └── resident_list_1min.pkl
  ├── model
  │   ├── loss.py
  │   ├── metrics.py
  │   └── models.py
  ├── model_pruning_compression
  │   ├── Edge_Running
  │   │   ├── Edge_Running&Test.py
  │   │   ├── test_data_.pkl
  │   │   └── test_label_.pkl
  │   └── model_pruning_compression.py
  ├── model_transfer
  │   ├── transfer_model.py
  │   ├── transfer_sampling_rate.py
  │   └── federated_model.py
  ├── experiment.py
  ├── experiment_record
  ├── global-state.pth
  ├── pickle_data
  │   ├── 3000.pkl
  │   ├── 3000_session.pkl
  │   ├── 661.pkl
  │   └── 661_session.pkl
  └── utils.py
 ```
## License

  MIT License.

---

#### Contact:
- Xudong Wang (xudongwang@link.cuhk.edu.cn)
- Guoming Tang (tangguo1999@gmail.com)
