# High_Capacity_Complex_Networks
## About
Code for, "HIGH-CAPACITY COMPLEX CONVOLUTIONAL NEURAL NETWORKS FOR I/Q MODULATION CLASSIFICATION".

## Data
Data for this submission (RML2016.10a.tar.bz2) can be found at: https://www.deepsig.io/datasets. To ensure proper execution of the code, be sure the data is saved as 'RML2016.10a_dict.pkl' or 'RML2016.10a_dict.dat'.

## Code

The following code will execute an example experiment training and testing across all SNR levels: (be sure to include the path to the dataset)
```
python3 run.py --data_directory path_to_data --train_SNRs -20 18 --test_SNRs -20 18 --arch Complex
```
The code automatically saves and stores results into folders in the local directory. 

*Disclaimer: All code is written in PyTorch
