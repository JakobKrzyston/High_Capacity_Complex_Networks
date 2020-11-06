def get(dataset, SPLIT, BATCH_SIZE):
    """
    # Author
    Jakob Krzyston (jakobk@gatech.edu)

    # Input
    dataset    - (str) name of the dataset
    SPLIT      -
    BATCH_SIZE -

    # Output (dict)
    train_dataloader - Dataloader for training 
    valid_dataloader - Dataloader for validation
    test_dataloader  - Dataloader for testing
    mods             - List of the modulations in the dataset
    test_snrs        - List of the SNRs in the test set
    """
    import data
    import numpy as np
    from torch.utils.data import DataLoader, random_split
    
    # Obtain the specified train and test sets
    if dataset == 'RML2016':
        print('Dataset: ' + dataset)
        mods = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK',\
                'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        samples = 1000 # number of samples per modulation per SNR
        print('Train SNRs: [-20, 18]')
        print('Test SNRs:  [-20, 18]')
        RML2016_dataset = data.RML2016_neg_20_18.Data()
        
        
    # Define variables for downstream processing
    test_snrs = np.arange(-20,20,2)
    
    # Split into train into train/val sets
    total = len(RML2016_dataset)
    lengths = [int(len(RML2016_dataset)*SPLIT)]
    lengths.append(total - lengths[0])
    print("Splitting data: {} training samples, {} validation samples".format(lengths[0], lengths[1]))
    train_set, val_set = random_split(RML2016_dataset, lengths)

    # Setup Dataloaders
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers = 2)
    val_dataloader   = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers = 2)
    ## SNR Dataloader is organized by SNR and unshuffled
    snr_dataloader  = DataLoader(RML2016_dataset, batch_size=samples, num_workers = 2)
   
    # Export Dataloaders via dictionary
    out = {'test_snrs':test_snrs,'mods':mods,'train_dataloader':train_dataloader,\
                     'val_dataloader':val_dataloader,'snr_dataloader':snr_dataloader}

    return out