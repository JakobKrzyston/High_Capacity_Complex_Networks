def eval(test_SNRs, model, tag, classes, snr_dataloader):
    """
    # Author
    Jakob Krzyston
    
    # Purpose 
    (1) Compute overall accuracy 
    (2) Compute overall confusion matrix  
    
    # Inputs
    test_SNRs      - (int) List of SNRs to be tested
    model          - Built architecture 
    tag            - (str) Namimg convention for the experiment
    classes        - (str) List on modulation classes
    snr_dataloader - SNR Dataloader 
    
    # Outputs
    Saved overall accuracy, confusion matrix, and avg inference time per batch
    """
    # Import Packages
    import os, time
    import numpy as np
    import torch
    
    # Function to plot the confusion matrix
    import confusion_matrix
    
    # Make an array of the test SNRs
    test_SNRs = np.arange(test_SNRs[0], test_SNRs[-1]+2,2)

    # Check for CUDA
    if torch.cuda.is_available():
        use_cuda = torch.cuda.is_available()

    # Compute overall accuracy, plot confusion matrix, and compute avg inference time per batch
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    total_time = 0
    y_all = []
    y_hat_all = []
    for x,y in snr_dataloader:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        x = x.to(device)
        model.eval()
        time_0 = time.time()
        y_hat = model(x)
        time_1 = time.time()
        total_time += time_1 - time_0
        device = torch.device("cpu")
        y = y.to(device)
        y_all.extend(y.numpy())
        y_hat = y_hat.to(device)
        y_hat_all.extend(y_hat.numpy())

    # Fill in the confusion matrix
    for i in range(len(y_all)):
        j = y_all[i]
        k = int(np.argmax(y_hat_all[i]))
        conf[j,k] = conf[j,k] + 1

    for i in range(len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    cor  = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    acc  = cor/(cor+ncor)

    # Save results
    np.save(os.getcwd()+'/'+tag+'/Computed_Values/Overall_Accuracy', acc.astype('float32'))
    print('Overall Accuracy: ' + str(round(acc,4)))
    np.save(os.getcwd()+'/'+tag+'/Computed_Values/Overall_Confusion_Matrix', confnorm.astype('float32'))
    filename = os.getcwd()+'/'+tag+'/Figures/Overall_Confusion_Matrix.png'
    confusion_matrix.plot(confnorm, filename, labels=classes)
    avg_inf_time = total_time/((len(snr_dataloader)*snr_dataloader.batch_size)/snr_dataloader.batch_size)
    print('Inference time per batch of test data (ms): ' + str(round(avg_inf_time*1000,6)))
    np.save(os.getcwd()+'/'+tag+'/Computed_Values/Avg_Infer_Time',avg_inf_time)