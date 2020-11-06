def eval(test_SNRs, model, tag, classes, snr_dataloader):
    """
    # Author
    Jakob Krzyston
    
    # Purpose
    At each SNR in the test set: 
    (1) Compute the accuracy 
    (2) Compute the confusion matrix  
    
    # Inputs
    test_SNRs      - (int) List of SNRs to be tested
    model          - Built architecture 
    tag            - (str) Namimg convention for the experiment
    classes        - (str) List on modulation classes
    snr_dataloader - SNR Dataloader 
    
    # Outputs
    Saved accuracies and confusion matrices at every SNR
    """
    # Import Packages
    import os
    import numpy as np
    import torch
    
    # Function to plot the confusion matrices
    import confusion_matrix
    
    # Make an array of the test SNRs
    test_SNRs = np.arange(test_SNRs[0], test_SNRs[-1]+2,2)
    
    # Check for CUDA
    if torch.cuda.is_available():
        use_cuda = torch.cuda.is_available()
    
    # Compute Acuracy and plot confusion matrices by SNR
    count = 1
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    acc = np.zeros(len(test_SNRs))
    s = 0
    for x,y in snr_dataloader:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        x = x.to(device)
        model.eval()
        y_hat = model(x)
        device = torch.device("cpu")
        y = y.to(device)
        y = y.numpy()
        y_hat = y_hat.to(device)
        y_hat = y_hat.numpy()

        # Fill in the confusion matrix
        for i in range(y.shape[0]):
            j = y[i]
            k = int(np.argmax(y_hat[i]))
            conf[j,k] = conf[j,k] + 1

        # After each SNR
        if count % len(classes) == 0:
            for i in range(len(classes)):
                confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
            if s == 0 or s == len(test_SNRs)-1:
                filename = os.getcwd()+'/'+tag+'/Figures/Confusion_Matrix_SNR_' + str(test_SNRs[s])+'.png'
                confusion_matrix.plot(confnorm, filename, labels=classes)
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            acc[s] = 1.0*cor/(cor+ncor)
            conf = np.zeros([len(classes),len(classes)])
            confnorm = np.zeros([len(classes),len(classes)])
            count = 1
            s += 1
        else:
            count += 1
    
    # Save results
    np.save(os.getcwd()+'/'+tag+'/Computed_Values/SNR_Accuracy', acc.astype('float32'))
    print('Accuracy by SNR:\n' + str(acc))