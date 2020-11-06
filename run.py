"""
# Author
Jakob Krzyston (jakobk@gatech.edu)

"""

def main():
    ## Import packages and functions
    import os, argparse, time
    import numpy as np
    import matplotlib.pyplot as plt
    import get_data, get_model, snr_acc, overall_acc
    import torch
    from torch import nn
    from torch import optim

    # Handle input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False, default = 'RML2016' )
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--train_pct', type=int, required=False, default = 50)
    parser.add_argument('--load_weights', type=int, required=False, default = 0)
    parser.add_argument('--trial', type=int, required=False, default=0)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    args = parser.parse_args()

    # Extract the data
    data_set = get_data.get(args.dataset, args.train_pct/100, BATCH_SIZE=256)
    train_dataloader = data_set['train_dataloader']
    val_dataloader   = data_set['val_dataloader']

    # If loading weights
    load_weights = args.load_weights

    # Specify file tag to ID the results from this run
    tag = args.dataset+'_train_-20_18_test_-20_18'+'_arch_'+args.arch+\
                         '_trial_'+str(args.trial)

    # Setup directories to organize results if training
    if args.load_weights == 0:
        sub_folders = ['Figures', 'Computed_Values']
        for i in range(len(sub_folders)):
            path = os.path.join(os.getcwd(),tag+'/'+sub_folders[i])
            os.makedirs(path, exist_ok = True)

    # Get the model
    model = get_model.get(args.arch)
    model_path = os.path.join(os.getcwd(),'./Weights/'+tag+'.pth') # Where to save weights
    
    # Hardware Check, Loss Function, and Optimizer
    criterion = nn.CrossEntropyLoss()
    if args.arch == 'Complex': # to match the settings of Krzyston et al., 2020
        optimizer = optim.Adam(params = model.parameters(), lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
    else:
        optimizer = optim.SGD(params = model.parameters(), lr = 0.1, momentum = 0.9)
    if torch.cuda.is_available():
        print('CUDA is available')
        use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = False
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Train model or load saved model
    if args.load_weights == 1:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.to(device)
        model.eval()
        print('Model Loaded')
    else: # Train the model

        # Setup early stopping
        patience_counter = 0
        patience = 5

        # Setup Training
        epochs = args.epochs
        train_losses = []
        valid_losses = []
        val_best = np.Inf
        best_ep = 0
        
        # Train
        start_all = time.time()
        for e in range(args.epochs):
            start_ep = time.time()
            running_loss = 0
            rl = 0
            model.train()
            for data, labels in train_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            with torch.no_grad():
                for dv,lv in val_dataloader:
                    dv = dv.to(device)
                    lv = lv.to(device)
                    model.eval()
                    op = model(dv)
                    ll = criterion(op, lv)
                    rl += ll.item()

            train_loss = running_loss/len(train_dataloader)
            val_loss = rl/len(val_dataloader)
            train_losses.append(train_loss)
            valid_losses.append(val_loss)

            if val_loss<val_best:
                val_best = val_loss
                checkpoint = {'state_dict':model.state_dict()}
                torch.save(checkpoint,model_path)
                best_ep = e
                patience_counter = 0
            else: #early stopping
                patience_counter += 1
                if patience_counter == patience-1:
                    end_ep = time.time()
                    print('Epoch: '+str(e))
                    print(' - ' + str(round(end_ep-start_ep,3)) + 's - train_loss: '+\
                          str(round(running_loss/len(train_dataloader),4))+' - val_loss: '\
                          +str(round(rl/len(val_dataloader),4)))
                    break

            end_ep = time.time()
            print('Epoch: '+str(e))
            print(' - ' + str(round(end_ep-start_ep,3)) + 's - train_loss: '+\
              str(round(running_loss/len(train_dataloader),4))+' - val_loss: '\
              +str(round(rl/len(val_dataloader),4)))
        end_all = time.time()
        print('Total training time = ' + str(round((end_all-start_all)/60,3)) + ' minutes')

        # PLot training and validation losses
        plt.plot(train_losses,label='Train')
        plt.plot(valid_losses,label='Valid')
        plt.legend()
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(os.getcwd(),tag+'/Figures/Train_Valid_Losses.png'))
        
        # Load best performing weights for inference
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.to(device)
        model.eval()
        print('Weights Loaded for Inference')

    # Overall accuracy and speed test
    overall_acc.eval(data_set['test_snrs'], model, tag, data_set['mods'], data_set['snr_dataloader'])

    # Accuracy by SNR
    snr_acc.eval(data_set['test_snrs'], model, tag, data_set['mods'], data_set['snr_dataloader'])
   

if __name__ == '__main__':
    # from torch.multiprocessing import freeze_support
    # freeze_support()
    main()
