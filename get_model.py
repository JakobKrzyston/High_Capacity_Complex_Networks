def get(arch):
    """
    Author: Jakob Krzyston
    
    Purpose:
    Get specified model.
    
    Input:
    arch - (str) shorthand name for a model
            see the respective imported scripts for model details
    Ouput:
    model - PyTorch model
    
    """
    import models_ieee_icc
    import models_res
    import models_dense
    import dense_res
    
    if arch == 'Complex':
        print('Architecture: Krzyston et al. 2020')
        model = models_ieee_icc.Complex() 
    elif arch   == 'res_18':
        print('Architecture: ResNet-18')
        model = models_res.resnet18()
    elif arch == 'res_18_c':
        print('Architecture: Complex ResNet-18')
        model = models_res.resnet18_c()
    elif arch == 'res_34':
        print('Architecture: ResNet-34')
        model = models_res.resnet34()
    elif arch == 'res_34_c':
        print('Architecture: Complex ResNet-34')
        model = models_res.resnet34_c()
    elif arch == 'dense_57':
        print('Architecture: DenseNet-57')
        model = models_dense.densenet57()
    elif arch == 'dense_57_c':
        print('Architecture: Complex DenseNet-57')
        model = models_dense.densenet57_c()
    elif arch == 'dense_73':
        print('Architecture: DenseNet-73')
        model = models_dense.densenet73()
    elif arch == 'dense_73_c':
        print('Architecture: Complex DenseNet-73')
        model = models_dense.densenet73_c()
    elif arch == 'dense_res_35':
        print('Architecture: DenseResNet-35')
        model = dense_res.denseresnet35()
    elif arch == 'dense_res_35_c':
        print('Architecture: Complex DenseResNet-35')
        model = dense_res.denseresnet35_c()
    elif arch == 'dense_res_68':
        print('Architecture: DenseResNet-68')
        model = dense_res.denseresnet68()
    elif arch == 'dense_res_68_c':
        print('Architecture: Complex DenseResNet-68')
        model = dense_res.denseresnet68_c()
    else:
        raise Exception('Invalid Architecture Name: ' + str(arch))
        
    return model