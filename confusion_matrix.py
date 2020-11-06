def plot(cm, filename, labels=[]):
        """
        # Author
        Jakob Krzyston

        # Purpose
        Plot confusion matrices
        Much of this borrowed from:
        https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb

        # Inputs
        cm         - (double) Array containing values for confusion matrix
        filename   - (str) Name of the saved image
        labels     - (str) List of the class/modulation names 

        # Output
        Saved confusion matrix
        """
        # Import Packages
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        
        #Confusion matrix
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(filename,transparent = True, bbox_inches = 'tight', pad_inches = 0.02)