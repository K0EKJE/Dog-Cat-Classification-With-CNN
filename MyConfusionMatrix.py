from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def my_confusion_matrix(model):
    '''
    This function creates a confusion matrix to visualize result on validation set.
    
    model - the model used for prediction
    '''
    predict = model.predict_generator(validation_generator,
                                      steps=np.ceil(val_set.shape[0] / 64))

    val_set['predict'] = np.argmax(predict, axis=-1)
    labels = dict((v,k) for k,v in train_generator.class_indices.items())
    val_set['cat/dog'] = val_set['predict'].map(labels)

    fig, ax = plt.subplots(figsize = (9, 6))

    cm = confusion_matrix(val_set["categories"], val_set["cat/dog"])
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["cat", "dog"])
    disp.plot(cmap = plt.cm.Blues, ax = ax)

    ax.set_title("Validation Set")
    plt.show()
