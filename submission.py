#!/usr/bin/env python
# coding: utf-8

# In[1]:


def submit_model(model):
    '''
    This is a function to create a file submission.csv
    to see performance on test set
    
    model - the predictor 
    '''
    # create a test data frame
    test_filenames = os.listdir("/kaggle/files/images/test1")
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    nb_samples = test_df.shape[0]


    test_gen = ImageDataGenerator()
    test_generator = test_gen.flow_from_dataframe(
        test_df, 
        "/kaggle/files/images/test1/", 
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=64
    )


    predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

    test_df['category'] = np.argmax(predict, axis=-1)

    label_map = dict((v,k) for k,v in train_generator.class_indices.items())
    test_df['category'] = test_df['category'].replace(label_map)

    test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['label'] = submission_df['category']
    submission_df.drop(['filename', 'category'], axis=1, inplace=True)
    submission_df.to_csv('submission.csv', index=False)
    
 
