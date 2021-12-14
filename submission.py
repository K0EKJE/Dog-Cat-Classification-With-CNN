{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fea41a9c-68ac-46fe-8ea5-cc37751d5435",
   "metadata": {},
   "outputs": [],
   "source": [
    "def submission(model):\n",
    "    \n",
    "    test = pd.DataFrame({'file': os.listdir('/kaggle/files/images/test1')})\n",
    "    \n",
    "    nb_samples = test.shape[0]\n",
    "    test_gen = ImageDataGenerator(rescale=1./255)\n",
    "    test_generator = test_gen.flow_from_dataframe(\n",
    "        test, \n",
    "        directory = destination + '/test/',\n",
    "        x_col='file',\n",
    "        y_col=None,\n",
    "        class_mode=None,\n",
    "        target_size=(224,224,3),\n",
    "        batch_size=64\n",
    "    )\n",
    "\n",
    "    predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/64))\n",
    "    test['categories'] = np.argmax(predict, axis=-1)\n",
    "    \n",
    "    label_map = dict((v,k) for k,v in train_generator.class_indices.items())\n",
    "    test['categories'] = test['categories'].replace(label_map)\n",
    "    test['categories'] = test['categories'].replace({ 'dog': 1, 'cat': 0 })\n",
    "    \n",
    "    submission_df = test.copy()\n",
    "    submission_df['id'] = submission_df['filename'].str.split('.').str[0]\n",
    "    submission_df['label'] = submission_df['category']\n",
    "    submission_df.drop(['filename', 'category'], axis=1, inplace=True)\n",
    "    submission_df.to_csv('submission.csv', index=False)\n",
    "    "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
