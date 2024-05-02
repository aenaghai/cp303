from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import h5py

print("SVM for preamble detection")

import time
start_time = time.time()
print("Starting Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
print("starting the code \n")

print("Fetching data \n")
print("Fetching labels first \n")

labels = pd.read_csv('./TRAINING_LABELS.csv', header=None)
length = labels.shape[1]
print('shape of training labels : ', labels.shape)

print("Fetching data \n")
data_real = pd.read_csv('./TRAINING_DATA_REAL.csv', header=None)
print('shape of training data real : ', data_real.shape)

original_array = data_real.values
reshaped_data = original_array.reshape((65, 2, length), order='F')
reshaped_data_new = np.transpose(reshaped_data, (2, 0, 1))
reshaped_data_new = reshaped_data_new.reshape(length, 65, 2)
print('shape of reshaped training data : ', reshaped_data_new.shape)

original_labels = labels.values
reshaped_labels = original_labels.reshape((length,), order='F')
print('shape of reshaped training labels : ', reshaped_labels.shape)

print("Fetching validation data \n")
test_data_real = pd.read_csv('./VALID_DATA_REAL.csv', header=None)
test_labels = pd.read_csv('./VALID_LABELS.csv', header=None)
length_test = test_labels.shape[1]
test_array = test_data_real.values

reshaped_data_test = test_array.reshape((65, 2, length_test), order='F')
reshaped_data_test = np.transpose(reshaped_data_test, (2, 0, 1))
reshaped_data_test = reshaped_data_test.reshape(length_test, 65, 2)
test_labels_original = test_labels.values
reshaped_labels_test = test_labels_original.reshape((length_test,), order='F')
print('shape of reshaped testing data : ', reshaped_data_test.shape)
print('shape of reshaped testing labels : ', reshaped_labels_test.shape)

print("Splitting data into train and validation sets \n")
X_train, X_val, y_train, y_val = train_test_split(reshaped_data_new, reshaped_labels, test_size=0.2, random_state=42)

print("Training SVM model \n")
svm_model = SVC(class_weight={0: 1, 1: 2})  # Class weights
svm_model.fit(X_train.reshape((X_train.shape[0], -1)), y_train)

print("Saving the SVM model as svm.h5 \n")
with h5py.File('svm.h5', 'w') as f:
    # Save model parameters
    f.create_dataset('support_vectors', data=svm_model.support_vectors_)
    f.create_dataset('dual_coef', data=svm_model.dual_coef_)
    f.create_dataset('intercept', data=svm_model.intercept_)
    f.create_dataset('classes', data=svm_model.classes_)

print("Evaluating the model \n")
train_accuracy = svm_model.score(X_train.reshape((X_train.shape[0], -1)), y_train)
val_accuracy = svm_model.score(X_val.reshape((X_val.shape[0], -1)), y_val)

print(f"Accuracy on training data: {train_accuracy}")
print(f"Accuracy on validation data: {val_accuracy}")

end_time = time.time()
elapsed_time = end_time - start_time
print("Starting Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
print("Ending Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
print("Elapsed Time:", elapsed_time, "seconds")

print("Process ends here")
