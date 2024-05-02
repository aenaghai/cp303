import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dropout, Dense, Activation, Input, Multiply
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def main():
    print("Attention with real data only and using class weights")

    start_time = time.time()
    print("Starting Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("starting the code \n")

    # Fetching data
    labels = pd.read_csv('./TRAINING_LABELS.csv', header=None)
    data_real = pd.read_csv('./TRAINING_DATA_REAL.csv', header=None)
    original_array = data_real.values
    reshaped_data = original_array.reshape((65, 2, labels.shape[1]), order='F')
    reshaped_data_new = np.transpose(reshaped_data, (2, 0, 1))
    reshaped_data_new = reshaped_data_new.reshape(labels.shape[1], 65, 2, 1)

    original_labels = labels.values
    reshaped_labels = original_labels.reshape((labels.shape[1], 1), order='F')

    # Fetching validation data
    test_data_real = pd.read_csv('./VALID_DATA_REAL.csv', header=None)
    test_labels = pd.read_csv('./VALID_LABELS.csv', header=None)
    test_array = test_data_real.values
    reshaped_data_test = test_array.reshape((65, 2, test_labels.shape[1]), order='F')
    reshaped_data_test = np.transpose(reshaped_data_test, (2, 0, 1))
    reshaped_data_test = reshaped_data_test.reshape(test_labels.shape[1], 65, 2, 1)
    test_labels_original = test_labels.values
    reshaped_labels_test = test_labels_original.reshape((test_labels.shape[1], 1), order='F')

    val_data, val_labels = reshaped_data_test, reshaped_labels_test  # Use test data as validation

    # Model architecture and compilation
    inputs = Input(shape=(65, 2, 1))
    conv1 = Conv2D(32, (3, 2), activation='relu')(inputs)
    batch_norm1 = BatchNormalization()(conv1)
    conv2 = Conv2D(32, (3, 1), activation='relu')(batch_norm1)
    batch_norm2 = BatchNormalization()(conv2)
    flatten = Flatten()(batch_norm2)
    dropout1 = Dropout(0.05)(flatten)
    dense1 = Dense(128, activation='relu')(dropout1)
    dense2 = Dense(64, activation='relu')(dense1)
    dropout2 = Dropout(0.05)(dense2)
    dense3 = Dense(32, activation='relu')(dropout2)
    dense4 = Dense(16, activation='relu')(dense3)
    dense5 = Dense(8, activation='relu')(dense4)
    dense6 = Dense(1, activation='sigmoid')(dense5)

    # Apply attention to the output of the second dense layer
    attention_probs = Dense(1, activation='softmax')(dense5)
    attention_mul = Multiply()([dense5, attention_probs])

    model = Model(inputs=inputs, outputs=attention_mul)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    class_weights = {0: 1, 1: 2}
    model.fit(reshaped_data_new, reshaped_labels, validation_data=(val_data, val_labels), epochs=500, batch_size=500,
              class_weight=class_weights)

    # Evaluate the model
    loss, accuracy = model.evaluate(reshaped_data_new, reshaped_labels)
    print(f"Loss on training data: {loss}")
    print(f"Accuracy on training data: {accuracy}")

    loss, accuracy = model.evaluate(val_data, val_labels)
    print(f"Loss on validation data: {loss}")
    print(f"Accuracy on validation data: {accuracy}")

    # Save the model
    model.save("attention")

    # Predictions and evaluation on test data
    print("Testing the trained model")
    model_test_output = model.predict(val_data)

    # Compute precision, recall, and F1-score
    threshold = 0.5
    temp_res_2 = (model_test_output >= threshold).astype(int)
    print("Obtaining classification report for threshold =", threshold)
    report = classification_report(val_labels, temp_res_2, zero_division=1)
    print(report)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Starting Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("Ending Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print("Elapsed Time:", elapsed_time, "seconds")

    print("Process ends here")

if __name__ == "__main__":
    main()
