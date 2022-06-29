#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from curses import meta
from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import os
import tqdm
import numpy as np
import tensorflow as tf
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################



def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights
# Train your model.
def cv_challenge_model(data_folder, result_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')


    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(result_folder, exist_ok=True)
    #TODO: remove this:
    #classes = ['Present', 'Unknown', 'Absent']
    #num_classes = len(classes)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    data = []
    #labels = list()
    murmurs = list()
    outcomes = list()
    frequency = 500
    div = 2
    new_freq = frequency/div
    #new_sig_len = 550

    data = np.zeros((num_patient_files,4,int(32256/div)))

    for i in tqdm.tqdm(range(num_patient_files)):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings, freq = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        
        indx = get_lead_index(current_patient_data)
        extracted_recordings = np.asarray(current_recordings)[indx]
        extracted_freq = np.asarray(freq)[indx]
        for j in range(len(extracted_recordings)):
            
            data_temp = signal.resample(extracted_recordings[j], int((len(extracted_recordings[j])/extracted_freq[j]) * new_freq))
            data[i,j,:len(data_temp)] = data_temp


        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)

        #TODO: remove this:
        # Extract labels and use one-hot encoding.
        #current_labels = np.zeros(num_classes, dtype=int)
        #label = get_label(current_patient_data)
        #if label in classes:
        #    j = classes.index(label)
        #    current_labels[j] = 1
        #labels.append(current_labels)

    #labels = np.vstack(labels)
    murmurs = np.vstack(murmurs)
    outcomes = np.argmax(np.vstack(outcomes),axis=1)
    data_numpy = np.asarray(data)
    print(f"Number of signals = {data_numpy.shape[0]}")
    '''
    # Loop through all data and find the sound recording lengths
    sig_len = []
    for i in tqdm.tqdm(data):
        sig_len.append(len(i))
    sig_len = np.asarray(sig_len)    
    print(f"Signal max length: {np.asarray(sig_len).max()}")
    data_padded = np.zeros((data_numpy.shape[0],np.asarray(sig_len).max()))
    #data_padded = np.zeros((data_numpy.shape[0],new_sig_len))
    for i in tqdm.tqdm(range(data_numpy.shape[0])):
        data_padded [i] = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(data_numpy[i],0),
                                                                        maxlen=sig_len.max(),
                                                                        padding='post',truncating='post', value=0.0)
    '''
    # The prevalence of the 3 different labels
    print("Murmurs prevalence:")
    print(f"Present = {np.where(np.argmax(murmurs,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(murmurs,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(murmurs,axis=1)==2)[0].shape[0]}")

    print("Outcomes prevalence:")
    print(f"Abnormal = {len(np.where(outcomes==0)[0])}, Normal = {len(np.where(outcomes==1)[0])}")
    
    #TODO: Implement weighting
    #new_weights=calculating_class_weights(labels)
    #keys = np.arange(0,labels.shape[1],1)
    #weight_dictionary = dict(zip(keys, new_weights.T[1]))
    
    data_numpy = np.moveaxis(data_numpy, 1, -1)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    FOLDS = 10
    skf = StratifiedKFold(n_splits=FOLDS)

    murmur_probas = []
    outcome_probas = []
    murmur_trues = []
    outcome_trues = []

    for train_index, val_index in skf.split(data_numpy, np.argmax(murmurs,axis=1)):
        X_train, X_val = data_numpy[train_index], data_numpy[val_index]
        y1_train, y1_val = murmurs[train_index], murmurs[val_index]
        y2_train, y2_val = outcomes[train_index], outcomes[val_index]
        # Train the model.
        clinical_model = build_clinical_model(data_numpy.shape[1],data_numpy.shape[2])

        murmur_model = build_murmur_model(data_numpy.shape[1],data_numpy.shape[2])
        #model = inception_model(data_padded.shape[1],1,labels.shape[1])
        #TODO: Add GPU strategy

        epochs = 25
        batch_size = 20
        murmur_model.fit(x=X_train, y=y1_train, epochs=epochs, batch_size=batch_size,   
                verbose=1,
                #class_weight=weight_dictionary,
                callbacks=[lr_schedule])

        clinical_model.fit(x=X_train, y=y2_train, epochs=epochs, batch_size=batch_size,   
                verbose=1,
                #class_weight=weight_dictionary,
                callbacks=[lr_schedule])
        #model.save(os.path.join(model_folder, 'model.h5'))

        murmur_probabilities = murmur_model.predict(X_val)

        outcome_probabilities = clinical_model.predict(X_val)


        print(accuracy_score(y1_val,(murmur_probabilities>0.5)*1))
        print(accuracy_score(y2_val,(outcome_probabilities>0.5)*1))

        murmur_probas.append(murmur_probabilities)
        outcome_probas.append(outcome_probabilities)
        murmur_trues.append(y1_val)
        outcome_trues.append(y2_val)

    return murmur_probas, outcome_probas, murmur_trues, outcome_trues

        # Save the model.
        #save_challenge_model(model_folder, classes, imputer, classifier)


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.h5'))
    model.compile(loss=[tf.keras.losses.CategoricalCrossentropy(),tf.keras.losses.BinaryCrossentropy()], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    frequency = 500
    div = 2
    new_freq = frequency/div
    #classes = ['Present', 'Unknown', 'Absent']
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']
    #label = np.zeros(len(classes),dtype=int)
    # Load the data.
    indx = get_lead_index(data)
    extracted_recordings = np.asarray(recordings)[indx]
    data_padded = np.zeros((1,4,int(32256/div)))
    new_sig_len = model.get_config()['layers'][0]['config']['batch_input_shape'][1]
    freq = get_frequency(data)

    for i in range(len(extracted_recordings)):
        rec = np.asarray(extracted_recordings[i])
        resamp_sig = signal.resample(rec, int((len(rec)/freq) * new_freq))
        data_padded[0,i,:]  = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(resamp_sig,0),
                                    maxlen=int(new_sig_len),padding='post',truncating='post', value=0.0)
    

    data_padded = np.moveaxis(data_padded,1,-1)                                                                       
    murmur_probabilities, outcome_probabilities = model.predict(data_padded)
    
    
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = (outcome_probabilities<0.5)*1
    outcome_labels[idx] = 1

    outcome_probabilities = np.array([outcome_probabilities[0],1-outcome_probabilities[0]])
    #print("labels:")
    #print(murmur_labels.shape)
    #print(outcome_labels.shape)


    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    #print("outcome_probabilities:")
    #print(murmur_probabilities.ravel().shape)
    #print(outcome_probabilities.ravel().shape)
    probabilities = np.concatenate((murmur_probabilities.ravel(), outcome_probabilities.ravel()))
    
    # Choose label with higher probability.
    #idx = np.argmax(probabilities, axis=1)
    #label[idx] = 1
    #print(f"Predicted label = {label}")
    #print(f"Predicted class: {classes[np.argmax(label)]}")

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)

def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    murmur_output = tf.keras.layers.Dense(3, activation='softmax', name="murmur_output")(gap_layer)
    clinical_output = tf.keras.layers.Dense(1, activation='sigmoid', name="clinical_output")(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[murmur_output,clinical_output])
    model.compile(loss={'murmur_output': "categorical_crossentropy", 'clinical_output': "binary_crossentropy"}, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def build_murmur_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    murmur_output = tf.keras.layers.Dense(3, activation='softmax', name="murmur_output")(gap_layer)
    #clinical_output = tf.keras.layers.Dense(1, activation='sigmoid', name="clinical_output")(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=murmur_output)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def build_clinical_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    clinical_output = tf.keras.layers.Dense(1, activation='sigmoid', name="clinical_output")(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=clinical_output)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    return model

def get_lead_index(patient_metadata):    
    lead_name = []
    lead_num = []
    cnt = 0
    for i in patient_metadata.splitlines(): 
        if i.split(" ")[0] == "AV" or i.split(" ")[0] == "PV" or i.split(" ")[0] == "TV" or i.split(" ")[0] == "MV":
            if not i.split(" ")[0] in lead_name:
                lead_name.append(i.split(" ")[0])
                lead_num.append(cnt)
            cnt += 1
    return np.asarray(lead_num)

def scheduler(epoch, lr):
    if epoch == 10:
        return lr * 0.1
    elif epoch == 15:
        return lr * 0.1
    elif epoch == 20:
        return lr * 0.1
    else:
        return lr