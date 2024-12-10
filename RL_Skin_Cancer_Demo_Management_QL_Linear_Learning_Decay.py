from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import math
import sklearn.metrics as metrics
from tqdm import tqdm
from gym import Env, spaces
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow.python.keras import backend
from tensorflow.keras.backend import clear_session

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv

import tensorflow as tf
import tensorflow.compat.v1 as tf1
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

image_size = 315

info_type = 'both' #prob,features,both
balance_episode = True

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

from sklearn.feature_extraction.text import CountVectorizer

def read_and_decode(dataset, batch_size, is_training, data_size,n_patients):
    if is_training:
        dataset = dataset.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.prefetch(buffer_size=data_size // batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.repeat(None)
    return dataset

def initialize_clinical_practice(clinical_cases_feat,clinical_cases_labels,dataset_size,n_classes,is_training,n_patients,set_distribution):

    if is_training and balance_episode:
        _, counts = np.unique(clinical_cases_labels, return_counts=True)

        akiec = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 0), axis=0))
        akiec_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 0), axis=0))

        bcc = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 1), axis=0))
        bcc_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 1), axis=0))

        bkl = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 2), axis=0))
        bkl_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 2), axis=0))

        df = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 3), axis=0))
        df_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 3), axis=0))

        mel = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 4), axis=0))
        mel_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 4), axis=0))

        nv = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 5), axis=0))
        nv_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 5), axis=0))

        vasc = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 6), axis=0))
        vasc_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 6), axis=0))

        akiec_set = tf.data.Dataset.from_tensor_slices((akiec, akiec_labels)).shuffle(buffer_size=counts[0],
                                                                                      reshuffle_each_iteration=True).repeat()
        bcc_set = tf.data.Dataset.from_tensor_slices((bcc, bcc_labels)).shuffle(buffer_size=counts[1],
                                                                                reshuffle_each_iteration=True).repeat()
        bkl_set = tf.data.Dataset.from_tensor_slices((bkl, bkl_labels)).shuffle(buffer_size=counts[2],
                                                                                reshuffle_each_iteration=True).repeat()
        df_set = tf.data.Dataset.from_tensor_slices((df, df_labels)).shuffle(buffer_size=counts[3],
                                                                             reshuffle_each_iteration=True).repeat()
        mel_set = tf.data.Dataset.from_tensor_slices((mel, mel_labels)).shuffle(buffer_size=counts[4],
                                                                                reshuffle_each_iteration=True).repeat()
        nv_set = tf.data.Dataset.from_tensor_slices((nv, nv_labels)).shuffle(buffer_size=counts[5],
                                                                             reshuffle_each_iteration=True).repeat()
        vasc_set = tf.data.Dataset.from_tensor_slices((vasc, vasc_labels)).shuffle(buffer_size=counts[6],
                                                                                   reshuffle_each_iteration=True).repeat()

        dataset_train = tf.data.Dataset.sample_from_datasets([akiec_set, bcc_set, bkl_set, df_set, mel_set, nv_set, vasc_set], weights=set_distribution)
        dataset_train = dataset_train.batch(1)

    else:
        dataset_train = tf.data.Dataset.from_tensor_slices((clinical_cases_feat,clinical_cases_labels))
        dataset_train = read_and_decode(dataset_train, 1, is_training, dataset_size,n_patients)

    patients = iter(dataset_train)

    return patients

def get_next_patient(patients):
    patient_scores,patient_diagnostics = patients.get_next()

    return np.squeeze(patient_scores),patient_diagnostics.numpy()[0]

class Dermatologist(Env):

    def __init__(self,patients,n_classes,vocab,n_actions):
        # Actions we can take, either skin lesion classes or don't know
        self.action_space = spaces.Discrete(n_actions)
        # Observation space - softmax + features after GAP
        self.observation_space = spaces.Box(-1*math.inf*np.ones((n_classes,)),math.inf*np.ones((n_classes,)))
        # Initialize state
        n_state,n_gt = get_next_patient(patients)
        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt
        # Set shower length
        self.number_of_patients = 0

    def step(self,patients,n_patients,n_actions,action):

        if n_actions == 2:
            # 2 actions
            reward_table = np.array([[-3, -4,  4,  4, -5,  4,  4], #dismiss
                                     [ 3,  4, -1, -1,  5, -1, -1]],np.float32)
         ## PERSONAL (Expert 2) - 2 actions
            #reward_table = np.array([[-5, -5, 20, 20, -20, 20, 20], #dismiss
            #                         [ 5,  5, -5, -5,  20, -5, -5], #excise
            #                         ],np.float32)
            
            # ## PERSONAL (Expert 3) - 2 actions
            # reward_table = np.array([[-1, -5,  2,  2, -10,  2,  2], #dismiss
            #                          [ 2,  5, -3, -3,  10, -3, -3], #excise
            #                         ],np.float32)
            
            # ## PERSONAL (Expert 4) - 2 actions
            # reward_table = np.array([[-3, -4,  3,  3, -5,  3,  3], #dismiss
            #                          [ 1,  2, -3, -3,  5, -3, -3], #excise
            #                         ],np.float32)
            
            # ## PERSONAL (Expert 5) - 2 actions
            # reward_table = np.array([[-3, -5,  5,  5, -5,  5,  5], #dismiss
             #                         [ 5,  5, -5, -5,  5, -5, -5], #excise
             #                        ],np.float32)
            
            # ## PERSONAL (Expert 6) - 2 actions
            #reward_table = np.array([[-3, -3,  5,  5, -5,  5,  5], #dismiss
            #                       [ 1,  4, -1, -1,  5, -1, -1], #excise
            #                       ],np.float32)
            
            # ## PERSONAL (Expert 7) - 2 actions
            #reward_table = np.array([[  1,  1,  5,  5, -3,  5,  5], #dismiss
            #                          [ -3,  5, -5, -5,  5, -5, -5], #excise
            #                         ],np.float32)
            
            # ## PERSONAL (Expert 8) - 2 actions
             #reward_table = np.array([[ -1, -3,  5,  5, -5,  5,  5], #dismiss
            #                          [  1,  3, -1, -1,  5, -1, -1], #excise
              #                       ],np.float32)
            
            # ## PERSONAL (Expert 9) - 2 actions
            #reward_table = np.array([[-1, -2,  5,  5, -5,  5,  5], #dismiss
            #                         [ 2,  4, -4, -4,  5, -4, -4], #excise
            #                         ],np.float32)

            # ## PERSONAL (Expert 10) - 2 actions
             #reward_table = np.array([[-2, -3,  4,  4, -5,  4,  4], #dismiss
             #                         [ 3,  5, -3, -3,  5, -3, -3], #excise
             #                        ],np.float32)
        else:
            # 3 actions
            reward_table = np.array([[  -2,  -3,   5,   5, -5,   5,    5], #dismiss
                                     [   3,   1,  -1,  -1, -5,  -1,   -1], #cryo
                                     [   2, 4.5,  -3,  -3,  5,  -3,   -3]],np.float32)

        self.revised_state = tf.one_hot(action,n_actions)

        reward = reward_table[action,self.gt]

        n_state, n_gt = get_next_patient(patients)

        old_gt = self.gt

        self.state = n_state
        self.gt = n_gt

        self.number_of_patients += 1

        # Check if checking patients is done
        if self.number_of_patients >= n_patients:
            done = 1
        else:
            done = 0

        return self.revised_state, self.state, reward,done,old_gt

    def reset(self,clinical_cases_feat,clinical_cases_labels,n_classes,dataset_size,vocab,is_training,n_patients,sample_distribution):
        # Reset clinical practice
        patients = initialize_clinical_practice(clinical_cases_feat,clinical_cases_labels, dataset_size,n_classes,is_training,n_patients,sample_distribution)
        n_state, n_gt = get_next_patient(patients)
        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt
        # Reset new practice
        self.number_of_patients = 0

        return self.state,patients

def main(_):
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 0.2  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 0.2  # Maximum epsilon greedy parameter
    epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
    initial_alpha = 0.1  # Initial learning rate
    decay_rate = 0.01  # Decay rate for learning rate

    #### Import Datasets ####
    tf1.enable_eager_execution()

    database = pd.read_csv('data/vectorDB.csv')
    labels = np.asarray(database['dx'])
    labels[labels == 'scc'] = 'akiec'

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    vocab = le.classes_
    n_words = len(vocab)

    features1 = np.load("data/nmed_rn34_ham10k_vectors.npy")
    features2 = pd.read_csv("data/vectorDB.csv")
    features2.pop('dx')
    features2 = np.asarray(features2, dtype='float32')

    features = np.concatenate([features1, features2], axis=1)
    
    _, counts = np.unique(labels, return_counts=True)
    counts = counts / np.sum(counts)
        
    labels_cat = le.transform(labels)
    train_feat, val_feat, train_labels, val_labels = train_test_split(
        features, labels_cat, test_size=0.2, random_state=111, stratify=labels_cat
    )

    patients = initialize_clinical_practice(train_feat, train_labels, train_labels.shape[0], True, n_words, Flags.n_patients, counts)
    derm = Dermatologist(patients, n_words, vocab, Flags.n_actions)

    state_dim = derm.state.shape[0]
    n_actions = Flags.n_actions
    W = np.zeros((state_dim, n_actions), dtype=np.float32)

    def get_Q_values(state):
        return np.dot(state, W)

    def update_weights(state, action, td_error, alpha):
        W[:, action] += alpha * td_error * state

    episode_reward_history = []
    episode_val_reward_history = []
    best_reward = -1 * math.inf
    best_actions_table = None

    epsilon_random_frames = 20
    epsilon_greedy_frames = 100000.0
    iter_count = 0

    for episode in range(Flags.n_episodes):
        alpha = initial_alpha / (1 + decay_rate * episode)  # Adaptive learning rate
        print(f"Episode {episode}: Learning Rate = {alpha:.4f}")

        i = 1
        print('Starting episode ', episode)

        done = False
        episode_score = 0

        state = derm.state
        n_not_random = 0

        while not done:
            try:
                iter_count += 1

                if iter_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    action = derm.action_space.sample()
                else:
                    q_values = get_Q_values(state)
                    action = np.argmax(q_values)
                    n_not_random += 1

                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

                revised_state, n_state, reward, done, _ = derm.step(patients, Flags.n_patients, Flags.n_actions, action)
                episode_score += reward

                q_values_next = get_Q_values(n_state)
                q_values_curr = get_Q_values(state)
                old_Q = q_values_curr[action]

                td_target = reward + gamma * np.max(q_values_next) * (1 - done)
                td_error = td_target - old_Q

                update_weights(state, action, td_error, alpha)

                state = n_state
                i += 1

            except tf.python.framework.errors_impl.OutOfRangeError:
                done = True
                break

        print('The episode duration was ', i - 1)
        print('The episode reward was ', episode_score)
        print('The number of not random actions was ', n_not_random)
        episode_reward_history.append(episode_score)

        state, patients_val = derm.reset(val_feat, val_labels, val_labels.shape[0], n_words, vocab, False, Flags.n_patients, counts)
        done = False
        management = np.array([])
        true_label = np.array([])
        actions_table = np.zeros([len(vocab), Flags.n_actions])
        episode_val_score = 0

        while not done:
            try:
                true_label = np.append(true_label, derm.gt)
                diag = derm.gt

                # Greedy policy for validation
                q_values = get_Q_values(state)
                action = np.argmax(q_values)

                management = np.append(management, action)
                _, state, reward, done, _ = derm.step(patients_val, val_labels.shape[0], Flags.n_actions, action)
                episode_val_score += reward
                actions_table[diag, action] += 1  # Track action counts per class

            except tf.python.framework.errors_impl.OutOfRangeError:
                done = True
                break

        print('The reward of the validation episode was ', episode_val_score)
        episode_val_reward_history.append(episode_val_score)

        # Update best reward and actions table
        if best_reward < episode_val_score:
            best_reward = episode_val_score
            best_actions_table = actions_table.copy()

        # Return to train
        _, patients = derm.reset(train_feat, train_labels, train_labels.shape[0], n_words, vocab, True, Flags.n_patients, counts)


    print('The scores for best validation Reward are:')
    print(best_actions_table)
    print('The best reward was ', best_reward)

    print('At the end of training the scores were')
    print(actions_table)


    plt.figure(1)
    plt.plot(episode_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Train')
    plt.show()

    plt.figure(2)
    plt.plot(episode_val_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Val')
    plt.show()

    print('The scores for best validation Reward are:')
    print(best_actions_table)
    print('The best reward was ', best_reward)

    print('At the end of training the scores were')
    print(actions_table)
    print('The final reward was ', episode_val_score)

    plt.figure(1)
    plt.plot(episode_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Train')
    plt.show()

    plt.figure(2)
    plt.plot(episode_val_reward_history)
    plt.plot(baseline_best_history)
    plt.plot(baseline_workse_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Val')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_patients',
        type=int,
        default= 100,
        help='Number of patients per episode.'
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default= 120,
        help='Number of episodes to play'
    )
    parser.add_argument(
        '--n_actions',
        type=int,
        default= 3,
        help='Number of actions available.'
    )
    Flags, unparsed = parser.parse_known_args()
    tf1.app.run(main=main)
