# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import random
# from collections import deque

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, InputLayer
# from tensorflow.keras.optimizers import Adam

# # Load your dataset
# vectorDB = pd.read_csv("vectorDB.csv")

# # Extract state vectors (probabilities) and labels ('dx')
# state_vectors = vectorDB[['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']].values
# labels = vectorDB['dx'].values  # 'dx' column contains the diagnosis label

# # Ensure labels are strings and handle any whitespace or case issues
# labels = np.array([str(label).strip().lower() for label in labels])

# # Check for unique labels in your dataset
# unique_labels = np.unique(labels)
# print("Unique labels in the dataset:", unique_labels)

# # Update the diagnosis labels to include all unique labels
# # Add 'scc' to the list if it's present in your dataset
# diagnosis_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'scc']
# label_to_index = {label: idx for idx, label in enumerate(diagnosis_labels)}
# index_to_label = {idx: label for idx, label in enumerate(diagnosis_labels)}
# num_actions = len(diagnosis_labels)
# print("Label to index mapping:", label_to_index)

# # Filter out any labels not in diagnosis_labels
# valid_labels = set(diagnosis_labels)
# filtered_indices = [i for i, label in enumerate(labels) if label in valid_labels]
# labels = labels[filtered_indices]
# state_vectors = state_vectors[filtered_indices]

# # Convert labels to indices
# labels_indices = np.array([label_to_index[label] for label in labels])

# # Split into training and testing sets
# train_states, test_states, train_labels, test_labels = train_test_split(
#     state_vectors, labels_indices, test_size=0.2, random_state=42)

# # Define the reward matrix
# reward_matrix = np.full((num_actions, num_actions), -1)  # default penalty

# # Reward table (adjusted to include 'scc')
# reward_table = {
#     ("mel", "mel"): 5,
#     ("mel", "nv"): -5,
#     ("mel", "bkl"): -5,
#     ("mel", "df"): -5,
#     ("mel", "vasc"): -5,
#     ("mel", "bcc"): -3,
#     ("mel", "akiec"): -3,
#     ("mel", "scc"): -3,

#     ("bcc", "bcc"): 3,
#     ("bcc", "nv"): -2,
#     ("bcc", "bkl"): -2,
#     ("bcc", "df"): -2,
#     ("bcc", "vasc"): -2,
#     ("bcc", "mel"): -3,
#     ("bcc", "akiec"): -2,
#     ("bcc", "scc"): -2,

#     ("akiec", "akiec"): 2,
#     ("akiec", "mel"): -3,
#     ("akiec", "bcc"): -2,
#     ("akiec", "nv"): -2,
#     ("akiec", "bkl"): -2,
#     ("akiec", "df"): -2,
#     ("akiec", "vasc"): -2,
#     ("akiec", "scc"): -2,

#     ("scc", "scc"): 4,
#     ("scc", "mel"): -3,
#     ("scc", "bcc"): -2,
#     ("scc", "akiec"): -2,
#     ("scc", "nv"): -2,
#     ("scc", "bkl"): -2,
#     ("scc", "df"): -2,
#     ("scc", "vasc"): -2,

#     # Rewards for benign conditions
#     ("nv", "nv"): 1,
#     ("bkl", "bkl"): 1,
#     ("df", "df"): 1,
#     ("vasc", "vasc"): 1,

#     # Penalties for misclassifying benign lesions as other benign lesions
#     ("nv", "bkl"): -1,
#     ("nv", "df"): -1,
#     ("nv", "vasc"): -1,
#     ("nv", "scc"): -2,
#     ("bkl", "nv"): -1,
#     ("bkl", "df"): -1,
#     ("bkl", "vasc"): -1,
#     ("bkl", "scc"): -2,
#     ("df", "nv"): -1,
#     ("df", "bkl"): -1,
#     ("df", "vasc"): -1,
#     ("df", "scc"): -2,
#     ("vasc", "nv"): -1,
#     ("vasc", "bkl"): -1,
#     ("vasc", "df"): -1,
#     ("vasc", "scc"): -2,

#     # Penalties for misclassifying benign lesions as malignant
#     ("nv", "mel"): -2,
#     ("nv", "bcc"): -2,
#     ("nv", "akiec"): -2,
#     ("bkl", "mel"): -2,
#     ("bkl", "bcc"): -2,
#     ("bkl", "akiec"): -2,
#     ("df", "mel"): -2,
#     ("df", "bcc"): -2,
#     ("df", "akiec"): -2,
#     ("vasc", "mel"): -2,
#     ("vasc", "bcc"): -2,
#     ("vasc", "akiec"): -2,
# }

# # Fill in the reward matrix
# for (actual_label, predicted_label), reward in reward_table.items():
#     actual_idx = label_to_index[actual_label]
#     predicted_idx = label_to_index[predicted_label]
#     reward_matrix[actual_idx, predicted_idx] = reward

# # Build the model
# def build_model(input_shape, num_actions):
#     model = Sequential()
#     model.add(InputLayer(input_shape=input_shape))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(num_actions, activation='linear'))
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#     return model

# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size  # Number of features in state vector
#         self.action_size = action_size  # Number of possible actions (diagnosis classes)
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.9  # Discount rate
#         self.epsilon = 0.2  # Exploration rate
#         self.epsilon_min = 0.1
#         self.epsilon_decay = 0.995

#         self.learning_rate = 0.001
#         self.model = build_model((self.state_size,), self.action_size)
#         self.target_model = build_model((self.state_size,), self.action_size)
#         self.update_target_model()

#     def update_target_model(self):
#         # Copy weights from model to target_model
#         self.target_model.set_weights(self.model.get_weights())

#     def remember(self, state, action, reward, next_state):
#         self.memory.append((state, action, reward, next_state))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         q_values = self.model.predict(state.reshape(1, -1), verbose=0)
#         return np.argmax(q_values[0])  # Return the action with the highest Q-value

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         states = np.array([experience[0] for experience in minibatch])
#         targets = self.model.predict(states, verbose=0)
#         next_states = np.array([experience[3] for experience in minibatch])
#         target_next = self.target_model.predict(next_states, verbose=0)

#         for i, (state, action, reward, next_state) in enumerate(minibatch):
#             t = targets[i]
#             t[action] = reward + self.gamma * np.amax(target_next[i])
#             targets[i] = t
#         self.model.fit(states, targets, epochs=1, verbose=0)
#         # Reduce epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

# # Initialize agent
# state_size = state_vectors.shape[1]  # Should be 7 (number of probabilities)
# action_size = num_actions  # Should be 8 (number of diagnosis classes)
# agent = DQNAgent(state_size, action_size)

# # Training parameters
# num_episodes = 1000
# batch_size = 32

# for e in range(num_episodes):
#     total_reward = 0
#     # Shuffle the training data each episode
#     indices = np.arange(len(train_states))
#     np.random.shuffle(indices)
#     for idx in indices:
#         state = train_states[idx]
#         action = agent.act(state)
#         # Predicted diagnosis
#         predicted_label_idx = action
#         # Actual diagnosis
#         actual_label_idx = train_labels[idx]
#         # Get reward
#         reward = reward_matrix[actual_label_idx, predicted_label_idx]
#         total_reward += reward
#         # Next state (could be the next sample or random)
#         next_idx = (idx + 1) % len(train_states)
#         next_state = train_states[next_idx]
#         # Remember experience
#         agent.remember(state, action, reward, next_state)
#         # Learn from experience
#         if len(agent.memory) > batch_size:
#             agent.replay(batch_size)
#     # Update target model
#     agent.update_target_model()
#     # Print progress
#     if e % 100 == 0:
#         print(f"Episode {e}, Total Reward: {total_reward}")
#         # Evaluate on the test set
#         correct_predictions = 0
#         for i in range(len(test_states)):
#             state = test_states[i]
#             q_values = agent.model.predict(state.reshape(1, -1), verbose=0)
#             action = np.argmax(q_values[0])
#             predicted_label_idx = action
#             actual_label_idx = test_labels[i]
#             if predicted_label_idx == actual_label_idx:
#                 correct_predictions += 1
#         accuracy = correct_predictions / len(test_states)
#         print(f"Test Accuracy after episode {e}: {accuracy:.4f}")

# # Final evaluation
# correct_predictions = 0
# for i in range(len(test_states)):
#     state = test_states[i]
#     q_values = agent.model.predict(state.reshape(1, -1), verbose=0)
#     action = np.argmax(q_values[0])
#     predicted_label_idx = action
#     actual_label_idx = test_labels[i]
#     if predicted_label_idx == actual_label_idx:
#         correct_predictions += 1
# accuracy = correct_predictions / len(test_states)
# print("Final Test Accuracy:", accuracy)



import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Define the reward table as in the original code
# Rows: ground truth label (akiec=0,bcc=1,bkl=2,df=3,mel=4,nv=5,vasc=6)
# Columns: chosen action (akiec=0,bcc=1,bkl=2,df=3,mel=4,nv=5,vasc=6,unkn=7 if used)
def get_reward_table(use_unknown=False):
    # UNKN_reward = -1
    # If use_unknown = False, we have 7 actions (0 to 6)
    # If use_unknown = True, we have 8 actions (0 to 7)
    base_table = np.array([
        [ 2, -2, -3, -3, -2, -3, -3, -1],
        [-2,  3, -4, -4, -2, -4, -4, -1],
        [-2, -2,  1, -2, -3, -2, -2, -1],
        [-2, -2, -2,  1, -3, -2, -2, -1],
        [-4, -3, -5, -5,  5, -5, -5, -1],
        [-2, -2, -2, -2, -3,  1, -2, -1],
        [-2, -2, -2, -2, -3, -2,  1, -1],
    ], dtype=np.float32)

    if use_unknown:
        return base_table
    else:
        # If unknown not used, just remove the last column
        return base_table[:, :7]

def load_data(csv_path='data/vectorDB.csv'):
    database = pd.read_csv(csv_path)
    # Extract labels
    labels = np.asarray(database['dx'])
    # Replace 'scc' with 'akiec' if needed
    labels[labels == 'scc'] = 'akiec'

    le = LabelEncoder()
    le.fit(labels)
    vocab = le.classes_
    labels_cat = le.transform(labels)

    # Extract features (not strictly necessary for tabular Q-learning if we only
    # use patient index as state, but we keep them for reference)
    database_feat = database.copy()
    database_feat.pop('dx')
    features = np.asarray(database_feat, dtype='float32')

    # For simplicity, we can just keep the entire dataset as "states".
    # state = index of the patient in the dataset
    # ground truth label = labels_cat[state]

    return features, labels_cat, vocab


def epsilon_greedy_action(Q, state, epsilon):
    # Q is shape (#states, #actions)
    # Choose a random action with probability epsilon, else best action
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of training episodes.')
    parser.add_argument('--n_patients_per_episode', type=int, default=1000, help='Number of patients per episode.')
    parser.add_argument('--use_unknown', type=bool, default=False, help='Whether to use unknown action.')
    parser.add_argument('--gamma', type=float, default=0.0, help='Discount factor. For a bandit-like problem, gamma=0.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Exploration rate.')
    args = parser.parse_args()

    # Load data
    features, labels_cat, vocab = load_data()
    n_classes = len(vocab)
    if args.use_unknown:
        # Add unknown action
        n_actions = n_classes + 1
    else:
        n_actions = n_classes

    reward_table = get_reward_table(args.use_unknown)

    # States: each patient is its own state
    n_states = labels_cat.shape[0]

    # Initialize Q-table
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    # Training loop
    for episode in range(args.n_episodes):
        # In each episode, we sample patients and update Q
        total_reward = 0.0
        for step in range(args.n_patients_per_episode):
            # Randomly pick a patient (state)
            s = np.random.randint(n_states)
            gt_label = labels_cat[s]

            # Select an action
            a = epsilon_greedy_action(Q, s, args.epsilon)

            # Get reward
            r = reward_table[gt_label, a]

            # Update Q
            # Since this is basically a bandit problem (each patient is independent),
            # we have no next state to consider or gamma for future reward.
            # Q[s,a] = Q[s,a] + alpha * (r - Q[s,a])
            Q[s,a] = Q[s,a] + args.alpha * (r - Q[s,a])

            total_reward += r

        avg_reward = total_reward / args.n_patients_per_episode
        print(f"Episode {episode+1}/{args.n_episodes}, Avg Reward: {avg_reward:.3f}")

    # After training, let's evaluate on the entire dataset
    # We take the best action for each patient and evaluate balanced accuracy
    predicted_actions = np.argmax(Q, axis=1)
    bacc = metrics.balanced_accuracy_score(labels_cat, predicted_actions)
    print("Final Balanced Accuracy:", bacc)
    print("Confusion Matrix:\n", metrics.confusion_matrix(labels_cat, predicted_actions))
    print("Classification Report:\n", metrics.classification_report(labels_cat, predicted_actions, digits=3))

if __name__ == "__main__":
    main()
