import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import math
import random
import numpy as np
from progress.bar import IncrementalBar
from sklearn.metrics import confusion_matrix

basedir = os.path.abspath(os.path.dirname(__file__))

number_of_features = 598
number_of_hidden   = 32   # size of hidden layer
number_of_gestures = 2    # output size
sequence_length    = 20   # refers to the amount of timeframes to check
batch_size         = 1    # how many different files to compute
learning_rate      = 0.001
num_epoch          = 30

# 0 - truth
# 1 - lie
truth_dir = basedir + "/training_data/truth"
lie_dir   = basedir + "/training_data/lie"
STORAGE_PATH = "parameters.model"

def get_numpy_arrays(test_path, tensor_hash, num):

    files = os.listdir(test_path)
    tensors = []

    for f in files:
        if f.endswith(".npy"):
            complete_path = test_path + "/" + f

            numpy_array = np.load(complete_path)
            tensor_array = torch.from_numpy(numpy_array)

            # TODO: TEMPORARY REMOVE LATER
            for _ in range (0, 100):
                tensors.append(tensor_array)

            tensor_hash[tensor_array] = num

    return tensors

tensor_hash = {}

truth_tensors = get_numpy_arrays(truth_dir, tensor_hash, 0)
lie_tensors   = get_numpy_arrays(lie_dir, tensor_hash, 1)

truth_len = len(truth_tensors)
lie_len   = len(lie_tensors)

min_length = truth_len if truth_len < lie_len else lie_len
truth_tensors = truth_tensors[0:min_length]
lie_tensors   = lie_tensors[0:min_length]

# PREPROCCESSING
file_tensors = truth_tensors + lie_tensors
num_files = min_length * 2
max_count = 0


random.shuffle(file_tensors)

# add cross validation here
CROSS_VAL_PORTION = 0.2
cross_val_index = int((1 - CROSS_VAL_PORTION) * num_files)

# LSTM model
class LieLSTM(nn.Module):
    def __init__(self, hidden_dim, input_size, gesture_size, seq_len):
        super(LieLSTM, self).__init__()

        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.seq_length = seq_len
        self.gesture_size = gesture_size
       
        # create the LSTM network
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=1 , batch_first=False)

        # linear space maps between hidden layer and output layer
        self.hidden2gesture = nn.Linear(hidden_dim * seq_len, gesture_size)

    def forward(self, seq_batch_input):

        batch_size = list(seq_batch_input.size())[1]

        hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
        cell_state = torch.zeros(1, batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)

        # run the lstm model
        lstm_out, (ht, ct) = self.lstm(seq_batch_input, hidden)

        #print(lstm_out.shape)
        #lstm_out = lstm_out[-1, :, :]

        # lstm_out = (seq_len, batch, hidden_size)
        # convert lstm_out to (batch_size, -1) to merge
        lstm_out = lstm_out.contiguous().view(batch_size, -1)

        # convert to the proper output dimensions
        gesture_out = self.hidden2gesture(lstm_out)

        # apply softmax function to normalize the values
        # double check dimension. prob wrong
        #gesture_probabilities = F.log_softmax(gesture_out, dim=1)
        #return gesture_probabilities
        return gesture_out

def epoch(number_of_gestures, batch_size, lstm_model, loss_function, optimizer, sequence_length, number_of_features, epoch_num, hidden_dim, tensor_hash):
    avg_total_loss = []
    
    # traverse through all of the 12 gesture training data
    with IncrementalBar("Training " + str(epoch_num) + "...", max=100) as increment_bar:
        bar_count = 0
        return_loss = 0
        count = 0
        loss_count = 0
        for num in range(0, (cross_val_index - batch_size), batch_size):

            # adjust bar to see progress
            if int(math.floor(count * 100 / cross_val_index)) > bar_count:
                bar_count += 1
                increment_bar.next()

            # make the targets and current_batch
            curr_batch = file_tensors[num]
            target = [tensor_hash[file_tensors[num]]]
            target = torch.LongTensor(target)

            tensor_size = curr_batch.size()

            for i in range (0, tensor_size[0] - sequence_length):
                smol_tensor = curr_batch[i:i+sequence_length,:]

                # clear the accumulated gradients
                #lstm_model.zero_grad()
                optimizer.zero_grad()

                # run forward pass
                #resulting_scores = model(sectioned_data, (h, c))
                resulting_scores = lstm_model(smol_tensor.view(sequence_length, batch_size, number_of_features).float())

                resulting_scores = resulting_scores.view(batch_size, number_of_gestures)

                # compute loss and backward propogate
                loss = loss_function(resulting_scores, target)
                loss.backward()

                optimizer.step()

                return_loss += loss.item()


            count += batch_size
            loss_count += 1

        # add the loss at the end and increment the progress bar
        avg_total_loss.append(float(return_loss / loss_count))

    increment_bar.next()
    torch.save(lstm_model.state_dict(), STORAGE_PATH)
    print("\nLoss: " + str((avg_total_loss)))

# create loss function, model, and optimizer
loss_function = nn.CrossEntropyLoss()
lstm_model = LieLSTM(number_of_hidden, number_of_features, number_of_gestures, sequence_length)
#lstm_model.load_state_dict(torch.load(STORAGE_PATH))
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
start_time = time.time()

# traverse through each epoch and train
lstm_model.train()
print("\nNumber of total tensors: " + str(num_files))
print("Storage path: " + str(STORAGE_PATH))
print(str(lstm_model) + "\n")
#for i in range (0, num_epoch):
#    epoch(number_of_gestures, batch_size, lstm_model, loss_function, optimizer, sequence_length, number_of_features, i, number_of_hidden, tensor_hash)

# save the model
lstm_model.eval()

labels = []
predictions = []

with torch.no_grad():
    # evaluate the model to come up with an accuracy
    correct = 0
    count = 0
    with IncrementalBar("Evaluating...", max=100) as increment_bar:
        bar_count = 0
        for num in range(0, cross_val_index):

            if int(math.floor(num * 100 / num_files)) > bar_count:
                bar_count += 1
                increment_bar.next()

            # batch_size x seq_length x num_gestures
            target = tensor_hash.get(file_tensors[num]) * torch.ones(1, dtype=torch.long)

            tensor_size = file_tensors[num].size()

            for i in range (0, tensor_size[0] - sequence_length):
                smol_tensor = file_tensors[num][i:i+sequence_length,:]

                resulting_tensor = lstm_model(smol_tensor.view(sequence_length, 1, number_of_features).float())

                #last_item = resulting_tensor.view(sequence_length, number_of_gestures)
                last_item = resulting_tensor
                #last_item = last_item[number_of_gestures - 1, :]
                last_item = torch.argmax(last_item)

                predictions.append(last_item.item())
                labels.append(tensor_hash[file_tensors[num]])

                if tensor_hash[file_tensors[num]] == last_item.item():
                    correct += 1

                count += 1

        test_correct = correct
        test_count = count

        correct = 0
        count = 0

        # cross validation
        for num in range(cross_val_index, num_files):
            if int(math.floor(num * 100 / num_files)) > bar_count:
                bar_count += 1
                increment_bar.next()

            # batch_size x seq_length x num_gestures
            target = tensor_hash.get(file_tensors[num]) * torch.ones(1, dtype=torch.long)

            tensor_size = file_tensors[num].size()

            for i in range (0, tensor_size[0] - sequence_length):
                smol_tensor = file_tensors[num][i:i+sequence_length,:]

                resulting_tensor = lstm_model(smol_tensor.view(sequence_length, 1, number_of_features).float())

                #last_item = resulting_tensor.view(sequence_length, number_of_gestures)
                last_item = resulting_tensor
                #last_item = last_item[number_of_gestures - 1, :]
                last_item = torch.argmax(last_item)

                predictions.append(last_item.item())
                labels.append(tensor_hash[file_tensors[num]])

                if tensor_hash[file_tensors[num]] == last_item.item():
                    correct += 1

                count += 1
            count += 1

        increment_bar.next()
        increment_bar.finish()

        print("Train Accuracy: " + str(test_correct) + "/" + str(test_count) + " = " + str(float(test_correct) / float(test_count) * 100)+ "%")
        print("Cross Accuracy: " + str(correct) + "/" + str(count) + " = " + str(float(correct) / float(count) * 100)+ "%")

conf_mat = confusion_matrix(labels, predictions)
print(str(conf_mat))