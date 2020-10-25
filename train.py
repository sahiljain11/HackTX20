import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import math
import random
from progress.bar import IncrementalBar
from sklearn.metrics import confusion_matrix

basedir = os.path.abspath(os.path.dirname(__file__))

number_of_features = 598
number_of_hidden   = 32   # size of hidden layer
number_of_gestures = 2    # output size
sequence_length    = 20   # refers to the amount of timeframes to check
batch_size         = 5    # how many different files to compute
learning_rate      = 0.001
num_epoch          = 30

# 0 - truth
# 1 - lie
training_data_name = basedir + "/file.npy"
STORAGE_PATH = "parameters.model"

def get_numpy_arrays(test_path):
    
     with open(test_path, 'wb'):
         numpy_array = np.load(f)
         tensor_array = torch.from_numpy(numpy_array)

    return

get_numpy_arrays(test_path)

# PREPROCCESSING
get_cols = os.path.join(basedir, 'csv_data/feature_cols.csv')
col_file = open(get_cols, "r")
col_data = col_file.readlines()[1].split(",")

feature_columns = []

for i in range(0, len(col_data)):
    if col_data[i].strip("\r\n") == "1":
        feature_columns.append(i)


file_tensors = []
file_hash = {}
num_files = 0
max_count = 0
with IncrementalBar("Determining Tensor Counts...", max=number_of_gestures) as increment_bar:
    for i in range(0, number_of_gestures):
        # navigate to subfolder
        name = folder_name[i]
        data_dir = os.path.join(basedir, 'csv_data/' + name + '/')
        files = os.listdir(data_dir)

        count = 0
        
        for j in range(0, len(files)):
            # traverse through each file in each sub folder
            test_string = name + str(j) + ".csv"
            training_file = os.path.join(data_dir, test_string)

            # convert them to tensors
            training_tensor = create_training_tensor(training_file, number_of_features, feature_columns)

            for k in range(sequence_length, training_tensor.shape[0]):
                if training_tensor.shape[1] != number_of_features:
                    break

                count += 1

        # determine max_count
        if count > max_count:
            max_count = count
        increment_bar.next()

with IncrementalBar("Preprocessing...", max=number_of_gestures) as increment_bar:
    for i in range(0, number_of_gestures):
        # navigate to subfolder
        name = folder_name[i]
        data_dir = os.path.join(basedir, 'csv_data/' + name + '/')
        files = os.listdir(data_dir)

        count = 0

        j = 0
        while True:
            # traverse through each file in each sub folder
            test_string = name + str(j) + ".csv"
            training_file = os.path.join(data_dir, test_string)

            # convert them to tensors
            training_tensor = create_training_tensor(training_file, number_of_features, feature_columns)
            #training_tensor = F.normalize(training_tensor, dim=0)

            for k in range(sequence_length, training_tensor.shape[0]):
                if training_tensor.shape[1] != number_of_features:
                    break
                count += 1

                if count > max_count:
                    break

                portion_tensor = training_tensor[k - sequence_length:k, :]

                file_tensors.append(portion_tensor)
                file_hash[portion_tensor] = i
                num_files += 1

            # increment j and see if enough tensors for this gesture have been made
            j = (j + 1) % len(files)
            if count > max_count:
                break

        #print("  " + str(count))
        increment_bar.next()

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

def epoch(folder_name, number_of_gestures, batch_size, lstm_model, loss_function, optimizer, sequence_length, number_of_features, epoch_num, hidden_dim):
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
            curr_batch = file_tensors[num].view(sequence_length, 1, number_of_features)
            target = [file_hash.get(file_tensors[num])]

            for i in range(1, batch_size):
                target.append(long(file_hash.get(file_tensors[num + i])))
                temp_matrix = file_tensors[num + i].view(sequence_length, 1, number_of_features)
                curr_batch = torch.cat((curr_batch, temp_matrix), dim=1)

            target = torch.LongTensor(target)
            #h = torch.randn(seq_length, hidden_dim).view(1, seq_length, hidden_dim)
            #c = torch.randn(seq_length, hidden_dim).view(1, seq_length, hidden_dim)

            # clear the accumulated gradients
            #lstm_model.zero_grad()
            optimizer.zero_grad()

            # run forward pass
            #resulting_scores = model(sectioned_data, (h, c))
            resulting_scores = lstm_model(curr_batch.view(sequence_length, batch_size, number_of_features))

            resulting_scores = resulting_scores.view(batch_size, number_of_gestures)

            # compute loss and backward propogate
            loss = loss_function(resulting_scores, target)
            loss.backward()

            optimizer.step()

            return_loss += loss.item()
            count += batch_size
            loss_count += 1

            #if count % 500 == 0:
            #    correct = 0
            #    total = 0

            #    for i in range(0, 1000):
            #        result = lstm_model(file_tensors[i].view(sequence_length, 1, number_of_features))

            #        #last_item = resulting_tensor.view(sequence_length, number_of_gestures)
            #        last_item = result
            #        #last_item = last_item[number_of_gestures - 1, :]
            #        last_item = torch.argmax(last_item)

            #        if file_hash.get(file_tensors[i]) == last_item.item():
            #            correct += 1
            #        total += 1

            #    accuracy = correct / total * 100
            #    print('Iteration: {}. Loss: {}. Accuracy: {}'.format(count, loss.item(), accuracy))
                

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
#    epoch(folder_name, number_of_gestures, batch_size, lstm_model, loss_function, optimizer, sequence_length, number_of_features, i, number_of_hidden)

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
            target = file_hash.get(file_tensors[num]) * torch.ones(1, dtype=torch.long)

            resulting_tensor = lstm_model(file_tensors[num].view(sequence_length, 1, number_of_features))

            #last_item = resulting_tensor.view(sequence_length, number_of_gestures)
            last_item = resulting_tensor
            #last_item = last_item[number_of_gestures - 1, :]
            last_item = torch.argmax(last_item)

            predictions.append(last_item.item())
            labels.append(file_hash.get(file_tensors[num]))

            if file_hash.get(file_tensors[num]) == last_item.item():
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
            target = file_hash.get(file_tensors[num]) * torch.ones(1, dtype=torch.long)

            resulting_tensor = lstm_model(file_tensors[num].view(sequence_length, 1, number_of_features))

            #last_item = resulting_tensor.view(sequence_length, number_of_gestures)
            last_item = resulting_tensor
            #last_item = last_item[number_of_gestures - 1, :]
            last_item = torch.argmax(last_item)

            predictions.append(last_item.item())
            labels.append(file_hash.get(file_tensors[num]))

            if file_hash.get(file_tensors[num]) == last_item.item():
                correct += 1
            count += 1

        increment_bar.next()
        increment_bar.finish()

        print("Train Accuracy: " + str(test_correct) + "/" + str(test_count) + " = " + str(float(test_correct) / float(test_count) * 100)+ "%")
        print("Cross Accuracy: " + str(correct) + "/" + str(count) + " = " + str(float(correct) / float(count) * 100)+ "%")

conf_mat = confusion_matrix(labels, predictions)
print(str(conf_mat))