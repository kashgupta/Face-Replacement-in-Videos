'''
  File name: p2_train.py
  Author: Kashish Gupta, Rajat Bhageria, and Rajiv Patel-O'Connor
  Date: 12-7-17
'''

import numpy as np
from matplotlib import pyplot as plt

import PyNet as net
from Project4BCNN.p3_dataloader import p3_dataloader

'''
  network architecture construction
  - Stack layers in order based on the architecture of your network
'''

layer_list = [
                net.Conv2d(output_channel=64, kernel_size=5, stride=1, padding=2),
                net.BatchNorm2D(momentum=0.99), #maybe dont need the momentum
                net.Relu(),
                net.Conv2d(output_channel=64, kernel_size=5, stride=1, padding=2),
                net.BatchNorm2D(momentum=0.99), #maybe dont need the momentum
                net.Relu(),
                net.MaxPool2d(kernel_size=2, padding=0, stride=2),
                net.Conv2d(output_channel=128, kernel_size=5, stride=1, padding=2),
                net.BatchNorm2D(momentum=0.99), #maybe dont need the momentum
                net.Relu(),
                net.Conv2d(output_channel=128, kernel_size=5, stride=1, padding=2),
                net.BatchNorm2D(momentum=0.99), #maybe dont need the momentum
                net.Relu(),
                net.MaxPool2d(kernel_size=2, padding=0, stride=2),
                net.Conv2d(output_channel=384, kernel_size=3, stride=1, padding=1),
                net.BatchNorm2D(momentum=0.99), #maybe dont need the momentum
                net.Relu(),
                net.Conv2d(output_channel=384, kernel_size=3, stride=1, padding=1),
                net.BatchNorm2D(momentum=0.99), #maybe dont need the momentum
                net.Relu(),
                net.Conv2d(output_channel=5, kernel_size=3, stride=1, padding=1),
                net.BatchNorm2D(momentum=0.99), #maybe dont need the momentum
                net.Sigmoid(), #always sigmoid
                net.Upsample(size=(40,40))
                #net.Relu(),
                #net.Flatten(), #TODO: This might be missing an upsample layer, idk what that is
                #net.Linear(128,1),
                #net.Sigmoid() #always sigmoid
             ]

'''
  Define loss function
'''
loss_layer = net.Binary_cross_entropy_loss() #change this based on whether we have L2 or binary cross entropy

'''
  Define optimizer 
'''
optimizer = net.SGD_Optimizer(1e-4, 5e-4, 0.99)


'''
  Build model
'''
my_model = net.Model(layer_list, loss_layer, optimizer)

'''
  Define the number of input channel and initialize the model
'''
num_features = 3 #this is correct
my_model.set_input_channel(num_features)

'''
  Input possible pre-trained model
'''
#my_model.load_model('preMolde.pickle')

def randomShuffle(data_set, label_set):
  n = data_set.shape[0]
  idx = np.arange(n)
  np.random.shuffle(idx)
  data_set_cur = data_set[idx]
  label_set_cur = label_set[idx]
  return data_set_cur, label_set_cur

def obtainMiniBatch(data_set_cur, label_set_cur, total_step, curr_j):
  n = data_set_cur.shape[0]
  start = (n*curr_j) / total_step
  end = (n*(curr_j+1)) / total_step #this may not be totally correct because of rounding error
  data_bt = data_set_cur[start:end] #but I've tested it on basic stuff and it works!, esp for Problem 2.1
  label_bt = label_set_cur[start:end]
  return [data_bt, label_bt]

'''
  pre-process data
  1. normalization
  2. convert ground truth data 
  3. resize data into the same size
'''
# obtain data
[data_set, label_set] = p3_dataloader()
#TODO: data loading has not been done for p3 at all
#TODO: shape for data_set needs to be [numpics, 3, width(40), height(40)]

'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''

max_epoch_num = 10
accuracies = np.zeros([max_epoch_num*500])
losses = np.zeros([max_epoch_num*500])
done = False
i=0
save_interval = 5 #TODO: this can be changed
while i < max_epoch_num and not done:
  '''
    random shuffle data 
  '''
  print("epoch number" + str(i))
  #data_set_250_cur, label_set_250_cur = randomShuffle(data_set_250, label_set_250) # design function by yourself
  # data_set_250_cur, label_set_250_cur = randomShuffle(data_set_250, label_set_250) # design function by yourself
  data_set_cur, label_set_cur = randomShuffle(data_set, label_set)

  step = 500  # choosing 1 because our  batch size is 64, so we can iterate through all 64 images at once
  for j in range (step):
    print("image number" + str(j))
    # obtain a mini batch for this step training
    [data_bt, label_bt] = obtainMiniBatch(data_set_cur, label_set_cur, step, j)  # design function by yourself

    # feedward data and label to the model
    loss, pred = my_model.forward(np.reshape(data_bt, (1,3,40,40)), label=label_bt)
    #if j == 498:
        #cv2.imshow(pred)
        #cv2.waitKey(0)
    losses[i] = loss
    rounded_pred = np.zeros(pred.shape)
    rounded_pred[pred>=0.5] = 1
    rounded_pred[pred<0.5] = 0
    rounded_pred = [b[0] for b in rounded_pred]
    accuracy = np.mean(label_bt == rounded_pred)
    accuracies[i] = accuracy

    i+=1
    if accuracy == 1.0:
      done = True
      i-=1

    # backward loss
    backward_loss = my_model.backward(loss)

    # update parameters in model
    my_model.update_param()

  '''
  save trained model, the model should be saved as a pickle file
  '''
  if i % save_interval == 0:
      my_model.save_model(str(i) + '.pickle')

iterations = np.arange(i)
plt.scatter(iterations, losses[:i])
plt.title("Loss for CNN with Actual Convolution")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

plt.scatter(iterations, accuracies[:i])
plt.title("Accuracy for CNN with Actual Convolution")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.show()