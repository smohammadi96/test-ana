import torch
import sys, os
import argparse
import numpy as np
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import time
import copy
import pickle
from GPUtil import showUtilization as gpu_usage
from numba import cuda
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import matplotlib.pyplot as plt
import torch
import sys, os
import argparse
import numpy as np
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import time
import copy
import pickle
from GPUtil import showUtilization as gpu_usage
from numba import cuda
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
torch.cuda.empty_cache()

sys.path.insert(1,'helpers')
sys.path.insert(1,'model')
sys.path.insert(1,'weight')

from helpers.augmentation  import Aug
from model.cvit import CViT
from helpers.loader import session
import optparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#Default
cession='g' # GPU runtime 
epoch = 1
dir_path = ""
batch_size = 16
lr=0.0001
weight_decay=0.0000001

parser = optparse.OptionParser("Train CViT model.")
parser.add_option("-e", "--epoch", type=int, dest='epoch', help='Number of epochs used for training the CViT model.')
parser.add_option("-v", "--version", dest='version', help='Version 0.1.')
parser.add_option("-s", "--cession", type="string",dest='session', help='Training session. Use g for GPU, t for TPU.')
parser.add_option("-d", "--dir", dest='dir', help='Training data path.')
parser.add_option("-b", "--batch", type=int, dest='batch', help='Batch size.')
parser.add_option("-l", "--rate",  type=float, dest='rate', help='Learning rate.')
parser.add_option("-w", "--decay", type=float, dest='decay', help='Weight decay.')

(options,args) = parser.parse_args()

if options.session:
    cession = options.session
if options.dir==None:
    print (parser.usage) 
    exit(0) 
else:
    dir_path = options.dir
if options.batch:
    batch_size = int(options.batch)
if options.epoch:
    epoch = int(options.epoch)
if options.rate:
    lr = float(options.rate)
if options.decay:
    weight_decay = float(options.decay)

if cession=='t':
    print('USING TPU.')
    device = xm.xla_device()

batch_size, dataloaders, dataset_sizes = session(cession, dir_path, batch_size)

#CViT model definition
model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512,
            dim=1024, depth=6, heads=8, mlp_dim=2048)
#model = nn.DataParallel(model)
model = nn.DataParallel(model, device_ids=[0, 1])
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#criterion = torch.nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss().to(device)

#criterion.to(device)
num_epochs = epoch
min_val_loss=10000
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)


def train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    min_loss = min_val_loss

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    #with open('weight/cvit_deepfake_detection_ep_50.pkl', 'rb') as f:
    #    train_loss, train_accu, val_loss, val_accu = pickle.load(f)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            phase_idx=0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #break
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step() # GPU || CPU

                if phase_idx%100==0:
                    print(phase,' loss:',phase_idx,':', loss.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, phase_idx * batch_size, dataset_sizes[phase], \
                        100. * phase_idx*batch_size / dataset_sizes[phase], loss.item()))
                phase_idx+=1                       

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accu.append(epoch_acc)
                writer.add_scalar("Acc/train", epoch_acc, epoch)
                writer.add_scalar("Loss/train", epoch_loss, epoch)
            else:
                val_loss.append(epoch_loss)
                val_accu.append(epoch_acc)
                writer.add_scalar("Acc/val", epoch_acc, epoch)
                writer.add_scalar("Loss/val", epoch_loss, epoch)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_loss < min_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_loss, min_loss))
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    with open('/kaggle/working/CViT/weight/weights.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu], f)

    state = {'epoch': num_epochs+1, 
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'min_loss':epoch_loss}
    torch.save(state, '/kaggle/working/CViT/weight/weights.pth')
    test(model)
    # summarize history for accuracy
    f1 = plt.figure()
    plt.plot([*range(0, num_epochs, 1)], [i.cpu().numpy() for i in train_accu])
    plt.plot([*range(0, num_epochs, 1)], [i.cpu().numpy() for i in val_accu])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig( 'accu_CViT_NeuralTexturs.png' )
    # # summarize history for loss
    f2 = plt.figure()
    plt.plot([*range(0, num_epochs, 1)], train_loss)
    plt.plot([*range(0, num_epochs, 1)], val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig( 'loss_CViT_NeuralTexturs.png' )
    
    return train_loss,train_accu,val_loss,val_accu, min_loss

def test(model):
    print('Test mode')
    with torch.no_grad():
        Sum = 0
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs).to(device).float()

            _,prediction = torch.max(output,1)

            pred_label = labels[prediction]
            pred_label = pred_label.detach().cpu().numpy()
            main_label = labels.detach().cpu().numpy()
            bool_list  = list(map(lambda x, y: x == y, pred_label, main_label))
            Sum += sum(np.array(bool_list)*1)

    print('Prediction: ', (Sum/dataset_sizes['test'])*100,'%')

if cession=='t':
    train_tpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss) #Train using TPU.
    writer.flush()
else:
    train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss) #Train using GPU.
    writer.flush()
