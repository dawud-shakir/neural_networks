# Convert sound to 2D CNN images, 
# train using Pytorch and predict test set. 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np 
import pandas as pd
import os
import librosa
import librosa.display

import IPython.display as ipd

import matplotlib.pyplot as plt
print(os.listdir("../input"))



import torchaudio
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, models, transforms
import torch.optim as optim

train_on_gpu=torch.cuda.is_available()

# Any results you write to the current directory are saved as output.
['sample_submission.csv', 'train_noisy.csv', 'train_noisy.zip', 'train_curated.zip', 'train_curated.csv', 'test.zip']
import zipfile

# Will unzip the files so that you can see them..
with zipfile.ZipFile("../input/"+"train_curated"+".zip","r") as z:
    z.extractall(".")
    
    
Labels = pd.read_csv("../input/train_curated.csv")
Labels.head()
WavPath = '/kaggle/working/'
Fils = os.listdir(WavPath)
sound, sample_rate = torchaudio.load(WavPath+Fils[2])
ipd.Audio(data=sound[0,:],rate=sample_rate) # load a local WAV file

#Create spectrum image
x, sr = librosa.load(WavPath+Fils[2])

plt.figure(figsize=(14, 5))



librosa.display.waveshow(x, sr=sr)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
Xdb.shape

S = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=23)
delta2_mfcc = librosa.feature.delta(MFCC, order=2)

#MFCC = librosa.feature.mfcc(y=x, sr=sample_rate,n_mfcc=23,dct_type=2)
librosa.display.specshow(log_S)
#print(np.max(MFCC),np.min(MFCC))
#MFCC = (MFCC+200)/500
#print(np.max(MFCC),np.min(MFCC))
plt.colorbar()
plt.tight_layout()



FilesS = np.zeros(len(Fils))
for i,File in enumerate(Fils):
    FilesS[i] = os.path.getsize(WavPath+File)

plt.figure(figsize=(20,8))
plt.hist(FilesS,bins=50)


#Labeling the data using one( or more then one) hot encoded:
Fils_2 = Labels['fname']
Fils_2

Class =set(Labels['labels'])
All_class= [] 
for i in Class:
    for j  in i.split(','):
        All_class.append(j)

All_class = set(All_class)

NumClasses = len(All_class)
OneHot_All = np.zeros((len(Fils_2),NumClasses))

for  i,file in enumerate(Labels['labels']):
    for j,clas in enumerate(All_class):
        OneHot_All[i,j] = np.int(clas in file)
np.mean(log_S/10+4)


###Train and validation split

# Encode classes
#ClassDict = dict(enumerate(set(Labels['labels'])))
#Class2int = {ch: ii for ii, ch in ClassDict.items()}
#encoded = np.array([Class2int[ch] for ch in Labels['labels']])

#NumClasses = len(Class2int) 
print(NumClasses)
## split data into training, validation, and test data (features and labels, x and y)
split_frac = 0.79
batch_size = 32

split_idx = int(len(Fils)*split_frac)
split_idx1 = int(batch_size*np.floor(split_idx/batch_size))
split_idx2 = int(batch_size*np.floor( (len(Fils) - split_idx1)/batch_size ))
train_x, val_x = Fils_2[:split_idx1], Fils_2[split_idx1:split_idx1+split_idx2]
train_y, val_y = OneHot_All[:split_idx1,:], OneHot_All[split_idx1:split_idx1+split_idx2,:]
print(len(train_x)/batch_size, len(val_x)/batch_size )

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split( Fils_2, OneHot_All, test_size=1-split_frac, random_state=42)
print(train_x.shape,val_x.shape,train_y.shape,val_y.shape)


#Data set pytorch loader
from scipy.io import wavfile
from librosa.feature import mfcc
class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels,DataPath,RecLen,DecNum=5,fft_Samp= 256,Im_3D= False):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.DataPath = DataPath
        self.RecLen = RecLen # length of most records
        self.fft_Samp = fft_Samp 
        self.Im_3D = Im_3D
        
        self.NFCC_Num = 128
        self.TimeSamp = 128
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]

        #y, sr = librosa.load(self.DataPath + ID)
        data,fs =  librosa.load(self.DataPath + ID)
        data = np.float32(data)
        S = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=128)
        Mel = librosa.power_to_db(S, ref=np.max)/10+4
        LabelOut = torch.from_numpy(self.labels[ID]).double()
        
        
        Im = torch.zeros((self.NFCC_Num,self.TimeSamp)).type(torch.FloatTensor)
        Ssum = np.sum(Mel,axis=0)
        MaxE = np.argmax(Ssum)
        if MaxE > Mel.shape[1]-64 : 
            MaxE = Mel.shape[1]-65
        if MaxE< 64 :
            MaxE = 64
        if Mel.shape[1] > self.TimeSamp :
            Im = torch.from_numpy(Mel[:,MaxE-64:MaxE+64])
        else: 
            Im[:,:Mel.shape[1]  ] = torch.from_numpy(Mel)
        
        

        Im = Im.double()
        return Im, LabelOut,ID
#Design CNN neural net
class CnnAudioNet(nn.Module):
    def __init__(self,NumClasses):
        super(CnnAudioNet,self).__init__()
        self.NumClasses = NumClasses
        self.Fc_features = 128
        self.C1 = nn.Conv2d(1,32,5,padding=1)
        self.C2 = nn.Conv2d(32,32,5,padding=1)
        self.C3 = nn.Conv2d(32,64,5,padding=1)
        self.C4 = nn.Conv2d(64,64,5,padding=1)
        
        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)
        self.BN3 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d((1,2),(1,2))
        
        
        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,self.NumClasses )
        self.dropout = nn.Dropout(0.25)
        self.Bat1 = nn.BatchNorm1d(128)

        
        
    def forward(self,x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.BN1(self.C1(x)))
        x = self.maxpool1(F.relu(self.BN1(self.C2(x))))
        x = F.relu(self.BN2(self.C3(x)))
        x = self.maxpool1(F.relu(self.BN2(self.C4(x))))
        x = F.relu(self.BN2(self.C4(x)))
        x = self.maxpool1(F.relu(self.BN2(self.C4(x))))
        x = F.relu(self.BN2(self.C4(x)))
        x = F.relu(self.BN3(self.C4(x)))
        # flatten image input
        x = self.dropout(x.view(-1,64*8*8))
        # add dropout layer
        x =  self.dropout(self.fc1(x))
        # add 1st hidden layer, with relu activation function
        # add dropout layer
        # add 2nd hidden layer, with relu activation function
        #x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x
        
#Transfer learning VGG net (currently not in use)
from torchvision import datasets, models, transforms


# Freeze training for all layers


class CnnTransferNet(nn.Module):
    def __init__(self):
        super(CnnTransferNet,self).__init__()
        
        self.vgg =  models.vgg16_bn().cuda()
        for param in self.vgg.features.parameters():
            param.require_grad = False

        
        self.fc1 = nn.Linear(1000,128)
        self.fc2 = nn.Linear(128,NumClasses)
        self.dropout = nn.Dropout(0.25)

        
        
    def forward(self,x):
        # add sequence of convolutional and max pooling layers
        Features = self.dropout(self.vgg(x))
        # flatten image input
        # add 1st hidden layer, with relu activation function
        Features = F.relu(self.fc1(Features))
        # add dropout layer
        # add 2nd hidden layer, with relu activation function
        Features = self.fc2(Features)
        return Features
Init model:
model = CnnAudioNet(NumClasses)
if train_on_gpu:
    model.cuda()
print(model)
# specify loss function (MSE)

#criterion = nn.MSELoss()
#criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.MultiLabelSoftMarginLoss()

optimizer = optim.Adam(params=model.parameters(), lr=0.001)# specify optimizer
#optimizer = optim.Adam(model.parameters(), lr=0.005)


a = train_x.tolist()
CnnAudioNet(
  (C1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
  (C2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
  (C3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
  (C4): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
  (BN1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (BN2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (BN3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (maxpool2): MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=4096, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=80, bias=True)
  (dropout): Dropout(p=0.25)
  (Bat1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
#Create Dataset:
#abelsDict = dict(zip(Fils,one_hot))
labelsDict_train = dict(zip(train_x,train_y))
labelsDict_val = dict(zip(val_x,val_y))

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 9}
params_v = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': 3}
RecLen = 176400

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

training_set = Dataset(train_x.tolist(), labelsDict_train,WavPath,RecLen,transforms.Compose(normalize))
training_generator = data.DataLoader(training_set, **params)

val_set = Dataset(val_x.tolist(),labelsDict_val,WavPath,RecLen,transforms.Compose(normalize))
val_generator = data.DataLoader(val_set, **params_v)
Learning:
import time
start_time = time.time()
#Warnings.filterwarnings('ignore')

# number of epochs to train the model
n_epochs = 1

valid_loss_min = np.Inf # track change in validation loss
print("Start training:")
idx = 0 
for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    TotMSE = 0 
    TotEl = 0
    
    ###################
    # train the model #
    ###################
    model.train()

    for dataBatch, target,_ in training_generator:
        
        idx+=1

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            dataBatch, target = dataBatch.unsqueeze(1).float().cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(dataBatch)
        # calculate the batch loss
        #loss = criterion(output, torch.squeeze(torch.argmax(target,dim=-1)))
        loss = criterion(output,target.float())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*dataBatch.size(0)
        #print(loss.item())
        #print('Finish batch')
        _,pred = torch.max(output,1)
        
        #Correct = torch.sum(torch.pow(output-target.float(),2))#
        ErrorS = torch.sum(torch.pow(torch.sigmoid(output)-target.float(),2))#
        TotMSE += ErrorS
        TotEl += output.numel()
        Correct =torch.sum(pred ==torch.squeeze(torch.argmax(target,dim=-1)))
        #print('Train batch loss: {:.6f},  Error: {:.4f},  Sum Correct: {} out of {}'.format(loss,ErrorS,Correct,output.shape[0]))
    print('Epoch: {} \t  Train batch loss: {:.6f} '.format(epoch,loss))

        
    ######################    
    # validate the model #
    ######################
    with torch.no_grad():
        model.eval()
        TotEl_v = 0
        valid_loss = 0 
        TotMSE_v = 0
        for dataBatch_v, target ,_ in val_generator  :

        # move tensors to GPU if CUDA is available
            if train_on_gpu:
                dataBatch_v, target = dataBatch_v.unsqueeze(1).float().cuda(),target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
            output = model(dataBatch_v)
        # calculate the batch loss
            loss = criterion(output,target.float())

            #loss = criterion(output, torch.squeeze(torch.argmax(target,dim=-1)))
        # update average validation loss 
            output.shape
            _,pred = torch.max(output,1)
            Correct = torch.sum(pred ==torch.squeeze(torch.argmax(target,dim=-1)))
            #SumCorrectVal += Correct
            valid_loss += loss.item()*dataBatch.size(0)
            #print(TotVal)

            ErrorS = torch.sum(torch.pow(torch.sigmoid(output)-target.float(),2))#
            TotMSE_v += ErrorS
            TotEl_v += output.numel()
        # calculate average losses
        train_lossM = train_loss/len(training_generator.dataset)
        valid_lossM = valid_loss/len(val_generator.dataset)
        MSE = TotMSE/TotEl
        MSE_V = TotMSE_v/TotEl_v

        # print training/validation statistics 
        print('Epoch: {} \t Training Loss: {:.6f}, Train MSE: {:.4f} \tValidation Loss: {:.6f},  Val MSE: {:.4f} '.format(
            epoch, train_lossM,MSE, valid_lossM,MSE_V))
        print("--- %s seconds ---" % (time.time() - start_time))
#Start training:
#Epoch: 1 	  Train batch loss: 0.075715 
#Epoch: 1 	 Training Loss: 0.098537, Train MSE: 0.0196 	Validation Loss: 0.061080,  Val MSE: 0.0152 
#--- 1352.824548482895 seconds ---
#Test data loader:
# data,target ,_= next(iter(val_generator))
# data = data.unsqueeze(1).float().cuda()
# output = model(dataBatch_v)
# plt.figure(figsize=(20,20))
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.plot(target[i,:].detach().cpu().numpy())
#     plt.plot(torch.sigmoid(output[i,:]).detach().cpu().numpy())
from glob import glob
F1 = glob('./*wav*')
for file in F1:
    os.remove(file)
# # Will unzip the files so that you can see them..
with zipfile.ZipFile("../input/"+"test"+".zip","r") as z:
    z.extractall("./test/")
WavPath_test =  './test/'

Fils_test = os.listdir(WavPath_test)


one_hot_test = np.zeros((len(Fils_test),NumClasses))

labelsDict = dict(zip(Fils_test,one_hot_test))

params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 4}

test_set = Dataset(Fils_test, labelsDict,WavPath_test,RecLen,transforms.Compose(normalize))
test_generator = data.DataLoader(test_set, **params)

#Test prediction
model.eval()
SoftM = torch.nn.Softmax()
Output_all = [] 
BatchRecs_all = [] 
with torch.no_grad():                   # operations inside don't track history

    for dataBatch, Lab,BatchRecs in test_generator:
        if train_on_gpu:
            dataBatch, Lab = dataBatch.unsqueeze(1).float().cuda(), Lab.cuda()
        output = model(dataBatch)
        outP = torch.sigmoid(output)
        #outP = output
        Output_all.append(outP)
        BatchRecs_all.append(BatchRecs)

  
#Submission

Dataout = np.zeros((4*len(Output_all)-3,80))
Names = []
for i in range(len(Output_all)):
    Dataout[i*4:(i+1)*4,:] = Output_all[i].cpu().detach().numpy()
    Names.append(BatchRecs_all[i][0])
    if i<840:
        Names.append(BatchRecs_all[i][1])
        Names.append(BatchRecs_all[i][2])
        Names.append(BatchRecs_all[i][3])
    
Cl = list(All_class)
#Cl.append('fname')

Output_all_DF =pd.DataFrame(columns=Cl,data = Dataout)
Output_all_DF['fname'] = Names
Output_all_DF.to_csv('submission.csv', index=False)
Output_all_DF.head()
Chirp_and_tweet	Hi-hat	Walk_and_footsteps	Keys_jangling	Cricket	Zipper_(clothing)	Accelerating_and_revving_and_vroom	Acoustic_guitar	Female_singing	Writing	Sink_(filling_or_washing)	Sigh	Skateboard	Church_bell	Fart	Waves_and_surf	Gasp	Microwave_oven	Clapping	Gurgling	Frying_(food)	Scissors	Hiss	Toilet_flush	Electric_guitar	Sneeze	Purr	Chink_and_clink	Traffic_noise_and_roadway_noise	Bark	Bicycle_bell	Gong	Cutlery_and_silverware	Squeak	Screaming	Strum	Crowd	Fill_(with_liquid)	Knock	Slam	...	Tap	Child_speech_and_kid_speaking	Cheering	Race_car_and_auto_racing	Cupboard_open_or_close	Run	Dishes_and_pots_and_pans	Male_singing	Drawer_open_or_close	Computer_keyboard	Raindrop	Whispering	Shatter	Accordion	Female_speech_and_woman_speaking	Finger_snapping	Buzz	Printer	Bus	Motorcycle	Marimba_and_xylophone	Water_tap_and_faucet	Drip	Chewing_and_mastication	Harmonica	Yell	Burping_and_eructation	Meow	Bathtub_(filling_or_washing)	Applause	Car_passing_by	Glockenspiel	Crackle	Trickle_and_dribble	Mechanical_fan	Bass_guitar	Stream	Male_speech_and_man_speaking	Bass_drum	fname
0	0.028487	0.027844	0.032224	0.113843	0.028028	0.035055	0.010620	0.005572	0.005470	0.024496	0.032933	0.031237	0.016457	0.007411	0.018344	0.011141	0.014786	0.046774	0.016522	0.029564	0.008643	0.018480	0.012803	0.015202	0.010898	0.033181	0.061771	0.026338	0.009488	0.022607	0.016411	0.009409	0.049610	0.037735	0.019231	0.005662	0.009992	0.014750	0.024764	0.023390	...	0.021701	0.020010	0.014929	4.683665e-03	0.032478	0.030491	0.023382	0.010705	0.080178	0.035622	0.045601	0.038099	0.032189	0.011613	0.039320	0.033508	0.009872	0.011684	0.015217	0.012337	0.038650	0.027374	0.031217	0.062876	0.008656	0.032481	0.017872	0.026738	0.015008	0.015567	0.010800	0.010213	0.062539	0.031236	0.021677	0.010749	0.008074	0.023878	0.022971	cb4391c5.wav
1	0.024022	0.011533	0.014695	0.021431	0.013886	0.014279	0.014731	0.001770	0.002876	0.011455	0.027568	0.013022	0.007871	0.009477	0.015415	0.020527	0.005563	0.022921	0.019746	0.012611	0.012304	0.012097	0.016659	0.023358	0.014105	0.018945	0.036824	0.013195	0.023725	0.017558	0.011811	0.013174	0.019094	0.014813	0.011817	0.004303	0.019381	0.012783	0.018742	0.007559	...	0.010124	0.006766	0.045385	4.894889e-03	0.016106	0.012862	0.022520	0.009402	0.019440	0.016012	0.015873	0.018501	0.026858	0.014734	0.009289	0.030116	0.012930	0.015889	0.024771	0.024860	0.054968	0.051295	0.010497	0.031880	0.006634	0.009424	0.009982	0.013243	0.027603	0.041034	0.050375	0.011081	0.014569	0.015291	0.023429	0.017647	0.024817	0.015880	0.017142	c72e8acd.wav
2	0.015488	0.008253	0.026712	0.017428	0.019624	0.018439	0.030629	0.000992	0.002712	0.013334	0.026900	0.016465	0.009851	0.007620	0.014869	0.026949	0.006555	0.018095	0.013229	0.018197	0.008485	0.013718	0.021925	0.019253	0.005593	0.008536	0.024712	0.005516	0.037297	0.012180	0.005684	0.008925	0.017207	0.015077	0.011601	0.003488	0.040675	0.015744	0.011798	0.005564	...	0.004609	0.005330	0.057955	6.308925e-03	0.019838	0.016726	0.011420	0.012238	0.034002	0.014456	0.010381	0.020203	0.017028	0.008448	0.008176	0.010331	0.015124	0.017221	0.032162	0.034683	0.018510	0.038815	0.006008	0.042106	0.004288	0.009329	0.007072	0.007147	0.021110	0.061139	0.056432	0.003324	0.023429	0.010931	0.024309	0.003494	0.014521	0.011552	0.006699	c011423b.wav
3	0.012534	0.013471	0.001670	0.007261	0.000940	0.004085	0.000488	0.032378	0.000311	0.001349	0.001842	0.002803	0.002174	0.006952	0.002641	0.000858	0.005531	0.021799	0.003621	0.004782	0.002310	0.001646	0.000524	0.001205	0.024021	0.006038	0.061609	0.031008	0.002640	0.042721	0.242064	0.014079	0.019282	0.003291	0.002391	0.021208	0.001013	0.001340	0.041978	0.004925	...	0.003871	0.002672	0.002349	1.343427e-04	0.008323	0.001170	0.005434	0.000655	0.011034	0.003244	0.002494	0.007301	0.027050	0.003298	0.001263	0.012288	0.002018	0.003011	0.001420	0.003308	0.074423	0.006437	0.005340	0.001044	0.002368	0.004115	0.012582	0.012370	0.001723	0.000850	0.000658	0.274215	0.002719	0.006064	0.001405	0.089563	0.001878	0.012666	0.010598	636df665.wav
4	0.000761	0.000585	0.000008	0.000094	0.000002	0.000021	0.000001	0.007536	0.000003	0.000004	0.000013	0.000012	0.000037	0.000492	0.000025	0.000003	0.000190	0.000738	0.000039	0.000041	0.000026	0.000018	0.000002	0.000008	0.004472	0.000055	0.001314	0.003703	0.000047	0.002951	0.068758	0.002424	0.000373	0.000029	0.000029	0.004290	0.000002	0.000006	0.002716	0.000064	...	0.000066	0.000025	0.000021	3.315759e-07	0.000440	0.000004	0.000162	0.000006	0.000066	0.000014	0.000010	0.000161	0.002030	0.000078	0.000004	0.000831	0.000047	0.000027	0.000009	0.000072	0.017180	0.000109	0.000103	0.000005	0.000135	0.000053	0.000365	0.000250	0.000017	0.000001	0.000003	0.187061	0.000017	0.000088	0.000005	0.245794	0.000020	0.000234	0.000267	bdf07823.wav
