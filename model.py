import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from collections import Counter
import mne
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import scale
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.autograd import Function
from sklearn.metrics import f1_score

import torchvision.models as models
from torchvision import datasets, transforms

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import random
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for GPU
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = r"./dataset"


csv_labels = pd.read_csv('allDataLabels.csv')




def upload_participant_data(sub_id):
    eeg_path_ima_de = os.path.join(base_path, "derivatives", sub_id, "ses-ima", "eeg",
                                f"{sub_id}_ses-ima_task-emotion_de.npy")
    eeg_path_ima_psd = os.path.join(base_path, "derivatives", sub_id, "ses-ima", "eeg",
                                f"{sub_id}_ses-ima_task-emotion_psd.npy")
    
    eeg_path_vid_de = os.path.join(base_path, "derivatives", sub_id, "ses-vid", "eeg",
                                f"{sub_id}_ses-vid_task-emotion_de.npy")
    eeg_path_vid_psd = os.path.join(base_path, "derivatives", sub_id, "ses-vid", "eeg",
                                f"{sub_id}_ses-vid_task-emotion_psd.npy")
    
    
    eeg_ima_de = np.load(eeg_path_ima_de, allow_pickle=True).item()["de"]
    eeg_ima_psd = np.load(eeg_path_ima_psd, allow_pickle=True).item()["psd"]

 

    eeg_vid_de = np.load(eeg_path_vid_de, allow_pickle=True).item()["de"]
    eeg_vid_psd = np.load(eeg_path_vid_psd, allow_pickle=True).item()["psd"]

    # print( eeg_ima_de[:30,0].shape )
    # print( eeg_vid_psd[17] )

    eeg_ima_de= eeg_ima_de.reshape(-1,21,30,5)
    eeg_ima_psd = eeg_ima_psd.reshape(-1,21,30,5)
    eeg_vid_de = eeg_vid_de.reshape(-1,21,30,5)
    eeg_vid_psd = eeg_vid_psd.reshape(-1,21,30,5)


    eeg_ima = np.concatenate([eeg_ima_de, eeg_ima_psd], axis=-1)
    eeg_vid = np.concatenate([eeg_vid_de, eeg_vid_psd], axis=-1)

    eeg_features = np.concatenate( [eeg_ima, eeg_vid], axis= 1)
    eeg_features = np.transpose(eeg_features, (1, 2, 0, 3 )) #42, 30, 64, 10

    # print(eeg_features.shape)
    # print( eeg_features[:,:,0,:5] )

    return  eeg_features





all_data = []
all_labels = []
all_labelsInitial = []
all_subjects = []
all_type = []

for subj in range(2, 55):
    if subj != 22:
        sub_id = f"sub-{subj:02d}"
        
        df_sub = upload_participant_data(sub_id)

        

        labelSujeto = csv_labels.loc[ csv_labels['subject']== subj ]

        # print( df_sub )

        # print('albels')
        # print( labelSujeto )
     
       
        original_emotion= labelSujeto["stimulus"]
        score_emotion = labelSujeto["labelAngie"].values 
        #df_sub["score_emotion"] = [ee[:2] for ee in labelSujeto["stimulus"] ]
        
        subjects = labelSujeto["subject"].values 
        typeS = labelSujeto["type"].values 

        all_data.append(df_sub)
        all_labels.append(score_emotion)
        all_subjects.append(subjects)
        all_labelsInitial.append(original_emotion)
        all_type.append(typeS)


all_data = np.concatenate( all_data , axis = 0)
all_labels = np.concatenate( all_labels)
all_subjects = np.concatenate( all_subjects )
all_labelsInitial = np.concatenate( all_labelsInitial )
all_type =  np.concatenate( all_type )

print(f'data {all_data.shape} {all_labels.shape} {all_subjects.shape} {all_labelsInitial.shape}')

#df_all = pd.concat(all_data, ignore_index=True)

s_init = pd.Series(all_labelsInitial)
idxLabels = np.where( (~s_init.str.lower().str.startswith("neu")) &
                       (all_labels!="Neutro") )[0]

all_data = all_data[ idxLabels ] 
all_labels = all_labels[ idxLabels ] 
all_subjects = all_subjects[ idxLabels ] 
all_labelsInitial = all_labelsInitial[ idxLabels ] 
all_type = all_type[ idxLabels ] 

print(f'data {all_data.shape} {all_labels.shape} {all_subjects.shape} {all_labelsInitial.shape}')

ch_names = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'F9', 'F10', 'TP9', 'TP10'
]


def make_unique_names(ch_names):
    counts = Counter()
    unique = []
    for ch in ch_names:
        counts[ch] += 1
        if counts[ch]==1:
            unique.append(ch)
        else:
            unique.append(f"{ch}-{counts[ch]-1}")
    return unique
ch_names_unique = make_unique_names(ch_names)

montage = mne.channels.make_standard_montage('brainproducts-RNP-BA-128')
pos = montage.get_positions()
posCh = np.empty([64,3])
for idx,ch in enumerate(ch_names):
    posCh[idx] = pos["ch_pos"].get(ch,[np.nan,np.nan,np.nan])
    

# Normalization 
minCoord = np.nanmin(posCh[:,:2], axis=0)
maxCoord = np.nanmax(posCh[:,:2], axis=0)
normalized = ((posCh[:,:2]-minCoord)/(maxCoord-minCoord)*15).round().astype(int)

plt.figure()
plt.plot(normalized[:, 0], normalized[:, 1], 'o')
for i, ch in enumerate(ch_names):
    plt.text(normalized[i, 0], normalized[i, 1], ch, fontsize=8)
plt.title("Standardized channel distribution")
plt.savefig('images/distribution.png', dpi=300)


def interpolate(mat):

    noise = np.random.normal(loc=0.0, scale=0.1*np.nanstd(mat), size=mat.shape)
    mat = noise + mat
    H, W = mat.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    known_mask = ~np.isnan(mat)
    known_points = np.stack([xx[known_mask], yy[known_mask]], axis=-1)
    known_values = mat[known_mask]
    
    missing_mask = np.isnan(mat)
    missing_points = np.stack([xx[missing_mask], yy[missing_mask]], axis=-1)
    
    filled = mat.copy()
    
    interp_values = griddata(known_points, known_values, missing_points, method='cubic')
    

    # interp_values[~np.isnan(interp_values)] = scale( interp_values[~np.isnan(interp_values)] )


    filled[missing_mask] = interp_values

    #but there are still nan

    known_mask = ~np.isnan(filled)
    known_points = np.stack([xx[known_mask], yy[known_mask]], axis=-1)
    known_values = filled[known_mask]
    nan_mask = np.isnan(filled)
    nan_points = np.stack([xx[nan_mask], yy[nan_mask]], axis=-1)
    
    # filled = np.nan_to_num( filled )

    filled[nan_mask] = griddata(
            known_points, known_values, nan_points, method='nearest'
        )
    

    return filled


file_path = "dataAllImages.npz"
if os.path.exists(file_path):
    loaded = np.load(file_path, allow_pickle=True)  # allow string arrays
    dataAllImages = loaded["data"]
    all_labels= loaded["all_labels"]
    all_subjects = loaded["all_subjects"]
    all_labelsInitial = loaded["all_labelsInitial"]
    all_type =loaded["all_type"] 
    
else:

    samples = all_data.shape[0]
    windows = all_data.shape[1]
    features = all_data.shape[3]
    dataAllImages = np.full((samples, windows, 16, 16, features), np.nan)


    all_data = all_data.reshape(-1,64,features)
    dataAllImages = dataAllImages.reshape(-1,16,16,features)

    for idxFeat in range(all_data.shape[2]):
        dataAllImages[:, normalized[:,1], normalized[:,0], idxFeat] = all_data[:,:,idxFeat ]
        
        for iSample in range( all_data.shape[0] ):
            dataAllImages[iSample, :, :, idxFeat] = interpolate(dataAllImages[iSample, :, :, idxFeat])


    dataAllImages = dataAllImages.reshape(samples, windows,16,16,features) 
    np.savez_compressed("dataAllImages.npz", data=dataAllImages,
                        all_labels=all_labels, all_subjects=all_subjects, 
                        all_labelsInitial=all_labelsInitial, 
                         all_type = all_type )


useOriginals = False

if useOriginals:
    all_labels = all_labelsInitial

    all_labels[ (all_labelsInitial=='dis4') | 
                (all_labelsInitial=='dis5') | 
                (all_labelsInitial=='dis8') ] =  "Disgust"

    all_labels[ (all_labelsInitial=='fear4') | 
                (all_labelsInitial=='fear5') | 
                (all_labelsInitial=='fear8') ] =  "Fear"

    all_labels[ (all_labelsInitial=='ins4') | 
                (all_labelsInitial=='ins5') | 
                (all_labelsInitial=='ins8') ] =  "Inspiration"

    all_labels[ (all_labelsInitial=='ten4') | 
                (all_labelsInitial=='ten5') | 
                (all_labelsInitial=='ten8') ] =  "Tenderness"

    all_labels[ (all_labelsInitial=='joy4') | 
                (all_labelsInitial=='joy5') | 
                (all_labelsInitial=='joy8') ] =  "Joy"

    all_labels[ (all_labelsInitial=='sad4') | 
                (all_labelsInitial=='sad5') | 
                (all_labelsInitial=='sad8') ] =  "Sadness"

emotions = np.unique(all_labels) #["Joy", "Inspiration", "Tenderness", "Sadness", "Fear", "Disgust"]
print( emotions )
sessions = ["ima","vid"]
emo_sess_dict = {}



doPlot = False #change to false to do not plot figures

if doPlot:
    os.makedirs("images", exist_ok=True)
    featuresName = ['de_delta','de_theta','de_alpha','de_beta','de_gamma',
                    'psd_delta','psd_theta','psd_alpha','psd_beta','psd_gamma' ]
    for emo in np.unique(all_labels):
        for sess in np.unique(all_type):
            # if sess not in emo_sess_dict[emo]:
            #     continue
            
            idxEmo = np.where( (all_labels== emo) & 
                            (all_type== sess) )[0]
            dataAll = dataAllImages[idxEmo]
            mean_data = dataAll.mean(axis=1).mean(axis=0)  # promedio trials
            num_features = mean_data.shape[-1]

            # print( mean_data.shape )

            # Channel map   
            testMatrix = np.full((16,16), "", dtype=object)
            for i, ch in enumerate(ch_names):
                testMatrix[normalized[i,1], normalized[i,0]] = ch

            X, Y = np.meshgrid(np.arange(16), np.arange(16))
            bg = np.where(testMatrix!="", 1, 0)

            

            for idxFeat in range(len(featuresName)):
                plt.figure(figsize=(12,5))
                plt.suptitle(f"emotion: {emo} - Sesión: {sess}", fontsize=14)
                plt.subplot(1,2,1)
                plt.pcolor(X,Y,bg, edgecolors='k', linewidths=0.5)
                for i in range(16):
                    for j in range(16):
                        label = testMatrix[i,j]
                        if label!="":
                            plt.text(j,i,label,ha="center",va="center",fontsize=8)
                plt.title("channel map")
                plt.gca().invert_yaxis()
                
                plt.subplot(1,2,2)
                
                

                #smoothed = gaussian_filter(mean_data[:,:,idxFeat], sigma=1.5)
                # img = plt.imshow(mean_data[:,:,idxFeat], origin='lower', cmap='viridis'  )
                img = plt.pcolor(X,Y,mean_data[:,:,idxFeat], edgecolors='k', linewidths=0.5)
                cbar = plt.colorbar(img)
                vmin, vmax = img.get_clim()
                ticks = np.linspace(vmin, vmax, 20)  # 9 labeled levels
                cbar.set_ticks(ticks)

                #plt.colorbar(label='Intensidad')
                plt.title("topographic map")
                plt.savefig(f'images/topographic_{emo}_{sess}_{featuresName[idxFeat]}.png', dpi=300)
                plt.close()

X_ima, y_ima, sub_ima = dataAllImages[all_type=='imagen'], all_labels[all_type=='imagen'], all_subjects[all_type=='imagen']

X_vid, y_vid, sub_vid = dataAllImages[all_type=='video'], all_labels[all_type=='video'], all_subjects[all_type=='video']


print("*************************************")
print( X_ima.shape, y_ima.shape, sub_ima.shape )
print( X_vid.shape, y_vid.shape, sub_vid.shape )


class modelPaper(nn.Module):
    
    def __init__(self, num_classes):
        
        super().__init__()


        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=10, out_channels=16, kernel_size=(1,3,3), padding=0),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            
        )

    
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1,3,3), stride=(1,2,2), padding=0),
            nn.Tanh(),
            nn.LeakyReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1,3,3), padding=(0, 1, 1) ),
            nn.BatchNorm3d(16),
         
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1,3,3), padding=(0, 1, 1) ),
            nn.BatchNorm3d(16),
           nn.ReLU(),
        )

        self.gru = nn.GRU(
                input_size=576,
                hidden_size=128,
                num_layers=1,
                batch_first=True
            )

        self.flatten = nn.Flatten()

        ## Fully connected
        self.fc = nn.Sequential(
                    nn.Linear(3840, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),  # <- use 1d here
                    nn.Dropout(p=0.5),
                    nn.Linear(256, num_classes)
                )

        

    def forward(self, x_source ):
        
        B, T, W, H, F = x_source.shape # dimensions

   
        x_source = x_source.permute(0, 4, 1, 2, 3) # N, F, T, W, H  

       

        x_source = self.conv1(x_source)
        #print(x_source.shape )
        x_source_b5 = self.conv2(x_source)
        #print(x_source_b5.shape )#

   
        x_source = self.conv3(x_source_b5)
        #print('conv2',x_source.shape )#
        x_source_b9 = self.conv4(x_source)
        #print('conv3', x_source_b9.shape )#

        x_source_s10 =  x_source_b5 + x_source_b9
        #print(x_source_s10.shape )#
        
        ########
        x_source = x_source_s10.reshape(B, T, -1)
        #print("after cnns ", x_source.shape )#


        out, h_n = self.gru(x_source)      # out: (B, seq_len, hidden_size)
        #print(out.shape, h_n.shape)

        x_source = self.flatten(out)
        #print(x_source.shape )
       
        x_source = self.fc(x_source)
        #print(x_source.shape )

        return x_source
    


class modelCNN(nn.Module):
    
    def __init__(self, filters, num_classes):
        
        super().__init__()

        activationFunction = nn.LeakyReLU()# 

        outFCL =  5 #32
        # filters = 32
        self.feat = nn.LayerNorm(outFCL)
        
        # self.feat =  nn.Sequential( nn.Linear(10, outFCL),  
        #                          nn.ReLU(), nn.LayerNorm(outFCL) )

        drCNN = .2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=outFCL, out_channels=filters, 
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            activationFunction,
            nn.Dropout(p = drCNN ),
            nn.MaxPool2d(2)
            #nn.AvgPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters*2, 
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(filters*2),
            activationFunction,
            nn.Dropout(p = drCNN ),
            nn.MaxPool2d(2)
            #nn.AvgPool2d(2)
        )
   
        self.lstm = nn.LSTM(
            input_size= filters*2*(4*4), # 1152, 
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            # dropout=0.2,
        )

        ## Fully connected
        self.fc = nn.Sequential(
            
            # nn.Dropout(p=0.5),
            # nn.Linear(16, 64),
            # activationFunction,
            nn.Dropout(p=0.5),
            nn.Linear(32, num_classes) )

        

    def forward(self, x_source ):
        
        x_source = x_source[:,:,:,:,:5]



        B, T, W, H, F = x_source.shape

        

        x_source = x_source.reshape(-1, W, H, F)
       

        x_source = self.feat( x_source )
   
        x_source = x_source.permute(0, 3, 1, 2)  

        x_source = self.conv1(x_source)
  
        
        x_source = self.conv2(x_source)

  
        # print(x_source.shape)
        x_source = x_source.reshape(B, T, -1)

        out, (h_n, c_n) = self.lstm(x_source) # out: (B, seq_len, dim) # Use final hidden state 
      
        forward_last = h_n[-2]   # Layer 1 forward
        backward_last = h_n[-1]  # Layer 1 backward
        x_source = torch.cat((forward_last, backward_last), dim=1)

        # x_source =  h_n[-1] #

        # print( out.shape )

        x_source = self.fc(x_source)

        return x_source
    

BATCH_SIZE =  32

def train_and_evaluate(X, y, subjects, stim_name, filters=64, rep=42):
    print(f"\ Type of stimulus: {stim_name} ")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print("aaaaaa Classes:", le.classes_)

    X = X.astype("float32")

    print(X.shape) 
    # Train: 80%, Test: 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=rep
    )

    transformer = StandardScaler()

    N, T, H, W, F = X_train.shape

    X_train = X_train.reshape(-1, T* H* W* F)
    X_test = X_test.reshape(-1, T* H* W* F)



    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    X_train = X_train.reshape(-1,T, H, W, F)
    X_test = X_test.reshape(-1, T, H, W, F)

    
    X_train = torch.from_numpy(  X_train )
    X_test = torch.from_numpy(  X_test )   
    y_train = torch.from_numpy(  y_train )
    y_test = torch.from_numpy(  y_test )

    # X_val = torch.from_numpy(  X_val )
    # y_val = torch.from_numpy(  y_val )
    
    # y_train is already a tensor of class indices
    num_classes = torch.unique(y_train).numel()
    
    # # Count how many samples per class
    class_counts = torch.bincount(y_train)
    print("Class counts:", class_counts)
    
    # Compute weights: inverse of frequency
    class_weights = 1.0 / class_counts.float()
    
    # Normalize so weights sum to number of classes (optional but common)
    class_weights = class_counts.sum() / (num_classes * class_counts.float())
    

    # class_weights /= class_weights.sum()
    
    # Move to device
    class_weights = class_weights.to(device)
    print("Class weights:", class_weights)
    loss_fn = nn.CrossEntropyLoss( class_weights )


    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), 
                              batch_size=BATCH_SIZE, shuffle=False)

    model = modelCNN(filters=filters, num_classes=num_classes).to(device)


    
    
    initial_lr = 1e-4 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, 
                                  weight_decay= 1e-1)  
    
    numEpochs = 60
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=initial_lr*10, steps_per_epoch=len(train_loader), epochs=numEpochs
    )
        

    accuracy = None

    for epoch in range(numEpochs):


        model.train()
        train_loss, preds_train, labels_train = 0, [], []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = batch_x.float()
            batch_y = batch_y.long()

    
            optimizer.zero_grad()

            
        

            logits  = model(batch_x )

            task_loss = loss_fn(logits, batch_y )  

         


            loss = task_loss  
            loss.backward()
           
            optimizer.step()
            #Save training results
            train_loss += loss.item()
            preds_train.append(torch.argmax(logits,1).cpu().numpy())
            labels_train.append(batch_y.cpu().numpy())
        

            scheduler.step()

        #training metrics
        preds_train = np.concatenate(preds_train)
        labels_train = np.concatenate(labels_train)
        train_acc = np.mean(np.diag(confusion_matrix(labels_train, preds_train, normalize="true")))
        train_loss /= len(train_loader)

        #evaluation
        model.eval()
    
        
        preds, labels, test_loss = [], [], 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb.float()  )
                test_loss += loss_fn(logits, yb.long()).item()
                preds.append(torch.argmax(logits,1).cpu().numpy())
                labels.append(yb.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        test_acc = np.mean(np.diag(confusion_matrix(labels, preds, normalize="true")))
        test_loss /= len(test_loader)


        

        print(f"Epoch {epoch+1:03d} | Train Loss={train_loss:.4f} Acc={train_acc:.2f} | Test Loss={test_loss:.4f} Acc={test_acc:.2f}")
    
   
    accuracy = np.diag(confusion_matrix(labels, preds, normalize="true"))

    # print(f" Mejor accuracy alcanzado en test: {best_acc:.2f}")

    print(f"accuracy achieved in test: {accuracy} global {test_acc}")

    return accuracy


imgLabels, countLabels = np.unique(y_ima, return_counts=True)
for (il, cl) in zip(imgLabels, countLabels):
    print(il, cl)

print("videos")
imgLabels, countLabels = np.unique(y_vid, return_counts=True)
for (il, cl) in zip(imgLabels, countLabels):
    print(il, cl)

runMultiClass = False

if runMultiClass:
    os.makedirs("results", exist_ok=True)

    file_image = "results/image_accuracy.csv"
    file_video  = "results/video_accuracy.csv"

    image_results= np.empty([30, 7]) * np.nan
    video_results = np.empty([30, 7]) * np.nan
    emotions = np.append(emotions, "Overall") 
    print(emotions)

    repetitions = 30
    for i in range(repetitions):
        print('***************'*20)
        print(f'repetion {i}')

        set_seed(i)
        image_accuracy = train_and_evaluate(X_ima, y_ima,sub_ima,  "Images", 
                                             filters=64, rep=i)
        set_seed(i)
        video_accuracy = train_and_evaluate(X_vid, y_vid,sub_vid, "Video", 
                                             filters=64, rep=i)

        image_results[i,:6] = image_accuracy
        image_results[i,-1] = np.mean( image_accuracy )
        video_results[i,:6] = video_accuracy
        video_results[i,-1] = np.mean( video_accuracy )

        df_img = pd.DataFrame(image_results, columns=emotions)
        df_vid = pd.DataFrame(video_results, columns=emotions)

        df_img.to_csv(file_image, index=False)
        df_vid.to_csv(file_video, index=False)

        print( file_image )




    print("\n=== Statistical analysis (Wilcoxon) ===")

    df_img = pd.read_csv(file_image, index_col=0)
    df_vid = pd.read_csv(file_video, index_col=0)

    wilcoxon_results = {
        "Emotion": [],
        "Media_Image": [],
        "Media_Video": [],
        "Statistical": [],
        "p_value": [],
        "Significant(p<0.05)": [],
        "Best_Model": []
    }

    for emo in df_img.columns:
        data_img = df_img[emo].dropna()
        data_vid = df_vid[emo].dropna()

        if len(data_img) == len(data_vid) and len(data_img) > 0:
            stat, p = wilcoxon(data_img, data_vid)
            signif = "Sí" if p < 0.05 else "No"
            better = "Images" if np.mean(data_img) > np.mean(data_vid) else "Videos"

            wilcoxon_results["Emotion"].append(emo)
            wilcoxon_results["Media_Image"].append(np.mean(data_img))
            wilcoxon_results["Media_Video"].append(np.mean(data_vid))
            wilcoxon_results["Statistical"].append(stat)
            wilcoxon_results["p_value"].append(p)
            wilcoxon_results["Significant(p<0.05)"].append(signif)
            wilcoxon_results["Best_Model"].append(better)

            print(f"{emo}: better={better}, p={p:.4f}, Significant={signif}")
        else:
            print(f"{emo}: There is not enough data for comparison.")

    df_wilcoxon = pd.DataFrame(wilcoxon_results)

    file_wilcoxon = "results/wilcoxon_results.csv"
    df_wilcoxon.to_csv(file_wilcoxon, index=False)

    print(f"\n/ Wilcoxon results saved in: {file_wilcoxon}")
    print(df_wilcoxon)



####################################################################################3
# BINARY CLASSIFICATION

positivas = [ 'Inspiration', 'Joy',  'Tenderness']
negativas = ['Disgust', 'Fear', 'Sadness']

def convert_to_binary(y):
    y_bin = y.copy()
    y_bin[np.isin(y_bin, positivas)] = "Positive"
    y_bin[np.isin(y_bin, negativas)] = "Negative"

       
    return y_bin # np.array(y_bin)

y_ima_bin = convert_to_binary(y_ima)
y_vid_bin = convert_to_binary(y_vid)

print(f"Binary labels (Images): {np.unique(y_ima_bin, return_counts=True)}")
print(f"Binary labels  (Videos): {np.unique(y_vid_bin, return_counts=True)}")

os.makedirs("binary_results", exist_ok=True)

image_file_bin = "binary_results/accuracy_binary_image.csv"
file_video_bin  = "binary_results/accuracy_binary_video.csv"

image_results_bin = np.empty([30, 3]) * np.nan  
video_results_bin = np.empty([30, 3]) * np.nan


repetitions = 30
for i in range(repetitions):
    print('***************'*20)
    print(f"\n--- BINARY: Iteration {i+1}/30 ---")
    # Train with binary labels
    set_seed(i)
    acc_img_bin = train_and_evaluate(X_ima, y_ima_bin,sub_ima, "Images(Binary)", 
                                      filters=64, rep=i)

    set_seed(i)
    acc_vid_bin = train_and_evaluate(X_vid, y_vid_bin, sub_ima, "Video(Binary)",
                                      filters=64, rep=i)

    le_bin = LabelEncoder()
    le_bin.fit(y_ima_bin) 
    clases_bin = list(le_bin.classes_)  


    image_results_bin[i,:2] = acc_img_bin
    image_results_bin[i,-1] = np.mean( acc_img_bin )
    video_results_bin[i,:2] = acc_vid_bin
    video_results_bin[i,-1] = np.mean( acc_vid_bin )

    pd.DataFrame(image_results_bin, columns=["Positive","Negative", "Overall"]).to_csv(image_file_bin)
    pd.DataFrame(video_results_bin, columns=["Positive","Negative", "Overall"]).to_csv(file_video_bin)

    print(f"→Binary results iter {i+1} saved.")

# wilcoxon for binary
print("\n=== Statistical analysis (Wilcoxon) - Binary ===")
df_img_bin = pd.read_csv(image_file_bin, index_col=0)
df_vid_bin = pd.read_csv(file_video_bin, index_col=0)

wilcoxon_results_bin = {
    "Label": [], "Media_Image": [], "Media_Video": [], "Statistical": [], "p_value": [], "Significant(p<0.05)": [], "Best_Model": []
}

for Label in df_img_bin.columns:
    data_img = df_img_bin[Label].dropna()
    data_vid = df_vid_bin[Label].dropna()
    if len(data_img) == len(data_vid) and len(data_img) > 0:
        stat, p = wilcoxon(data_img, data_vid)
        signif = "Yes" if p < 0.05 else "No"
        better = "Images" if np.mean(data_img) > np.mean(data_vid) else "Videos"

        wilcoxon_results_bin["Label"].append(Label)
        wilcoxon_results_bin["Media_Image"].append(np.mean(data_img))
        wilcoxon_results_bin["Media_Video"].append(np.mean(data_vid))
        wilcoxon_results_bin["Statistical"].append(stat)
        wilcoxon_results_bin["p_value"].append(p)
        wilcoxon_results_bin["Significant(p<0.05)"].append(signif)
        wilcoxon_results_bin["Best_Model"].append(better)

        print(f"{Label}: better={better}, p={p:.4f}, Significant={signif}")
    else:
        print(f"{Label}: There is not enough data for comparison.")

df_wilcoxon_bin = pd.DataFrame(wilcoxon_results_bin)
df_wilcoxon_bin.to_csv("binary_results/wilcoxon_binary.csv", index=False)
print("\n/ Binary Wilcoxon results saved in: binary_results/wilcoxon_binary.csv")
print(df_wilcoxon_bin)
