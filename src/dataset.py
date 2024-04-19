import os
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt

DAIC = "/scratch1/tereeves/DAIC"

def get_labels(DAIC_loc):
    train_labels = pd.read_csv(os.path.join(DAIC_loc,"labels","train_split.csv"))
    dev_labels = pd.read_csv(os.path.join(DAIC_loc,"labels","dev_split.csv"))
    test_labels = pd.read_csv(os.path.join(DAIC_loc,"labels","test_split.csv"))
    return [train_labels,dev_labels,test_labels]
    

def create_spectrograms(labels,DAIC_loc,split_size=1000,name="Train"):
    index = 0
    save_fldr = os.path.join(DAIC_loc,"data","spectrograms")
    new_labels = []
    for x in tqdm(labels.values):
        patien_num, ptsd = x[0], x[2]
        path = os.path.join(DAIC_loc,"data",f"{patien_num}_P",f"{patien_num}_AUDIO.wav")
        audio_data, sample_rate = librosa.load(path)
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate,n_mels=80,fmax=8000)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)


        #split variable len spectrogram into even subsections
        split_idxs = np.arange(start=split_size,stop=spectrogram.shape[1],step=split_size)
        spec_splits = np.array_split(spectrogram,split_idxs,axis=1)

        for spec in spec_splits:
            save_path = os.path.join(save_fldr,str(index)) 
            np.save(save_path,spec)
            new_labels.append(ptsd)
            index +=1 
    pd.Series(new_labels).to_csv(os.path.join(save_fldr,name))
class DAICSpecSpliceDataset(Dataset):
    def __init__(self, DAIC_loc, train):
        self.save_fldr = os.path.join(DAIC_loc,"data","spectrograms")
        data = "train" if train else "dev"

        labels_folders = os.path.join(DAIC_loc,"labels","spectrograms")

        self.labels = pd.read_csv(os.path.join(labels_folders,f"{data}.csv")).to_numpy()

        idx = np.where(self.labels == 1)[0]
        self.labels = np.concatenate((self.labels, self.labels[idx]))
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        true_index, label = self.labels[index]
        spec = np.load(os.path.join(self.save_fldr,f"{true_index}.npy"))

        if spec.shape[1] != 500:
            spec = np.pad(spec,((0,0),(0,500-spec.shape[1])),'mean')
        
        return spec, label


class DAICSpectrogramDataset(Dataset):
    def __init__(self, DAIC_loc,train,max_len):
        # get labels and patient nums for split
        train_split,dev_split,_ = get_labels(DAIC_loc)
        data = train_split if train else dev_split
        self.labels = data.loc[:,"PCL-C (PTSD)"]
        self.patient_nums = data.iloc[:,0]
        self.severity = data.loc[:,"PTSD Severity"]
        
        
        
        self.DAIC_loc = DAIC_loc # data location
        self.max_len = max_len # max length of spectogram segment in time axis

    def __len__(self):
        return len(self.patient_nums)

    def __getitem__(self, idx):
        pat_num = self.patient_nums[idx] # get patient number
        label = self.labels[idx] # get binary depression label
        severity = self.severity[idx]

        # Use patient num to load spetrogram
        spec_path = os.path.join(self.DAIC_loc,"data",f"{pat_num}_P",f"{pat_num}_spectrogram.png.npy")
        spectrogram = np.load(spec_path)

        #split variable len spectrogram into even subsections
        split_idxs = np.arange(start=self.max_len,stop=spectrogram.shape[1],step=self.max_len)
        spec_splits = np.array_split(spectrogram,split_idxs,axis=1)
        
        
        # pad last spectrogram if short
        pad_len = self.max_len - spec_splits[-1].shape[1]
        spec_splits[-1] = np.pad(spec_splits[-1], ((0,0),(0,pad_len)), 'mean') 


        return spec_splits, label, severity


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # import pdb;pdb.set_trace()
    ## get sequence lengths
    ## padd
    sequences = [ torch.tensor(np.array(x)) for (x,_,_) in batch ]
    labels = torch.tensor([ y for (_,y,_) in batch ]).to(device)
    severities = torch.tensor([ s for (_,_,s) in batch ]).to(device)
    lengths = torch.tensor([ len(t) for t in sequences ]).to(device)

    sequences = torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True).to(device)
    ## compute mask
    mask = (sequences != 0).to(device)
    return sequences, lengths, mask, labels, severities

def get_sampler():
    train_labels, _, _ = get_labels(DAIC)
    zero = train_labels[train_labels.loc[:,'PCL-C (PTSD)']==0].index
    one = train_labels[train_labels.loc[:,'PCL-C (PTSD)']==1].index
    
    one_weight = len(zero) / len(one)

    weights = torch.ones((len(train_labels),1))
    weights[one] = one_weight
    weights = (weights / weights.sum()).reshape(-1)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_labels),
        replacement=False
    )
    return sampler



    


if __name__ == "__main__":
    labels = get_labels(DAIC)
    names = ("train","dev","test")
    for i, label in enumerate(labels):
        print(f"{i+1} / 3")
        create_spectrograms(label,DAIC,split_size=500,name=names[i])
    device = "cpu"
    spectrogram_dataset = DAICSpecSpliceDataset(DAIC,True)
    train_dataloader = DataLoader(spectrogram_dataset, batch_size=5)

    sequences,  labels = next(iter(train_dataloader))
    print(sequences.shape)
    plt.imsave()
    
