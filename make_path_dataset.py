
import os
#from tqdm import tqdm
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
#import scipy.io

train_df= pd.read_csv("labels/train_split.csv")
dev_df= pd.read_csv("labels/dev_split.csv")
test_df= pd.read_csv("labels/test_split.csv")

train_df=train_df.drop(['PHQ_Score','PCL-C (PTSD)','PTSD Severity'], axis=1)
dev_df=dev_df.drop(['PHQ_Score','PCL-C (PTSD)','PTSD Severity'], axis=1)
test_df=test_df.drop(['PHQ_Score','PCL-C (PTSD)','PTSD Severity'], axis=1)

#upsampling
df_class0= train_df[train_df['PHQ_Binary']==0]
df_class1= train_df[train_df['PHQ_Binary']==1]
df_upsample1=resample(df_class1, n_samples=126)
train_df=pd.concat([df_class0,df_upsample1])

df_class0= dev_df[dev_df['PHQ_Binary']==0]
df_class1= dev_df[dev_df['PHQ_Binary']==1]
df_upsample1=resample(df_class1, n_samples=126)
dev_df=pd.concat([df_class0,df_upsample1])

df_class0= test_df[test_df['PHQ_Binary']==0]
df_class1= test_df[test_df['PHQ_Binary']==1]
df_upsample1=resample(df_class1, n_samples=126)
test_df=pd.concat([df_class0,df_upsample1])


#get path
def get_path(df, feature_type):
    paths=[]
    for i in range(len(df)):
        idx= str(df['Participant_ID'][i])
        path= "data/"+ idx + "_P/features/" + idx + '_' + feature_type
        paths.append(path)
    return paths



#best audio features from the baslines established
train_df['Path_audio'] = get_path(train_df,"vgg16.csv")
dev_df['Path_audio'] = get_path(dev_df,"vgg16.csv")
test_df['Path_audio'] = get_path(test_df,"vgg16.csv")
#best visual features from the baslines established
#train_df['Path_video'] = get_path(train_df,"CNN_ResNet.mat")

train_df['Gender'] = train_df['Gender'].replace({'female': 0,'female ': 0, 'male': 1, 'male ': 1})
dev_df['Gender'] = dev_df['Gender'].replace({'female': 0,'female ': 0, 'male': 1, 'male ': 1})
test_df['Gender'] = test_df['Gender'].replace({'female': 0,'female ': 0, 'male': 1, 'male ': 1})
#there were some values with a whitespace 

def get_features(df):
    all_features=[]
    for i in range(2):
        features=[]
        file= pd.read_csv(df['Path_audio'][i])
        file=file.drop(['name','timeStamp'], axis=1)
        features=file.values
        #print(features)
        torch_features=torch.tensor(features)
        all_features.append(torch_features)
    #print(all_features)    
    return (all_features)

train_features= get_features(train_df)
#print(abcd[:2])
train_features= pad_sequence(train_features, batch_first=True, padding_value=0.0)
torch.save(train_features, 'audio_train_features_sample.pt')
#print(train_features)
print(train_features.dtype)
print(train_features.shape)
print("DONE WITH TRAIN")
#print("next")
#print(abcd[:2])
#dev_df['Audio_Features']= get_features(dev_df)
#test_df['Audio_Features']=get_features(test_df)
dev_features= get_features(dev_df)
dev_features= pad_sequence(dev_features, batch_first=True, padding_value=0.0)
torch.save(dev_features, 'audio_dev_features_sample.pt')
print("DONE WITH DEV")

test_features= get_features(test_df)
test_features= pad_sequence(test_features, batch_first=True, padding_value=0.0)
torch.save(test_features, 'audio_test_features_sample.pt')
print("DONE WITH TEST")

train_df= train_df.drop('Path_audio', axis=1)
dev_df= dev_df.drop('Path_audio', axis=1)
test_df= test_df.drop('Path_audio', axis=1)



gender_tensor= torch.tensor(train_df['Gender'])
torch.save(gender_tensor, 'gender_train_features.pt')
y_tensor= torch.tensor(train_df['PHQ_Binary'])
torch.save(y_tensor, 'y_train.pt')
#print(y_tensor.shape)
#print(gender_tensor.shape)
gender_tensor= torch.tensor(dev_df['Gender'])
torch.save(gender_tensor, 'gender_dev_features.pt')
y_tensor= torch.tensor(dev_df['PHQ_Binary'])
torch.save(y_tensor, 'y_dev.pt')

gender_tensor= torch.tensor(test_df['Gender'])
torch.save(gender_tensor, 'gender_test_features.pt')
y_tensor= torch.tensor(test_df['PHQ_Binary'])
torch.save(y_tensor, 'y_test.pt')



#print(y_tensor[:2])

#train_df.to_csv('audio_train_dataset_aayushi.csv', index=False)
#dev_df.to_csv('audio_dev_dataset_aayushi.csv', index=False )
#test_df.to_csv('audio_test_dataset_aayushi.csv', index=False )
