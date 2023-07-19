import os
import time
import torch
import argparse
import pandas as pd
import scipy.io as scio
import seaborn as sns
import matplotlib.pyplot as plt
import scipy,scipy.io
from sklearn.preprocessing import normalize
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from collections import defaultdict
import numpy as np
from models import VAE2,Classifier

domainSet =['regu','xray']
class TwoModalDataset(Dataset):
    def __init__(self,phase='train',sourceDomainIndex=0, targetDomainIndex = 0,trialIndex=0):
        self.load_mat(sourceDomainIndex,targetDomainIndex,trialIndex)
        self.phase = phase
        if self.phase == 'train':
            flag = 1
        #if self.phase == 'val':
        #    flag = 2
        if self.phase == 'test':
            flag = 2
        #print(self.feature_B)
        #print(len(self.feature_B))
        
        self.feature_B = self.feature_B[self.splitFlag_B==flag,]
        print(self.label_B)
        print(self.splitFlag_B==flag)
        
        
        self.label_B=np.array(self.label_B)
        
        
        self.label_B = self.label_B[self.splitFlag_B==flag]
        if self.phase == 'train':
            self.features = np.concatenate((self.feature_A,self.feature_B))
            self.labels = np.concatenate((self.label_A,self.label_B))
        if self.phase == 'val' or self.phase == 'test':
            self.features = self.feature_B
            self.labels = self.label_B
            
    def load_mat(self,sourceDomainIndex=0,targetDomainIndex=0,trialIndex=0):
        # load features and labels
        data_dir = './data/XrayBaggage20/'
        #data_A = scipy.io.loadmat(data_dir+'XrayDataset-'+domainSet[sourceDomainIndex]+'-resnet101-noft.mat')
        #data_A = scipy.io.loadmat('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/CRISPR.mat')
        '''
        data1=scio.loadmat('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/XrayDataset-regu-resnet101-noft.mat')
        data2=scio.loadmat('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/XrayDataset-xray-resnet101-noft.mat')
        
        
        df1 = pd.read_csv('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/CRISPR.csv')
        df2 = pd.read_csv('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/Compound.csv')
        df1=df1.iloc[:,1:]
        df2=df2.iloc[:,1:]
        df3=pd.concat([df1,df2])
        df3['Metadata_gene']=pd.factorize(df3['Metadata_gene'])[0]
        df3=df3.reset_index()
        df11=df3.iloc[:5759,1:]
        df11=df11.reset_index()
        df11=df11.iloc[:,1:]
        df22=df3.iloc[5759:,1:]
        df22=df22.reset_index()
        df22=df22.iloc[:,1:]
        
        
        

        df11['Metadata_gene']=df11['Metadata_gene'].astype('int64')
        dfx=df11.iloc[:,14:]
        dfx=np.array(dfx)
        dfy=df11['Metadata_gene']
        dfy=np.array(dfy)
        dfy=dfy.reshape(1,-1)
        data1['labels']=dfy
        data1['resnet101_features']=dfx

        df22['Metadata_gene']=df22['Metadata_gene'].astype('int64')
        dfx=df22.iloc[:,14:]
        dfx=np.array(dfx)
        dfy=df22['Metadata_gene']
        dfy=np.array(dfy)
        dfy=dfy.reshape(1,-1)
        data2['labels']=dfy
        data2['resnet101_features']=dfx
        
        data_A=data1
        data_B=data2
        '''
        data_A=scio.loadmat('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/data_A.mat')
        data_B=scio.loadmat('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/data_B.mat')        

                
        
        
        
        feature_A = data_A['resnet101_features']
        self.feature_A = normalize(feature_A,norm='l2')
        self.label_A = data_A['labels'][0,]
        self.num_class = len(np.unique(self.label_A))
        
        
        #data_B = scipy.io.loadmat(data_dir+'XrayDataset-'+domainSet[targetDomainIndex]+'-resnet101-noft.mat')
        #data_B = scipy.io.loadmat('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/Compound.mat')
        feature_B = data_B['resnet101_features']
        self.feature_B = normalize(feature_B,norm='l2')
        self.label_B = data_B['labels'][0,]
        
        #dataSplit = scipy.io.loadmat(data_dir+'instanceSplit_xrayDataset_unseen10.mat')# default data split for office-home, 30 unseen classes
        
        dataSplit = scipy.io.loadmat('/data/datacenter/H3C_GPU/projects/yuchen/gzsda-main/gzsda-main/data/XrayBaggage20/dataSplit1.mat')
        
        self.splitFlag_B = dataSplit['targetDomain_splitFlag'][0,trialIndex][0,targetDomainIndex][0,] # [0, index of trial][0, index of domain], 1--train, 2--test, 0--not used
        self.unseenClass_B = dataSplit['targetDomain_unseenClass'][0,trialIndex][0,targetDomainIndex][0,]
            
            
            
            
        data_A['resnet101_features'] = data_A['resnet101_features'].astype(np.float32)
        data_B['resnet101_features'] = data_B['resnet101_features'].astype(np.float32)
        
        
        #dataSplit['targetDomain_splitFlag']=dataSplit['targetDomain_splitFlag'].astype(np.uint8)
        #dataSplit['targetDomain_unseenClass']=dataSplit['targetDomain_unseenClass'].astype(np.uint8)
        #print('data_A',type(data_A['resnet101_features'][0][0]))
        #print('data_A',type(data_A['labels'][0][0]))
        #print('data_B',type(data_B['resnet101_features'][0][0]))
        #print('data_B',type(data_B['labels'][0][0]))
        #print('dataSplit',type(dataSplit['targetDomain_splitFlag'][0][0][0][0][0][0]))
        #print('dataSplit',type(dataSplit['targetDomain_unseenClass'][0][0][0][0][0][0]))
        
        
        
        
        
        
        
    def __len__(self):
        if self.phase == 'train': #or self.phase == 'val':
            return self.feature_A.shape[0]
        if self.phase == 'test':
            return self.feature_B.shape[0]
    def __getitem__(self,idx):

        # return a pair of regular and xray image features, which are paired randomly
        label = self.label_A[idx]
        indicesB_this_label = np.argwhere(self.label_B == label)
        if self.phase == 'test':
            idx_B = idx
            return self.feature_B[idx_B,:],self.label_B[idx_B]
        if len(indicesB_this_label) > 0:
            idx_B = np.random.choice(indicesB_this_label[:,0])
            return self.feature_A[idx,:], self.feature_B[idx_B,:],self.label_A[idx],self.label_B[idx_B]
        else:
            #print(np.asarray(np.ones_like(self.label_A[idx]), dtype='float64')  * -1)
            return self.feature_A[idx,:], np.zeros_like(self.feature_A[idx,:]), self.label_A[idx], np.asarray(np.ones_like(self.label_A[idx]), dtype='int64')  * -1


def test_model(model,dataset,dataloader,device,model_type='mlp'):
    since = time.time()
    
    num_class = dataset.num_class
    running_corrects = np.zeros((num_class,))
    num_sample_per_class = np.zeros((num_class,))
    # Iterate over data.
    for index, (features,labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            if model_type=='knn':
                preds = model.predict(features)
            if model_type=='mlp':
                model.eval()
                preds = model(features)
                preds = preds.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
            if index == 0:
                outputs_test = preds
                labels_test = labels
            else:
                outputs_test = np.concatenate((outputs_test, preds), 0)
                labels_test = np.concatenate((labels_test, labels), 0)
        if model_type=='mlp':
            preds = np.argmax(outputs_test,1)
        if model_type=='knn':
            preds = outputs_test
   
    for i in range(len(labels_test)):
        num_sample_per_class[labels_test[i]] += 1
        if preds[i]==labels_test[i]:
            running_corrects[labels_test[i]] += 1

    acc_per_class = running_corrects / num_sample_per_class
    print('running_corrects',running_corrects)
    print('num_sample_per_class',num_sample_per_class)
    
    
    #print('acc_per_class111111',acc_per_class)
    lst_acc=[x for x in acc_per_class if not np.isnan(x)]
    acc = np.mean(lst_acc)
    #print('acc111111',acc)
    #print('dataset.unseenClass_B',dataset.unseenClass_B)
    #print('acc_per_class[dataset.unseenClass_B==0]',acc_per_class[dataset.unseenClass_B==0])
    #print('acc_per_class[dataset.unseenClass_B==1]',acc_per_class[dataset.unseenClass_B==1])
    lst_seen=[x for x in acc_per_class[dataset.unseenClass_B==0] if not np.isnan(x)]
    lst_unseen=[x for x in acc_per_class[dataset.unseenClass_B==1] if not np.isnan(x)]
    
    #print('lstseen',lst_seen)
    #print('lst_unseen',lst_unseen)
    
    #print('bad')
    #print('acc_per_class',acc_per_class)
    #print(acc_per_class.shape)
    #print('dataset',dataset.unseenClass_B==0)
    #print((dataset.unseenClass_B==0).shape)
    #print('1111111',acc_per_class[dataset.unseenClass_B==0])
    acc_seen = np.mean(lst_seen)
  
    acc_unseen = np.mean(lst_unseen)
    
    time_elapsed = time.time() - since
    #print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('per-class acc:{:2.4f}, seen acc:{:2.4f}, unseen acc:{:2.4f}'.format(acc,acc_seen,acc_unseen))
    return acc_per_class,acc,acc_seen,acc_unseen

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    datasets = {x: TwoModalDataset(phase=x,sourceDomainIndex=args.sourceDomainIndex,targetDomainIndex=args.targetDomainIndex,trialIndex=args.trialIndex) for x in ['train','test']}
    data_loaders={}
    data_loaders['train'] = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, shuffle=True)
    data_loaders['trainall'] = DataLoader(dataset=datasets['train'], batch_size=len(datasets['train']), shuffle=False)
    data_loaders['test'] = DataLoader(dataset=datasets['test'], batch_size=len(datasets['test']), shuffle=False)

    def loss_fn(recon_xS, xS, recon_xT, xT, meanS, log_varS, meanT, log_varT):
        criterion = torch.nn.MSELoss(size_average=False)
        reconstruction_loss = criterion(recon_xS, xS) + criterion(recon_xT, xT)
        KLD = -0.5 * torch.sum(1 + log_varS - meanS.pow(2) - log_varS.exp())  -0.5 * torch.sum(1 + log_varT - meanT.pow(2) - log_varT.exp())
        #print(reconstruction_loss, KLD)
        return (reconstruction_loss + 0*KLD) / xS.size(0)
    def loss_fn2(recon_xS,recon_xS2, xS, recon_xT,recon_xT2, xT, meanS, log_varS, meanT, log_varT,yT,epoch):
        criterion = torch.nn.MSELoss(size_average=False)
        mask = yT!=-1
        reconstruction_loss = criterion(recon_xS, xS) + criterion(recon_xT[mask,:], xT[mask,:])
        cross_reconstruction_loss = criterion(recon_xS2[mask,:], xT[mask,:]) + criterion(recon_xT2[mask,:], xS[mask,:])
        KLD = -0.5 * torch.sum(1 + log_varS - meanS.pow(2) - log_varS.exp())  -0.5 * torch.sum(1 + log_varT[mask,:] - meanT[mask,:].pow(2) - log_varT[mask,:].exp())
        weight = epoch*5e-4
        return (reconstruction_loss+cross_reconstruction_loss+weight*KLD) / xS.size(0)
        
    vae = VAE2(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_domains = 2,
        num_labels=160 if args.conditional else 0).to(device)   ######  
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    ############################################################
    # train CVAE
    ############################################################
    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (xS,xT,yS,yT) in enumerate(data_loaders['train']):

            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)

            if args.conditional:
                recon_xS, meanS, log_varS, zS = vae(xS, yS, d=torch.zeros_like(yS).to(device))
                recon_xT, meanT, log_varT, zT = vae(xT, yT, d=torch.ones_like(yT).to(device))
                loss = loss_fn(recon_xS, xS, recon_xT, xT, meanS, log_varS, meanT, log_varT)
            else:
                
                xS=xS.to(torch.float32)
                #yS=yS.to(torch.int64)
                #print('xS',xS)
                #print('yS',yS)
                #print('dddddd',torch.zeros_like(yS).to(device))
                
                #print('xS',xS[0][0].dtype)
                #print('yS',yS[0].dtype)
                #print('dddddd',torch.zeros_like(yS).to(device)[0].dtype)
                #print(xS.shape)
                #print(yS.shape)
                #print(torch.zeros_like(yS).to(device)[0].shape)
                
                
                recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, yS, d=torch.zeros_like(yS).to(device))
                
                xT=xT.to(torch.float32)
                
                
                #print('xTTTTTTT',xT[0][0].dtype)
                #print('yTTTTTTTTTTTTTTTT',yT[0].dtype)
                #print('ddddddTTT',torch.ones_like(yT).to(device).dtype)
                
                
                recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, yT, d=torch.ones_like(yT).to(device))
                #loss = loss_fn(recon_xS, xS, recon_xT, xT, meanS, log_varS, meanT, log_varT)
                loss = loss_fn2(recon_xS, recon_xS2, xS, recon_xT,recon_xT2, xT, meanS, log_varS, meanT, log_varT,yT,epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ############################################################
    #Generating pseudo training samples and train/test a classifier
    ############################################################
    def next_batch(vae,batch_size=64):
        y = np.random.randint(160,size=batch_size)#######
        y = torch.from_numpy(y)        
        pseudo_x = vae.inference(n=batch_size,c=y,d = torch.ones_like(y))
        return pseudo_x, y
    def generate_z(xS,xT,vae):
        vae.eval()
        xS=xS.to(torch.float32)
        recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, 
                                       torch.zeros(xS.shape[0],).to(device), 
d=torch.zeros(xS.shape[0],dtype=torch.int64).to(device))
        xT=xT.to(torch.float32) 
        recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, 
                                                       torch.ones(xT.shape[0],).to(device), 
                                                       d=torch.ones(xT.shape[0],dtype=torch.int64).to(device))
        return recon_xS2, recon_xT2
    def save_generated_data_for_visualisation(datasets,vae,device):
        realSourceX = datasets['train'].feature_A
        realSourceY = datasets['train'].label_A
        realTargetX = datasets['train'].feature_B
        realTargetY = datasets['train'].label_B
        realTargetTestX = datasets['test'].feature_B
        realTargetTestY = datasets['test'].label_B
        recon_xS, recon_xT = generate_z(torch.tensor(realSourceX).to(device), 
                                        torch.tensor(realTargetX).to(device), 
                                        vae)
        for i in range(10):
            temp1, temp2 = generate_z(torch.tensor(realSourceX).to(device), 
                                        torch.tensor(realTargetX).to(device), 
                                        vae)
            #temp1, temp2 = np.array(temp1.detach().cpu()), np.array(temp2.detach().cpu())
            recon_xS, recon_xT = torch.cat((recon_xS,temp1),dim=0), torch.cat((recon_xT,temp2),dim=0)
        recon_xS, recon_xT = np.array(recon_xS.detach().cpu()), np.array(recon_xT.detach().cpu())
        scipy.io.savemat('./results/'+args.filename+'_generated_data.mat',
                                mdict={'realSourceX': realSourceX, 'realSourceY':realSourceY,
                                       'realTargetX':realTargetX, 'realTargetY':realTargetY,
                                       'fakeSourceX': recon_xT,
                                       'fakeTargetX': recon_xS,
                                       'realTargetTestX':realTargetTestX, 
                                       'realTargetTestY':realTargetTestY})


    #save_generated_data_for_visualisation(datasets,vae,device)
    
    classifier = Classifier(input_dim=300,num_labels=160).to(device) # train and test a classifier
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.01)
    lr_scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls,25, gamma=0.1)
    
    num_epochs = 50
    acc_per_class = np.zeros((num_epochs,160))
    acc = np.zeros((num_epochs,))
    acc_seen = np.zeros((num_epochs,))
    acc_unseen = np.zeros((num_epochs,))
    for epoch in range(num_epochs):
        print(epoch)
        for iteration, (xS,xT,yS,yT) in enumerate(data_loaders['train']):
            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)
            #x,y = next_batch(vae,batch_size=1024)
            recon_xS,recon_xT = generate_z(xS,xT,vae)
            mask = yT!=-1
            xT = xT[mask,:]
            yT = yT[mask]
            recon_xT = recon_xT[mask,:]
            xtrain = torch.cat((xS,xT,recon_xS,recon_xT),dim=0)
            ytrain = torch.cat((yS,yT,yS,yT),dim=0)
            output = classifier(xtrain)
            loss_cls = classifier.lossfunction(output, ytrain)
            optimizer_cls.zero_grad()
            loss_cls.backward()
            optimizer_cls.step()
        lr_scheduler_cls.step()
        acc_per_class[epoch,],acc[epoch],acc_seen[epoch],acc_unseen[epoch] = \
                test_model(classifier, datasets['test'], data_loaders['test'],device,model_type='mlp')
        #print('acc_per_class',acc_per_class)
        #print('acc',acc)
        
    scipy.io.savemat('./results/'+args.filename+'.mat',
                     mdict={'acc_per_class':acc_per_class,'acc':acc,
                            'acc_seen':acc_seen,'acc_unseen':acc_unseen})
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[300, 512])#########
    parser.add_argument("--decoder_layer_sizes", type=list, default=[512, 300])##########
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sourceDomainIndex", type=int, default=0)
    parser.add_argument("--targetDomainIndex", type=int, default=0)
    parser.add_argument("--trialIndex", type=int, default=0)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()
    source = domainSet[args.sourceDomainIndex]
    target = domainSet[args.targetDomainIndex]
    args.filename = 'xray10-'+source+'-'+target+'-trial'+str(args.trialIndex)+'-vaeEpochs-'+str(args.epochs)+'-latSize-'+str(args.latent_size)+'lr'+str(args.learning_rate)
    print(args.filename)

    main(args)
