import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
from sklearn.preprocessing import MinMaxScaler
from scipy import io

sc = MinMaxScaler()
count_t = time.time()

def imshow(img,name,text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('/home/iichsk/workspace/siameseNet/picture_L1/{}_{}.png'.format(name,count_t))    
    plt.close()

def show_plot(iteration,loss,name):
    plt.plot(iteration,loss)
    plt.savefig('/home/iichsk/workspace/siameseNet/picture_L1/{}_{}.png'.format(name,count_t))
    plt.close()
class Config():
    training_dir = "/home/iichsk/workspace/dataset/facedatabase/faces/training/"
    testing_dir = "/home/iichsk/workspace/dataset/facedatabase/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 100

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated),'example test')
print(example_batch[2].numpy())



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.Ref_pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.batch1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.batch2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3)
        self.batch3 = nn.BatchNorm2d(8)
        

        self.fc1 = nn.Linear(8*100*100, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 5)
        self.relu = nn.ReLU(inplace=True)

    def forward_once(self, x):

        output = self.Ref_pad(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.batch1(output)
        cnn_tmp1 = output

        output = self.Ref_pad(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.batch2(output)
        cnn_tmp2 = output

        output = self.Ref_pad(output)
        output = self.conv3(output)
        output = self.relu(output)
        output = self.batch3(output)
        cnn_tmp3 = output

        output = output.view(output.size()[0], -1)
        output = self.relu(self.fc1(output))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output, cnn_tmp1, cnn_tmp2, cnn_tmp3

    def forward(self, input1, input2):
        output1, subset_cnn_tmp1, subset_cnn_tmp2, subset_cnn_tmp3 = self.forward_once(input1)
        output2, querry_cnn_tmp1, querry_cnn_tmp2, querry_cnn_tmp3 = self.forward_once(input2)

        return output1, output2, subset_cnn_tmp1, subset_cnn_tmp2, subset_cnn_tmp3, querry_cnn_tmp1, querry_cnn_tmp2, querry_cnn_tmp3


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 1) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 1))


        return loss_contrastive

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1,output2, _, _, _, _, _, _ = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
show_plot(counter,loss_history,'loss/loss_graph')
io.savemat(
        "loss/loss{}.mat".format(count_t),
        {'loss': loss_history})
folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)
print('dataiter',dataiter)
for i, data in enumerate(test_dataloader,0):
    _,x1,label2 = data
    concatenated = torch.cat((x0,x1),0)
    output1,output2, subset_cnn_tmp1, subset_cnn_tmp2, subset_cnn_tmp3, querry_cnn_tmp1, querry_cnn_tmp2, querry_cnn_tmp3 = net(Variable(x0).cuda(),Variable(x1).cuda())
    

    subset_cnn_tmp1 = torch.transpose(subset_cnn_tmp1,0,1)
    subset_cnn_tmp2 = torch.transpose(subset_cnn_tmp2,0,1)
    subset_cnn_tmp3 = torch.transpose(subset_cnn_tmp3,0,1)
    
    subset_cnn_tmp1 = subset_cnn_tmp1.cpu().detach()
    subset_cnn_tmp2 = subset_cnn_tmp2.cpu().detach()
    subset_cnn_tmp3 = subset_cnn_tmp3.cpu().detach()

    subset_cnn_tmp1_np = np.zeros((int(subset_cnn_tmp1.shape[0]), int(subset_cnn_tmp1.shape[2]), int(subset_cnn_tmp1.shape[3])))
    subset_cnn_tmp2_np = np.zeros((int(subset_cnn_tmp2.shape[0]), int(subset_cnn_tmp2.shape[2]), int(subset_cnn_tmp2.shape[3])))
    subset_cnn_tmp3_np = np.zeros((int(subset_cnn_tmp3.shape[0]), int(subset_cnn_tmp3.shape[2]), int(subset_cnn_tmp3.shape[3])))

    querry_cnn_tmp1 = torch.transpose(querry_cnn_tmp1,0,1)
    querry_cnn_tmp2 = torch.transpose(querry_cnn_tmp2,0,1)
    querry_cnn_tmp3 = torch.transpose(querry_cnn_tmp3,0,1)
    
    querry_cnn_tmp1 = querry_cnn_tmp1.cpu().detach()
    querry_cnn_tmp2 = querry_cnn_tmp2.cpu().detach()
    querry_cnn_tmp3 = querry_cnn_tmp3.cpu().detach()

    querry_cnn_tmp1_np = np.zeros((int(querry_cnn_tmp1.shape[0]), int(querry_cnn_tmp1.shape[2]), int(querry_cnn_tmp1.shape[3])))
    querry_cnn_tmp2_np = np.zeros((int(querry_cnn_tmp2.shape[0]), int(querry_cnn_tmp2.shape[2]), int(querry_cnn_tmp2.shape[3])))
    querry_cnn_tmp3_np = np.zeros((int(querry_cnn_tmp3.shape[0]), int(querry_cnn_tmp3.shape[2]), int(querry_cnn_tmp3.shape[3])))
   

    for ii in range(int(subset_cnn_tmp1.shape[0])):
        subset_cnn_tmp1_np[ii, :, :]=sc.fit_transform(torch.squeeze(subset_cnn_tmp1[ii, :, :]))
        querry_cnn_tmp1_np[ii, :, :]=sc.fit_transform(torch.squeeze(querry_cnn_tmp1[ii, :, :]))

    for iii in range(int(subset_cnn_tmp2.shape[0])):
        subset_cnn_tmp2_np[iii, :, :]=sc.fit_transform(torch.squeeze(subset_cnn_tmp2[iii, :, :]))
        subset_cnn_tmp3_np[iii, :, :]=sc.fit_transform(torch.squeeze(subset_cnn_tmp3[iii, :, :]))
        querry_cnn_tmp2_np[iii, :, :]=sc.fit_transform(torch.squeeze(querry_cnn_tmp2[iii, :, :]))
        querry_cnn_tmp3_np[iii, :, :]=sc.fit_transform(torch.squeeze(querry_cnn_tmp3[iii, :, :]))    

    subset_cnn_tmp1_torch = torch.from_numpy(np.expand_dims(subset_cnn_tmp1_np,axis=1))
    subset_cnn_tmp2_torch = torch.from_numpy(np.expand_dims(subset_cnn_tmp2_np,axis=1))
    subset_cnn_tmp3_torch = torch.from_numpy(np.expand_dims(subset_cnn_tmp3_np,axis=1))
    querry_cnn_tmp1_torch = torch.from_numpy(np.expand_dims(querry_cnn_tmp1_np,axis=1))
    querry_cnn_tmp2_torch = torch.from_numpy(np.expand_dims(querry_cnn_tmp2_np,axis=1))
    querry_cnn_tmp3_torch = torch.from_numpy(np.expand_dims(querry_cnn_tmp3_np,axis=1))   

    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'test_result/test_{}'.format(i),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
    imshow(torchvision.utils.make_grid(querry_cnn_tmp1_torch),'querry_middle_value/querry_cnn_tmp1_torch_{}'.format(i))
    imshow(torchvision.utils.make_grid(querry_cnn_tmp2_torch),'querry_middle_value/querry_cnn_tmp2_torch_{}'.format(i))
    imshow(torchvision.utils.make_grid(querry_cnn_tmp3_torch),'querry_middle_value/querry_cnn_tmp3_torch_{}'.format(i))
imshow(torchvision.utils.make_grid(subset_cnn_tmp1_torch),'subset_middle_value/subset_cnn_tmp1_torch')
imshow(torchvision.utils.make_grid(subset_cnn_tmp2_torch),'subset_middle_value/subset_cnn_tmp2_torch')
imshow(torchvision.utils.make_grid(subset_cnn_tmp3_torch),'subset_middle_value/subset_cnn_tmp3_torch')

    