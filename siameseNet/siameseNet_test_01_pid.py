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
sc = MinMaxScaler()

def imshow(img,name,text=None,text2=None,text3=None,text4=None):
    npimg = img.numpy()
    count_t = time.time()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    if text2:
        plt.text(0, 115, text2, fontsize=6, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    if text3:
        plt.text(0, 130, text3, fontsize=6, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    if text4:
        plt.text(75, 100, text4, fontsize=6, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('/home/iichsk/workspace/siameseNet/picture/{}_{}.png'.format(name,count_t))    
    plt.close()

def show_plot(iteration,loss,name):
    plt.plot(iteration,loss)
    plt.savefig('/home/iichsk/workspace/siameseNet/picture/{}.png'.format(name))
    plt.close()
class Config():
    training_dir = "/home/iichsk/workspace/dataset/facedatabase/waveform/training/"
    testing_dir = "/home/iichsk/workspace/dataset/facedatabase/waveform/testing/"
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
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)), img0_tuple[0], img1_tuple[0], img0_tuple[1],img1_tuple[1]
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

'''class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        cnn_tmp = output
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output, cnn_tmp

    def forward(self, input1, input2):
        output1, cnn_tmp1 = self.forward_once(input1)
        output2, cnn_tmp2 = self.forward_once(input2)

        return output1, output2, cnn_tmp1, cnn_tmp2'''

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
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


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
        img0, img1 , label, _, _,_,_ = data
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
show_plot(counter,loss_history,'loss_graph')

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
'''dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    output1,output2, middle_value1, middle_value2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    middle_value1 = torch.transpose(middle_value1,0,1)
    middle_value2 = torch.transpose(middle_value2,0,1)


    middle_value1 = middle_value1.cpu().detach()
    middle_value2 = middle_value2.cpu().detach()

    middle_value1_np = np.zeros((int(middle_value1.shape[0]), int(middle_value1.shape[2]), int(middle_value1.shape[3])))
    middle_value2_np = np.zeros((int(middle_value2.shape[0]), int(middle_value2.shape[2]), int(middle_value2.shape[3])))

    for ii in range(int(middle_value1.shape[0])):
        middle_value1_np[ii, :, :]=sc.fit_transform(torch.squeeze(middle_value1[ii, :, :]))
        middle_value2_np[ii, :, :]=sc.fit_transform(torch.squeeze(middle_value2[ii, :, :]))
    middle_value1_torch = torch.from_numpy(np.expand_dims(middle_value1_np,axis=1))
                middle_value2_torch = torch.from_numpy(np.expand_dims(middle_value2_np,axis=1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'test result','Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
    imshow(torchvision.utils.make_grid(middle_value1_torch),'middle_value/cnn_middle_feature1')
                imshow(torchvision.utils.make_grid(middle_value2_torch),'middle_value/cnn_middle_feature2')'''
#allFiles, _ = map(list, zip(*test_dataloader.dataset.samples))
dataiter = iter(test_dataloader)

for i, data in enumerate(test_dataloader,0):
    x0,x1,label2,fname0,fname1,c0,c1 = data
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
    
    imshow(torchvision.utils.make_grid(concatenated),'waveform2_test_result/test_{}'.format(i),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()),fname0,fname1,'label={}'.format(label2))
    imshow(torchvision.utils.make_grid(querry_cnn_tmp1_torch),'querry_middle_value/querry_cnn_tmp1_torch_{}'.format(i))
    imshow(torchvision.utils.make_grid(querry_cnn_tmp2_torch),'querry_middle_value/querry_cnn_tmp2_torch_{}'.format(i))
    imshow(torchvision.utils.make_grid(querry_cnn_tmp3_torch),'querry_middle_value/querry_cnn_tmp3_torch_{}'.format(i))
imshow(torchvision.utils.make_grid(subset_cnn_tmp1_torch),'subset_middle_value/subset_cnn_tmp1_torch')
imshow(torchvision.utils.make_grid(subset_cnn_tmp2_torch),'subset_middle_value/subset_cnn_tmp2_torch')
imshow(torchvision.utils.make_grid(subset_cnn_tmp3_torch),'subset_middle_value/subset_cnn_tmp3_torch')
