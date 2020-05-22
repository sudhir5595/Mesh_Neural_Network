import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from pytorch_model_summary import summary
import numpy as np

#n = 10
class Net(nn.Module):
    def __init__(self,k):                                  # k is the number of output classes needed
        super(Net,self).__init__()
        self.spatial = spatial_Des()
        self.structural = structural_Des()
        self.Mesh1 = Mesh1()
        self.Mesh2 = Mesh2()
        #self.conv1 = nn.Conv1d(1024,1024,1)
        self.linear = nn.Linear(1024,1024)           
        self.mlp2 = mlp2()
        self.mlp3 = mlp3(k)
        self.k = k
        
    
    def forward(self,x):
        n = x.size()[0] 
        center = x[:,0:3]
       # print(center.shape)
        corner = x[:,3:12]
        normal = x[:,12:15]
        neighbour_index = x[:,15:18]                  

        y = self.spatial(center)
        z = self.structural(corner,normal,neighbour_index)
        out1,out2 = self.Mesh1(y,z,neighbour_index)
        #print(out1.shape)
        #print(out2.shape)
        out3,out4 = self.Mesh2(out1,out2,neighbour_index)
        
        out5 = torch.cat((out3,out4),dim=1)
        #print(out5.shape)
        #out5 = out5.view(n,1024,1)
        out5 = self.linear(out5)
        #out5 = out5.view(n,1024)
        
        out6 = torch.cat((out5,out3,out1),dim=1)
        #print(out6.shape)
        
        out6 = self.mlp2(out6)
        
        final_output = self.mlp3(out6)
        
        return final_output
         
        


class mlp2(nn.Module):
    def __init__(self):
        super(mlp2,self).__init__()
        #self.conv1 = nn.Conv1d(1792,1024,1)
        self.linear = nn.Linear(1792,1024)

    def forward(self,x):
        n = x.size()[0]
        #x = x.view(n,1792,1)
        #x = self.conv1(x)
        #x = x.view(n,1024)
        x = self.linear(x)
        l = torch.max(x,0)
        return l.values

class mlp3(nn.Module):
    def __init__(self,k):
        super(mlp3,self).__init__()
        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256,k)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self,x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.linear3(x)
        #print(x.shape)
        return x.unsqueeze(dim=0)


class spatial_Des(nn.Module):                                                            #as described in the paper, this is just a multilayer perceptron
    def __init__(self):
        super(spatial_Des,self).__init__()
        #self.conv1 = nn.Conv1d(3,64,1)
        #self.conv2 = nn.Conv1d(64,64,1)
        self.linear1 = nn.Linear(64,64)
        self.linear2 = nn.Linear(3,64)
        #self.bn1 = nn.BatchNorm1d(64)
        
        
    def forward(self,x):
        n = x.size()[0]
        #x = x.view(n,3,1)
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = self.conv2(x)
        x = self.linear1(F.relu(self.linear2(x)))
        #x = x.view(n,64)
        return x



class structural_Des(nn.Module):
    def __init__(self):
        super(structural_Des,self).__init__()
        self.kc = Kernel_Correlation(4)                  
        self.frc = Face_Rotate_Conv()
        self.conv1 = nn.Conv1d(131,131,1)
        self.conv2 = nn.Conv1d(131,131,1)
        #self.linear1 = nn.Linear(131,131)
        #self.linear2 = nn.Linear(131,131)
        self.bn1 = nn.BatchNorm1d(131)
        
    def forward(self,corner,normal,neighbour):
        n = corner.size()[0]
        x = self.frc(corner)
        z = self.kc(normal,neighbour)

        x = torch.cat((normal,x,z),dim=1)        
        x = x.view(n,131,1)
        x = self.conv2(F.relu(self.bn1(self.conv1(x))))
        #x = self.linear2(F.relu(self.linear1(x)))
        x = x.view(n,131)

        return x
        
        
        



class Face_Rotate_Conv(nn.Module):
    def __init__(self):
        super(Face_Rotate_Conv,self).__init__()
        self.linear3 = nn.Linear(32,64)
        self.linear4 = nn.Linear(64,64)
        self.conv = nn.Conv1d(1,32,6,3)

        
    def forward(self,corner):                # corner is (n,9) . we take these face wise and apply 1-d convolution of 6 features out of 9 in 3 turns
        n = corner.size()[0]
        """
        final_array = torch.zeros(n,64)
        for i in range(n):
            corner_1_i = corner[i,0:3]      #first corner vector
            corner_2_i = corner[i,3:6]    #second
            corner_3_i = corner[i,6:9]    #third

            corner_1_2 = torch.cat((corner_1_i,corner_2_i),0)           #appending 1st and 2nd corner
            corner_2_3 = torch.cat((corner_2_i,corner_3_i),0)           #2nd and 3rd
            corner_3_1 = torch.cat((corner_3_i,corner_1_i),0)           #3rd and 1st

            corner_1_2 = corner_1_2.view(1,1,6)                         #tensor.view function reshapes the 6 length tensor to 1,1,6(batch_size,input_channels,no_of_features) . This is the format of input to be sent in conv1d.
            corner_2_3 = corner_2_3.view(1,1,6)
            corner_3_1 = corner_3_1.view(1,1,6)

            conv1_2 = self.conv(corner_1_2)
            conv1_2 = conv1_2.view(-1)                  # a 32 length feature tensor

            conv2_3 = self.conv(corner_2_3)
            conv2_3 = conv2_3.view(-1)

            conv3_1 = self.conv(corner_3_1)
            conv3_1 = conv3_1.view(-1)


            vec4 = torch.mean(torch.stack([conv1_2,conv2_3,conv3_1]),dim=0)     #simply (conv_1_2 + conv_2_3 + conv_3_1)/3


            vec4 = self.linear4(F.relu(self.linear3(vec4)))
            final_array[i,:] = vec4
        """
        #final_array = torch.zeros(n,6,3)
        final_array = torch.cat((corner[:,0:3],corner[:,3:6],corner[:,6:9],corner[:,0:3]),dim=1)
        #final_array[:,:,1] = torch.cat((corner[:,3:6],corner[:,6:9]),dim=1)
        #final_array[:,:,2] = torch.cat((corner[:,6:9],corner[:,0:3]),dim=1)
        final_array = final_array.view(n,1,12)
        final_array = self.conv(final_array)
        #print(final_array.shape)

        new_arr = torch.mean(final_array,dim=2)
        vec4 = self.linear4(F.relu(self.linear3(new_arr)))


        return vec4


class Kernel_Correlation(nn.Module):
    def __init__(self,k):
        super(Kernel_Correlation,self).__init__()
        self.theta_par = nn.Parameter(torch.rand(1,64, 4) * np.pi)
        self.phi_par = nn.Parameter(torch.rand(1,64, 4) *2* np.pi)
        #self.learnable_kernel = nn.Parameter(torch.empty(64,k,3))
        #self.learnable_kernel.requires_grad = True                           #since this matrix is learnable it is initialised by torch.ones and nn.parameter is applied.
        self.k = k
        #nn.init.xavier_uniform_(self.learnable_kernel, gain=nn.init.calculate_gain('relu'))                                                                                        #change initialisation mode to xavier or he.        

    def forward(self,normal,neighbour):
        sigma = 0.2
        n = normal.size()[0]
        neighbour = neighbour.long()
        normal_neigh1 = normal[neighbour[:,0]]
        normal_neigh2 = normal[neighbour[:,1]]
        normal_neigh3 = normal[neighbour[:,2]]

        face_normal = normal.unsqueeze(2).expand(-1,-1,64).unsqueeze(3)
        normal_neigh1 = normal_neigh1.unsqueeze(2).expand(-1,-1,64).unsqueeze(3)
        normal_neigh2 = normal_neigh2.unsqueeze(2).expand(-1,-1,64).unsqueeze(3)
        normal_neigh3 = normal_neigh3.unsqueeze(2).expand(-1,-1,64).unsqueeze(3)

        fea = torch.cat((face_normal,normal_neigh1,normal_neigh2,normal_neigh3),dim=3)
        fea = fea.unsqueeze(4).expand(-1,-1,-1,-1,4)

        kernel = torch.cat((torch.sin(self.theta_par)*torch.sin(self.phi_par) , torch.sin(self.theta_par)*torch.cos(self.phi_par) , torch.cos(self.theta_par)) ,dim=0 )
        kernel = kernel.unsqueeze(0).expand(n,-1,-1,-1)
        kernel = kernel.unsqueeze(3).expand(-1,-1,-1,4,-1)

        correl = torch.sum((fea - kernel)**2,1)

        fea = torch.sum(torch.sum(np.e**(correl / (-2*sigma**2)), 3), 2) / 16
        """
        final_array = torch.zeros(n,64)
        for i in range(n):
            neighbours = neighbour[i,:] 
            a = int(neighbours[0].item())
            b = int(neighbours[1].item())         
            c = int(neighbours[2].item())                            # a,b,c are the three neighbour indexes
            
            #print(i)
            #print(a,b,c)
            #print(normal.shape)
            
            normal_i = normal[i,:]
            normal_a = normal[a,:]
            normal_b = normal[b,:]
            normal_c = normal[c,:]

            for m in range(64):
                sum = 0
                for l in range(self.k):
                    x = normal_i - self.learnable_kernel[m,l,:]
                    norm = torch.norm(x)
                    sum = sum + exp(-1*norm*norm)/(2*sigma*sigma)

                final_array[i,m] = sum/(self.k*4)
        """
        return fea




class Mesh1(nn.Module):
    def __init__(self):
        super(Mesh1,self).__init__()
        self.aggregation1 = Aggregation1()
        self.combination1 = Combination1()
        
    def forward(self,spatial,structural,neighbour):
        out1 = self.combination1(spatial,structural)
        out2 = self.aggregation1(structural,neighbour)
        return out1,out2
        
    


class Combination1(nn.Module):
    def __init__(self):
        super(Combination1,self).__init__()
        self.conv1 = nn.Conv1d(64+131,256,1)
        
        
    def forward(self,spatial,structural):
        n = structural.size()[0]
        a1 = torch.cat((spatial,structural),dim=1)    # a1 has size (n,64+131) now
        a1 = a1.view(n,131+64,1)
        a1 = self.conv1(a1)
        a1 = a1.view(n,256)            #a1 has size (n,256) now
        
        
        return a1
        


class Aggregation1(nn.Module):
    def __init__(self):
        super(Aggregation1,self).__init__()
        self.conv1 = nn.Conv1d(131,256,1)

        
    def forward(self,structural,neighbour):
        n = structural.size()[0]
        """                   #neighbour has size n,3 , structural has size n,131
        final_array = torch.zeros(n,131)
        
        for i in range(n):
            neighbours = neighbour[i,:] 
            a = int(neighbours[0].item())
            b = int(neighbours[1].item())         
            c = int(neighbours[2].item())                            # a,b,c are the three neighbour indexes
            
            
            features_i = structural[i,:]
            features_a = structural[a,:]
            features_b = structural[b,:]
            features_c = structural[c,:]
            
            vec4 = torch.mean(torch.stack([features_i,features_a,features_b,features_c]),dim=0)         #implementing avg. aggregation and not concatenation

            final_array[i,:] = vec4
        
        final_array = final_array.view(n,131,1)
        final_array_1 = self.conv1(final_array)

        final_array_2 = final_array_1.view(n,256)
        """
        neighbour = neighbour.long()
        first_neigh = structural[neighbour[:,0]]
        second_neigh = structural[neighbour[:,1]]
        third_neigh = structural[neighbour[:,2]]

        final_array = torch.mean(torch.stack([structural,first_neigh,second_neigh,third_neigh]),dim=0)

        final_array = final_array.view(n,131,1)
        final_array_1 = self.conv1(final_array)

        final_array_2 = final_array_1.view(n,256)
        
        return final_array_2
            
        


        

class Mesh2(nn.Module):
    def __init__(self):
        super(Mesh2,self).__init__()
        self.aggregation2 = Aggregation2()
        self.combination2 = Combination2()
        
    def forward(self,out1,out2,neighbour):
        out3 = self.combination2(out1,out2)
        out4 = self.aggregation2(out2,neighbour)
        return out3,out4
        
    


class Combination2(nn.Module):
    def __init__(self):
        super(Combination2,self).__init__()
        self.conv1 = nn.Conv1d(512,512,1)
        
    def forward(self,out1,out2):
        n = out1.size()[0]
        #print(out1.shape)
        #print(out2.shape)
        a1 = torch.cat((out1,out2),dim=1)       # a1 has size n,512 now
        a1 = a1.view(n,512,1)    
        a1 = self.conv1(a1)
        a1 = a1.view(n,512)            
        
        return a1
        


class Aggregation2(nn.Module):
    def __init__(self):
        super(Aggregation2,self).__init__()
        self.conv1 = nn.Conv1d(256,512,1)
        
    def forward(self,out2,neighbour):
        n = neighbour.size()[0]
        """                  
        final_array = torch.zeros(n,256)
        for i in range(n):
            neighbours = neighbour[i,:] 
            a = int(neighbours[0].item())
            b = int(neighbours[1].item())         
            c = int(neighbours[2].item())                            # a,b,c are the three neighbour indexes
            
            
            features_i = out2[i,:]
            features_a = out2[a,:]
            features_b = out2[b,:]
            features_c = out2[c,:]
            
            
            
            vec4 = torch.mean(torch.stack([features_i,features_a,features_b,features_c]),dim=0)         #implementing avg. aggregation and not concatenation
            final_array[i,:] = vec4
        
        final_array = final_array.view(n,256,1)
        final_array_1 = self.conv1(final_array)

        final_array_2 = final_array_1.view(n,512)
        """

        neighbour = neighbour.long()
        first_neigh = out2[neighbour[:,0]]
        second_neigh = out2[neighbour[:,1]]
        third_neigh = out2[neighbour[:,2]]

        final_array = torch.mean(torch.stack([out2,first_neigh,second_neigh,third_neigh]),dim=0)

        final_array = final_array.view(n,256,1)
        final_array_1 = self.conv1(final_array)

        final_array_2 = final_array_1.view(n,512)
        return final_array_2
        


#model = Net(5)

#inp = torch.randn(316,18)
#print(model(inp))
#print(summary(model,inp))