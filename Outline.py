import torch
import torch.nn as nn
import torch.nn.functional as F

n = 10
class Net(nn.Module):
    def __init__(self,k):                                  # k is the number of output classes needed
        super(Net,self).__init__()
        self.spatial = spatial_Des()
        self.structural = structural_Des()
        self.Mesh1 = Mesh1()
        self.Mesh2 = Mesh2()
        self.linear1 = nn.Linear(1024,1024)
        self.linear2 = nn.Linear(1024,1024)             # mlp left to implement
        self.mlp2 = mlp2()
        self.mlp3 = mlp3(k)
        
    
    def forward(self,x):
       # center = x[:,3]
       # corner = x[:,3:12]
       # normal = x[:,12:15]
       # neighbour_index = x[:,15:18]                  #ignore this,will fix this once i get the data format
        
        y = self.spatial(center)
        z = self.structural(corner,normal,neighbour_index)
        out1,out2 = self.Mesh1(y,z,neighbour_index)
        out3,out4 = self.Mesh2(out1,out2,neighbour_index)
        
        out5 = torch.cat((out3,out4),0)
        out5 = self.linear2(F.relu(self.Linear1(out5)))
        
        out6 = torch.cat((out5,out3),0)
        out6 = torch.cat((out6,out1),0)
        
        out6 = self.mlp2(out6)
        
        final_output = self.mlp3(out6,k)
        
        return final_output
         
        


class mlp2(nn.Module):
    def __init__(self):
        super(mlp2,self).__init__()
        self.linear1 = nn.Linear(n*1792,n*1024)
        self.linear2 = nn.Linear(n*1024,n*1024)

    def forward(self,x):
        x = self.linear2(F.relu(self.linear1))
        x = x.view(n,1024)
        l = torch.max(x,0)
        return l.values

class mlp3(nn.Module):
    def __init__(self,k):
        super(mlp3,self).__init__()
        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256,k)

    def forward(self,x):
        x = self.linear3(F.relu(self.linear2(F.relu(self.linear1(x)))))
        return x


class spatial_Des(nn.Module):                                                            #as described in the paper, this is just a multilayer perceptron
    def __init__(self):
        super(spatial_Des,self).__init__()
        self.linear1 = nn.Linear(n*3,n*64)
        self.linear2 = nn.Linear(n*64,n*64)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# In[ ]:


class structural_Des(nn.Module):
    def __init__(self):
        super(structural_Des,self).__init__()
        self.kc = Kernel_Correlation(4)                   #the 2 arguments are k(number of vectors you want to learn for each kernel) and sigma(both are hyperparameters to be adjusted)
        self.frc = Face_Rotate_Conv()
        self.linear1 = nn.Linear(131,131)
        self.linear2 = nn.Linear(131,131)
        
    def forward(self,corner,normal,neighbour):
        x = self.frc(corner)
        z = self.kc(normal,neighbour)
        z = torch.cat((normal,z),0)
        x = torch.cat((z,x),0)                              #torch.cat((a1,a2),0) basically appends two tensors a1 and a2 and returs the new
        
        x = self.linear2(F.relu(self.linear1(x)))
        
        return x
        
        
        


# In[ ]:



class Face_Rotate_Conv(nn.Module):
    def __init__(self):
        super(Face_Rotate_Conv,self).__init__()
       # self.linear1 = nn.Linear(9,32)
        #self.linear2 = nn.Linear(32,32)
        self.linear3 = nn.Linear(32,64)
        self.linear4 = nn.Linear(64,64)
        self.conv = nn.Conv1d(1,32,6)

        
    def forward(self,corner):                # corner is n*9 . we take these face wise and apply 1-d convolution of 6 features out of 9 in 3 turns
        final_array = torch.zeros(n*64)
        for i in range(n):
            corner_1_i = corner[i*9:i*9+3]      #first corner vector
            corner_2_i = corner[i*9+3:i*9+6]    #second
            corner_3_i = corner[i*9+6:i*9+9]    #third

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
            final_array[i*64:i*64+64] = vec4

        return final_array


class Kernel_Correlation(nn.Module):
    def __init__(self,k):
        super(Kernel_Correlation,self).__init__()
        self.learnable_kernel = nn.Parameter(torch.randn(64,k,3))
        self.learnable_kernel.requires_grad = True                           #since this matrix is learnable it is initialised by torch.ones and nn.parameter is applied.
                                                                                                #change initialisation mode to xavier or he.        

    def forward(self,normal,neighbour):
        sigma = 1
        final_array = torch.zeros(n*64)
        for i in range(n):
            neighbours = neighbour[3*i:3*i+3] 
            a = neighbours[0]
            b = neighbours[1]         
            c = neighbours[2]                            # a,b,c are the three neighbour indexes
            
            
            normal_i = normal[i*3:i*3+3]
            normal_a = normal[a*3:a*3+3]
            normal_b = normal[b*3:b*3+3]
            normal_c = normal[c*3:c*3+3]

            for m in range(64):
                sum = 0
                for l in range(k):
                    x = normal_i - learnable_kernel[m,l,:]
                    norm = torch.norm(x)
                    sum = sum + exp(-1*norm*norm)/(2*sigma*sigma)

                final_array[i*64,i*64+m] = sum/(k*4)

        return final_array












        
        


# In[ ]:


class Mesh1(nn.Module):
    def __init__(self):
        super(Mesh1,self).__init__()
        self.aggregation1 = Aggregation1()
        self.combination1 = Combination1()
        
    def forward(self,spatial,structural,neighbour):
        out1 = self.combination1(spatial,structural)
        out2 = self.aggregation1(structural,neighbour)
        return out1,out2
        
    


# In[ ]:


class Combination1(nn.Module):
    def __init__(self):
        super(Combination1,self).__init__()
        self.linear1 = nn.Linear(n*(64+131),n*200)
        self.linear2 = nn.Linear(n*200,n*256)
        
    def forward(self,spatial,structural):
        a1 = torch.cat((spatial,structural),0)    # a1 has size n*(64+131 now)
        a1 = F.relu(self.linear1(a1))            #a1 has size n*200 now
        a1 = self.linear2(a1)
        
        return a1
        
       # a1 = torch.cat((spatial,structural),dim=0)


# In[ ]:


class Aggregation1(nn.Module):
    def __init__(self):
        super(Aggregation1,self).__init__()
        #self.linear1 = nn.Linear(128,256)
        #self.linear2 = nn.Linear(256,128)
        self.linear3 = nn.Linear(n*131,n*512)
        self.linear4 = nn.Linear(n*512,n*256)
        
    def forward(self,structural,neighbour):
        n = len(neighbour)//3                   #neighbour has size n*3 , structural has size n*131
        final_array = torch.zeros(n*131)
        for i in range(n):
            neighbours = neighbour[3*i:3*i+3] 
            a = neighbours[0]
            b = neighbours[1]         
            c = neighbours[2]                            # a,b,c are the three neighbour indexes
            
            
            features_i = structural[i*131:i*131+131]
            features_a = structural[a*131:a*131+131]
            features_b = structural[b*131:b*131+131]
            features_c = structural[c*131:c*131+131]
            
            vec4 = torch.mean(torch.stack([features_i,features_a,features_b,features_c]),dim=0)         #implementing avg. aggregation and not concatenation

            final_array[i*131:i*131+131] = vec4
        
        final_array_1 = F.relu(self.linear3(final_array))
        final_array_2 = self.linear4(final_array_1)
        
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
        
    


# In[ ]:


class Combination2(nn.Module):
    def __init__(self):
        super(Combination2,self).__init__()
        self.linear1 = nn.Linear(n*512,n*512)
        self.linear2 = nn.Linear(n*512,n*512)
        
    def forward(self,out1,out2):
        a1 = torch.cat((out1,out2),0)    # a1 has size n*(512) now
        a1 = F.relu(self.linear1(a1))            
        a1 = self.linear2(a1)
        
        return a1
        
       # a1 = torch.cat((spatial,structural),dim=0)


# In[ ]:


class Aggregation2(nn.Module):
    def __init__(self):
        super(Aggregation2,self).__init__()
        #self.linear1 = nn.Linear(128,256)
        #self.linear2 = nn.Linear(256,128)
        self.linear3 = nn.Linear(n*256,n*512)
        self.linear4 = nn.Linear(n*512,n*512)
        
    def forward(self,out2,neighbour):
        #n = len(neighbour)//3                   #neighbour has size n*3 , out2 has size n*256
        final_array = torch.zeros(n*256)
        for i in range(n):
            neighbours = neighbour[3*i:3*i+3] 
            a = neighbours[0]
            b = neighbours[1]         
            c = neighbours[2]                            # a,b,c are the three neighbour indexes
            
            
            features_i = structural[i*256:i*256+256]
            features_a = structural[a*256:a*256+256]
            features_b = structural[b*256:b*256+256]
            features_c = structural[c*256:c*256+256]
            
            
            
            vec4 = torch.mean(torch.stack([features_i,features_a,features_b,features_c]),dim=0)         #implementing avg. aggregation and not concatenation
            final_array[i*256:i*256+256] = vec4
        
        final_array_1 = F.relu(self.linear3(final_array))
        final_array_2 = self.linear4(final_array_1)
        
        return final_array_2


model = Net(5)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)