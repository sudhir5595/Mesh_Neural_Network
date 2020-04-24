import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.module):
    def __init__(self):
        super(Net,self).__init__()
        self.spatial = Spatial_Des()
        self.structural = Structural_Des()
        self.Mesh1 = Mesh1()
        self.Mesh2 = Mesh2()
        self.Mesh3 = Mesh3()
        self.mlp1 = mlp1()
        self.mlp2 = mlp2()
        self.mlp3 = mlp3()
        
    
    def forward(self,x):
       # center = x[:,3]
       # corner = x[:,3:12]
       # normal = x[:,12:15]
       # neighbour_index = x[:,15:18]                  #ignore this,will fix this once i get the data format
        
        y = self.spatial(center)
        z = self.structural(corner,normal,neighbour_index)
        out1,out2 = self.Mesh1(y,z,neighbour_index)
        out3,out4 = self.Mesh2(out1,out2,neighbour_index)
        
        out5 = np.append(out3,out4)
        out5 = self.mlp1(out5)
        
        out6 = np.append(out5,out3)
        out6 = np.append(out6,out1)
        
        out6 = self.mlp2(out6)
        
        final_output = self.mlp3(out6)
        
        return final_output
         
        


# In[ ]:


class spatial_Des(nn.module):
    def __init__(self):
        super(spatial_Des,self).__init__()
        self.linear1 = nn.Linear(n*3,n*32)
        self.linear2 = nn.Linear(n*32,n*64)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear(x)
        return x


# In[ ]:


class structural_Des(nn.module):
    def __init__(self):
        super(structural_Des,self).__init__()
        self.kc = Kernel_Correlation()
        self.frc = Face_Rotate_Conv()
        self.linear1 = nn.Linear(131,131)
        self.linear2 = nn.Linear(131,131)
        
    def forward(self,corner,normal,neighbour):
        x = self.frc(corner)
        z = self.kc(normal,neighbour)
        z = np.append(normal,z)
        x = np.append(z,x)
        
        x = self.linear2(F.relu(self.linear1(x)))
        
        return x
        
        
        


# In[ ]:


'''
class Kernel_Correlation(nn.module):
    def __init__(self):
        super(Kernel_Correlation,self).__init__()
        self.f = 
        
'''        


# In[ ]:


class Mesh1(nn.module):
    def __init__(self):
        super(Mesh1,self).__init__()
        self.aggregation1 = Aggregation1()
        self.combination1 = Combination1()
        
    def forward(self,spatial,structural,neighbour):
        out1 = self.combination1(spatial,structural)
        out2 = self.aggregation1(structural,neighbour)
        return out1,out2
        
    


# In[ ]:


class Combination1(nn.module):
    def __init__(self):
        super(Combination1,self).__init__()
        self.linear1 = nn.Linear(n*(64+131),n*200)
        self.linear2 = nn.Linear(n*200,n*256)
        
    def forward(self,spatial,structural):
        a1 = np.append(spatial,structural)    # a1 has size n*(64+131 now)
        a1 = F.relu(linear1(a1))            #a1 has size n*200 now
        a1 = linear2(a1)
        
        return a1
        
       # a1 = torch.cat((spatial,structural),dim=0)


# In[ ]:


class Aggregation1(nn.module):
    def __init__(self):
        super(Aggregation1,self).__init__()
        self.linear1 = nn.Linear(128,256)
        self.linear2 = nn.Linear(256,128)
        self.linear3 = nn.Linear(n*128,n*512)
        self.linear4 = nn.Linear(n*512,n*256)
        
    def forward(self,structural,neighbour):
        n = len(neighbour)//3                   #neighbour has size n*3 , structural has size n*64
        final_array = np.zeros(n*128)
        for i in range(n):
            neighbours = neighbour[3*i:3*i+3] 
            a = neighbours[0]
            b = neighbours[1]         
            c = neighbours[2]                            # a,b,c are the three neighbour indexes
            
            
            features_i = structural[i*64:i*64+64]
            features_a = structural[a*64:a*64+64]
            features_b = structural[b*64:b*64+64]
            features_c = structural[c*64:c*64+64]
            
            inp_1 = np.append(features_i,features_a)
            inp_2 = np.append(features_i,features_b)
            inp_3 = np.append(features_i,features_c)
            
            inp_1 = F.relu(self.linear1(inp_1))
            inp_1 = self.linear2(inp_1)
            
            inp_2 = F.relu(self.linear1(inp_2))
            inp_2 = self.linear2(inp_2)
            
            inp_3 = F.relu(self.linear1(inp_3))
            inp_3 = self.linear2(inp_3)
            
            max_1_2 = np.maximum(inp_1,inp_2)
            max_1_2_3 = np.maximum(max_1_2, inp_3)
            
            final_array[i*128:i*128+128] = max_1_2_3
        
        final_array_1 = F.relu(self.linear3(final_array))
        final_array_2 = self.linear4(final_array_1)
        
        return final_array_2
            
        
        


# In[1]:


'''import numpy as np
a1 = np.array([1,2,3])
a2 = np.array([2,3,4])

a3 = np.append(a1,a2)
print(a3)
'''

