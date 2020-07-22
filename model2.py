class Net(nn.Module):
    def __init__(self,k):                                  # k is the number of output classes needed
        super(Net,self).__init__()
        self.spatial = spatial_Des()
        self.structural = structural_Des()
        self.Mesh1 = Mesh1()
        self.Mesh2 = Mesh2()
        #self.conv1 = nn.Conv1d(1024,1024,1)
        self.linear = nn.Linear(512,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.mlp2 = mlp2()
        self.mlp3 = mlp3(k)
        self.k = k
        
    
    def forward(self,x):
        n = x.size()[0] 
        center = x[:,0:3]
        corner = x[:,3:12]
        normal = x[:,12:15]
        neighbour_index = x[:,15:18]                  

        y = self.spatial(center)
        z = self.structural(corner,normal,neighbour_index)
        out1,out2 = self.Mesh1(y,z,neighbour_index)
        out3,out4 = self.Mesh2(out1,out2,neighbour_index)
        
        out5 = torch.cat((out3,out4),dim=1)

        out5 = self.linear(out5)
        out5 = self.bn1(out5)
        
        out6 = torch.cat((out5,out3,out1),dim=1)        
        out6 = self.mlp2(out6)
        
        final_output = self.mlp3(out6)
        
        return final_output
         
        


class mlp2(nn.Module):
    def __init__(self):
        super(mlp2,self).__init__()
        #self.conv1 = nn.Conv1d(1792,1024,1)
        self.linear = nn.Linear(640,256)              #Linear layer is theoretically same as conv1d layer with kernel_size = 1. By experimenting on performance and time taken by the two layers we can decide which to keep
        self.bn1 = nn.BatchNorm1d(256)
        
    def forward(self,x):
        n = x.size()[0]
        x = self.bn1(self.linear(x))
        l = torch.max(x,0)
        return l.values

class mlp3(nn.Module):
    def __init__(self,k):
        super(mlp3,self).__init__()
        self.linear1 = nn.Linear(256,128)
        self.linear2 = nn.Linear(128,32)
        self.linear3 = nn.Linear(32,k)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self,x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x.unsqueeze(dim=0)


class spatial_Des(nn.Module):                                                            #as described in the paper, this is just a multilayer perceptron
    def __init__(self):
        super(spatial_Des,self).__init__()
        self.linear1 = nn.Linear(32,32)
        self.bn1 = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(3,32)
        self.bn2 = nn.BatchNorm1d(32)
        #self.bn1 = nn.BatchNorm1d(64)
        
        
    def forward(self,x):
        n = x.size()[0]
        x = self.bn2(self.linear1(F.relu(self.bn1(self.linear2(x)))))
        return x



class structural_Des(nn.Module):
    def __init__(self):
        super(structural_Des,self).__init__()
        self.kc = Kernel_Correlation(4)                  
        self.frc = Face_Rotate_Conv()
        self.conv1 = nn.Conv1d(67,67,1)
        self.bn2 = nn.BatchNorm1d(67)
        self.conv2 = nn.Conv1d(67,67,1)
        #self.linear1 = nn.Linear(131,131)
        #self.linear2 = nn.Linear(131,131)
        self.bn1 = nn.BatchNorm1d(67)
        
    def forward(self,corner,normal,neighbour):
        n = corner.size()[0]
        x = self.frc(corner)
        z = self.kc(normal,neighbour)

        x = torch.cat((normal,x,z),dim=1)        
        x = x.view(n,67,1)
        x = self.conv2(F.relu(self.bn1(self.conv1(x))))
        #x = self.linear2(F.relu(self.linear1(x)))
        x = self.bn2(x.view(n,67))

        return x
        
        
        



class Face_Rotate_Conv(nn.Module):
    def __init__(self):
        super(Face_Rotate_Conv,self).__init__()
        self.linear3 = nn.Linear(16,32)
        self.linear4 = nn.Linear(32,32)
        self.conv = nn.Conv1d(1,16,6,3)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)

        
    def forward(self,corner):                # corner is (n,9) . we take these face wise and apply 1-d convolution of 6 features out of 9 in 3 turns
        n = corner.size()[0]
        final_array = torch.cat((corner[:,0:3],corner[:,3:6],corner[:,6:9],corner[:,0:3]),dim=1)
        final_array = final_array.view(n,1,12)
        final_array = self.conv(final_array)

        new_arr = torch.mean(final_array,dim=2)
        vec4 = self.linear4(self.bn2(F.relu(self.linear3(self.bn1(new_arr)))))


        return vec4


class Kernel_Correlation(nn.Module):
    def __init__(self,k):
        super(Kernel_Correlation,self).__init__()
        self.theta_par = nn.Parameter(torch.rand(1,32, 4) * np.pi)
        self.phi_par = nn.Parameter(torch.rand(1,32, 4) *2* np.pi)
        self.k = k
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

    def forward(self,normal,neighbour):
        sigma = 0.2
        n = normal.size()[0]
        neighbour = neighbour.long()
        normal_neigh1 = normal[neighbour[:,0]]
        normal_neigh2 = normal[neighbour[:,1]]
        normal_neigh3 = normal[neighbour[:,2]]

        face_normal = normal.unsqueeze(2).expand(-1,-1,32).unsqueeze(3)
        normal_neigh1 = normal_neigh1.unsqueeze(2).expand(-1,-1,32).unsqueeze(3)
        normal_neigh2 = normal_neigh2.unsqueeze(2).expand(-1,-1,32).unsqueeze(3)
        normal_neigh3 = normal_neigh3.unsqueeze(2).expand(-1,-1,32).unsqueeze(3)

        fea = torch.cat((face_normal,normal_neigh1,normal_neigh2,normal_neigh3),dim=3)
        fea = fea.unsqueeze(4).expand(-1,-1,-1,-1,4)

        kernel = torch.cat((torch.sin(self.theta_par)*torch.sin(self.phi_par) , torch.sin(self.theta_par)*torch.cos(self.phi_par) , torch.cos(self.theta_par)) ,dim=0 )
        kernel = kernel.unsqueeze(0).expand(n,-1,-1,-1)
        kernel = kernel.unsqueeze(3).expand(-1,-1,-1,4,-1)

        correl = torch.sum((fea - kernel)**2,1)

        fea = torch.sum(torch.sum(np.e**(correl / (-2*sigma**2)), 3), 2) / 16
        return self.relu(self.bn(fea))




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
        self.conv1 = nn.Conv1d(32+67,128,1)
        self.bn1 = nn.BatchNorm1d(128)
        
        
    def forward(self,spatial,structural):
        n = structural.size()[0]
        a1 = torch.cat((spatial,structural),dim=1)    # a1 has size (n,64+131) now
        a1 = a1.view(n,32+67,1)
        a1 = self.conv1(a1)
        a1 = a1.view(n,128)            #a1 has size (n,256) now
        
        
        return self.bn1(a1)
        


class Aggregation1(nn.Module):
    def __init__(self):
        super(Aggregation1,self).__init__()
        self.conv1 = nn.Conv1d(67,128,1)
        self.bn1 = nn.BatchNorm1d(128)

        
    def forward(self,structural,neighbour):
        n = structural.size()[0]
        neighbour = neighbour.long()
        first_neigh = structural[neighbour[:,0]]
        second_neigh = structural[neighbour[:,1]]
        third_neigh = structural[neighbour[:,2]]

        final_array = torch.mean(torch.stack([structural,first_neigh,second_neigh,third_neigh]),dim=0)

        final_array = final_array.view(n,67,1)
        final_array_1 = self.conv1(final_array)

        final_array_2 = final_array_1.view(n,128)
        
        
        return self.bn1(final_array_2)
            
        


        

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
        self.conv1 = nn.Conv1d(256,256,1)
        self.bn1 = nn.BatchNorm1d(256)
        
    def forward(self,out1,out2):
        n = out1.size()[0]
        #print(out1.shape)
        #print(out2.shape)
        a1 = torch.cat((out1,out2),dim=1)       # a1 has size n,512 now
        a1 = a1.view(n,256,1)    
        a1 = self.conv1(a1)
        a1 = a1.view(n,256)            
        
        return self.bn1(a1)
        


class Aggregation2(nn.Module):
    def __init__(self):
        super(Aggregation2,self).__init__()
        self.conv1 = nn.Conv1d(128,256,1)
        self.bn1 = nn.BatchNorm1d(256)
        
    def forward(self,out2,neighbour):
        n = neighbour.size()[0]

        neighbour = neighbour.long()
        first_neigh = out2[neighbour[:,0]]
        second_neigh = out2[neighbour[:,1]]
        third_neigh = out2[neighbour[:,2]]

        final_array = torch.mean(torch.stack([out2,first_neigh,second_neigh,third_neigh]),dim=0)

        final_array = final_array.view(n,128,1)
        final_array_1 = self.conv1(final_array)

        final_array_2 = final_array_1.view(n,256)
        return self.bn1(final_array_2)
