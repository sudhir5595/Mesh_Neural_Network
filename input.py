import numpy as np
import open3d as o3d

def input_stl(path):
    triangular_mesh = o3d.io.read_triangle_mesh(path)
    
    parameters = {}
    parameters["normals"] = np.array(triangular_mesh.triangle_normals)
    parameters["vertices"] = np.array(triangular_mesh.vertices)
    num_tri = int(parameters["normals"].shape[0])
    
    parameters["centroids"] = np.zeros((num_tri,3))                 
    parameters["corners"] = np.zeros((num_tri, 9))
    parameters["neigh_index"] = np.zeros((num_tri,3))
    parameters["triangles"] = np.zeros((num_tri, 3,3))
    parameters["tri_to_check"] = np.zeros((num_tri, 3,3))
    
    
    for i in range(num_tri):
        
        parameters["triangles"][i] = [ parameters["vertices"][3*i],
                                            parameters["vertices"][3*i+1],
                                            parameters["vertices"][3*i+2] ]
        parameters["tri_to_check"][i] = [ parameters["vertices"][3*i],
                                            parameters["vertices"][3*i+1],
                                            parameters["vertices"][3*i+2] ]
        #sorting vertices(rows) according to x-coordinate, then y-coordinate, then z-coordinate
        parameters["triangles"][i].view('i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)
        parameters["tri_to_check"][i].view('i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)
        #centroid formula        
        parameters["centroids"][i] = ((1 / 3) * np.sum(parameters["triangles"][i], axis = 0))
        # Corner: vectors from the center point to three vertices, v1, v2, v3 are those 3 vectors
        parameters["corners"][i] = np.reshape( parameters["triangles"][i] - parameters["centroids"][i], 9 )    
        #initializing neighbouring indices with index of triangle itself
        parameters["neigh_index"][i] = [i,i,i]  
        
    
    for i in range(num_tri):
        #print()
        #print(i)
        #print()
        curr_update_index = 0
        for j in range(num_tri):
            if i == j:
                continue
            
            if ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()):
                #print(str(j)+'a')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'b')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'c')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            
            parameters["tri_to_check"][j][[0,1,2]] = parameters["tri_to_check"][j][[2,0,1]]
            
            if ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()):
                #print(str(j)+'d')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'e')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'f')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
                
            parameters["tri_to_check"][j][[0,1,2]] = parameters["tri_to_check"][j][[2,0,1]]
            
            if ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()):
                #print(str(j)+'g')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'h')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'i')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
                
            parameters["tri_to_check"][j][[0,1,2]] = parameters["tri_to_check"][j][[1,0,2]]
            
            if ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()):
                #print(str(j)+'j')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'k')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'l')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
                
            parameters["tri_to_check"][j][[0,1,2]] = parameters["tri_to_check"][j][[2,0,1]]
            
            if ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()):
                #print(str(j)+'m')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'n')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'p')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
                
            parameters["tri_to_check"][j][[0,1,2]] = parameters["tri_to_check"][j][[2,0,1]]
            
            if ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()):
                #print(str(j)+'q')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][0] == parameters["tri_to_check"][j][0]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'r')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
            elif ((parameters["triangles"][i][1] == parameters["tri_to_check"][j][1]).all()
                and (parameters["triangles"][i][2] == parameters["tri_to_check"][j][2]).all()):
                #print(str(j)+'s')
                parameters["neigh_index"][i][curr_update_index] = j
                curr_update_index = curr_update_index + 1
             
    return parameters


#o = input_stl("/home/prathmesh/Desktop/SoC-2020p/copy_cube.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020p/ArtecSpiderNerfGunmm.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020p/test0.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020p/test1.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020p/test2.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020p/test3.stl")
#print(o["neigh_index"])



