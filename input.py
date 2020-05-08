import numpy as np
import open3d as o3d

def switch_rows(parameter, perm):
    parameter[[0,1,2]] = parameter[perm]

def common_side(tri1, tri2, row1, row2):
    bool_val = ((tri1[row1] == tri2[row1]).all()
                and (tri1[row2] == tri2[row2]).all())
    return bool_val    

def check_and_assign(tri_arr1, tri_arr_temp, neigh_arr, i , j, curr_update_index):
    
    if common_side(tri_arr1[i], tri_arr_temp[j], 0, 1):
        print(str(j))
        neigh_arr[i][curr_update_index] = j
        curr_update_index = curr_update_index + 1
    elif common_side(tri_arr1[i], tri_arr_temp[j], 0, 2):
        print(str(j))
        neigh_arr[i][curr_update_index] = j
        curr_update_index = curr_update_index + 1
    elif common_side(tri_arr1[i], tri_arr_temp[j], 1, 2):
        print(str(j))
        neigh_arr[i][curr_update_index] = j
        curr_update_index = curr_update_index + 1
    
    return curr_update_index
    
    
def assign_neigh_index(tri_arr1, tri_arr_temp, num_tri, neigh_arr):
    
    for i in range(num_tri):
        print()
        print(i)
        print()
        curr_update_index = 0
        for j in range(num_tri):
            if i == j:
                continue
            
            curr_update_index = check_and_assign(tri_arr1, tri_arr_temp, neigh_arr, i , j, curr_update_index)
            
            switch_rows(tri_arr_temp[j], perm = [2,0,1])
            
            curr_update_index = check_and_assign(tri_arr1, tri_arr_temp, neigh_arr, i , j, curr_update_index)
                
            switch_rows(tri_arr_temp[j], perm = [2,0,1])
            
            curr_update_index = check_and_assign(tri_arr1, tri_arr_temp, neigh_arr, i , j, curr_update_index)
                
            switch_rows(tri_arr_temp[j], perm = [1,0,2])
            
            curr_update_index = check_and_assign(tri_arr1, tri_arr_temp, neigh_arr, i , j, curr_update_index)
                
            switch_rows(tri_arr_temp[j], perm = [2,0,1])
            
            curr_update_index = check_and_assign(tri_arr1, tri_arr_temp, neigh_arr, i , j, curr_update_index)
                
            switch_rows(tri_arr_temp[j], perm = [2,0,1])
            
            curr_update_index = check_and_assign(tri_arr1, tri_arr_temp, neigh_arr, i , j, curr_update_index)
            
            if curr_update_index == 3:
                break

    
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
        #triangles and tri_to_check are same
        parameters["triangles"][i], parameters["tri_to_check"][i] = [ parameters["vertices"][3*i],
                                                                     parameters["vertices"][3*i+1],
                                                                     parameters["vertices"][3*i+2] ], [ parameters["vertices"][3*i],
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
        
    assign_neigh_index(parameters["triangles"], parameters["tri_to_check"], num_tri, parameters["neigh_index"])
        
    return parameters


#o = input_stl("/home/prathmesh/Desktop/SoC-2020/copy_cube.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020/ArtecSpiderNerfGunmm.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020/test0.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020/test1.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020/test2.stl")
#o = input_stl("/home/prathmesh/Desktop/SoC-2020/test3.stl")
#print(o["neigh_index"])



