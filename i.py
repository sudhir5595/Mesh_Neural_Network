import numpy as np
import open3d as o3d

def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def input_stl(path):
    triangular_mesh = o3d.io.read_triangle_mesh(path)

    parameters = {}
    parameters["normals"] = np.array(triangular_mesh.triangle_normals)
    parameters["vertices"] = np.array(triangular_mesh.vertices)
    parameters["num_tri"] = int(parameters["normals"].shape[0])

    parameters["triangles"] = np.zeros((parameters["num_tri"], 3,3))
    parameters["faces"] = np.zeros((parameters["num_tri"], 3))
    parameters["centroids"] = np.zeros((parameters["num_tri"],3))                 
    parameters["corners"] = np.zeros((parameters["num_tri"], 9))
    faces_contain_this_vertex = []
        
    for i in range(parameters["num_tri"]):

        parameters["triangles"][i] = [ parameters["vertices"][3*i],
                                      parameters["vertices"][3*i+1],
                                      parameters["vertices"][3*i+2] ]
        parameters["faces"][i] = [3*i, 3*i+1, 3*i+2] 
        #sorting vertices(rows) according to x-coordinate, then y-coordinate, then z-coordinate
        parameters["triangles"][i].view('i8,i8,i8').sort(order=['f0', 'f1', 'f2'], axis=0)
        #centroid formula        
        parameters["centroids"][i] = ((1 / 3) * np.sum(parameters["triangles"][i], axis = 0))
        # Corner: vectors from the center point to three vertices, v1, v2, v3 are those 3 vectors
        parameters["corners"][i] = np.reshape( parameters["triangles"][i] - parameters["centroids"][i], 9 )    
        
        for j in range(3):
            faces_contain_this_vertex.append(set([]))

        faces_contain_this_vertex[3*i].add(i)
        faces_contain_this_vertex[3*i+1].add(i)
        faces_contain_this_vertex[3*i+2].add(i)
        
        

    parameters["neigh_index"] = []
    for i in range(parameters['num_tri']):
        n1 = find_neighbor(parameters["faces"], faces_contain_this_vertex, 3*i, 3*i+1, i)
        n2 = find_neighbor(parameters['faces'], faces_contain_this_vertex, 3*i+1, 3*i+2, i)
        n3 = find_neighbor(parameters['faces'], faces_contain_this_vertex, 3*i+2, 3*i, i)
        parameters["neigh_index"].append([n1, n2, n3])


    parameters["neigh_index"] = np.array(parameters["neigh_index"])

    return parameters

# Root for saving the npy file
o = input_stl("/home/prathmesh/Desktop/SoC-2020/test1.stl")
print(o["neigh_index"])
#np.savez(root + '.npz',faces=faces, neighbors=neighbors)
