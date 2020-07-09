import numpy as np
import open3d as o3d
from downsizing import downsize

def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def input_stl(path):
    
    try:
        triangular_mesh = o3d.io.read_triangle_mesh(path)
        parameters = {}
        parameters["normals"] = np.array(triangular_mesh.triangle_normals)
        triangular_mesh.remove_duplicated_vertices()
        parameters["vertices"] = np.array(triangular_mesh.vertices)
        parameters["num_tri"] = int(parameters["normals"].shape[0])
        parameters["triangles"] = np.array(triangular_mesh.triangles)
        parameters["centroids"] = np.zeros((parameters["num_tri"],3))                 
        parameters["corners"] = np.zeros((parameters["num_tri"], 9))

        parameters = downsize(parameters, 1024)
        faces_contain_this_vertex = []    
        for i in range(parameters["vertices"].shape[0]):
            faces_contain_this_vertex.append(set([]))

        for i in range(parameters["num_tri"]):
            [v1, v2, v3] = parameters["triangles"][i]
            x1, y1, z1 = parameters["vertices"][v1]
            x2, y2, z2 = parameters["vertices"][v2]
            x3, y3, z3 = parameters["vertices"][v3]
            
            Gx = (x1 + x2 + x3) / 3
            Gy = (y1 + y2 + y3) / 3
            Gz = (z1 + z2 + z3) / 3

            parameters["centroids"][i] = [Gx, Gy, Gz]

            c1 = parameters["vertices"][v1] - parameters["centroids"][i]
            c2 = parameters["vertices"][v2] - parameters["centroids"][i]
            c3 = parameters["vertices"][v3] - parameters["centroids"][i]

            parameters['corners'][i] = np.concatenate((c1, c2, c3))

            faces_contain_this_vertex[v1].add(i)
            faces_contain_this_vertex[v2].add(i)
            faces_contain_this_vertex[v3].add(i)

            
        parameters["neigh_index"] = []
        for i in range(parameters['num_tri']):
            [v1, v2, v3] = parameters["triangles"][i]
            n1 = find_neighbor(parameters["triangles"], faces_contain_this_vertex, v1, v2, i)
            n2 = find_neighbor(parameters["triangles"], faces_contain_this_vertex, v2, v3, i)
            n3 = find_neighbor(parameters["triangles"], faces_contain_this_vertex, v3, v1, i)
            parameters["neigh_index"].append([n1, n2, n3])


        parameters["neigh_index"] = np.array(parameters["neigh_index"])
        #parameters["neigh_index"].sort(axis = 1)
        
        return parameters
    except:
        return False    
#o = input_stl('/media/prathmesh/PPB External HDD/IITB/SoC/SoC-2020/ModelNet10_stl (copy)/ModelNet10/chair/train/chair_0889.stl')
#print(o["triangles"])
#print(o)
