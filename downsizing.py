import numpy as np 
import open3d as o3d 
import random

def downsize(para, size):
    if para['num_tri'] < size:
	    triangles = []
	    centroid = []
	    corners = []
	    normals = []


	    for i in range(size - para['num_tri']):
		    index = random.randint(0, para['num_tri']-1)
		    triangles.append(para['triangles'][index])
		    centroid.append(para['centroids'][index])
		    corners.append(para['corners'][index])
		    normals.append(para['normals'][index])

		
	    para['num_tri'] = size
	    para['triangles'] = np.concatenate((para['triangles'], np.array(triangles)))
	    para['centroids'] = np.concatenate((para['centroids'], np.array(centroid)))
	    para['corners'] = np.concatenate((para["corners"], np.array(corners)))
	    para['normals'] = np.concatenate((para["normals"], np.array(normals)))


	    return para

    elif para['num_tri'] ==  size:
	    return para

    else:
	    triangles = []
	    centroid = []
	    corners = []
	    normals = []


	    for i in range(size):
		    index = random.randint(0, para['num_tri']-1)
		    triangles.append(para['triangles'][index])
		    centroid.append(para['centroids'][index])
		    corners.append(para['corners'][index])
		    normals.append(para['normals'][index])
		

	    para['num_tri'] = size
	    para['triangles'] = np.array(triangles)
	    para['centroids'] = np.array(centroid)
	    para['corners'] = np.array(corners)
	    para['normals'] = np.array(normals)
	

	    return para 
