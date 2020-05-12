import open3d as o3d

for i in range(1,345):
	 mesh = o3d.io.read_triangle_mesh("/home/prathmesh/Desktop/SoC-2020/ModelNet10/ModelNet10/toilet/train/toilet_"+str(i).zfill(4)+".off")
	 mesh.compute_vertex_normals()
	 o3d.io.write_triangle_mesh("/home/prathmesh/Desktop/SoC-2020/new/ModelNet10/ModelNet10/toilet/train/toilet_"+str(i).zfill(4)+".stl",mesh)

	