path = 'ModelNet10'

import os
lst = os.listdir(path)

lst.sort()

for x in lst:
	folder = path + '/' + x
	if(os.path.isdir(folder)):
		hello = os.listdir(folder+'/train/')
		for a in hello:
			file = folder + '/train/' + a
			yu = a.split('.')
			file_name = yu[0] + '.stl'
			file_name = folder + '/train/' + file_name

			os.system('meshio-convert ' + file + ' ' + file_name)
			os.system('rm ' + file)
			print("Done")