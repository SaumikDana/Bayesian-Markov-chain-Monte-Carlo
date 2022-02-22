
import numpy as np
import vtk

class parse_vtk:
   
   def parse(self, filename):
	"""
	Method to parse the file `filename`. It returns a matrix with all the coordinates.

	:param string filename: name of the input file.

	:return: mesh_points: it is a `n_points`-by-3 matrix containing the coordinates of
		the points of the mesh
	:rtype: numpy.ndarray

	.. todo::

		- specify when it works
	"""
	self.infile = filename

	reader = vtk.vtkDataSetReader()
	reader.SetFileName(self.infile)
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	data = reader.GetOutput()

	n_points = data.GetNumberOfPoints()
	mesh_points = np.zeros([n_points, 3])

	for i in range(n_points):
		mesh_points[i][0], mesh_points[i][1], mesh_points[i][
			2
		] = data.GetPoint(i)

	return mesh_points

if __name__ == '__main__':

   filename = "./vtk_plots/test_t0010.vtk"
   parser = parse_vtk()
   coords = parser.parse(filename)

#import numpy as np
#import vtk
#from vtk.util.numpy_support import vtk_to_numpy
#
#filename = "./vtk_plots/test_t0010.vtk"
#reader = vtk.vtkDataSetReader()
#reader.SetFileName(filename)
#reader.ReadAllVectorsOn()
#reader.ReadAllScalarsOn()
#reader.Update()
#data = reader.GetOutput()
#
#print('--- Mesh information ---')
#
#print('Number of cells is %s, Number of pieces is %s, Number of points is %s' % (data.GetNumberOfCells(), data.GetNumberOfPieces(), data.GetNumberOfPoints()))  
#
#print('--- Mesh information ---')
#
#d=data.GetPointData()
#
#array=d.GetArray('displacement')
#
#print(type(array))
#
#print('Number of scalars in file is %s' % reader.GetNumberOfScalarsInFile())
#
#print('Number of vectors in file is %s' % reader.GetNumberOfVectorsInFile())
#
#print(reader.GetVectorsNameInFile(0))
