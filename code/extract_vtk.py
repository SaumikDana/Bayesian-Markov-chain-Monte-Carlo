
import numpy as np
import vtk
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3

class parse_vtk:

   def __init__(self,filename):

       self.infile = filename
   
   def parse(self, write = False):
	"""
	Method to parse the file `filename`. It returns a matrix with all the coordinates.

	:param string filename: name of the input file.

	:return: mesh_points: it is a `n_points`-by-3 matrix containing the coordinates of
		the points of the mesh
	:rtype: numpy.ndarray

	.. todo::

		- specify when it works
	"""
	reader = vtk.vtkDataSetReader()
	reader.SetFileName(self.infile)
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	data = reader.GetOutput()

	n_points = data.GetNumberOfPoints()
	mesh_points = np.zeros([n_points, 3])

	for i in range(n_points):
	    mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)

        if write: # write if needed
           self.write(mesh_points)

	return mesh_points

   def write(self, mesh_points):
	"""
	Writes a vtk file, called filename, copying all the structures from self.filename but
	the coordinates. mesh_points is a matrix that contains the new coordinates to
	write in the vtk file.

	:param numpy.ndarray mesh_points: it is a `n_points`-by-3 matrix containing
		the coordinates of the points of the mesh
	:param string filename: name of the output file.

	.. todo:: DOCS
	"""
	self.outfile = self.infile + "_out"

	reader = vtk.vtkDataSetReader()
	reader.SetFileName(self.infile)
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	data = reader.GetOutput()

	points = vtk.vtkPoints()

	for i in range(data.GetNumberOfPoints()):
		points.InsertNextPoint(mesh_points[i, :])

	data.SetPoints(points)

	writer = vtk.vtkDataSetWriter()
	writer.SetFileName(self.outfile)

	if vtk.VTK_MAJOR_VERSION <= 5:
		writer.SetInput(data)
	else:
		writer.SetInputData(data)

	writer.Write()

   def plot(self, plot_file=None, show_fig=True):
	"""
	Method to plot a vtk file. If `plot_file` is not given it plots `self.infile`.

	:param string plot_file: the vtk filename you want to plot.
	:param bool save_fig: a flag to save the figure in png or not. If True the
		plot is not shown.
		
	:return: figure: matlplotlib structure for the figure of the chosen geometry
	:rtype: matplotlib.pyplot.figure
	"""
	plot_file = self.infile

	# Read the source file.		
	reader = vtk.vtkDataSetReader()
	reader.SetFileName(plot_file)
	reader.Update()

	data = reader.GetOutput()
	points = data.GetPoints()
	ncells = data.GetNumberOfCells()

	# for each cell it contains the indeces of the points that define the cell
	figure = plt.figure()
	axes = a3.Axes3D(figure)
	vtx = np.zeros((ncells, 3, 3))
	for i in range(0, ncells):
		for j in range(0, 3):
			cell = data.GetCell(i).GetPointId(j)
			vtx[i][j][0], vtx[i][j][1], vtx[i][j][2] = points.GetPoint(int(cell))
		tri = a3.art3d.Poly3DCollection([vtx[i]])
		tri.set_color('b')
		tri.set_edgecolor('k')
		axes.add_collection3d(tri)
	
	## Get the limits of the axis and center the geometry
	max_dim = np.array([np.max(vtx[:,:,0]), \
					np.max(vtx[:,:,1]), \
					np.max(vtx[:,:,2])])
	min_dim = np.array([np.min(vtx[:,:,0]), \
					np.min(vtx[:,:,1]), \
					np.min(vtx[:,:,2])])
	
	max_lenght = np.max(max_dim - min_dim)
	axes.set_xlim(-.6*max_lenght + (max_dim[0]+min_dim[0])/2, .6*max_lenght + (max_dim[0]+min_dim[0])/2)
	axes.set_ylim(-.6*max_lenght + (max_dim[1]+min_dim[1])/2, .6*max_lenght + (max_dim[1]+min_dim[1])/2)
	axes.set_zlim(-.6*max_lenght + (max_dim[2]+min_dim[2])/2, .6*max_lenght + (max_dim[2]+min_dim[2])/2)

	# Show the plot to the screen
	if show_fig: #default
           plt.show()
	else:
	   figure.savefig(plot_file.split('.')[0] + '.png')
		
	return figure


if __name__ == '__main__':

   filename = "./vtk_plots/test_t0010.vtk"

   parser = parse_vtk(filename)
   coords = parser.parse()
   parser.plot()
 
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
