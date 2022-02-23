import os
import numpy as np
import vtk
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3

def plot(T, UX, UY, linewidth = 1.0, markersize = 4.0, rate = 100):

   plt.rcParams.update({'font.size': 14})

   plt.figure()
   plt.plot(T, UX, '-o', color = (0.76, 0.01, 0.01), linewidth = linewidth, markersize = markersize, label = 'UX Target Max X')
   plt.xlabel('Time stamp')
   plt.ylabel('DispX $(m)$')
   plt.title('Injection rate %s MSCF/day' % rate)
   plt.legend(frameon=False)
   plt.tight_layout()
   plt.savefig('plots/ux_%s.png' % rate)

   plt.figure()
   plt.plot(T, UY, '-o', color = (0.76, 0.01, 0.01), linewidth = linewidth, markersize = markersize, label = 'UY Target Min X')
   plt.xlabel('Time stamp')
   plt.ylabel('DispY $(m)$')
   plt.title('Injection rate %s MSCF/day' % rate)
   plt.legend(frameon=False)
   plt.tight_layout()
   plt.savefig('plots/uy_%s.png' % rate)

   plt.show()
   plt.close('all')

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

   def get_surface_information(self, vector_name):
        """
        Method to get surface information.
        
        :param vector_name: the vector you want to process.
        	
        :return: displacement components
        :rtype: array
        """
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(self.infile)
        reader.Update()
        data = reader.GetOutput()
        
        npoints = data.GetNumberOfPoints()
        point = data.GetPoint(0)
        d = data.GetPointData()
        
        array = d.GetArray(vector_name)
        
        u, v, w, x, y, z = np.zeros(npoints),np.zeros(npoints),np.zeros(npoints),np.zeros(npoints),np.zeros(npoints),np.zeros(npoints)
        
        for n in range(npoints):
            x[n], y[n], z[n] = data.GetPoint(n)
            u[n], v[n], w[n] = array.GetTuple(n)
        
        # Surface information at max y
        u = u[np.where((x==max(x)) & (y==max(y)))[0]]
        v = v[np.where((x==min(x)) & (y==max(y)))[0]]
        
        del x, y, z
            
        return u, v

if __name__ == '__main__':

   UX, UY, T = [], [], []
   rate = 400
   directory = './vtk_plots' + '/%s' % rate
   count_ = 0
   for file_name in sorted(os.listdir(directory)):
       filename = os.path.join(directory, file_name)
       if os.path.isfile(filename):
           count_ += 1
           parser = parse_vtk(filename)
           u, v = parser.get_surface_information("displacement")
           UX.append(u[0]); UY.append(v[0]); T.append(count_)
   plot(T,UX,UY,rate=rate)
