import numpy as np
import vtk

class parse_vtk:

   def __init__(self,filename):

        self.infile = filename
   
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
            
        return u[0], v[0]

