import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support

filename = "./vtk_plots/test_t0010.vtk"
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(filename)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()

cell2point = vtk.vtkCellDataToPointData()
cell2point.SetInputData(reader.GetOutput())
cell2point.Update()

#u = numpy_support.vtk_to_numpy(reader.GetOutput().GetCellData().GetArray('displacement'))

coord = numpy_support.vtk_to_numpy(cell2point.GetOutput().GetPoints().GetData())
x = coord[:,0]
y = coord[:,1]
z = coord[:,2]

#temperature = numpy_support.vtk_to_numpy(cell2point.GetOutput().GetPointData().GetAbstractArray(1))
#plt.tricontourf(x,y,temperature,15,cmap="jet")
#plt.gca().set_aspect("equal")
#plt.colorbar()
#plt.show()
