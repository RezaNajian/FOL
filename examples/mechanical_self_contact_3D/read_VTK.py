import meshio
mesh = meshio.read('Structure_0_1.vtk')

print((mesh.point_data["DISPLACEMENT"].shape))