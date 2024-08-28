import meshio

# read FOL solutions
fol_sols = meshio.read('FOL_KR_solutions/self_contact_FOL.vtk')

# read kratos solutions
for ind in [0,5,10,15]:
    kr_sol = meshio.read(f'FOL_KR_solutions/self_contact_KR_ind_{ind}.vtk').point_data["DISPLACEMENT"]
    fol_sols.point_data[f"test_U_KR_bc_{ind}"] = kr_sol
    fol_sol = fol_sols.point_data[f"test_U_FOL_bc_{ind}"]
    fol_sols.point_data[f"test_U_ERR_bc_{ind}"] = abs(kr_sol-fol_sol)
    
fol_sols.write('FOL_KR_solutions/KR_FOL_comp.vtk',file_format="vtk")