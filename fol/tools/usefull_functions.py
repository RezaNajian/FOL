
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math
import gmsh
import meshio
import os
import shutil

def plot_mesh_vec_data(L, vectors_list, subplot_titles=None, fig_title=None, cmap='viridis',
                       block_bool=False, colour_bar=True, colour_bar_name=None,
                       X_axis_name=None, Y_axis_name=None, show=False, file_name=None):
    num_vectors = len(vectors_list)
    if num_vectors < 1 or num_vectors > 4:
        raise ValueError("vectors_list must contain between 1 and 4 elements.")

    if subplot_titles is not None and len(subplot_titles) != num_vectors:
        raise ValueError("subplot_titles must have the same number of elements as vectors_list if provided.")

    # Determine the grid size for the subplots
    grid_size = math.ceil(math.sqrt(num_vectors))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(5*grid_size, 5*grid_size), squeeze=False)
    
    # Flatten the axs array and hide unused subplots if any
    axs = axs.flatten()
    for ax in axs[num_vectors:]:
        ax.axis('off')

    for i, squared_mesh_vec_data in enumerate(vectors_list):
        N = int((squared_mesh_vec_data.reshape(-1, 1).shape[0])**0.5)
        im = axs[i].imshow(squared_mesh_vec_data.reshape(N, N), cmap=cmap, extent=[0, L, 0, L])

        if subplot_titles is not None:
            axs[i].set_title(subplot_titles[i])
        else:
            axs[i].set_title(f'Plot {i+1}')

        if colour_bar:
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

        if X_axis_name is not None:
            axs[i].set_xlabel(X_axis_name)

        if Y_axis_name is not None:
            axs[i].set_ylabel(Y_axis_name)

    if fig_title is not None:
        plt.suptitle(fig_title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show(block=block_bool)

    if file_name is not None:
        plt.savefig(file_name)

def plot_data_input(input_morph, num_columns, filename):

    N = int(input_morph.shape[1]**0.5)
    L = 1

    # Calculate the number of rows based on the number of columns and the length of input_morph
    num_rows = int(np.ceil(len(input_morph) / num_columns))

    # Create a new figure with variable subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns, num_rows))

    # Flatten the axes array to handle variable numbers of subplots
    axes = axes.flatten()

    # Loop through the input_morph rows and plot each row in a separate subplot
    for i in range(len(input_morph)):
        ax = axes[i] if i < len(axes) else None  # Handle cases with fewer subplots than data
        if ax:
            Z = input_morph[i].reshape(N, N)  # Reshape the vectorized Z to a 2D array
            min_val = np.min(Z)
            max_val = np.max(Z)
            im = ax.imshow(Z, cmap='viridis', extent=[0, L, 0, L], vmin=min_val, vmax=max_val)
            ax.set_title(f'Row {i+1}')
            ax.set_xticks([])
            ax.set_yticks([])

    # Add a color bar at the top of the figure
    cbar_ax = fig.add_axes([0.3, 1.02, 0.4, 0.005])  # Define position and size of the color bar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    # Remove any unused subplots
    for i in range(len(input_morph), len(axes)):
        fig.delaxes(axes[i])

    # Adjust subplot spacing
    plt.tight_layout()

    # Save the plot as a PDF and PNG file with user-defined filename
    plt.savefig(f'{filename}.png')

def create_2D_square_model_info_thermal(L,N,T_left,T_right):
    # FE init starts here
    Ne = N - 1  # Number of elements in each direction
    nx = Ne + 1  # Number of nodes in the x-direction
    ny = Ne + 1  # Number of nodes in the y-direction
    ne = Ne * Ne    # Total number of elements
    # Generate mesh coordinates
    x = jnp.linspace(0, L, nx)
    y = jnp.linspace(0, L, ny)
    X, Y = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = jnp.zeros((Y.shape[-1]))
    # Gauss quadrature points and weights (for a 2x2 integration)
    # Create a matrix to store element nodal information
    elements_nodes = jnp.zeros((ne, 4), dtype=int)
    # Fill in the elements_nodes with element and node numbers
    for i in range(Ne):
        for j in range(Ne):
            e = i * Ne + j  # Element index
            # Define the nodes of the current element
            nodes = jnp.array([i * (Ne + 1) + j, i * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j])
            # Store element and node numbers in the matrix
            elements_nodes = elements_nodes.at[e].set(nodes) # Node numbers

    element_ids = jnp.arange(0,elements_nodes.shape[0])

    # Identify boundary nodes on the left and right edges
    left_boundary_nodes = jnp.arange(0, ny * nx, nx)  # Nodes on the left boundary
    left_boundary_nodes_values = T_left * jnp.ones(left_boundary_nodes.shape)
    right_boundary_nodes = jnp.arange(nx - 1, ny * nx, nx)  # Nodes on the right boundary
    right_boundary_nodes_values = T_right * jnp.ones(right_boundary_nodes.shape)
    boundary_nodes = jnp.concatenate([left_boundary_nodes, right_boundary_nodes])
    boundary_values = jnp.concatenate([left_boundary_nodes_values, right_boundary_nodes_values])
    non_boundary_nodes = []
    for i in range(N*N):
        if not jnp.any(boundary_nodes == i):
            non_boundary_nodes.append(i)
    non_boundary_nodes = jnp.array(non_boundary_nodes)

    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}
    elements_dict = {"elements_ids":element_ids,"elements_nodes":elements_nodes}
    dofs_dict = {"T":{"non_dirichlet_nodes_ids":non_boundary_nodes,"dirichlet_nodes_ids":boundary_nodes,"dirichlet_nodes_dof_value":boundary_values}}
    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict}

def box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, case_dir):

    cell_type = 'hexahedron'
    degree= 1
    msh_dir = case_dir
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'box.msh')

    offset_x = 0.
    offset_y = 0.
    offset_z = 0.
    domain_x = Lx
    domain_y = Ly
    domain_z = Lz

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format
    if cell_type.startswith('tetra'):
        Rec2d = False  # tris or quads
        Rec3d = False  # tets, prisms or hexas
    else:
        Rec2d = True
        Rec3d = True
    p = gmsh.model.geo.addPoint(offset_x, offset_y, offset_z)
    l = gmsh.model.geo.extrude([(0, p)], domain_x, 0, 0, [Nx], [1])
    s = gmsh.model.geo.extrude([l[1]], 0, domain_y, 0, [Ny], [1], recombine=Rec2d)
    v = gmsh.model.geo.extrude([s[1]], 0, 0, domain_z, [Nz], [1], recombine=Rec3d)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(msh_file)
    gmsh.finalize()

    mesh = meshio.read(msh_file)
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict[cell_type] # (num_cells, num_nodes)
    meshio_obj = meshio.Mesh(points=points, cells={cell_type: cells})

    return meshio_obj

def create_3D_box_model_info_thermal(Nx,Ny,Nz,Lx,Ly,Lz,T_left,T_right,case_dir):

    settings = box_mesh(Nx,Ny,Nz,Lx,Ly,Lz,case_dir)
    X = settings.points[:,0]
    Y = settings.points[:,1]
    Z = settings.points[:,2]

    left_boundary_node_ids = []
    right_boundary_node_ids = []
    none_boundary_node_ids = []
    for node_id,node_corrds in enumerate(settings.points):
        if np.isclose(node_corrds[0], 0., atol=1e-5):
            left_boundary_node_ids.append(node_id)
        elif np.isclose(node_corrds[0], Lx, atol=1e-5):
            right_boundary_node_ids.append(node_id)
        else:
            none_boundary_node_ids.append(node_id)

    left_boundary_node_ids = jnp.array(left_boundary_node_ids)
    right_boundary_node_ids = jnp.array(right_boundary_node_ids)
    none_boundary_node_ids = jnp.array(none_boundary_node_ids)

    left_boundary_nodes_values = T_left * jnp.ones(left_boundary_node_ids.shape)
    right_boundary_nodes_values = T_right * jnp.ones(right_boundary_node_ids.shape)

    boundary_nodes = jnp.concatenate([left_boundary_node_ids, right_boundary_node_ids])
    boundary_values = jnp.concatenate([left_boundary_nodes_values, right_boundary_nodes_values])

    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}
    elements_dict = {"elements_ids":jnp.arange(len(settings.cells_dict['hexahedron'])),
                     "elements_nodes":jnp.array(settings.cells_dict['hexahedron'])}
    dofs_dict = {"T":{"non_dirichlet_nodes_ids":none_boundary_node_ids,"dirichlet_nodes_ids":boundary_nodes,"dirichlet_nodes_dof_value":boundary_values}}
    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict},settings
def create_3D_box_model_info_mechanical(model_settings,case_dir):

    settings = box_mesh(model_settings["Nx"],model_settings["Ny"],
                        model_settings["Nz"],model_settings["Lx"],
                        model_settings["Ly"],model_settings["Lz"],case_dir)
    X = settings.points[:,0]
    Y = settings.points[:,1]
    Z = settings.points[:,2]

    left_boundary_node_ids = []
    left_non_boundary_node_ids = []
    right_boundary_node_ids = []
    right_non_boundary_node_ids = []
    left_right_non_boundary_node_ids = []
    for node_id,node_corrds in enumerate(settings.points):
        if np.isclose(node_corrds[0], 0., atol=1e-5):
            left_boundary_node_ids.append(node_id)
        else:
            left_non_boundary_node_ids.append(node_id)

        if np.isclose(node_corrds[0], model_settings["Lx"], atol=1e-5):
            right_boundary_node_ids.append(node_id)
        else:
            right_non_boundary_node_ids.append(node_id)

        if not np.isclose(node_corrds[0], 0., atol=1e-5):
            if not np.isclose(node_corrds[0], model_settings["Lx"], atol=1e-5):
                left_right_non_boundary_node_ids.append(node_id)

    dofs_dict = {"Ux":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]},
                 "Uy":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]},
                 "Uz":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]}}

    for dof in ["Ux","Uy","Uz"]:
        if model_settings[f"{dof}_left"] !="" and model_settings[f"{dof}_right"] !="":

            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(left_right_non_boundary_node_ids)

            dofs_dict[dof]["dirichlet_nodes_ids"].extend(left_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_left"]] * len(left_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)

            dofs_dict[dof]["dirichlet_nodes_ids"].extend(right_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_right"]] * len(right_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)

        elif model_settings[f"{dof}_right"] !="":
            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(right_non_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_ids"].extend(right_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_right"]] * len(right_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)  

        elif model_settings[f"{dof}_left"] !="":
            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(left_non_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_ids"].extend(left_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_left"]] * len(left_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values) 

        dofs_dict[dof]["dirichlet_nodes_dof_value"] = np.array(dofs_dict[dof]["dirichlet_nodes_dof_value"])
        dofs_dict[dof]["non_dirichlet_nodes_ids"] = np.array(dofs_dict[dof]["non_dirichlet_nodes_ids"])
        dofs_dict[dof]["dirichlet_nodes_ids"] = np.array(dofs_dict[dof]["dirichlet_nodes_ids"])


    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}
    elements_dict = {"elements_ids":jnp.arange(len(settings.cells_dict['hexahedron'])),
                     "elements_nodes":jnp.array(settings.cells_dict['hexahedron'])}

    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict},settings


def import_box_model_info_mechanical(file_name,case_dir,model_settings):

    mesh = meshio.read(os.path.join(case_dir, file_name))

    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict['tetra'] # (num_cells, num_nodes)
    meshio_obj = meshio.Mesh(points=points, cells={'tetra': cells})

    X = meshio_obj.points[:,0]
    Y = meshio_obj.points[:,1]
    Z = meshio_obj.points[:,2]

    left_boundary_node_ids = []
    left_non_boundary_node_ids = []
    right_boundary_node_ids = []
    right_non_boundary_node_ids = []
    left_right_non_boundary_node_ids = []
    for node_id,node_corrds in enumerate(meshio_obj.points):
        if np.isclose(node_corrds[0], 0., atol=1e-5):
            left_boundary_node_ids.append(node_id)
        else:
            left_non_boundary_node_ids.append(node_id)

        if np.isclose(node_corrds[0], model_settings["Lx"], atol=1e-5):
            right_boundary_node_ids.append(node_id)
        else:
            right_non_boundary_node_ids.append(node_id)

        if not np.isclose(node_corrds[0], 0., atol=1e-5):
            if not np.isclose(node_corrds[0], model_settings["Lx"], atol=1e-5):
                left_right_non_boundary_node_ids.append(node_id)

    dofs_dict = {"Ux":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]},
                 "Uy":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]},
                 "Uz":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]}}

    for dof in ["Ux","Uy","Uz"]:
        if model_settings[f"{dof}_left"] !="" and model_settings[f"{dof}_right"] !="":

            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(left_right_non_boundary_node_ids)

            dofs_dict[dof]["dirichlet_nodes_ids"].extend(left_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_left"]] * len(left_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)

            dofs_dict[dof]["dirichlet_nodes_ids"].extend(right_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_right"]] * len(right_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)

        elif model_settings[f"{dof}_right"] !="":
            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(right_non_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_ids"].extend(right_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_right"]] * len(right_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)  

        elif model_settings[f"{dof}_left"] !="":
            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(left_non_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_ids"].extend(left_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_left"]] * len(left_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values) 

        dofs_dict[dof]["dirichlet_nodes_dof_value"] = np.array(dofs_dict[dof]["dirichlet_nodes_dof_value"])
        dofs_dict[dof]["non_dirichlet_nodes_ids"] = np.array(dofs_dict[dof]["non_dirichlet_nodes_ids"])
        dofs_dict[dof]["dirichlet_nodes_ids"] = np.array(dofs_dict[dof]["dirichlet_nodes_ids"])


    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}
    elements_dict = {"elements_ids":jnp.arange(len(meshio_obj.cells_dict['tetra'])),
                     "elements_nodes":jnp.array(meshio_obj.cells_dict['tetra'])}

    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict},meshio_obj


def create_2D_square_model_info_mechanical(L,N,Ux_left,Ux_right,Uy_left,Uy_right):
    # FE init starts here
    Ne = N - 1  # Number of elements in each direction
    nx = Ne + 1  # Number of nodes in the x-direction
    ny = Ne + 1  # Number of nodes in the y-direction
    ne = Ne * Ne    # Total number of elements
    # Generate mesh coordinates
    x = jnp.linspace(0, L, nx)
    y = jnp.linspace(0, L, ny)
    X, Y = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = jnp.zeros((Y.shape[-1]))
    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}

    # Create a matrix to store element nodal information
    elements_nodes = jnp.zeros((ne, 4), dtype=int)
    # Fill in the elements_nodes with element and node numbers
    for i in range(Ne):
        for j in range(Ne):
            e = i * Ne + j  # Element index
            # Define the nodes of the current element
            nodes = jnp.array([i * (Ne + 1) + j, i * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j])
            # Store element and node numbers in the matrix
            elements_nodes = elements_nodes.at[e].set(nodes) # Node numbers

    element_ids = jnp.arange(0,elements_nodes.shape[0])
    elements_dict = {"elements_ids":element_ids,"elements_nodes":elements_nodes}

    # Identify boundary nodes on the left and right edges
    left_boundary_nodes = jnp.arange(0, ny * nx, nx)  # Nodes on the left boundary
    right_boundary_nodes = jnp.arange(nx - 1, ny * nx, nx)  # Nodes on the right boundary

    left_ux_values = Ux_left * jnp.ones(left_boundary_nodes.shape)
    right_ux_values = Ux_right * jnp.ones(right_boundary_nodes.shape)
    ux_boundary_nodes = jnp.concatenate([left_boundary_nodes, right_boundary_nodes])
    ux_boundary_values = jnp.concatenate([left_ux_values, right_ux_values])
    ux_non_boundary_nodes = []
    for i in range(N*N):
        if not (jnp.any(ux_boundary_nodes == i)):
            ux_non_boundary_nodes.append(i)
    ux_non_boundary_nodes = jnp.array(ux_non_boundary_nodes)

    dofs_dict = {"Ux":{"non_dirichlet_nodes_ids":ux_non_boundary_nodes,"dirichlet_nodes_ids":ux_boundary_nodes,"dirichlet_nodes_dof_value":ux_boundary_values}}

    left_uy_values = Uy_left * jnp.ones(left_boundary_nodes.shape)
    right_uy_values = Uy_right * jnp.ones(left_boundary_nodes.shape)
    uy_boundary_nodes = jnp.concatenate([left_boundary_nodes, right_boundary_nodes])
    uy_boundary_values = jnp.concatenate([left_uy_values, right_uy_values])
    uy_non_boundary_nodes = []
    for i in range(N*N):
        if not (jnp.any(uy_boundary_nodes == i)):
            uy_non_boundary_nodes.append(i)
    uy_non_boundary_nodes = jnp.array(uy_non_boundary_nodes)

    dofs_dict["Uy"] = {"non_dirichlet_nodes_ids":uy_non_boundary_nodes,"dirichlet_nodes_ids":uy_boundary_nodes,"dirichlet_nodes_dof_value":uy_boundary_values}
    
    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict}

def create_random_fourier_samples(fourier_control,numberof_sample):
    N = int(fourier_control.GetNumberOfControlledVariables()**0.5)
    num_coeffs = fourier_control.GetNumberOfVariables()
    coeffs_matrix = np.zeros((0,num_coeffs))
    for i in range (numberof_sample):
        coeff_vec = np.random.normal(size=num_coeffs)
        coeffs_matrix = np.vstack((coeffs_matrix,coeff_vec))

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # also add uniform dstibuted K of value 0.5
    coeff_vec = 1e-4 * np.zeros((num_coeffs))
    coeff_vec[0] = 10
    coeffs_matrix = np.vstack((coeffs_matrix,coeff_vec))
    K_matrix = np.vstack((K_matrix,fourier_control.ComputeControlledVariables(coeff_vec)))
    # plot_data_input(K_matrix,10,'K distributions')    

    return coeffs_matrix,K_matrix

def create_random_voronoi_samples(voronoi_control,numberof_sample,mean,std_dev):
    number_seeds = voronoi_control.numberof_seeds
    rangeofValues = voronoi_control.k_rangeof_values
    numberofVar = voronoi_control.GetNumberOfVariables()
    coeffs_matrix = np.zeros((0,numberofVar))
    
    for _ in range(numberof_sample):
        x_coords = np.random.normal(mean, std_dev, number_seeds)
        y_coords = np.random.normal(mean, std_dev, number_seeds)
        if isinstance(rangeofValues, tuple):
            K_values = np.random.uniform(rangeofValues[0],rangeofValues[-1],number_seeds)
        if isinstance(rangeofValues, list):
            K_values = np.random.choice(rangeofValues, size=number_seeds)
        
        Kcoeffs = np.zeros((0,numberofVar))
        Kcoeffs = np.concatenate((x_coords.reshape(1,-1), y_coords.reshape(1,-1), K_values.reshape(1,-1)), axis=1)
        coeffs_matrix = np.vstack((coeffs_matrix,Kcoeffs))
    K_matrix = voronoi_control.ComputeBatchControlledVariables(coeffs_matrix)
    return coeffs_matrix,K_matrix

def create_clean_directory(case_dir):
    # Check if the directory exists
    if os.path.exists(case_dir):
        # Remove the directory and all its contents
        shutil.rmtree(case_dir)
    
    # Create the new directory
    os.makedirs(case_dir)

def plot_mesh_vec_data_paper_temp(vectors_list):
    fontsize = 16
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))  # Adjusted to 4 columns

    # Plot the first entity in the first row
    data = vectors_list[0]
    N = int((data.reshape(-1, 1).shape[0]) ** 0.5)
    im = axs[0, 0].imshow(data.reshape(N, N), cmap='viridis', aspect='equal')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_title('Elasticity Morph.', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the same entity with mesh grid in the first row, second column
    im = axs[0, 1].imshow(data.reshape(N, N), cmap='bone', aspect='equal')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xticklabels([])  # Remove text on x-axis
    axs[0, 1].set_yticklabels([])  # Remove text on y-axis
    axs[0, 1].set_title(r'Mesh Grid: $51 \times 51$', fontsize=fontsize)
    axs[0, 1].grid(True, color='red', linestyle='-', linewidth=1)  # Adding solid grid lines with red color
    axs[0, 1].xaxis.grid(True)
    axs[0, 1].yaxis.grid(True)

    x_ticks = np.linspace(0, N, N)
    y_ticks = np.linspace(0, N, N)
    axs[0, 1].set_xticks(x_ticks)
    axs[0, 1].set_yticks(y_ticks)

    cbar = fig.colorbar(im, ax=axs[0, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Zoomed-in region
    zoom_region = data.reshape(N, N)[20:40, 20:40]
    im = axs[0, 2].imshow(zoom_region, cmap='bone', aspect='equal')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_xticklabels([])  # Remove text on x-axis
    axs[0, 2].set_yticklabels([])  # Remove text on y-axis
    axs[0, 2].set_title('Zoomed-in: $x \in [0.4, 0.8], y \in [0.2, 0.6]$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the mesh grid
    axs[0, 2].xaxis.set_major_locator(plt.LinearLocator(21))
    axs[0, 2].yaxis.set_major_locator(plt.LinearLocator(21))
    axs[0, 2].grid(color='red', linestyle='-', linewidth=2)

    # Plot cross-sections along x-axis at y=0.5 for U (FOL and FEM) in the second row, fourth column
    y_idx = int(N * 0.5)
    U1 = vectors_list[0].reshape(N, N)
    axs[0, 3].plot(np.linspace(0, 1, N), U1[y_idx, :], label='Conductivity', color='black')
    axs[0, 3].set_xlim([0, 1])
    #axs[0, 3].set_ylim([min(U1[y_idx, :].min()), max(U1[y_idx, :].max())])
    axs[0, 3].set_aspect(aspect='auto')
    axs[0, 3].set_title('Cross-section of K at y=0.5', fontsize=fontsize)
    axs[0, 3].legend(fontsize=fontsize)
    axs[0, 3].grid(True)
    axs[0, 3].set_xlabel('x', fontsize=fontsize)
    axs[0, 3].set_ylabel('K', fontsize=fontsize)


    # Plot the second entity in the second row
    data = vectors_list[1]
    im = axs[1, 0].imshow(data.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_title('$T$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the fourth entity in the second row
    data = vectors_list[2]
    im = axs[1, 1].imshow(data.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_title('$T$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the absolute difference between vectors_list[1] and vectors_list[3] in the third row, second column
    diff_data_1 = np.abs(vectors_list[1] - vectors_list[2])
    im = axs[1, 2].imshow(diff_data_1.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_title('Abs. Difference $T$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot cross-sections along x-axis at y=0.5 for U (FOL and FEM) in the second row, fourth column
    y_idx = int(N * 0.5)
    U1 = vectors_list[1].reshape(N, N)
    U2 = vectors_list[2].reshape(N, N)
    axs[1, 3].plot(np.linspace(0, 1, N), U1[y_idx, :], label='U FOL', color='blue')
    axs[1, 3].plot(np.linspace(0, 1, N), U2[y_idx, :], label='U FEM', color='red')
    axs[1, 3].set_xlim([0, 1])
    axs[1, 3].set_ylim([min(U1[y_idx, :].min(), U2[y_idx, :].min()), max(U1[y_idx, :].max(), U2[y_idx, :].max())])
    axs[1, 3].set_aspect(aspect='auto')
    axs[1, 3].set_title('Cross-section of T at y=0.5', fontsize=fontsize)
    axs[1, 3].legend(fontsize=fontsize)
    axs[1, 3].grid(True)
    axs[1, 3].set_xlabel('x', fontsize=fontsize)
    axs[1, 3].set_ylabel('T', fontsize=fontsize)

    plt.tight_layout()

    # Save the figure in multiple formats
    plt.savefig('plot_mesh_vec_data.png', dpi=300)
    plt.savefig('plot_mesh_vec_data.pdf')

    plt.show()



def plot_mesh_vec_grad_data_thermal(vectors_list):
    fontsize = 16
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))

    data = vectors_list[0]
    L = 1
    N = int((data.reshape(-1, 1).shape[0])**0.5)

    dx = L / (N - 1)

    U_fem = vectors_list[2]
    dU_dx_fem = -1 * data.reshape(N, N) * np.gradient(U_fem.reshape(N, N), dx, axis=1)
    dU_dy_fem = -1 * data.reshape(N, N) * np.gradient(U_fem.reshape(N, N), dx, axis=0)

    im = axs[0, 1].imshow(dU_dx_fem, cmap='plasma')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_title('Heat flux $q_x$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    im = axs[1, 1].imshow(dU_dy_fem, cmap='plasma')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_title('Heat flux $q_y$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    U_fol = vectors_list[1]
    dU_dx_fol = -1 * data.reshape(N, N) * np.gradient(U_fol.reshape(N, N), dx, axis=1)
    dU_dy_fol = -1 * data.reshape(N, N) * np.gradient(U_fol.reshape(N, N), dx, axis=0)

    min_v = np.min(dU_dx_fem)
    max_v = np.max(dU_dx_fem)
    im = axs[0, 0].imshow(dU_dx_fol, cmap='plasma', vmin=min_v, vmax=max_v)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_title('Stress $q_{x}$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    min_v = np.min(dU_dy_fem)
    max_v = np.max(dU_dy_fem)
    im = axs[1, 0].imshow(dU_dy_fol, cmap='plasma', vmin=min_v, vmax=max_v)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_title('Stress $q_{y}$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    diff_data_2 = np.abs(dU_dx_fem - dU_dx_fol)
    im = axs[0, 2].imshow(diff_data_2, cmap='plasma')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_title('Abs. Difference $q_{x}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    diff_data_2 = np.abs(dU_dy_fem - dU_dy_fol)
    im = axs[1, 2].imshow(diff_data_2, cmap='plasma')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_title('Abs. Difference $q_{y}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    # Extract cross-sections at y = 0.5
    y_index = N // 2
    stress_x_cross_fem = dU_dx_fem[y_index, :]
    stress_y_cross_fem = dU_dy_fem[y_index, :]
    stress_x_cross_fol = dU_dx_fol[y_index, :]
    stress_y_cross_fol = dU_dy_fol[y_index, :]

    # Plot cross-sections in the fourth column
    axs[0, 3].plot(np.linspace(0, L, N), stress_x_cross_fem, label='FEM', color='r')
    axs[0, 3].plot(np.linspace(0, L, N), stress_x_cross_fol, label='FOL', color='b')
    axs[0, 3].set_title('Cross-section $q_x$', fontsize=fontsize)
    axs[0, 3].legend()

    axs[1, 3].plot(np.linspace(0, L, N), stress_y_cross_fem, label='FEM', color='r')
    axs[1, 3].plot(np.linspace(0, L, N), stress_y_cross_fol, label='FOL', color='b')
    axs[1, 3].set_title('Cross-section $q_y$', fontsize=fontsize)
    axs[1, 3].legend()

    # Save cross-section data to a text file
    with open('cross_section_data.txt', 'w') as f:
        f.write('x, stress_x_fem, stress_x_fol, stress_y_fem, stress_y_fol, stress_xy_fem, stress_xy_fol\n')
        for i in range(N):
            f.write(f'{i*dx}, {stress_x_cross_fem[i]}, {stress_x_cross_fol[i]}, {stress_y_cross_fem[i]}, {stress_y_cross_fol[i]}\n')


    plt.tight_layout()
    plt.savefig('plot_mesh_vec_grad_data.png', dpi=300)
    plt.savefig('plot_mesh_vec_grad_data.pdf')
    plt.show()

def TensorToVoigt(tensor):
    if tensor.size == 4:
        voigt = jnp.zeros((3,1))
        voigt = voigt.at[0,0].set(tensor[0,0])
        voigt = voigt.at[1,0].set(tensor[1,1])
        voigt = voigt.at[2,0].set(tensor[0,1])
    return voigt
    
def fourth_order_identity_tensor(dim=3):
    I = jnp.zeros((dim, dim, dim, dim))
    I = jnp.einsum('ik,jl->ijkl',jnp.eye(dim),jnp.eye(dim))
    return I
    
def diad_special(A,B,dim):
    C = jnp.zeros((dim, dim, dim, dim))
    C = 0.5*(jnp.einsum('ik,jl->ijkl',A,B) + jnp.einsum('il,jk->ijkl',A,B))
    return C
    
def FourthTensorToVoigt(Cf):
    if Cf.size == 16:
        C = jnp.zeros((3,3))
        C = C.at[0,0].set(Cf[0,0,0,0])
        C = C.at[0,1].set(Cf[0,0,1,1])
        C = C.at[0,2].set(Cf[0,0,0,1])
        C = C.at[1,0].set(C[0,1])
        C = C.at[1,1].set(Cf[1,1,1,1])
        C = C.at[1,2].set(Cf[1,1,0,1])
        C = C.at[2,0].set(C[0,2])
        C = C.at[2,1].set(C[1,2])
        C = C.at[2,2].set(Cf[0,1,0,1])
    return C


def Neo_Hooke(F,k,mu):
    C = jnp.dot(F.T,F)
    invC = jnp.linalg.inv(C)
    J0 = jnp.linalg.det(F)
    eps = 1e-12
    J = jnp.where(jnp.abs(J0)<eps, eps, J0)
    # if jnp.abs(J) < 1e-12:
    #     raise ValueError("Deformation gradient determinant is too small, possible degenerate element.")
    p = (k/4)*(2*J-2*J**(-1))
    dp_dJ = (k/4)*(2 + 2*J**(-2))
    # Strain Energy
    xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
    I1_bar = (J**(-2/3))*jnp.trace(C)
    xsie_iso = 0.5*mu*(I1_bar - 3)
    loss_positive_bias = 100     # To prevent loss to become a negative number
    xsie = xsie_vol + xsie_iso + loss_positive_bias

    # Stress Tensor
    S_vol = J*p*invC
    I_fourth = fourth_order_identity_tensor(C.shape[0])
    P = I_fourth - (1/3)*jnp.einsum('ij,kl->ijkl', invC, C)
    S_bar = mu*jnp.eye(C.shape[0])
    S_iso = (J**(-2/3))*jnp.einsum('ijkl,kl->ij',P,S_bar)
    Se = S_vol + S_iso

    P_bar = diad_special(invC,invC,invC.shape[0]) - (1/3)*jnp.einsum('ij,kl->ijkl',invC,invC)
    C_vol = (J*p + dp_dJ*J**2)*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*p*diad_special(invC,invC,invC.shape[0])
    C_iso = (2/3)*(J**(-2/3))*jnp.vdot(S_bar,C)*P_bar - \
            (2/3)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
    C_tangent_fourth = C_vol + C_iso
    
    return xsie, Se, C_tangent_fourth