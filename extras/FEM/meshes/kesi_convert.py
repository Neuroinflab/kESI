import sys
import meshio
# conda install h5py lxml
# pip install meshio

src_name = sys.argv[1]
dst_name = sys.argv[2]
msh = meshio.read(src_name)

meshio.write(dst_name,
             meshio.Mesh(points=msh.points,
                         cells={"tetra": msh.cells["tetra"]}))
meshio.write(dst_name[:-5] + "_boundaries.xdmf",
             meshio.Mesh(points=msh.points,
                         cells={"triangle": msh.cells["triangle"]},
                         cell_data={"triangle": {"boundaries": msh.cell_data["triangle"]["gmsh:physical"]}}))

meshio.write(dst_name[:-5] + "_subdomains.xdmf",
             meshio.Mesh(points=msh.points,
                         cells={"tetra": msh.cells["tetra"]},
                         cell_data={"tetra": {"subdomains":
                            msh.cell_data["tetra"]["gmsh:physical"]}}))
