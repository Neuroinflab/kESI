import mfem.ser as mfem
import numpy as np

device = mfem.Device("cpu")
device.Print()

mesh = mfem.Mesh("/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/fine_with_electrode_scratch.msh", 1, 1)
# mesh = mfem.Mesh("/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/fine_with_plane_scratch.msh", 1, 1)
mesh.UniformRefinement()
# mesh = mfem.Mesh("/home/mdovgialo/projects/halje_data_analysis/kESI/extras/mfem_solver/fine_with_plane.msh")
# import IPython
# IPython.embed()
dim = mesh.Dimension()

order = 1

fec = mfem.H1_FECollection(order, dim)

fespace = mfem.FiniteElementSpace(mesh, fec)
print('Number of finite element unknowns: ' +
      str(fespace.GetTrueVSize()))

# import IPython
# IPython.embed()

# this is a masking list which decides, which boundaries of the mesh are Dirichlet boundaries
# in this case we have 4 spheres, and for some reason index of the outside boundary is 5
# for now I want to set the outside boundary of the 4 spheres as essential boundary and having a 0 potential.
ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
ess_bdr.Assign(0)
ess_bdr[mesh.bdr_attributes[-1]-1] = 1


ess_tdof_list = mfem.intArray()
fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

# setup boundary potential coeff (piecewise coefficient)
boundary_potential = mfem.Vector([1.0] * mesh.bdr_attributes.Max())
boundary_potential_coeff = mfem.PWConstCoefficient(boundary_potential)


b = mfem.LinearForm(fespace)

# is this supposed to be conductivity? or inverse times pi???

# Expression(f'''
#                                             {-0.25 / np.pi / conductivity}
#                                             / {r_src}
# conductivities_vector = mfem.Vector([0.33, 1.65, 0.0165, 0.33, 1e-10])
# conductivities_vector = mfem.Vector([0.33, 1.65, 0.0165, 0.33])


# conductivities_vector = mfem.Vector(list(0.25 / np.pi / np.array([0.33, 1.65, 0.0165, 0.33])))
conductivities_vector = mfem.Vector(list(1 / np.array([0.33, 1.65, 0.0165, 0.33, 1e-10])))
# conductivities_vector = mfem.Vector([0.33] * 4)
conductivities_coeff = mfem.PWConstCoefficient(conductivities_vector)

# b.AddBoundaryIntegrator(mfem.BoundaryLFIntegrator(boundary_potential_coeff))

# for each point charge
point_charge_coeff = mfem.DeltaCoefficient()
point_charge_coeff.SetScale(1)
point_charge_coeff.SetDeltaCenter(mfem.Vector([0.0, 0.08, 0.00]))
b.AddDomainIntegrator(mfem.DomainLFIntegrator(point_charge_coeff))



b.Assemble()
x = mfem.GridFunction(fespace)
x.Assign(0.0)

a = mfem.BilinearForm(fespace)

a.AddDomainIntegrator(mfem.DiffusionIntegrator(conductivities_coeff))
a.Assemble()

A = mfem.OperatorPtr()
B = mfem.Vector()
X = mfem.Vector()

a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
print("Size of linear system: " + str(A.Height()))

AA = A.AsSparseMatrix()
M = mfem.GSSmoother(AA)
mfem.PCG(A, M, B, X, 1, 10000, 1e-12, 0.0)


# cg = mfem.CGSolver()
# cg.SetRelTol(1e-12)
# cg.SetMaxIter(2000)
# cg.SetPrintLevel(1)
# cg.SetPreconditioner(M)
# cg.SetOperator(A)
# cg.Mult(B, X)

a.RecoverFEMSolution(X, b, x)

mesh.Print('refined_brain_test.mesh', 8)
x.Save('sol_brain_test.gf', 8)

paraview_dc = mfem.ParaViewDataCollection("test", mesh)
paraview_dc.SetPrefixPath("ParaView")
paraview_dc.SetLevelsOfDetail(order)
paraview_dc.SetDataFormat(mfem.VTKFormat_BINARY)
paraview_dc.SetHighOrderOutput(True)
paraview_dc.SetCycle(0)
paraview_dc.SetTime(0.0)
paraview_dc.RegisterField("potential", x)
paraview_dc.Save()

plate_z = -0.088