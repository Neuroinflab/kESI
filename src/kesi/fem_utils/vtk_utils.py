from mfem import ser as mfem
from tqdm import tqdm


def grid_function_save_vtk(gf, os, field_name):
    fes = gf.FESpace()
    mesh = fes.GetMesh()
    vec_dim = gf.VectorDim()
    assert gf.Size()/vec_dim == mesh.GetNV()
    if vec_dim == 1:
        os.write("SCALARS "+field_name+ " double 1\n LOOKUP_TABLE default\n")
        for i in tqdm(range(fes.GetNDofs()), total=fes.GetNDofs(), desc='saving mesh data scalars {}'.format(field_name)):
            os.write(str(gf[i])+"\n")
    elif (vec_dim == 2 or vec_dim == 3) and mesh.SpaceDimension() > 1:
        print('Vector field', field_name)
        os.write("VECTORS "+field_name+" double\n")
        vdofs = mfem.intArray(vec_dim)
        for i in tqdm(range(fes.GetNDofs()), total=fes.GetNDofs(), desc='saving mesh data vectors {}'.format(field_name)):
            vdofs.SetSize(1)
            vdofs[0] = i
            fes.DofsToVDofs(vdofs)
            os.write(str(gf[vdofs[0]]) +' '+str(gf[vdofs[1]]) +' ')
            if vec_dim == 2:
                os.write("0.0\n")
            else:
                os.write(str(gf[vdofs[2]])+"\n")
    else:
        pass
