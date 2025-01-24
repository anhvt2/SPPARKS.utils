
# SPPARKS Logo

The purpose of this repository is to create a SPPARKS microstructure that makes up SPPARKS in words, while retaining different grain size at different locations.

# Workflow

```shell
python3 img2phase.py
python3 maskSpkVti.py
```

# Input

Adopted from `examples/potts_grad/in.potts_temp_grad.periodic_x.off`. 

# Geometric modeling

Adopted from the dogbone specimen modeling from DAMASK person repository. Create and mask according to phase. 

# Convert VTI to Numpy array

```python
import pyvista
data = pyvista.read('potts_3d.20.vti') # a uniform grid object: https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UniformGrid.html
# points = data.points
Nx, Ny, Nz = np.array(data.dimensions) - 1
spin = data.get_array('Spin')
mod_spin = spin
data['Spin'] = mod_spin
data.save('test.vti')
```
