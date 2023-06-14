
import pyvista
data = pyvista.read('potts_3d.20.vti') 
phase = np.load('spparks_phase.npy')
Nx, Ny, Nz = np.array(data.dimensions) - 1

spin = data.get_array('Spin').reshape(Nz, Ny, Nx).T

spin += 1
for i in range(Nx):
	for j in range(Ny):
		if i >= phase.shape[0] or j >= phase.shape[1]:
			spin[i,j,:] = 1
		else:
			if phase[i,j] != 0:
				spin[i,j,:] = 1

data['Spin'] = spin.T.flatten()
data.save('spparks_logo.vti')
