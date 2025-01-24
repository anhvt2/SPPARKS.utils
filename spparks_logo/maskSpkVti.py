
import pyvista
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.get_cmap('coolwarm')

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
# data.save('spparks_logo.vti')

pl = pyvista.Plotter(off_screen=True)
pl.add_mesh(data.threshold(1+0.1), show_edges=False, line_width=1, cmap=cmap)
pl.background_color = "white"
pl.camera_position = 'yx'
pl.camera.elevation += 180
# pl.camera.azimuth += 180
# pl.camera.roll += 180
# pl.add_axes(color='k')
# pl.show_axes()
pl.remove_scalar_bar()
pl.screenshot('spparks_horizontal.png', window_size=[1860*6,968*6])
