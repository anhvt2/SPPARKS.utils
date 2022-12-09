
# Instruction

1. build `libjpeg`
```shell
./configure
make
sudo make install
```

2. build SPPARKS

```shell
mpirun -np 16 spk < in.potts_additive_small3d
```

3. run SPPARKS
```shell
mpirun -np 16 spk < in.potts_additive_small3d
```

4. visualize
	1. through ParaView: load `*.vti` files
		* animate
		* save animation
	2. through OVITO:
		* load `dump.additive_small3d.*`
		* on the `Pipelines` (the first out of 3 icons on top right)
			* `Edit Column Mapping`: `i1` as `Particle Type`
			* `x`, `y`, `z` should automatically load
			* add `Assign color` filter: operate on `Particles`
		* on the `Rendering`:
			* animate
			* choose a view point by clicking on one of four views
			* `Save to file` with file name `test_` or `perspective_small3d_` at a designated location
			* click `Render active viewport`
			* save animation
