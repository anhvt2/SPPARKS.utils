
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
		* `Edit Column Mapping`: `i1` as `Particle Type`
		* `x`, `y`, `z` should automatically load
		* add `Assign color` filter: operate on `Particles`
		* animate
		* save animation
