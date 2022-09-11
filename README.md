# SPPARKS

## General guide
* from [http://spparks.sandia.gov](spparks.sandia.gov)
* include `libjpeg` into installation for `dump image` command
* to look for `libjpeg` library `ldconfig -p | grep libjpeg`
* modify `Makefile.mpi` to include `jpeg` library; see page 15/35 at [this slide](http://lammps.sandia.gov/tutorials/italy14/Compiling_LAMMPS.pdf)

```Makefile
SPK_INC =       -DSPPARKS_GZIP -DSPPARKS_JPEG
...

# JPEG and/or PNG library, OPTIONAL 
# see discussion in doc/Section_start.html#2_2 (step 7) 
# only needed if -DLAMMPS_JPEG or -DLAMMPS_PNG listed with LMP_INC 
# INC = path(s) for jpeglib.h and/or png.h 
# PATH = path(s) for JPEG library and/or PNG library 
# LIB = name(s) of JPEG library and/or PNG library

# JPG_INC =
# JPG_PATH =
# JPG_LIB = -ljpeg

JPG_INC =     -I/usr/lib/x86_64-linux-gnu/
JPG_PATH =    -L/usr/lib/x86_64-linux-gnu/
JPG_LIB =     -ljpeg
```

```shell
cd STUBS
make
cd ..
# make mpi-stubs # make in src/STUBS/
make mpi # serial/g++
```

An example of Makefile.mpi on Solo
```Makefile
# solo = ECN Linux cluster

SHELL = /bin/sh

# ---------------------------------------------------------------------
# compiler/linker settings
# specify flags and libraries needed for your compiler

CC =        mpicxx
CCFLAGS =   -O2 -std=c++11 -axCORE-AVX512
SHFLAGS =   -fPIC
DEPFLAGS =  -M

LINK =      ${CC}
LINKFLAGS = -O2 
LIB =       -lstdc++ -lm
SIZE =      size

ARCHIVE =   ar
ARFLAGS =   -rc
SHLIBFLAGS =    -shared

# ---------------------------------------------------------------------
# SPPARKS-specific settings
# specify settings for SPPARKS features you will use

# SPPARKS ifdef options, see doc/Section_start.html

SPK_INC =    -DSPPARKS_GZIP  -DSPPARKS_JPEG -DSPPARKS_BIGBIG
#SPK_INC =  -DSPPARKS_GZIP  -DSPPARKS_JPEG -DSTITCH_PARALLEL
#SPK_INC =  -DSPPARKS_GZIP  -DSPPARKS_JPEG 
SPK_INC =    -DSPPARKS_GZIP  -DSPPARKS_JPEG -DSPPARKS_BIGBIG -DSPPARKS_UNORDERED_MAP


# MPI library, can be src/STUBS dummy lib
# INC = path for mpi.h, MPI compiler settings
# PATH = path for MPI library
# LIB = name of MPI library

MPI_INC =       #-I${MPI_HOME}/include
MPI_PATH =      #-L${MPI_HOME}/lib 
MPI_LIB =   
MPI_INC = 
MPI_PATH =
MPI_LIB =

# JPEG library, only needed if -DLAMMPS_JPEG listed with LMP_INC
# INC = path for jpeglib.h
# PATH = path for JPEG library
# LIB = name of JPEG library

JPG_INC = -I/usr/include
JPG_PATH = -L/usr/lib64 
JPG_LIB = -ljpeg


# ---------------------------------------------------------------------
# build rules and dependencies
# no need to edit this section

include Makefile.package.settings
include Makefile.package

EXTRA_INC = $(SPK_INC) $(PKG_INC) $(MPI_INC) $(JPG_INC) $(PKG_SYSINC)
EXTRA_PATH = $(PKG_PATH) $(MPI_PATH) $(JPG_PATH) $(PKG_SYSPATH)
EXTRA_LIB = $(PKG_LIB) $(MPI_LIB) $(JPG_LIB) $(PKG_SYSLIB)
EXTRA_CPP_DEPENDS = $(PKG_CPP_DEPENDS)
EXTRA_LINK_DEPENDS = $(PKG_LINK_DEPENDS)

# Path to src files

vpath %.cpp ..
vpath %.h ..

# Link target

$(EXE): $(OBJ)
    $(LINK) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(EXTRA_LIB) $(LIB) -o $(EXE)
    $(SIZE) $(EXE)

# Library targets

lib:    $(OBJ)
    $(ARCHIVE) $(ARFLAGS) $(EXE) $(OBJ)

shlib:  $(OBJ)
    $(CC) $(CCFLAGS) $(SHFLAGS) $(SHLIBFLAGS) $(EXTRA_PATH) -o $(EXE) \
        $(OBJ) $(EXTRA_LIB) $(LIB)

# Compilation rules

%.o:%.cpp
    $(CC) $(CCFLAGS) $(SHFLAGS) $(EXTRA_INC) -c $<

%.d:%.cpp
    $(CC) $(CCFLAGS) $(EXTRA_INC) $(DEPFLAGS) $< > $@

# Individual dependencies

depend : fastdep.exe $(SRC)
    @./fastdep.exe $(EXTRA_INC) -- $^ > .depend || exit 1

fastdep.exe: ../DEPEND/fastdep.c
    cc -O -o $@ $<

sinclude .depend
```

## Compile with `libjpeg`
include `libjpeg` in SPPARKS

https://github.com/idaholab/SPPARKS/blob/master/doc/Section_start.txt


A standard JPEG library usually goes by the name libjpeg.a and has an
associated header file jpeglib.h.  Whichever JPEG library you have on
your platform, you'll need to set the appropriate JPG_INC, JPG_PATH,
and JPG_LIB variables in Makefile.foo so that the compiler and linker
can find it.


Solutions: 
1. download and compile `libjpeg`: e.g. `jpeg-9e` works
    ```shell
    ./configure
    make
    sudo make install
    ```

2. include in `Makefile.mpi` for `JPG_INC`, `JPG_PATH`, and `JPG_LIB`
    ```shell
    JPG_INC  = -I/usr/local/lib/
    JPG_PATH = -L/usr/local/lib/
    JPG_LIB  = /usr/local/lib/libjpeg.a
    ```

3. edit `SPK_INC`: make sure `-DSPPARKS_JPEG` is included
```
SPK_INC =    -DSPPARKS_GZIP  -DSPPARKS_JPEG -DSPPARKS_BIGBIG -DSPPARKS_UNORDERED_MAP -DSTITCH_PARALLEL
```

