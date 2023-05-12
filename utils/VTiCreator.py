# Theron Rodgers, Sandia National Labs, August 2015
# Python script to convert SPPARKS output files to Paraview VTI files. The current version only supports dump files with two variables. However, this can be modified by adding additional instancesof lines 29-33 (and modifying the Name specifed on line 29).
# An example line from a compatible SPPARKS simulation is : dump 1 text 100.0 dump.potts.* id i1 d1
from pandas import *
import io

baseFileName = 'dump.additive4.' #Base file name, file names must be formatted as "baseFileNameN" where N is a sequential integer
numberOfFiles = 1 #Number of files to process
Nx = 300 #Number of x-lattice sites
Ny = 300 #Number of y-lattice site
Nz = 200 #Number of z-lattice sites
firstFile = 5 # N of the first processed file
lastFile = numberOfFiles + firstFile


#Define a function to create an XML-style vti file.
def to_vti(df, fileName, mode='w'):
	
	#Create the VTI file header in the XML style
	xml = '<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n  <ImageData WholeExtent="0 ' + str(Nx) + ' 0 ' + str(Ny) + ' 0 ' + str(Nz) + '" Origin="0 0 0" Spacing="1 1 1">\n  <Piece Extent="0 ' + str(Nx) + ' 0 ' + str(Ny) + ' 0 ' + str(Nz) + '">\n  <PointData>\n  </PointData>\n   <CellData Scalars="Spin, Mobility">\n    <DataArray type="Int32" Name="Spin" format="ascii">\n'
	
	#Open the file and write the header
	with open(fileName, mode) as f:
        
		f.write(xml)
		#Write the first column of data
		df.to_csv(f,columns=[1],header=False, index=False)
		#Write the break header
		f.write('    </DataArray>\n\n    <DataArray type="Float32" Name="Stored Energy" format="ascii">\n')
		f.flush()
        
        #Write the second column of data
		df.to_csv(f,columns=[2],header=False, index=False)
        #To include additional variables in the VTI file, add additional versions of the above structure.
		f.write('    </DataArray>\n   </CellData>\n  </Piece>\n  </ImageData>\n</VTKFile>')
        f.close()


	

for i in range(firstFile, lastFile):
	currentInFile = baseFileName + str(i)
	currentOutFile = baseFileName + str(i) + '.vti'

	#Read in the file, assuming the standard 9 header lines
	dataInGood = read_csv(currentInFile, sep=' ', skiprows = 9, header=None,index_col = 0)#, iterator = True, chunksize  = 1000)
	#dataInGood = concat(dataIn, ignore_index = True)
	
	#Sort the data to have increasing index (This is needed due to MPI)
	
	to_vti(dataInGood.sort_index(axis=0), currentOutFile)
	print('Finished file ' + str(i))
