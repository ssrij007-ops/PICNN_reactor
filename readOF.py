"""
This is for cases read openfoam result on a squared mesh 2D
"""
import foamFileAddNoise
import numpy as np
from foamFileOperation import readVectorFromFile,readScalarFromFile
def convertOFMeshToImage_StructuredMesh(nx,ny,MeshFile,FileName,ext,mriLevel=0,plotFlag=True,ReshapeOrder = 'F'):
	title=['x','y']
	OFVector=None
	OFScalar=None
	for i in range(len(FileName)):
		if FileName[i][-1]=='U':
			OFVector=readVectorFromFile(FileName[i])
			title.append('u')
			title.append('v')
		elif FileName[i][-1]=='p':
			OFScalar=readScalarFromFile(FileName[i])
			title.append('p')
		elif 'H2' in FileName[i]:
			OFScalar=readScalarFromFile(FileName[i])
			title.append('H2')
		elif 'O2' in FileName[i]:
			OFScalar=readScalarFromFile(FileName[i])
			title.append('O2')
		elif 'N2' in FileName[i]:
			OFScalar=readScalarFromFile(FileName[i])    		
			title.append('N2')    		
		elif 'H2O' in FileName[i]:
			OFScalar=readScalarFromFile(FileName[i])
			title.append('H2O')
		else:
			print('Variable name is not clear')
			exit()
		#elif FileName[i][-1]=='T':
			#OFScalar=readScalarFromFile(FileName[i])
			#title.append('T')

	nVar=len(title)
	OFMesh=readVectorFromFile(MeshFile)
	Ng=OFMesh.shape[0]
	OFCase=np.zeros([Ng,nVar])
	OFCase[:,0:2]=np.copy(OFMesh[:,0:2])
	if OFVector is not None and OFScalar is not None:
		if mriLevel>1e-16:
			OFVector=foamFileAddNoise.addMRINoise(OFVector,mriLevel)
		OFCase[:,2:4]=np.copy(OFVector[:,0:2])
		OFCase[:,4]=np.copy(OFScalar)
		#OFCase[:,5]=np.copy(OFScalar)
		#OFCase[:,6]=np.copy(OFScalar)
	elif OFScalar is not None:
		OFCase[:,2]=np.copy(OFScalar)
	OFPic=np.reshape(OFCase, (ny,nx,nVar), order=ReshapeOrder)
	if plotFlag:
		pass	#plt.show()
	return OFPic #torch.from_numpy(OFPic)






