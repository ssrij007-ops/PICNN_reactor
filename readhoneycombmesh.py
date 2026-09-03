import sys
sys.path.append(r"C:\Users\SS6391\anaconda3\envs\cuda_env")  # Add the folder path
import Ofpp
import numpy as np
from pyMesh import hcubeMesh

def read_honeycomb_mesh(ofmeshfile):
	h = 0.01  # Grid spacing
	OFBCCoord = Ofpp.parse_boundary_field(ofmeshfile)  # Corrected variable name    
    
    # Extract boundary points (fixed indentation)
	OFINERTWALLC = OFBCCoord[b'inertWall'][b'value']
	OFREACTINGWALLC = OFBCCoord[b'reactingWall'][b'value']
	OFINLETC = OFBCCoord[b'inlet'][b'value']
	OFOUTLETC = OFBCCoord[b'outlet'][b'value']

    # Separate X and Y coordinates (Fixed variable names)
	inletX, inletY = OFINLETC[:, 0], OFINLETC[:, 1]
	inertWallX, inertWallY = OFINERTWALLC[:, 0], OFINERTWALLC[:, 1]  # Fixed incorrect assignment
	outletX, outletY = OFOUTLETC[:, 0], OFOUTLETC[:, 1]
	reactingWallX, reactingWallY = OFREACTINGWALLC[:, 0], OFREACTINGWALLC[:, 1]

    # Ensure dimensions are correctly assigned
	ny=len(inletX);nx=len(inertWallX)
	#ny, nx = len(inletX), len(inertWallX)
    
   # Create structured computational mesh
	myMesh = hcubeMesh(inletX, inletY, outletX, outletY,
				inertWallX, inertWallY, reactingWallX, reactingWallY,h, True, True,
				tolMesh=1e-10, tolJoint=1e-1) #hcubeMesh    
	return myMesh, nx, ny

def proj_solu_from_ofmesh_2_mymesh(OFPic, myMesh):
	OFX = OFPic[:, :, 0]
	OFY = OFPic[:, :, 1]
	OFU = OFPic[:, :, 2] 
	OFV = OFPic[:, :, 3] 
	OFP = OFPic[:, :, 4]
    
    # Initialize arrays for additional species
	OFH2 = OFPic[:, :, 5] #if OFPic.shape[2] > 5 else None
	OFO2 = OFPic[:, :, 6] #if OFPic.shape[2] > 6 else None
	OFN2 = OFPic[:, :, 7] #if OFPic.shape[2] > 7 else None
	OFH2O = OFPic[:, :, 8] #if OFPic.shape[2] > 8 else None

    # Initialize projected solution arrays
	OFU_sb = np.zeros_like(OFU)
	OFV_sb = np.zeros_like(OFV)
	OFP_sb = np.zeros_like(OFP)
	OFH2_sb = np.zeros_like(OFH2) #if OFH2 is not None else None
	OFO2_sb = np.zeros_like(OFO2) #if OFO2 is not None else None
	OFN2_sb = np.zeros_like(OFN2) #if OFN2 is not None else None
	OFH2O_sb = np.zeros_like(OFH2O) #if OFH2O is not None else None

	ny, nx = myMesh.x.shape
	for i in range(nx):
		for j in range(ny):
			dist = (myMesh.x[j, i] - OFX)**2 + (myMesh.y[j, i] - OFY)**2
			idx_min = np.unravel_index(np.argmin(dist), dist.shape)
			OFU_sb[j, i] = OFU[idx_min]
			OFV_sb[j, i] = OFV[idx_min]
			OFP_sb[j, i] = OFP[idx_min]
			OFH2_sb[j, i] = OFH2[idx_min]
			OFO2_sb[j, i] = OFO2[idx_min]
			OFN2_sb[j, i] = OFN2[idx_min]
			OFH2O_sb[j, i] = OFH2O[idx_min]
		#if OFH2 is not None:
			#OFH2_sb[j, i] = OFH2[idx_min]
		#if OFO2 is not None:
			#OFO2_sb[j, i] = OFO2[idx_min]
		#if OFN2 is not None:
			#OFN2_sb[j, i] = OFN2[idx_min]
		#if OFH2O is not None:
			#OFH2O_sb[j, i] = OFH2O[idx_min]

    # Return the projected solution arrays
	projected_solution = [OFU_sb, OFV_sb, OFP_sb, OFH2_sb, OFO2_sb, OFN2_sb, OFH2O_sb]
	#if OFH2 is not None:
		#projected_solution.append(OFH2_sb)
	#if OFO2 is not None:
		#projected_solution.append(OFO2_sb)
	#if OFN2 is not None:
		#projected_solution.append(OFN2_sb)
	#if OFH2O is not None:
		#projected_solution.append(OFH2O_sb)

	return projected_solution
