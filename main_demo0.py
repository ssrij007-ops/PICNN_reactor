import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
import tikzplotlib
from sklearn.metrics import mean_squared_error as calMSE

from readhoneycombmesh import read_honeycomb_mesh,\
                              proj_solu_from_ofmesh_2_mymesh
                              

from dataset import VaryGeoDataset
from pyMesh import visualize2D, setAxisLabel, to4DTensor
from model import USCNNSep
from readOF import convertOFMeshToImage_StructuredMesh
#torch.autograd.set_detect_anomaly(True)
h=0.1
Itol=0
#Prepare data


mesh_hf,nx_hf,ny_hf=read_honeycomb_mesh('Fine/2/C_corrected')

solu_hf1=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/U',
	                                             'Fine/2/p'],

	                                            [0,1,0,1],0.0,False)
solu_hf2=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/H2'],
	                                            [0,1,0,1],0.0,False)
solu_hf3=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/O2'],
	                                            [0,1,0,1],0.0,False)
solu_hf4=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/N2'],
	                                            [0,1,0,1],0.0,False)
solu_hf5=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/H2O'],
	                                            [0,1,0,1],0.0,False)
solu_hf=np.concatenate([solu_hf1,solu_hf2[:,:,2:],solu_hf3[:,:,2:],solu_hf4[:,:,2:],solu_hf5[:,:,2:]],axis=2) 

#solu_hf=np.concatenate([solu_hf1],axis=2)

mesh_lf,nx_lf,ny_lf=read_honeycomb_mesh('Coarse/2/C_corrected')

solu_lf1=convertOFMeshToImage_StructuredMesh(nx_lf,ny_lf,'Coarse/2/C_corrected',
	                                           ['Coarse/2/U',
	                                            'Coarse/2/p'],
	                                            [0,1,0,1],0.0,False)
solu_lf2=convertOFMeshToImage_StructuredMesh(nx_lf,ny_lf,'Coarse/2/C_corrected',
	                                             ['Coarse/2/H2'],
	                                            [0,1,0,1],0.0,False)
solu_lf3=convertOFMeshToImage_StructuredMesh(nx_lf,ny_lf,'Coarse/2/C_corrected',
	                                             ['Coarse/2/O2'],
	                                            [0,1,0,1],0.0,False)
solu_lf4=convertOFMeshToImage_StructuredMesh(nx_lf,ny_lf,'Coarse/2/C_corrected',
	                                             ['Coarse/2/N2'],
	                                            [0,1,0,1],0.0,False)
solu_lf5=convertOFMeshToImage_StructuredMesh(nx_lf,ny_lf,'Coarse/2/C_corrected',
	                                             ['Coarse/2/H2O'],
	                                            [0,1,0,1],0.0,False)
solu_lf=np.concatenate([solu_lf1,solu_lf2[:,:,2:],solu_lf3[:,:,2:],solu_lf4[:,:,2:],solu_lf5[:,:,2:]], axis =2)

[OFU_lf,OFV_lf,OFP_lf, OFH2_lf, OFO2_lf, OFN2_lf, OFH2O_lf]=proj_solu_from_ofmesh_2_mymesh(solu_lf,mesh_lf) 

relativeNoise=0 #np.random.uniform(low=0.0, high=1.0, size=1)
inputRand=relativeNoise*np.random.rand(1, 5, 10, 10)
inputRand=torch.from_numpy(inputRand)
inputRand=inputRand.float().to('cuda')
print("✅ inputRand shape (fixed):", inputRand.shape)
u_lf=torch.from_numpy(OFU_lf).float().to('cuda').reshape([1,1,OFU_lf.shape[0],OFU_lf.shape[1]])
v_lf=torch.from_numpy(OFV_lf).float().to('cuda').reshape([1,1,OFV_lf.shape[0],OFV_lf.shape[1]])
p_lf=torch.from_numpy(OFP_lf).float().to('cuda').reshape([1,1,OFP_lf.shape[0],OFP_lf.shape[1]])
H2_lf=torch.from_numpy(OFH2_lf).float().to('cuda').reshape([1,1,OFH2_lf.shape[0],OFH2_lf.shape[1]])
O2_lf=torch.from_numpy(OFO2_lf).float().to('cuda').reshape([1,1,OFO2_lf.shape[0],OFO2_lf.shape[1]])
N2_lf=torch.from_numpy(OFN2_lf).float().to('cuda').reshape([1,1,OFN2_lf.shape[0],OFN2_lf.shape[1]])
H2O_lf=torch.from_numpy(OFH2O_lf).float().to('cuda').reshape([1,1,OFH2O_lf.shape[0],OFH2O_lf.shape[1]])
velofunc = lambda argc: np.full_like(argc, 0.2)

#velofunc=lambda argc: 100*argc*(0.2-argc)

infertruth=torch.from_numpy(velofunc(mesh_hf.x[0,:])).float().to('cuda')
infertruth=infertruth[1:29].reshape([28,1])

solu_hf1=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/U',
	                                             'Fine/2/p'],

	                                            [0,1,0,1],0.0,False)
solu_hf2=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/H2'],
	                                            [0,1,0,1],0.0,False)
solu_hf3=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/O2'],
	                                            [0,1,0,1],0.0,False)
solu_hf4=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/N2'],
	                                            [0,1,0,1],0.0,False)
solu_hf5=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/H2O'],
	                                            [0,1,0,1],0.0,False)
solu_hf1=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'Fine/2/C_corrected',
	                                             ['Fine/2/U',
	                                              'Fine/2/p'],
	                                            [0,1,0,1],0.0,False)
solu_hf=np.concatenate([solu_hf1,solu_hf2[:,:,2:],solu_hf3[:,:,2:],solu_hf4[:,:,2:],solu_hf5[:,:,2:]],axis=2) 

#solu_hf=np.concatenate([solu_hf1],axis=2) 

HFMeshList=[mesh_hf]
train_set=VaryGeoDataset(HFMeshList)

def finite_difference(f, dx, dim=-1, order=1):
		if order == 1:
			return (f.roll(-1, dims=dim) - f.roll(1, dims=dim)) / (2 * dx)
		elif order == 2:
			return (f.roll(-1, dims=dim) - 2 * f + f.roll(1, dims=dim)) / (dx**2)
		else:
			raise ValueError("Only 1st and 2nd order derivatives are supported.")


#Set up paramter
batchSize=1
NvarInput=5
NvarOutput=7
nEpochs=50000
lr=0.001
Ns=1
nu= 4.97e-5 # 0.01
criterion = nn.MSELoss()
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)

#Set up model
model=USCNNSep(h,nx_hf,ny_hf,NvarInput,NvarOutput,'ortho').to('cuda')
model=model.to('cuda')
optimizer = optim.Adam(model.parameters(),lr=lr)
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)

#Project OFSolution 2 MyMesh
[OFU_sb,OFV_sb,OFP_sb,OFH2_sb,OFO2_sb, OFN2_sb, OFH2O_sb]=proj_solu_from_ofmesh_2_mymesh(solu_hf,mesh_hf)
#[BICUBICU,BICUBICV,_]=proj_solu_from_ofmesh_2_mymesh(solu_lf,mesh_hf)

OFU_sb_cuda=torch.from_numpy(OFU_sb).reshape([1,1,OFU_sb.shape[0],OFU_sb.shape[1]]).float().to('cuda')
OFH2_sb_cuda=torch.from_numpy(OFH2_sb).reshape([1,1,OFH2_sb.shape[0],OFH2_sb.shape[1]]).float().to('cuda')
OFO2_sb_cuda=torch.from_numpy(OFO2_sb).reshape([1,1,OFO2_sb.shape[0],OFO2_sb.shape[1]]).float().to('cuda')
OFN2_sb_cuda=torch.from_numpy(OFN2_sb).reshape([1,1,OFN2_sb.shape[0],OFN2_sb.shape[1]]).float().to('cuda')
OFH2O_sb_cuda=torch.from_numpy(OFH2O_sb).reshape([1,1,OFH2O_sb.shape[0],OFH2O_sb.shape[1]]).float().to('cuda')


nobs=200
# For a 30 (rows) × 70 (columns) mesh = 2100 total cells
# Define vertical strips at columns: 14, 28, 42, 56
idxpool = [i for i in range(30 * 70) if i > 70]  # Skip first row if desired
#idxsample=sample(idxpool,nobs)
idxsample = [i for i in idxpool if (i in range(70*2 + 14, 70*28 + 14, 70)) or (i in range(70*2 + 28, 70*28 + 28, 70)) or (i in range(70*2 + 42, 70*28 + 42, 70)) or (i in range(70*2 + 56, 70*28 + 56, 70))]
idx1 = [i // 70 for i in idxsample]  # 30 rows
idx2 = [i % 70 for i in idxsample]   # 70 columns
idx1=np.asarray(idx1)
np.savetxt('idx1.txt',idx1)
idx2=np.asarray(idx2)
np.savetxt('idx2.txt',idx2)


#Define the geometry transformation
def dfdx(f,dydeta,dydxi,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h	
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
	return dfdx

def dfdy(f,dxdxi,dxdeta,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h	
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdy=Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
	return dfdy

# Define the model to train
def train(epoch):
	#print("✅ Entered train()")   # <-- This should always print if you enter the function
	#startTime=time.time()
	xRes=0
	yRes=0
	mRes=0
	SpRes=0
	#TRes=0
	eU=0
	eV=0
	eP=0
	eH2=0
	eO2=0
	eN2=0
	eH2O=0
	
	for iteration, batch in enumerate(training_data_loader):
		[_,cord,_,_,_,Jinv,dxdxi,dydxi,dxdeta,dydeta]=to4DTensor(batch)
		optimizer.zero_grad()
		input=torch.cat([u_lf,v_lf, H2_lf, O2_lf,H2O_lf],axis=1)
		input=input*(1+inputRand)
		if epoch==1:
			np.savetxt('InputU.txt',input[0,0,:,:].detach().cpu().numpy())
			np.savetxt('InputV.txt',input[0,1,:,:].detach().cpu().numpy())
			np.savetxt('InputH2.txt',input[0,2,:,:].detach().cpu().numpy())
			np.savetxt('InputO2.txt',input[0,3,:,:].detach().cpu().numpy())
			np.savetxt('InputH2O.txt',input[0,4,:,:].detach().cpu().numpy())


		BICUBICU=model.US(input)[0,0,:,:].detach().cpu().numpy()
		BICUBICV=model.US(input)[0,1,:,:].detach().cpu().numpy()

		output=model(input)#model(cord)  # This is where the model (CNN/PICNN) makes predictions
		output_pad=udfpad(output)
		UTEMP=torch.zeros([1,1,ny_hf,nx_hf]).float().to('cuda')
		USparseOBS=torch.zeros([1,1,ny_hf,nx_hf]).float().to('cuda')
		outputU_tmep=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		for ii in range(len(idx1)):
			UTEMP[0,0,idx1[ii],idx2[ii]]=OFU_sb_cuda[0,0,idx1[ii],idx2[ii]]-outputU_tmep[0,0,idx1[ii],idx2[ii]]
			USparseOBS[0,0,idx1[ii],idx2[ii]]=OFU_sb_cuda[0,0,idx1[ii],idx2[ii]]
    
		CHAN_U    = 0
		CHAN_V    = 1
		CHAN_P    = 2
		CHAN_H2   = 3
		CHAN_O2   = 4
		CHAN_N2   = 5
		CHAN_H2O  = 6

		outputU   = UTEMP +output_pad[:, CHAN_U, :, :].unsqueeze(1)
		outputV   = output_pad[:, CHAN_V, :, :].unsqueeze(1)
		outputP   = output_pad[:, CHAN_P, :, :].unsqueeze(1)
		outputH2  = output_pad[:, CHAN_H2, :, :].unsqueeze(1)
		outputO2  = output_pad[:, CHAN_O2, :, :].unsqueeze(1)
		outputN2  = output_pad[:, CHAN_N2, :, :].unsqueeze(1)
		outputH2O = output_pad[:, CHAN_H2O, :, :].unsqueeze(1)


		#outputU=UTEMP+output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
			                                #output_pad.shape[2],
			                                #output_pad.shape[3])

		#outputV=output_pad[:,1,:,:].reshape(output_pad.shape[0],1,
			                                #output_pad.shape[2],
			                                #output_pad.shape[3])
		#outputP=output_pad[:,2,:,:].reshape(output_pad.shape[0],1,
			                                #output_pad.shape[2],
			                               # output_pad.shape[3])
		#outputH2=output_pad[:,3,:,:].reshape(output_pad.shape[0],1,
			                                #output_pad.shape[2],
			                                #output_pad.shape[3])
		#outputO2=output_pad[:,4,:,:].reshape(output_pad.shape[0],1,
			                                #output_pad.shape[2],
			                                #output_pad.shape[3])
		#outputN2=output_pad[:,5,:,:].reshape(output_pad.shape[0],1,
			                                #output_pad.shape[2],
			                                #output_pad.shape[3])
		#outputH2O=output_pad[:,6,:,:].reshape(output_pad.shape[0],1,
			                                #output_pad.shape[2],
			                                #output_pad.shape[3])


		for j in range(batchSize):
			#Impose BC
            
			#print("🟨 target shape:",output[j,2,0,:].reshape(1,nx_hf-2*padSingleSide))
			#print("🟦 model.source shape:", model.source.shape)
            
# Top wall (last rows)

			outputU[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0.5* (output[j, CHAN_U, -1, :] + output[j, CHAN_U, -2, :]).reshape(1, -1)

# Bottom wall (first rows)

			outputU[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0.5* (output[j, CHAN_U, 0, :] + output[j, CHAN_U, 1, :]).reshape(1, -1)

# Right wall (last cols)

			outputU[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0.5 * (output[j, CHAN_U, :, -1].reshape(-1, 1) + output[j, CHAN_U, :, -2].reshape(-1, 1))

# Left wall (first cols)

			outputU[j, 0, padSingleSide:-padSingleSide, :padSingleSide] = 0.5 * (output[j, CHAN_U, :, 0].reshape(-1, 1) + output[j, CHAN_U, :, 1].reshape(-1, 1))

# Corners

			outputU[j, 0, 0, 0]     = (outputU[j, 0, 0, 1] + outputU[j, 0, 1, 0])

			outputU[j, 0, 0, -1]    =  (outputU[j, 0, 0, -2] + outputU[j, 0, 1, -1])

			outputU[j, 0, -1, 0]    =  (outputU[j, 0, -1, 1] + outputU[j, 0, -2, 0])

			outputU[j, 0, -1, -1]   = (outputU[j, 0, -1, -2] + outputU[j, 0, -2, -1])

			#outputU[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0 # up reacting wall
			#outputU[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0  # down inert wall
			#outputU[j,0,padSingleSide:-padSingleSide,-padSingleSide:]= output[j, 0,:, -1].reshape(ny_hf -2*padSingleSide,1) # right outlet bc
			#bc_value = torch.abs(model.source[:68]).view(-1, 1)  # [28, 1]
			#outputU[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = torch.abs(model.source)#left inlet bc

            #outputU[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=torch.abs(model.source)   #left inlet bc
			einfer=torch.sqrt(criterion(infertruth,torch.abs(model.source))/criterion(torch.abs(model.source),torch.abs(model.source)*0))
			print('Infer velocity====================',model.source[0])
			try:
				print('>>>>>>>Infer error<<<<<<< =======================',einfer.item())
				print('>>>>>>>model source<<<<<<< =======================',model.source)
			except:
				pass

			#outputU[j,0,0,0]=0.5*(outputU[j,0,0,1]+outputU[j,0,1,0])
			#outputU[j,0,0,-1]=0.5*(outputU[j,0,0,-2]+outputU[j,0,1,-1])
			#outputU[j,0,0,0]=1*(outputU[j,0,0,1])
			#outputU[j,0,0,-1]=1*(outputU[j,0,0,-2])
			
            
# Top wall (last rows)
			outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0.5 * (output[j, CHAN_V, -1, :] + output[j, CHAN_V, -2, :]).reshape(1, -1)

# Bottom wall (first rows)
			outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0.5 * (output[j, CHAN_V, 0, :] + output[j, CHAN_V, 1, :]).reshape(1, -1)

# Right wall (last cols)
			outputV[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0.5 * (output[j, CHAN_V, :, -1].reshape(-1, 1) + output[j, CHAN_V, :, -2].reshape(-1, 1))

# Left wall (first cols)
			outputV[j, 0, padSingleSide:-padSingleSide, :padSingleSide] = 0.5 * (output[j, CHAN_V, :, 0].reshape(-1, 1) + output[j, CHAN_V, :, 1].reshape(-1, 1))

# Corners
			outputV[j, 0, 0, 0]     = (outputV[j, 0, 0, 1] + outputV[j, 0, 1, 0])
			outputV[j, 0, 0, -1]    = (outputV[j, 0, 0, -2] + outputV[j, 0, 1, -1])
			outputV[j, 0, -1, 0]    = (outputV[j, 0, -1, 1] + outputV[j, 0, -2, 0])
			outputV[j, 0, -1, -1]   = (outputV[j, 0, -1, -2] + outputV[j, 0, -2, -1])


			#outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0 #up reacting wall
			#outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0 # down inert bc
			#outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=output[j,1,:,-1].reshape(ny_hf -2*padSingleSide,1) # right outlet bc
			#outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 # left inlet bc
			
			#outputV[j,0,0,0]=0.5*(outputV[j,0,0,1]+outputV[j,0,1,0])
			#outputV[j,0,0,-1]=0.5*(outputV[j,0,0,-2]+outputV[j,0,1,-1])
			#outputV[j,0,0,0]=1*(outputV[j,0,0,1])
			#outputV[j,0,0,-1]=1*(outputV[j,0,0,-2])
			

			
			outputP[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,2, -1, :].reshape(1,nx_hf-2*padSingleSide) # up reacting wall
			outputP[j,0,:padSingleSide,padSingleSide:-padSingleSide]=output[j,2,0,:].reshape(1,nx_hf-2*padSingleSide) # down inert wall			
			outputP[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=torch.abs(model.source) #output[j,2,:,-1].reshape(ny_hf-2*padSingleSide,1)    # right outlet bc			
			outputP[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=output[j,2,:,0].reshape(ny_hf-2*padSingleSide,1)  # left inlet bc
			
			#outputP[j,0,0,0]=0.5*(outputP[j,0,0,1]+outputP[j,0,1,0])
			#outputP[j,0,0,-1]=0.5*(outputP[j,0,0,-2]+outputP[j,0,1,-1])
			outputP[j,0,0,0]=1*(outputP[j,0,0,1])
			outputP[j,0,0,-1]=1*(outputP[j,0,0,-2])

#########################
### Smooth Padding BCs ##
#########################

# === H2 ===
			outputH2[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0.5 *(output[j, CHAN_H2, -1, :] + output[j, CHAN_H2, -2, :]).reshape(1, -1)
			outputH2[j, 0, :padSingleSide, padSingleSide:-padSingleSide]  = 0.5 * (output[j, CHAN_H2, 0, :] + output[j, CHAN_H2, 1, :]).reshape(1, -1)
			outputH2[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0.5 * (output[j, CHAN_H2, :, -1].reshape(-1, 1) + output[j, CHAN_H2, :, -2].reshape(-1, 1))
			outputH2[j, 0, padSingleSide:-padSingleSide, :padSingleSide]  = 0.5 * (output[j, CHAN_H2, :, 0].reshape(-1, 1) + output[j, CHAN_H2, :, 1].reshape(-1, 1))

			outputH2[j, 0, 0, 0]     = 0.5 * (outputH2[j, 0, 0, 1] + outputH2[j, 0, 1, 0])
			outputH2[j, 0, 0, -1]    = 0.5 * (outputH2[j, 0, 0, -2] + outputH2[j, 0, 1, -1])
			outputH2[j, 0, -1, 0]    = 0.5 * (outputH2[j, 0, -1, 1] + outputH2[j, 0, -2, 0])
			outputH2[j, 0, -1, -1]   = 0.5 * (outputH2[j, 0, -1, -2] + outputH2[j, 0, -2, -1])

# === O2 ===
			outputO2[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0.5 * (output[j, CHAN_O2, -1, :] + output[j, CHAN_O2, -2, :]).reshape(1, -1)
			outputO2[j, 0, :padSingleSide, padSingleSide:-padSingleSide]  = 0.5 * (output[j, CHAN_O2, 0, :] + output[j, CHAN_O2, 1, :]).reshape(1, -1)
			outputO2[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0.5 * (output[j, CHAN_O2, :, -1].reshape(-1, 1) + output[j, CHAN_O2, :, -2].reshape(-1, 1))
			outputO2[j, 0, padSingleSide:-padSingleSide, :padSingleSide]  = 0.5 * (output[j, CHAN_O2, :, 0].reshape(-1, 1) + output[j, CHAN_O2, :, 1].reshape(-1, 1))

			outputO2[j, 0, 0, 0]     = (outputO2[j, 0, 0, 1] + outputO2[j, 0, 1, 0])
			outputO2[j, 0, 0, -1]    = (outputO2[j, 0, 0, -2] + outputO2[j, 0, 1, -1])
			outputO2[j, 0, -1, 0]    = (outputO2[j, 0, -1, 1] + outputO2[j, 0, -2, 0])
			outputO2[j, 0, -1, -1]   = (outputO2[j, 0, -1, -2] + outputO2[j, 0, -2, -1])

# === N2 ===
			outputN2[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0.5 * (output[j, CHAN_N2, -1, :] + output[j, CHAN_N2, -2, :]).reshape(1, -1)
			outputN2[j, 0, :padSingleSide, padSingleSide:-padSingleSide]  = 0.5 * (output[j, CHAN_N2, 0, :] + output[j, CHAN_N2, 1, :]).reshape(1, -1)
			outputN2[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0.5 * (output[j, CHAN_N2, :, -1].reshape(-1, 1) + output[j, CHAN_N2, :, -2].reshape(-1, 1))
			outputN2[j, 0, padSingleSide:-padSingleSide, :padSingleSide]  = 0.5 * (output[j, CHAN_N2, :, 0].reshape(-1, 1) + output[j, CHAN_N2, :, 1].reshape(-1, 1))
            
			outputN2[j, 0, 0, 0]     = (outputN2[j, 0, 0, 1] + outputN2[j, 0, 1, 0])
			outputN2[j, 0, 0, -1]    = (outputN2[j, 0, 0, -2] + outputN2[j, 0, 1, -1])
			outputN2[j, 0, -1, 0]    = (outputN2[j, 0, -1, 1] + outputN2[j, 0, -2, 0])
			outputN2[j, 0, -1, -1]   = (outputN2[j, 0, -1, -2] + outputN2[j, 0, -2, -1])

# === H2O ===
			outputH2O[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0.5 * (output[j, CHAN_H2O, -1, :] + output[j, CHAN_H2O, -2, :]).reshape(1, -1)
			outputH2O[j, 0, :padSingleSide, padSingleSide:-padSingleSide]  = 0.5 * (output[j, CHAN_H2O, 0, :] + output[j, CHAN_H2O, 1, :]).reshape(1, -1)
			outputH2O[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0.5 * (output[j, CHAN_H2O, :, -1].reshape(-1, 1) + output[j, CHAN_H2O, :, -2].reshape(-1, 1))
			outputH2O[j, 0, padSingleSide:-padSingleSide, :padSingleSide]  = 0.5 * (output[j, CHAN_H2O, :, 0].reshape(-1, 1) + output[j, CHAN_H2O, :, 1].reshape(-1, 1))

			outputH2O[j, 0, 0, 0]     = (outputH2O[j, 0, 0, 1] + outputH2O[j, 0, 1, 0])
			outputH2O[j, 0, 0, -1]    = (outputH2O[j, 0, 0, -2] + outputH2O[j, 0, 1, -1])
			outputH2O[j, 0, -1, 0]    = (outputH2O[j, 0, -1, 1] + outputH2O[j, 0, -2, 0])
			outputH2O[j, 0, -1, -1]   = (outputH2O[j, 0, -1, -2] + outputH2O[j, 0, -2, -1])


# === 1. Reacting wall mask ===

# === 1. Reacting wall mask ===
		reacting_wall_mask = torch.zeros_like(outputH2)
		reacting_wall_mask[:, :, -1, :] = 1.0  # top wall = reacting surface

# === 2. Partial derivatives for U, V, P === x
		dudx = dfdx(outputU, dydeta, dydxi, Jinv)
		d2udx2 = dfdx(dudx, dydeta, dydxi, Jinv)
		dudy = dfdy(outputU, dxdxi, dxdeta, Jinv)
		d2udy2 = dfdy(dudy, dxdxi, dxdeta, Jinv)
		dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
		d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
		dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
		d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
		dpdx = dfdx(outputP, dydeta, dydxi, Jinv)
		dpdy = dfdy(outputP, dxdxi, dxdeta, Jinv)

# === 3. Surface reaction rate ===
		k_surface = 1.935e2
		Y_H2 = outputH2
		Y_O2 = outputO2
		reaction_rate = k_surface * Y_H2 * torch.sqrt(torch.clamp(Y_O2, min=1e-4))
		reaction_source_H2  = -reacting_wall_mask * reaction_rate
		reaction_source_O2  = -reacting_wall_mask * (0.5 * reaction_rate)
		reaction_source_H2O = reacting_wall_mask * reaction_rate
		reaction_source_N2  = torch.zeros_like(outputN2)

		reaction_sources = [reaction_source_H2, reaction_source_O2, reaction_source_N2, reaction_source_H2O]

# === 4. Species Residuals with spatial weighting ===
		dx = 1.0 / (outputH2.shape[-1] - 1)
		dy = 1.0 / (outputH2.shape[-2] - 1)
		species_outputs = [outputH2, outputO2, outputN2, outputH2O]
		Dk = [7e-5, 2e-5, 1.8e-5, 6e-5]

		speciesResiduals = []
		weighted_species_loss = 0.0

		for k in range(len(species_outputs)):
			outputYk = species_outputs[k]
			dYkdx = finite_difference(outputYk, dx, dim=-1, order=1)
			dYkdy = finite_difference(outputYk, dy, dim=-2, order=1)
			d2Ykdx2 = finite_difference(outputYk, dx, dim=-1, order=2)
			d2Ykdy2 = finite_difference(outputYk, dy, dim=-2, order=2)
    
			advYk = outputU * dYkdx + outputV * dYkdy
			diffYk = Dk[k] * (d2Ykdx2 + d2Ykdy2)
			res_k = advYk - diffYk - reaction_sources[k].to(outputU.device)
			speciesResiduals.append(res_k)
    
		if k in [0, 1, 3]:  # H2, O2, H2O
			weighted_res = reacting_wall_mask * res_k**2 + 0.1 * (1 - reacting_wall_mask) * res_k**2
			weighted_species_loss += torch.mean(weighted_res)
		else:
			 weighted_species_loss += torch.mean(res_k**2)

# === 5. PDE Residuals ===
		continuity = dudx + dvdy
		momentumX = outputU * dudx + outputV * dudy
		forceX = -dpdx + nu * (d2udx2 + d2udy2)
		Xresidual = momentumX - forceX
		momentumY = outputU * dvdx + outputV * dvdy
		forceY = -dpdy + nu * (d2vdx2 + d2vdy2)
		Yresidual = momentumY - forceY

        
		import torch.nn.functional as F

# Interpolate low-res H2 and H2O tensors to match high-res output shapes (e.g., [1,1,30,70])
		true_H2_tensor = F.interpolate(H2_lf, size=(outputH2.shape[-2], outputH2.shape[-1]), mode='bilinear', align_corners=False)
		true_O2_tensor = F.interpolate(O2_lf, size=(outputO2.shape[-2], outputO2.shape[-1]), mode='bilinear', align_corners=False)
		true_H2O_tensor = F.interpolate(H2O_lf, size=(outputH2O.shape[-2], outputH2O.shape[-1]), mode='bilinear', align_corners=False)

# Compute supervised losses
		loss_eH2 = torch.mean((outputH2 - true_H2_tensor) ** 2)
		loss_eH2O = torch.mean((outputH2O - true_H2O_tensor) ** 2)
		loss_eO2 = torch.mean((outputO2 - true_O2_tensor) ** 2)

# === 7. Total loss ===
		λ_momentum = 10.0
		λ_continuity = 10.0
		loss = (
		λ_momentum *criterion(Xresidual, torch.zeros_like(Xresidual)) +
		λ_momentum*criterion(Yresidual, torch.zeros_like(Yresidual)) +
		λ_continuity*criterion(continuity, torch.zeros_like(continuity)) +
		weighted_species_loss +
        (1.0 / (eH2 + 1e-3)) * loss_eH2 + \
        (1.0 / (eO2 + 1e-3)) * loss_eO2 + \
        (1.0 / (eH2O + 1e-3)) * loss_eH2O
        )


		loss.backward()
		optimizer.step()



# === 7. Logging ===

		loss_xm = criterion(Xresidual, torch.zeros_like(Xresidual))
		loss_ym = criterion(Yresidual, torch.zeros_like(Yresidual))
		loss_mass = criterion(continuity, torch.zeros_like(continuity))
		xRes += loss_xm.item()
		yRes += loss_ym.item()
		mRes += loss_mass.item()
		SpRes += weighted_species_loss.item()


		CNNUNumpy = outputU[0,0,:,:].cpu().detach().numpy()
		CNNVNumpy = outputV[0,0,:,:].cpu().detach().numpy()
		CNNPNumpy = outputP[0,0,:,:].cpu().detach().numpy()
		CNNH2Numpy = outputH2[0,0,:,:].cpu().detach().numpy()
		CNNO2Numpy = outputO2[0,0,:,:].cpu().detach().numpy()
		CNNN2Numpy = outputN2[0,0,:,:].cpu().detach().numpy()
		CNNH2ONumpy = outputH2O[0,0,:,:].cpu().detach().numpy()

		eU += np.sqrt(calMSE(OFU_sb, CNNUNumpy) / calMSE(OFU_sb, OFU_sb * 0))
		eV += np.sqrt(calMSE(OFV_sb, CNNVNumpy) / calMSE(OFV_sb, OFV_sb * 0))
		eP += np.sqrt(calMSE(OFP_sb, CNNPNumpy) / calMSE(OFP_sb, OFP_sb * 0))
        
		eH2 += np.sqrt(calMSE(OFH2_sb, CNNH2Numpy) / calMSE(OFH2_sb, OFH2_sb * 0))
		eO2 += np.sqrt(calMSE(OFO2_sb, CNNO2Numpy) / calMSE(OFO2_sb, OFO2_sb * 0))
		eN2 += np.sqrt(calMSE(OFN2_sb, CNNN2Numpy) / calMSE(OFN2_sb, OFN2_sb * 0))
		eH2O += np.sqrt(calMSE(OFH2O_sb, CNNH2ONumpy) / calMSE(OFH2O_sb, OFH2O_sb * 0))
        
		eUmag = np.sqrt(calMSE(np.sqrt(OFU_sb**2 + OFV_sb**2), np.sqrt(CNNUNumpy**2 + CNNVNumpy**2)) / calMSE(np.sqrt(OFU_sb**2 + OFV_sb**2), np.sqrt(OFU_sb**2 + OFV_sb**2) * 0))
		eUBICUIC = np.sqrt(calMSE(np.sqrt(OFU_sb[1:-1,1:-1]**2 + OFV_sb[1:-1,1:-1]**2), np.sqrt(BICUBICU**2 + BICUBICV**2)) / calMSE(np.sqrt(OFU_sb[1:-1,1:-1]**2 + OFV_sb[1:-1,1:-1]**2), np.sqrt(OFU_sb[1:-1,1:-1]**2 + OFV_sb[1:-1,1:-1]**2) * 0))

	print('VelMagError_CNN =', eUmag)
	print('VelMagError_BI =', eUBICUIC)
	print('P_err_CNN =', eP)
	print('H2_err_CNN =', eH2)
	print('O2_err_CNN =', eO2)
	print('N2_err_CNN =', eN2)
	print('H2O_err_CNN =', eH2O)

	print('Epoch is', epoch)
	print('xRes Loss is', xRes / len(training_data_loader))
	print('yRes Loss is', yRes / len(training_data_loader))
	print('mRes Loss is', mRes / len(training_data_loader))
	print('SpRes Loss is', SpRes / len(training_data_loader))
	print('eU Loss is', eU / len(training_data_loader))
	print('eV Loss is', eV / len(training_data_loader))
	print('eP Loss is', eP / len(training_data_loader))
	print('eH2 Loss is', eH2 / len(training_data_loader))
	print('eO2 Loss is', eO2 / len(training_data_loader))
	print('eN2 Loss is', eN2 / len(training_data_loader))
	print('eH2O Loss is', eH2O / len(training_data_loader))



	if epoch==1:
		np.savetxt('BIErrorVmag.txt',eUBICUIC*np.ones([4,4]))
	if (einfer.item()<0.05) or epoch%nEpochs==0 or epoch%5000==0 or epoch==100: # eP<0.15 and eVmag<0.04 and epoch==100 or epoch%5000==0 or epoch%nEpochs==0 or 
		torch.save(model, str(epoch)+'.pth')
		fig0=plt.figure(figsize=(14,15)) #fig0=plt.figure()
        
		ax=plt.subplot(5,3,1)
		_,cbar=visualize2D(ax,mesh_lf.x,
			           		  mesh_lf.y,
			           np.sqrt(input[0,0,:,:].cpu().detach().numpy()**2+\
			           		   input[0,1,:,:].cpu().detach().numpy()**2),'vertical',[0,0.28])
		cbar.set_ticks([0,0.05, 0.1, 0.15, 0.2, 0.25, 0.28]) #,1.2,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('Input')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8)
        
		ax=plt.subplot(5,3,2)
		_,cbar=visualize2D(ax,mesh_hf.x,
			                  mesh_hf.y,
			           np.sqrt(outputU[0,0,:,:].cpu().detach().numpy()**2),'vertical',[0,0.28]) #+\	           		   outputV[0,0,:,:].cpu().detach().numpy()**2
		setAxisLabel(ax,'p')
		ax.set_title('CNN')
		cbar.set_ticks([0,0.05, 0.1, 0.15, 0.2, 0.25, 0.28]) #,1.2,1.5])
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8)
        

		ax=plt.subplot(5,3,3)
		_,cbar=visualize2D(ax,mesh_hf.x,
			           		  mesh_hf.y,
			           np.sqrt(OFU_sb**2+\
			           		   OFV_sb**2),'vertical',[0,0.28])
		cbar.set_ticks([0,0.05, 0.1, 0.15, 0.2, 0.25, 0.28]) #,1.2,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('Truth')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8)

		ax=plt.subplot(5,3,4)
		_,cbar=visualize2D(ax,mesh_hf.x,
			           		  mesh_hf.y,
			                  2*USparseOBS[0,0,:,:].cpu().detach().numpy(),'vertical',[0,0.28])
		cbar.set_ticks([0,0.05, 0.1, 0.15, 0.2, 0.25, 0.28]) #,1.2,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('Observation')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8)


		#ax=plt.subplot(2,3,6)
		#visualize2D(ax,mesh_hf.x,
			           #mesh_hf.y,
			           # OFN2_sb[:,:],'vertical',[0,0.988])
		#setAxisLabel(ax,'p')
		#ax.set_title('True'+'H2')
		#ax.set_aspect('equal')

		ax=plt.subplot(5,3,5)
		pdb.set_trace()
		_,cbar=visualize2D(ax,mesh_hf.x[1:-1,1:-1],
			           		  mesh_hf.y[1:-1,1:-1],
			           		  np.sqrt(BICUBICU**2+BICUBICV**2),
							  'vertical',[0,0.28])
		setAxisLabel(ax,'p')
		ax.set_title('Bicubic')
		ax.set_aspect('equal')

		#ax_=plt.subplot(2,3,6)
		#ax_.plot(mesh_hf.x[0,1:29].reshape([28,1]),model.source.cpu().detach().numpy(),'x',label='Inferred',color='blue')
		#ax_.plot(mesh_hf.x[0,:],velofunc(mesh_hf.y[0,:]),'--',label='True')
		#setAxisLabel(ax_,'p')
		#ax_.set_ylabel(r'$v$')
		#ax_.set_title('Inlet '+'Velocity Profile')


		
		#ax_=plt.subplot(2,3,6)		
		#ax_.plot(mesh_hf.x[0,1:29].reshape([28,1]),model.source.cpu().detach().numpy(),'x',label='Inferred',color='blue')
		#ax_.plot(mesh_hf.x[0,:],velofunc(mesh_hf.y[0,:]),'--',label='True')
		#setAxisLabel(ax_,'p')
		#ax_.set_ylabel(r'$v$')
		#ax_.set_title('Inlet '+'Velocity Profile')
		

		#ax=plt.subplot(2,3,6)
		#_,cbar=visualize2D(ax, mesh_hf.x, mesh_hf.y,OFP_sb) #,'vertical',[0,101325])
		#plt.title('OFP_sb')
		#plt.colorbar()
		#plt.axis('equal')
        
		#ax=plt.subplot(2,3,6)
		#plt.contourf(mesh_lf.x, mesh_lf.y,OFP_lf),'vertical',[0,101325]
		#plt.title('OFP_lf')
		#plt.colorbar()
		#plt.axis('equal')
        
        
		ax=plt.subplot(5,3,6)
		_,cbar=visualize2D(ax,mesh_hf.x,
			           		  mesh_hf.y,
			                  (OFU_sb),
			           		  'vertical',[0,0.28])
		cbar.set_ticks([0,0.05, 0.1, 0.15, 0.2, 0.25, 0.28]) #,1.2,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('OFU_sb')
		ax.set_aspect('equal')

		#ax=plt.subplot(2,3,5)
		#visualize2D(ax,mesh_hf.x,
			           #mesh_hf.y,
			           #outputP[0,0,:,:].cpu().detach().numpy(),'vertical',[0,101325])
		#setAxisLabel(ax,'p')
		#ax.set_title('Super-resolved '+'Pressure')
		#ax.set_aspect('equal')

		#ax=plt.subplot(2,3,6)
		#visualize2D(ax,mesh_hf.x,
			           #mesh_hf.y,
			           #OFP_sb[:,:],'vertical',[0,101325])
		#setAxisLabel(ax,'p')
		#ax.set_title('True '+'Pressure')
		#ax.set_aspect('equal')
   
		ax=plt.subplot(5,3,7)
		_,cbar=visualize2D(ax,mesh_lf.x,
			           		  mesh_lf.y,
			             input[0,2,:,:].cpu().detach().numpy(),'vertical', [0,0.003])
			           		   
		setAxisLabel(ax,'p')
		ax.set_title('InputH2')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8)

		ax=plt.subplot(5,3,8)
		_,cbar= visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           outputH2[0,0,:,:].cpu().detach().numpy(),'vertical',[0,0.003])
		setAxisLabel(ax,'p')
		ax.set_title('S_R '+'H2')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8) 


		ax=plt.subplot(5,3,9)
		_,cbar= visualize2D(ax,mesh_hf.x,
			          mesh_hf.y,
			          OFH2_sb[:,:],'vertical',[0,0.003])
		setAxisLabel(ax,'p')
		ax.set_title('True '+'H2')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8) 
        
    
		#ax=plt.subplot(2,3,3)
		#_,cbar= visualize2D(ax,mesh_hf.x,
			           #mesh_hf.y,
			           #outputN2[0,0,:,:].cpu().detach().numpy(),'vertical',[0,1])
		#setAxisLabel(ax,'p')
		#ax.set_title('S_R '+'N2')
		#ax.set_aspect('equal')
		#cbar.ax.tick_params(labelsize=8) 

		#ax=plt.subplot(2,3,4)
		#_,cbar= visualize2D(ax,mesh_hf.x,
			           #mesh_hf.y,
			           #OFN2_sb[:,:],'vertical',[0,1])
		#setAxisLabel(ax,'p')
		#ax.set_title('True '+'N2')
		#ax.set_aspect('equal')
		#cbar.ax.tick_params(labelsize=8) 
        
		ax=plt.subplot(5,3,10)
		_,cbar=visualize2D(ax,mesh_lf.x,
			           		  mesh_lf.y,
			             input[0,3,:,:].cpu().detach().numpy(),'vertical', [0,0.012])
			           		   
		setAxisLabel(ax,'p')
		ax.set_title('InputO2')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8)
        
		ax=plt.subplot(5,3,11)
		_,cbar= visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           outputO2[0,0,:,:].cpu().detach().numpy(),'vertical',[0,0.012])
		setAxisLabel(ax,'p')
		ax.set_title('S_R '+'O2')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8) 

		ax=plt.subplot(5,3,12)
		_,cbar= visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           OFO2_sb[:,:],'vertical',[0,0.012])
		setAxisLabel(ax,'p')
		ax.set_title('True '+'O2')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8) 
        
		ax=plt.subplot(5,3,13)
		_,cbar=visualize2D(ax,mesh_lf.x,
			           		  mesh_lf.y,
			             input[0,4,:,:].cpu().detach().numpy(),'vertical', [0,0.013])
			           		   
		setAxisLabel(ax,'p')
		ax.set_title('InputH2O')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8)

		ax=plt.subplot(5,3,14)
		_,cbar= visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           outputH2O[0,0,:,:].cpu().detach().numpy(),'vertical',[0,0.013])
		setAxisLabel(ax,'p')
		ax.set_title('S_R '+'H2O')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8) 

		ax=plt.subplot(5,3,15)
		_,cbar= visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           OFH2O_sb[:,:],'vertical',[0,0.013])
		setAxisLabel(ax,'p')
		ax.set_title('True '+'H2O')
		ax.set_aspect('equal')
		cbar.ax.tick_params(labelsize=8) 


		fig0.tight_layout(pad=1)
		fig0.savefig(str(epoch)+'Transport.pdf',bbox_inches='tight')
		plt.close(fig0)

	return (xRes/len(training_data_loader)), (yRes/len(training_data_loader)),\
		   (mRes/len(training_data_loader)),(eU/len(training_data_loader)),\
		   (eV/len(training_data_loader)), (eP/len(training_data_loader)),\
		   (eH2/len(training_data_loader)), (eO2/len(training_data_loader)),(eH2O/len(training_data_loader)),(eN2/len(training_data_loader)),model.source.detach().cpu().numpy(),einfer.item()


	#return (xRes/len(training_data_loader)), (yRes/len(training_data_loader)),\
		   #(mRes/len(training_data_loader)),(eU/len(training_data_loader)),\
		   #(eV/len(training_data_loader)), (eP/len(training_data_loader)),\
		    #model.source.detach().cpu().numpy(),einfer.item()
			

XRes=[];YRes=[];MRes=[];CRes=[]; SpRes=[]
EU=[];EV=[];EP=[];EH2=[];EO2=[];EN2=[];EH2O=[]
Iinlet=[]
TotalstartTime=time.time()
EINFER=[]
for epoch in range(1,nEpochs+1):
	tic=time.time()
	xres,yres,mres,cres, eu,ev,ep,eH2,eO2,eN2,eH2O,einferr=train(epoch) #einfer=train(epoch) #,infer
	print('Time of this epoch=',time.time()-tic)
	EINFER.append(einferr)
	XRes.append(xres)
	YRes.append(yres)
	MRes.append(mres)
	CRes.append(cres)
	SpRes.append(SpRes)
	EU.append(eu)
	EV.append(ev)
	EP.append(ep)
	EH2.append(eH2)
	EO2.append(eO2)
	EN2.append(eN2)
	EH2O.append(eH2O)
	#Iinlet.append(infer)
	if einferr<0.01:
		break
TimeSpent=time.time()-TotalstartTime

plt.figure()
plt.plot(XRes,'-o',label='X-momentum Residual')
plt.plot(YRes,'-x',label='Y-momentum Residual')
plt.plot(MRes,'-*',label='Continuity Residual')
plt.plot(CRes,'-.',label='Transport Residual')
#plt.plot(SpRes,'-+',label='Species Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('convergence.pdf',bbox_inches='tight')
tikzplotlib.save('convergence.tikz')

plt.figure()
plt.plot(EU,'-o',label=r'$u$')
plt.plot(EV,'-x',label=r'$v$')
plt.plot(EP,'-*',label=r'$p$')

plt.plot(EH2,'-*',label=r'$H2$')
plt.plot(EO2,'-*',label=r'$O2$')
plt.plot(EN2,'-*',label=r'$N2$')
plt.plot(EH2O,'-*',label=r'$H2O$')

plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('error.pdf',bbox_inches='tight')
tikzplotlib.save('error.tikz')
EU=np.asarray(EU)
EV=np.asarray(EV)
EP=np.asarray(EP)

EH2=np.asarray(EH2)
EO2=np.asarray(EO2)
EN2=np.asarray(EN2)
EH2O=np.asarray(EH2O)

XRes=np.asarray(XRes)
YRes=np.asarray(YRes)
MRes=np.asarray(MRes)
CRes=np.asarray(CRes)
Iinlet=np.asarray(Iinlet)
np.savetxt('EU.txt',EU)
np.savetxt('EV.txt',EV)
np.savetxt('EP.txt',EP)
np.savetxt('EH2.txt',EH2)
np.savetxt('EO2.txt',EO2)
np.savetxt('EN2.txt',EN2)
np.savetxt('EH2O.txt',EH2O)
np.savetxt('Iinlet.txt',Iinlet.squeeze())
np.savetxt('XRes.txt',XRes)
np.savetxt('YRes.txt',YRes)
np.savetxt('MRes.txt',MRes)
np.savetxt('CRes.txt',CRes)
np.savetxt('TimeSpent.txt',np.zeros([2,2])+TimeSpent)
np.savetxt('EINFER.txt',np.asarray(EINFER))

