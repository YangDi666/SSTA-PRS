import sys
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import csv
import argparse

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Selective Spatio-Temporal Aggregation')
    parser.add_argument(
        '--pose1',
        help='Pose detected by estimator 1')
    parser.add_argument(
        '--pose2',
        help='Pose detected by estimator 2')
    parser.add_argument(
        '--pose3',
        help='Pose detected by estimator 3')
    parser.add_argument(
        '--gt',
        default=None,
        help='Ground-truth pose if have')

    parser.add_argument(
        '--vis-conf',
        default=True)

    parser.add_argument('--outname', default='ssta.npz')
    return parser

class PRS():
	def __init__(self, arg):
		self.arg=arg
		self.get_poses()

	def get_poses(self):
		p1=np.load(self.arg.pose1)
		p2=np.load(self.arg.pose2)
		p3=np.load(self.arg.pose3)
		
		self.pose1=p1['kpts']
		self.pose2=p2['kpts']
		self.pose3=p3['kpts']
		self.gt=None
		nb_f=max(self.pose1.shape[0], self.pose2.shape[0], self.pose3.shape[0])

		if self.arg.gt:
			gt=np.load(self.arg.gt)
			self.gt=gt['kpts']
			nb_f=self.gt.shape[0]
		
		#print(lcr_2d.shape[0], alp_2d.shape[0], op_2d.shape[0], gt_2d.shape[0])
		# Pad 0 for frames with no pose detected
		self.pose1=np.concatenate((self.pose1, np.zeros((nb_f-self.pose1.shape[0],17,2))), axis=0)
		self.pose2=np.concatenate((self.pose2, np.zeros((nb_f-self.pose2.shape[0],17,2))), axis=0)
		self.pose3=np.concatenate((self.pose3, np.zeros((nb_f-self.pose3.shape[0],17,2))), axis=0)
		print('Loading poses...', self.pose1.shape[0],self.pose2.shape[0],self.pose3.shape[0] )
		
		

	def save_pose(self, kpts_out):
		
		out_name=self.arg.outname
		kpts_out = np.array(kpts_out).astype(np.float32)
		np.savez_compressed(out_name, kpts=kpts_out)
		print('kpts_out npz saved in ', out_name)

	def D(self, p1, p2):
		# point: *2
		
		return np.sqrt(((p1-p2)**2).sum(axis=0))
			
	def D_pose(self, pose1, pose2):
		# pose: 17*2
		
		return np.sqrt(((pose1-pose2)**2).sum(axis=1)).sum(axis=0)

	def D_video(self, pose1, pose2):
		# pose: 17*2

		return np.sqrt(((pose1-pose2)**2).sum(axis=1))
		# *17

	def MPJPE(self, pose, gt, nb_joints, indexLs, indexRs):
		# pose: 17*2
		return (self.D_pose(pose, gt)/nb_joints)/(self.D( gt[indexLs], gt[indexRs] )+0.00000001)
		
	def PJPE_video(self, poses, gt, nb_joints, nb_frames, indexLs, indexRs ):
		# poses: n*17*2	
		M=np.array([])
		for i in range(nb_frames):
			#print(D_video(poses[i], gt[i]), D(  gt[i, indexLs], gt[i, indexRs]  ))
			M=np.append(M, self.D_video(poses[i], gt[i])/(self.D(  gt[i, indexLs], gt[i, indexRs]  )+0.00000001))

		return 	M

	def Confidence(self, lcr_2d, op_2d, alp_2d, ssta_2d):
		#pose2d: 12*2
		D_avg = ( self.MPJPE(lcr_2d, ssta_2d, 12, 0, 1)+self.MPJPE(op_2d, ssta_2d, 12, 0, 1)+self.MPJPE(alp_2d, ssta_2d, 12, 0, 1) )/3
		return np.exp(-1*D_avg)
		#return 1-D_avg
		
	def SSTA(self, gamma=-100)	:	
		ssta=np.zeros((self.pose1.shape[0], 17, 2))
		mpjpe=np.zeros(self.pose1.shape[0])
		confidence=np.zeros(self.pose1.shape[0])	
		if self.pose1.shape[0]==0:
			return 'No', mpjpe, confidence	
				
		
		for i in range(max(self.pose1.shape[0], self.pose2.shape[0], self.pose3.shape[0])-1):
			if i==0:
				
				#pose1=np.where(lcr_2d[i], lcr_2d[i], op_2d[i])
				#ssta[i]=np.where(pose1[i], pose1[i], alp_2d[i])
				#ssta[i]=lcr_2d[i]
				
				for j in range(self.pose1.shape[1]):
					k={self.D(self.pose2[i,j], self.pose3[i,j]): self.pose2[i,j], self.D(self.pose2[i,j], self.pose1[i,j]): self.pose2[i,j], self.D(self.pose1[i,j], self.pose3[i,j]): self.pose3[i,j]}
					ssta[i, j, :]=k[min(k.keys())]
				
				confidence[i]=self.Confidence(self.pose2[i, 5:,:], self.pose3[i, 5:,:], self.pose1[i, 5:,:], ssta[i, 5:, :])
				if self.arg.gt:
					mpjpe[i]= self.MPJPE(ssta[i, 5:, :], self.gt[i, 5:, :], 12, 0, 1)
				if confidence[i]<gamma:
					confidence[i]=0
			else:
				if i<min(self.pose2.shape[0], self.pose1.shape[0], self.pose3.shape[0]):
					for j in range(self.pose2.shape[1]):
						k={self.D(self.pose2[i,j], ssta[i-1,j]): self.pose2[i,j], self.D(self.pose3[i,j], ssta[i-1,j]): self.pose3[i,j], self.D(self.pose1[i,j], ssta[i-1,j]): self.pose1[i,j]}
						ssta[i, j, :]=k[min(k.keys())]
					confidence[i]=self.Confidence(self.pose2[i, 5:,:], self.pose3[i, 5:,:], self.pose1[i, 5:,:], ssta[i, 5:, :])
					if self.arg.gt:
						mpjpe[i]= self.MPJPE(ssta[i, 5:, :], self.gt[i, 5:, :], 12, 0, 1)
					if confidence[i]<gamma:
						confidence[i]=0
				else:
					ssta[i]=np.where(self.pose2[i], self.pose2[i], self.pose3[i])
					#pose1=np.where(lcr_2d[i], lcr_2d[i], op_2d[i])
					confidences[i]=self.Confidence(self.pose2[i, 5:,:], self.pose3[i, 5:,:], self.pose1[i, 5:,:], ssta[i, 5:, :])
					if self.arg.gt:
						mpjpe[i]= self.MPJPE(ssta[i, 5:, :], self.gt[i, 5:, :], 12, 0, 1)
					if confidence[i]<gamma:
						confidence[i]=0
			
		return ssta, mpjpe, confidence

	def vis(self, mpjpes, confidences):
		fig1=plt.figure()
		ax=fig1.add_subplot(111)
		ax.set_title('MPJPE with Confidence(C)')
		ax.set_xlabel('Confidence(C)')
		ax.set_ylabel('MPJPE')
		#confidences[np.where(confidences==1)]=0
		#mpjpes[np.where(mpjpes==0)]=1

		color=np.where(confidences>0, 1, 0)
		color[0]=2
		#mpjpes=mpjpes[np.where(confidences<1)]
		#color=color[np.where(confidences<1)]
		I1=len(np.where(color[np.where( mpjpes
	<=3)]!=0)[0])
		print(I1)
		#I1=np.where(confidences>=0 and confidences<0.05)
		ax.scatter(confidences, mpjpes, c=color)
		plt.show()

	def run(self):
		mpjpes=np.array([])
		confidences=np.array([])
		
		pjpe_video_p1=np.array([])
		pjpe_video_p2=np.array([])
		pjpe_video_p3=np.array([])
		pjpe_video_ssta=np.array([])
		
		ssta2d, mpjpe, confidence=self.SSTA(gamma=0.06)
		print(ssta2d.shape[0])
		if ssta2d=='No':
			print(line[:-1]+' has no people detected.')
		else:
			mpjpes=np.append(mpjpes, mpjpe)
			confidences=np.append(confidences, confidence)
			self.save_pose(ssta2d)
			if self.arg.gt:
				pjpe_video_p1=np.append(pjpe_video_p1, self.PJPE_video(self.pose1[:,5:,:], self.gt[:,5:,:], 12, self.pose1.shape[0], 0,1 ))
				pjpe_video_p2=np.append(pjpe_video_p2, self.PJPE_video(self.pose2[:,5:,:], self.gt[:,5:,:], 12, self.pose2.shape[0], 0,1))
				pjpe_video_p3=np.append(pjpe_video_p3, self.PJPE_video(self.pose3[:,5:,:], self.gt[:,5:,:], 12, self.pose3.shape[0], 0,1))
				pjpe_video_ssta=np.append(pjpe_video_ssta, self.PJPE_video(ssta2d[:,5:,:], self.gt[:,5:,:], 12, ssta2d.shape[0], 0,1))
		
		
		#print('Confidences: ', confidences) 
		#print('Errs: ', mpjpes)
		if self.arg.gt:
			print('PCK_alp: ', len(np.where(pjpe_video_p1<=2.0)[0])/len(pjpe_video_p1))
			print('PCK_lcr: ', len(np.where(pjpe_video_p2<=2.0)[0])/len(pjpe_video_p2))
			print('PCK_op: ', len(np.where(pjpe_video_p3<=2.0)[0])/len(pjpe_video_p3))
			print('PCK_ssta: ', len(np.where(pjpe_video_ssta<=2.0)[0])/len(pjpe_video_ssta))
		
		print('Visualization of Confidences...')
		if self.arg.vis_conf:
			self.vis(mpjpes[0:-1], confidences[0:-1])	
		#print(alp_2d, op_2d)
		#print(mpjpes.shape)
		f = open('statistic.csv','w')
		writer = csv.writer(f)
		writer.writerow(('MPJPE', 'C'))
		for i in range(mpjpes.shape[0]-1):
			writer.writerow((mpjpes[i], confidences[i]))
		f.close()
	
if __name__ == "__main__":
	parser = get_parser()
	arg = parser.parse_args()	
	processor = PRS(arg)
	processor.run()
	
