import sys
import numpy as np
import caffe
import argparse
import cv2
#from tqdm import tqdm
import os
from collections import OrderedDict
import subprocess
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image

name1='cam2.txt'
class ImageHelper:
    def __init__(self, S, L, means):
        self.S = S
        self.L = L
        self.means = means
    
    def prepare_image_and_grid_regions_for_network(self, fname, roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        I, im_resized = self.load_and_prepare_image(fname, roi)
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = []
            all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
            R = self.pack_regions_for_network(all_regions)
        return I, R

    def get_rmac_features(self, I, R, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end='rmac/normalized')
        return np.squeeze(net.blobs['rmac/normalized'].data)

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S)/np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means
        
        return I, im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        
        
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:             
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R

    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1
		
        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])
        
        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)
def extract_features(image_helper, net, args):
    
    fin = open(name1, 'r')
    fs1=[]
    for line in fin.readlines():
        line = line.strip()
        fs1.append(line)
    fin.close()
    Ss = [args.S, ] if not args.multires else [args.S - 250, args.S, args.S + 250]
    # First part, queries
    for S in Ss:
        # Set the scale of the image helper
        image_helper.S = S
        out_queries_fname = "queries.npy"
        
        dim_features = net.blobs['rmac/normalized'].data.shape[1]
        N_queries = len(fs1)
        features_queries = np.zeros((N_queries, dim_features), dtype=np.float32)
		#for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        for i in range(N_queries):
            print ('extract features %d'%i)
			# Load image, process image, get image regions, feed into the network, get descriptor, and store
            I, R = image_helper.prepare_image_and_grid_regions_for_network(fs1[i], roi=None)
            features_queries[i] = image_helper.get_rmac_features(I, R, net)
        np.save(out_queries_fname, features_queries)
    features_queries = np.load("queries.npy")    
    
    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]
    
    # Restore the original scale
    image_helper.S = args.S
    return features_queries

def ListFilesToTxt(dir,file,ends):
	files = os.listdir(dir)
	ends=ends.split(' ')
	for name in files:
		for end in ends:
			if name.endswith(end):
				file.write(os.path.basename(dir)+'/'+name+"\n")
def Test():
	dir=name1[0:4]
	outfile=name1
	files = open(outfile,"w+")
	ends=".jpg .png .bmp .jpeg .JPG"
	ListFilesToTxt(dir,files,ends)
	files.close()
# def rotate_image(imgp):
        # fd=open(imgp)
        # img=Image.open(imgp)
        # if img.size[1]>480:
            # width=480
            # height=int(img.size[0]*(float(480)/img.size[1]))
            # img=img.resize((height,width))
        # tags=exifread.process_file(fd) 
        # if tags.has_key("Image Orientation"):
            # if str(tags["Image Orientation"])=='Rotated 90 CW' :
                # img=img.transpose(Image.ROTATE_270)
            # if str(tags["Image Orientation"])=='Rotated 180':
                # img=img.transpose(Image.ROTATE_180)
        # line=imgp.split('/')
        # print line[1]
        # imgp1='cam3'+'/'+line[1]
        # img.save(imgp1)
        # return imgp1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Oxford / Paris')
    parser.add_argument('--gpu', type=int, required=False,default=0, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--S', type=int, required=False, default=800,help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--L', type=int, required=False, default=2,help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--eval_binary', type=str, required=False, help='Path to the compute_ap binary to evaluate Oxford / Paris')
    parser.add_argument('--multires', dest='multires', action='store_true', help='Enable multiresolution features')
    parser.add_argument('--aqe', type=int, required=False, help='Average query expansion with k neighbors')
    parser.add_argument('--dbe', type=int, required=False,help='Database expansion with k neighbors')
    parser.set_defaults(multires=False)
    args = parser.parse_args()
    
   

    # Load and reshape the means to subtract to the inputs
    args.means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]

    # Configure caffe and load the network
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net('model/deploy_resnet101_normpython.prototxt', 'model/model.caffemodel', caffe.TEST)
    #net = caffe.Net('model/deploy_resnet101_my.prototxt', 'model/model.caffemodel', caffe.TEST)

    # Load the dataset and the image helper
    #dataset = Dataset(args.dataset, args.eval_binary)
    Test()
    image_helper = ImageHelper(args.S, args.L, args.means)

    # Extract features
    features_queries = extract_features(image_helper, net, args)
	
    # Compute similarity
	
    # sim=np.zeros((features_queries.shape[0],features_queries.shape[0]))
    # for i in range(features_queries.shape[0]):
        # a=features_queries[i].reshape(1,-1)

        # b=euclidean_distances(a,features_queries)
        # sim[i]=b   
    sim = features_queries.dot(features_queries.T)
    if args.aqe is not None and args.aqe > 0:
        idx = np.argsort(sim, axis=1)[:, ::-1]
        features_queries = np.vstack([np.vstack((features_queries[i], features_queries[idx[i, :args.aqe]])).mean(axis=0) for i in range(len(features_queries))])
        #for i in range(features_queries.shape[0]):
        #    features_queries[i] = np.vstack((features_queries[i], features_dataset[idx[i, :args.aqe]])).mean(axis=0)
        sim = features_queries.dot(features_queries.T)
    fin1 = open(name1, 'r')
    fs1=[]
    for line1 in fin1.readlines():
        line1=line1.strip()
        fs1.append(line1)
        fin1.close()
        m=len(fs1)
  
    while True:  
		D=np.argsort(sim,axis=1)[:,::-1]
		index=raw_input('input a number between 0 and %d:' %(m-1))
		index=int(index)
		result_id1=D[index][0]
		result_id2=D[index][1]
		result_id3=D[index][2]
		result_id4=D[index][3]
		print sim[index][result_id1],sim[index][result_id2],sim[index][result_id3],sim[index][result_id4]
		
		#query_img=cv2.imread(fs1[index])
		result_img1=cv2.imread(fs1[result_id1])
		result_img2=cv2.imread(fs1[result_id2])
		result_img3=cv2.imread(fs1[result_id3])
		result_img4=cv2.imread(fs1[result_id4])
		
		#cv2.imwrite(str(index)+'.jpg',query_img)
		cv2.imwrite(str(result_id1)+'first.jpg',result_img1)
		cv2.imwrite(str(result_id2)+'second.jpg',result_img2)
		cv2.imwrite(str(result_id3)+'third.jpg',result_img3)
		cv2.imwrite(str(result_id4)+'fourth.jpg',result_img4)
		# Score
		#dataset.score(sim, args.temp_dir, args.eval_binary)
