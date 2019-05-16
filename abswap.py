import numpy as np 
import maxflow
import cv2
import random
import math

def smoothness(alpha,beta,int1 , int2):
	K = 20
	return ((alpha-beta)!=0)*((abs(int1-int2)>=5)*K + (abs(int1-int2)<5)*2*K) 

def is_out_of_bounds(x,y,xmax,ymax):
	if (x<0 or y<0 or x>=xmax or y>=ymax):
		return True
	else:
		return False		

def linear_interpolate(i, j, xmax, ymax, im):
	f = math.floor(j)
	c = math.ceil(j)
	diff = j - f
	if(not (is_out_of_bounds(i,f,xmax,ymax)) and not(is_out_of_bounds(i,c,xmax, ymax))): 
		intensity = diff * im[i, c] + (1 - diff) * im[i, f]
	elif(not(is_out_of_bounds(i,f,xmax,ymax)) and is_out_of_bounds(i,c,xmax, ymax)):
		intensity = im[i, f]
	else:
		intensity = im[i, c]

	return intensity

def data_cost_fwd(i,j,d,xmax,ymax,im1,im2):
	start = j + d - 0.5
	end = j + d + 0.5
	minimum = 10000
	while(start <= end):
		if(not(is_out_of_bounds(i,start,xmax,ymax))):
			m = abs(im1[i,j] - linear_interpolate(i,start,xmax,ymax,im2))
			if(minimum > m):
				minimum = m
		start += 0.1
	return minimum

def data_cost_rev(i,j,d,xmax,ymax,im1,im2):
	if(is_out_of_bounds(i,j+d,xmax,ymax)):
		return 20

	start = j - 0.5
	end = j + 0.5
	minimum = 10000

	while(start <= end):
		if(not(is_out_of_bounds(i,start,xmax,ymax))):
			m = abs(linear_interpolate(i,start, xmax, ymax, im1) - im2[i, j+d])
			if(minimum > m):
				minimum = m
		start += 0.1
	
	return minimum

def data_cost(i,j,d,img1,img2):
	rows,cols = img1.shape
	m1 = data_cost_fwd(i,j,d,rows,cols,img1,img2)
	m2 = data_cost_rev(i,j,d,rows,cols,img1,img2)
	m3 = 20
	kk = min(min(m1,m2),m3)
	return kk**2		

# def data_cost(i,j,d,img1,img2):
# 	rows,cols = img1.shape
# 	if (is_out_of_bounds(i,j-d,rows,cols)):
# 		return 20
# 	else :
# 		return min(abs(img1[i,j]-img2[i,j-d]),20)

def data_cost_precompute(img1,img2,dmax):
	rows,cols = img1.shape
	cost_mat = np.zeros((rows,cols,dmax+1))
	for i in range(rows):
		for j in range(cols):
			for k in range(dmax+1):
				cost_mat[i,j,k] = data_cost(i,j,k,img1,img2)
	return cost_mat			

def estimate_energy(labels,cost_mat,img2):
	rows,cols = labels.shape

	energy = 0 
	for i in range(rows):
		for j in range(cols):
			energy += cost_mat[i,j,int(labels[i,j])]
			if (i>=1):
				energy += smoothness(labels[i,j] , labels[i-1,j], img2[i,j], img2[i-1,j])
			if (j>=1):
				energy += smoothness(labels[i,j] , labels[i,j-1], img2[i,j], img2[i-1,j])
			# if (i<=rows-2):
			# 	energy += smoothness(labels[i,j] , labels[i+1,j])		# avoid repeat
			# if (j<=cols-2):	
			# 	energy += smoothness(labels[i,j] , labels[i,j+1])

	return energy
	
def estimate_labels(labels, g, nodes, cut_edges,pixels,node_loc,alpha ,beta):
	C = len(cut_edges)
	new_labels = labels.copy()

	# for i in range(C):
	# 	if (cut_edges[i][0]==pixels or cut_edges[i][1]==pixels):
	# 		if (cut_edges[i][0]==pixels):
	# 			idx = cut_edges[i][1]
	# 		else:
	# 			idx = cut_edges[i][0]
	# 		x,y = node_loc[idx]
	# 		new_labels[x,y] = alpha
	# 	elif (cut_edges[i][0]==pixels+1 or cut_edges[i][1]==pixels+1):
	# 		if (cut_edges[i][0]==pixels+1):
	# 			idx = cut_edges[i][1]
	# 		else:
	# 			idx = cut_edges[i][0]
	# 		x,y = node_loc[idx]
	# 		new_labels[x,y] = beta	

	# return new_labels	

	for i in range(len(nodes)):
		x,y = node_loc[i]
		# print (nodes[i])
		if (g.get_segment(nodes[i])==0):
			
			new_labels[x,y] = beta
		else :
			new_labels[x,y] = alpha 

	return new_labels		




def t_link_cost(idx,pix_disp,labels,node_loc,cost_mat,t_link_type,alpha,beta,img2):
	rows,cols = labels.shape
	x,y = node_loc[idx]

	cost = cost_mat[x,y,pix_disp]

	if (t_link_type == True):
		ref_label = alpha
	else:
		ref_label = beta

	if (x>=1 ):
		if (labels[x-1][y]!=alpha and labels[x-1][y]!=beta):
			cost += smoothness(ref_label, labels[x-1][y], img2[x,y], img2[x-1,y])
		

	if (x<=rows-2):
		if (labels[x+1][y]!=alpha and labels[x+1][y]!=beta):
			cost += smoothness(ref_label, labels[x+1][y], img2[x,y], img2[x+1,y])


	if (y>=1 ):
		if (labels[x][y-1]!=alpha and labels[x][y-1]!=beta):
			cost += smoothness(ref_label, labels[x][y-1], img2[x,y], img2[x,y-1])
		
	

	if (y<=cols-2):
		if (labels[x][y+1]!=alpha and labels[x][y+1]!=beta):
			cost += smoothness(ref_label, labels[x][y+1], img2[x,y], img2[x,y+1])
	

	return cost		 




def make_graph(labels,cost_mat,alpha,beta,img2):

	rows,cols = labels.shape
	node_loc = []
	loc_to_idx = {}
	n_nodes = 0
	for i in range(rows):
		for j in range(cols):
			if (labels[i,j]==alpha or labels[i,j]==beta):
				node_loc.append((i,j))
				loc_to_idx[(i,j)] = n_nodes
				n_nodes+=1

	

	pixels = n_nodes

	print (pixels)
	# print (node_loc)
	# print (loc_to_idx)


	g = maxflow.Graph[float](pixels,4*pixels)
	nodes = g.add_nodes(pixels)
	

	n_nodes +=2 

	# adj_mat = np.zeros((n_nodes,n_nodes))

	for i in range(pixels):
		x,y = node_loc[i]
		# sm = smoothness(alpha,beta)

		if (x>=1):
			if ((x-1,y) in loc_to_idx.keys()):
				nbr = loc_to_idx[(x-1,y)]
				# adj_mat[i,nbr] = adj_mat[nbr,i]= sm
				sm = smoothness(alpha,beta,img2[x,y], img2[x-1,y])
				g.add_edge(nodes[i],nodes[nbr],sm,sm)
				# print (i, ' ', nbr, ' ', sm)

		if (y>=1):
			if ((x,y-1) in loc_to_idx.keys()):
				nbr = loc_to_idx[(x,y-1)]
				# adj_mat[i,nbr] = adj_mat[nbr,i]= sm
				sm = smoothness(alpha,beta,img2[x,y], img2[x,y-1])
				g.add_edge(nodes[i],nodes[nbr],sm,sm)
				# print (i, ' ', nbr, ' ', sm)
		'''
		if (x<=rows-2):										#avoid repeat
			if ((x+1,y) in loc_to_idx.keys()):
				nbr = loc_to_idx[(x+1,y)]
				# adj_mat[i,nbr] = adj_mat[nbr,i]= sm
				g.add_edge(nodes[i],nodes[nbr],sm,sm)
				
		if (y<=cols-2):
			if ((x,y+1) in loc_to_idx.keys()):
				nbr = loc_to_idx[(x,y+1)]
				# adj_mat[i,nbr] = adj_mat[nbr,i]= sm
				g.add_edge(nodes[i],nodes[nbr],sm,sm)
		'''		

		# adj_mat[i,pixels] = adj_mat[pixels,i] = t_link_cost(i,labels[x,y],labels,node_loc,cost_mat,True)
		# adj_mat[i,pixels+1] = adj_mat[pixels+1,i] = t_link_cost(i,labels[x,y],labels,node_loc,cost_mat,False)

		t1 = t_link_cost(i,alpha,labels,node_loc,cost_mat,True,alpha,beta,img2)
		t2 = t_link_cost(i,beta,labels,node_loc,cost_mat,False,alpha,beta,img2)

		g.add_tedge(nodes[i],t1,t2)

	return (pixels,node_loc,loc_to_idx,g,nodes)

def boykovMincut(pixels,node_loc,loc_to_idx,g,nodes):
	rows,cols = img1.shape
	
	flow = g.maxflow()
	cut_edges = []

	for i in range(pixels):
		x,y = node_loc[i];
		
		node_label = g.get_segment(nodes[i])
		if (node_label == 0):
			cut_edges.append((i,pixels+1))
		else :
			cut_edges.append((i,pixels))

		if (x>=1):
			if ((x-1,y) in loc_to_idx.keys()):
				nbr = loc_to_idx[(x-1,y)]
				nbr_label = g.get_segment(nodes[nbr])
				if (g.get_segment(nodes[i])!=g.get_segment(nodes[nbr])):
					cut_edges.append((i,nbr))
			

		if (x<=rows-2):
			if ((x+1,y) in loc_to_idx.keys()):
				nbr = loc_to_idx[(x+1,y)]
				nbr_label = g.get_segment(nodes[nbr])
				if (g.get_segment(nodes[i])!=g.get_segment(nodes[nbr])):
					cut_edges.append((i,nbr))

		if (y>=1):
			if ((x,y-1) in loc_to_idx.keys()):
				nbr = loc_to_idx[(x,y-1)]
				nbr_label = g.get_segment(nodes[nbr])
				if (g.get_segment(nodes[i])!=g.get_segment(nodes[nbr])):
					cut_edges.append((i,nbr))

		if (y<=cols-2):
			if ((x,y+1) in loc_to_idx.keys()):
				nbr = loc_to_idx[(x,y+1)]
				nbr_label = g.get_segment(nodes[nbr])
				if (g.get_segment(nodes[i])!=g.get_segment(nodes[nbr])):
					cut_edges.append((i,nbr))

	return cut_edges				



if __name__ == '__main__':

	# dmax = 14
	dmax = 14

	# read images
	img1 = cv2.imread("tsukuba/scene1.row3.col1.ppm",0).astype(int)
	img2 = cv2.imread("tsukuba/scene1.row3.col2.ppm",0).astype(int)
	# img1 = cv2.imread("sawtooth/im0.ppm",0).astype(int)
	# img2 = cv2.imread("sawtooth/im1.ppm",0).astype(int)

	labels = np.zeros(img1.shape,dtype= int)
	rows,cols = img1.shape
	for i in range(rows):
		for j in range(cols):
			labels[i,j] = random.randint(0,dmax)

	step = np.floor(255/dmax)
	label_img = step*labels
	cv2.imwrite('disp_res.png',label_img)

	

	cost_mat = data_cost_precompute(img2,img1,dmax)			#changed
	# np.save('cost_mat_0.1_saw.npy',cost_mat)
	# cost_mat = np.load('cost_mat_0.1.npy')
	print ('precomputing done')

	success = False
	old_energy = estimate_energy(labels,cost_mat,img2)

	iter = 0

	energy_arr = []

	while(success == False):
		print ('iteration: ',iter)
		for alpha in range(dmax+1):
			for beta in range(alpha+1,dmax+1):

				print('alpha: ',alpha, ' beta: ',beta)

				pixels,node_loc,loc_to_idx,g,nodes = make_graph(labels,cost_mat,alpha,beta,img2)
				print ('Graph constructed')
				cut_edges = boykovMincut(pixels,node_loc,loc_to_idx,g,nodes)
				print ('mincut found')
				# labels1 = estimate_labels(labels, cut_edges,pixels,node_loc,alpha ,beta)
				labels1 = estimate_labels(labels, g, nodes,cut_edges,pixels,node_loc,alpha ,beta)
				print ('labels estimated')
				new_energy = estimate_energy(labels1,cost_mat,img2)

				print (old_energy, ' ', new_energy)

				if (old_energy > new_energy):
					old_energy = new_energy
					labels = labels1.copy()
					success = True

				energy_arr.append(old_energy)	

				del node_loc
				del loc_to_idx
				del g
				del nodes 
				del pixels

		iter+=1
		np.save('iter_'+str(iter)+'.npy', labels)
				
	
		step = np.floor(255/dmax)
		label_img = step*labels
		cv2.imwrite('disp_res' +str(iter)+ '.png',label_img)		
					
		if (success== True):
			success = False
		else :
			break

	np.save('energy_graph_saw.npy',energy_arr)