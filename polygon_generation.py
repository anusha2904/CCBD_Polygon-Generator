import math, random
from PIL import Image,ImageDraw
import time
import os 

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


f=open('wktfile.csv','w')


#GENERATING A SINGLE IRREGULAR MAP LIKE POLYGON WITH 10-500 VERTICES USING A BASE SHAPE AND ADDING IRREGULARITY 

def generatePolygon( ctrX, ctrY, aveRadius, base_shape_vertices, total_vertices ) :
	
	total_vertices  = total_vertices - base_shape_vertices
	vertices_counter = total_vertices
	vertices_array = []
	base_shape_randomness = 0
	main_vertice_flag = 0
	rand_to_Add = 0

	#Creating a base regular polygon shape

	for i in range(base_shape_vertices):
		
		main_vertice_distance = (random.uniform(1.09 * aveRadius, 0.9 * aveRadius))
		base_shape_randomness = main_vertice_distance / aveRadius

		if base_shape_randomness<1:
			tag = 1
		else: 
			tag = 0

		#introducing irregularity in the larger polygon shape by changing 1 or more vertices of the regular polygon

		x_cord = ctrX + main_vertice_distance * math.cos(2 * i * math.pi / base_shape_vertices)
		y_cord = ctrY + main_vertice_distance * math.sin(2 * i * math.pi / base_shape_vertices)
		main_vertice_flag = 1
		vertices_array.append((x_cord,y_cord))

		#distributing the total vertices randomly to each side of the polygon 
		
		if (i == base_shape_vertices - 1):
			edge_vertices = vertices_counter
		
		else:
			edge_vertices = int((total_vertices/base_shape_vertices) + random.uniform(-0.4, 0.4) * (total_vertices / base_shape_vertices))

		vertices_counter -= edge_vertices
		
		#adding irregularity and randomness to vertices between base shape vertices 

		for j in range(int(edge_vertices / 2)):

			s_angle = (2 * i * math.pi / base_shape_vertices) + (2 * j * math.pi / (base_shape_vertices * edge_vertices))
			distance = (aveRadius * math.cos(math.pi / (base_shape_vertices))) / (math.cos((math.pi / (base_shape_vertices)) - (2 * j * math.pi / (base_shape_vertices * edge_vertices))))
			
			random_distance = (random.uniform(1.03 * distance, 0.97 * distance))
			
			if  (j<20)& (tag == 0):
				rand_to_Add = ((0.6-(j/100))*base_shape_randomness*distance)/5 	
			else:
				rand_to_Add = 0

			if  (j<20)& (tag == 1):
				rand_to_Add = (((j/100))*base_shape_randomness*distance)	
			else:
				rand_to_Add = 0

			x = (ctrX + random_distance * math.cos(s_angle) + rand_to_Add * math.cos(s_angle)) 
			y = (ctrY + random_distance * math.sin(s_angle) + rand_to_Add * math.sin(s_angle)) 
			vertices_array.append((x,y))

		for j in range(int(edge_vertices / 2), edge_vertices):

			k = int(j - edge_vertices / 2)
			s_angle = (2 * i * math.pi / base_shape_vertices) + (2 * j * math.pi / (base_shape_vertices * edge_vertices))
			distance = (aveRadius * math.cos(math.pi / (base_shape_vertices))) / (math.cos(2 * k *math.pi / (base_shape_vertices * edge_vertices)))
			random_distance = (random.uniform(1.03 * distance, 0.97 * distance))
			
			if (j>(edge_vertices - 100))& (tag == 0):
				rand_to_Add = (((j/100)) * base_shape_randomness * distance)	
			else:
				rand_to_Add = 0

			if (j>(edge_vertices - 100))& (tag == 1):
				rand_to_Add = ((0.6-(j/100))*base_shape_randomness*distance)/5
			else:
				rand_to_Add = 0
			
			x = (ctrX + random_distance * math.cos(s_angle) + rand_to_Add * math.cos(s_angle)) 
			y = (ctrY + random_distance * math.sin(s_angle) + rand_to_Add * math.sin(s_angle))
			vertices_array.append((x,y))
	
	return vertices_array

#call generate polygon function to get the array of vertices' coordinates 

f.write('wkt;\n');
timer_start=time.time()

major = []
placement = 0

for i in range(10):
        sides=random.randint(3, 9)
        nvert=random.randint(10, 501)
        radius=random.randint(100, 250)
        l = generatePolygon(250+placement,250,radius,sides,nvert)
        l.append(l[0])
        major.append(l)
        placement = placement+radius+250
        

timer_end=time.time()-timer_start
print("time taken is: ",timer_end,"\n")

#writing it into a WKT file 
for pol in major:
	tupVerts = list(map(tuple,pol))
	listtostr=','.join([str(elem) for elem in tupVerts])
	listtostr1=''.join([listtostr[i] for i in range(len(listtostr)) if (listtostr[i]!='(' and listtostr[i]!=')') ])
	commacount=0
	listtostr2='POLYGON((';
	for i in range(len(listtostr1)):
		if (listtostr1[i]==','):
			if(commacount%2==0):
				listtostr2=listtostr2+''
			else:
				listtostr2=listtostr2+listtostr1[i]+' '
			commacount+=1
		else:
			listtostr2=listtostr2+listtostr1[i]
	listtostr2+='))'
	f.write(listtostr2)
	f.write('\n')
f.close()



#GENERATING A MAP USING VORONOI FUNCTION AND ADDING NOISY EDGES 

voronoi_input = [(250,250)]
vertices_for_voronoi = generatePolygon(250, 250, 200, 8, 100)
polygon = Polygon(vertices_for_voronoi)

#Getting random point to feed the voronoi function 

for i in range(180):
 	x = random.randint(90,440)
 	y = random.randint(90,440)
 	point = Point(x, y)
 	voronoi_input.append((x, y))
 	if polygon.contains(point):
 		voronoi_input.append((x, y))

fig, ax = plt.subplots(1)

def checkVertices(vertices):
    for couple in vertices:
        if (any(x < 0 or x > 256 for x in couple)):
            return False
    return True

#Adding noise to edges for shapes that are similar to maps

def perp_point_u(pnt_1, pnt_2, n):
	
	m = 0.3 + random.uniform(0,1)
	X = pnt_2[0] + (m*(pnt_1[1] - pnt_2[1]))/n
	Y = pnt_2[1] - (m*(pnt_1[0] - pnt_2[0]))/n
	
	return [X,Y]

def perp_point_d(pnt_1, pnt_2, n):
	
	m = 0.3 + random.uniform(0,1)
	X = pnt_2[0] - (m*(pnt_1[1] - pnt_2[1]))/n
	Y = pnt_2[1] + (m*(pnt_1[0] - pnt_2[0]))/n

	return [X,Y]

def dist(p_1, p_2):
	
	x1 = p_1[0]
	x2 = p_2[0]
	y1 = p_1[1]
	y2 = p_2[1]
	result= ((((x2 - x1)**2) + ((y2 - y1)**2))**0.5)
	
	return result 

def noisy_edge(vertices):
	
	copy_vertices = []
	copy_vertices = vertices.copy(); 
	index_counter = 0

	for i in range(len(copy_vertices)):
		
		if(i == len(copy_vertices) - 1):
			ged = 0
		else:
			ged = i+1

		n = random.randint(40,80)
		vertices_in_between = []
		
		for j in range(1, n):
			x = (((n-j) * copy_vertices[i][0] + j * copy_vertices[ged][0]) / n)
			y = (((n-j) * copy_vertices[i][1] + j * copy_vertices[ged][1]) / n)
			vertices_in_between.append([x,y])
	
		index_counter_2 = 1
		section = 0
		
		while (index_counter_2 < n/4):
			
			up_down = random.randint(0, 2)
			temp = []
			
			for h in range(section, section+4):
				temp.append(vertices_in_between[section])
			
			section = section+4
			random_vertice = random.choice(temp) 
			
			if(up_down == 1):
				perp = perp_point_u(copy_vertices[i], random_vertice, dist(copy_vertices[i], random_vertice))
			else:
				perp = perp_point_d(copy_vertices[i], random_vertice, dist(copy_vertices[i], random_vertice))
			
			vertices.insert(index_counter + index_counter_2, perp)
			index_counter_2 += 1
		
		index_counter += index_counter_2
			
	return vertices


from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib 
vor = Voronoi(voronoi_input)

N = 3
nfloors = np.random.rand(N) 

cmap = plt.get_cmap('terrain')
colors = cmap(nfloors) 

patches = []

#Drawing each of the polygons obtained from voronoi function

for region in vor.regions:
    if (region != []):
        vertices = vor.vertices[region]
        if (checkVertices(vertices)):
        	polygon = Polygon(noisy_edge(vertices.tolist()), closed=True)
        	patches.append(polygon)



collection = PatchCollection(patches, cmap=matplotlib.cm.jet)
b = os.path.getsize("wktfile.csv")
print("size of file : ",b," bytes\n")

ax.add_collection(collection)

#plotting with color 

cmap=plt.get_cmap('gist_earth_r')
colors=cmap(nfloors)
collection.set_color(colors)
collection.set_edgecolor('#4285f4')
collection.set_clim([3, 500])

ax.autoscale_view()
plt.axis('off')
fig.patch.set_facecolor('#4285f4')

fig.savefig('saved_figure.png')
plt.show()




from sklearn.cluster import KMeans
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76




image = cv2.imread('saved_figure.png')
#plt.imshow(image)





image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.imshow(image)





resized_image = cv2.resize(image, (1200, 600))
#plt.imshow(resized_image)




def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))



def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



def get_colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i]/255 for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]*255) for i in counts.keys()]
    rgb_colors = [ordered_colors[i]*255 for i in counts.keys()]
    
    if (show_chart):
        plt.figure(figsize = (8, 8))
        plt.pie(counts.values(), colors = ordered_colors,autopct='%1.1f%%',labeldistance=1.1,pctdistance=0.9,wedgeprops = {"edgecolor" : "black",
                      'linewidth': 1,
                      'antialiased': True})
        labels = hex_colors
        plt.legend(labels, loc = "upper right")
        plt.title('The distribution on the map is :')
        plt.savefig('piechart.png')
        plt.show()
    return rgb_colors



get_colors(get_image('saved_figure.png'), 4, True)


