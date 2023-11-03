import numpy as np
import numpy.linalg as nlin
import numpy.matlib as nmat
from Ray_original import Ray

class Plane:
	'''
	Scene class to define a rectangular transducer plane from which rays are cast into space. One ray is cast per pixel. The imaging plane is defined by a focal point that lies behind the plane, a straight line distance from the focal point to the center of the imaging plane, an azimuth angle, a coelevation angle, vertical and horizontal resolutions, and vertical and horizontal number of pixels.
	IMPORTANT: I've assumed that the azimuth lies on the x-y plane, with x to the right, y coming out of the screen, and z pointing down (this is how one would see it when taking measurements). Equivalently, x can point to the right and y can point into the screen, meaning z points up. The coelevation is measured from the z axis and ranges from 0 to 180°.
	Additionally, offsets for the pixel centers are chosen so that they make more sense with an odd number of pixels in each axis. Lastly, some functions are left over from an optics implementation in which separate light source planes can be defined.
	'''

	def __init__(self, center, distance, azimuth, coelevation, vres, hres, nv, nh, opening):
		'''
		constructor that builds an imaging plane out of a center point, a distance from it to the center of the imaging plane, an azimuth, a coelevation, horizontal resolution, vertical resolution, vertical number of pixels (num rows), and horizontal number of pixels (num cols)
		
		INPUT:
			center: center point of the plane 
			distance: straight line distance from focus to plane center
			azimuth: azimuth in x-y plane (x axis = 0) in radians
			coelevation: coelevation from z axis in radians
			vres: vertical resolution in meters
			hres: horizontal resolution in meters
			nv: number of vertical pixels (i.e. number of rows in plane)
			nh: number of horizontal pixels (i.e. number of columns in plane)
			opening: transducer opening angle in radians
		
		'''
		self.center = center
		self.distance = distance
		self.azimuth = azimuth
		self.coelevation = coelevation
		self.hres = hres
		self.vres = vres
		self.nh = nh
		self.nv = nv
		self.opening = opening

		#focus of the imaging plane, placed behind the plane's center
		self.focus = center - np.array([
			distance*np.sin(coelevation)*np.cos(azimuth), 
			distance*np.sin(coelevation)*np.sin(azimuth), 
			distance*np.cos(coelevation)])

		#imaging plane's normal vector
		self.normal = self.center - self.focus
		self.normal = self.normal/distance

		#imaging plane's basis vectors (tangent to the plane)
		self.tany = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
		self.tanx = np.cross(self.tany, self.normal) 


	def prepareImagingPlane(self):
		'''
		this function prepares the imaging plane by calculating the center of each pixel (called the throughPoint here) and using that information to create a ray for each pixel. it uses only information given in the constructor. the reason it was split into a separate function is that a light sink imaging plane and a light source plane are essentially the same, but these extra rays aren't needed for the light source plane.

		INPUT:
			none

		OUTPUT:
			none

		'''

		self.image = np.zeros((self.nv, self.nh))

		#imaging plane's center coordinate per pixel, ordered from upper left corner [0,0] to lower right corner [nv-1, nh-1]
		vertOffset = (self.nv-1)/2*self.vres*self.tany
		horzOffset = -(self.nh-1)/2*self.hres*self.tanx
		origin00 = self.center + vertOffset + horzOffset

		#ordered as row, col, x, y, z
		self.throughPoints = np.zeros((self.nv, self.nh, 3))
		for row in range(self.nv):
			#pixel center adjusted for this row
			rowpix = origin00 - row*self.vres*self.tany
			for col in range(self.nh):
				#final pixel center
				pix = rowpix + col*self.hres*self.tanx
				self.throughPoints[row,col,:] = pix

		#create rays that go from the source to the center of each pixel
		self.rays = np.array([ [ Ray(self.throughPoints[row, col, :], self.throughPoints[row, col, :] - self.focus) for col in range(self.nh) ] for row in range(self.nv) ])

		#create another set of through points from the perspective of the focal point; this will be useful later on when exploiting homogeneous coords
		#note that the z axis is negative to preserve right-handedness
		v_axis = self.vres*(np.arange(self.nv) - (self.nv-1)/2)[::-1]/(-self.distance)
		h_axis = self.hres*(np.arange(self.nh) - (self.nh-1)/2)/(-self.distance)
		self.homogeneous = np.zeros((self.nv, self.nh, 3))
		self.homogeneous[:,:,0] = nmat.repmat(h_axis, self.nv, 1)#x coord
		self.homogeneous[:,:,1] = nmat.repmat(v_axis[:,np.newaxis], 1, self.nh)#y coord
		self.homogeneous[:,:,2] = 1 
		
		#compute the directivity: since the rays and their directions are precomputed already, the directivities can also be computed and stored. This follows from the directivity function depending only on the azimuth and coelevation, and not from the actual coordinate at which the rays intersect with objects. Furthermore, the calculation is much easier using homogeneous coordinates.
		#the transducer radiating point is now at (0,0,0), while the center of the imaging plane through which the rays pass is at (0,0,1). The directivity then depends only on the coordinates of the through-points in homogeneous coordinates.
		numerator = np.power(self.homogeneous[:,:,0],2) + np.power(self.homogeneous[:,:,1],2)
		self.directivity = np.exp(-numerator/np.tan(self.opening)**2)
		print('a')


	def setLightPattern(self, lightPattern):
		'''
		this function takes an nv x nh light pattern and saves it

		INPUT:
			lightPattern: an nv x nh matrix that will be used as a light pattern

		OUTPUT:
			returns nothing, only saves.

		'''
		self.lightPattern = lightPattern



	def lighting(self, point, normal, objects):
		'''
		function that finds the lighting that this plane causes at a point in space. the point is used to find a direction vector. we take this vector and check whether there are any hits on any object with a smaller distance than the original point. if there are no hits, there's line of sight and we can use the corresponding point on the lightPattern and the normal to calculate the lighting.
	
		INPUTS:
			point: the point in space which we're tracing rays to
			normal: the object's normal at the point of interest
			objects: list of objects to test for hits

		OUTPUT:
			light: the illumination value that will be seen from the imaging plane

		'''

		direction = point - self.focus #ray direction
		dist = nlin.norm(self.focus - point) #distance to target point
		ray = Ray(self.focus, direction) #light source ray
		light = 0

		#only test if the ray points against the normal, i.e. if the illumination ray hits the point without going through the object
		if(direction.dot(normal) < 0):
			objDist = np.inf
			#find the distance to each object
			for obj in objects:
				(hits, intersection, t, nor) = obj.hit(ray)
				if(hits and t < objDist):
					objDist = t

			if(objDist < dist): #if a closer object was found
				return light

			#if we've made it to this point, no objects are in the way and the source illuminates the correct side of the object. we have to find which pixel of the lightPattern plane the ray is going through

			#first, find the length of the light ray from the source to the light pattern plane
			l = (self.center - ray.source).dot(self.normal)/ray.direction.dot(self.normal)
			intersect = ray.source + l*ray.direction

			#this point is given by a linear combination of a z-pointing vector and the tangent vector. these two are orthogonal and can be used to form perform a change of basis.
			basis = np.zeros((2,3))
			basis[0,:] = self.tanx #x component
			basis[1,:] = self.tany #y component taken in the neg direction due to how values are stored in matrices
			coords = basis.dot(intersect)
			col = int(coords[0]/self.hres + (self.nh-1)/2)
			row = int(coords[1]/self.vres + (self.nv-1)/2)

			if (col < 0 or row < 0 or col >= self.nh or row >= self.nv):
				return light

			light = self.lightPattern[row,col] * np.abs(direction.dot(normal)/nlin.norm(direction))

		return light



	def cast(self, objects, lightSource):
		'''
		function that casts rays on a set of objects that is passed as a parameter. one ray is cast per pixel of the plane, and for each pixel, the first intersected object is found.

		INPUT:
			objects: np array of objects with a hit function that returns the point of intersection and the normal at that point
			lightSource: object that defines the lighting in the scene. this is also a Plane object, it's just used differently.

		OUTPUT:
			returns nothing, but the hit map used to calculate and store the lighting per pixel.

		'''

		#for each pixel in the imaging plane, cast a ray and find which object it hits. then, find the lighting.
		for i in range(self.nv): #row
			for j in range(self.nh): #col
				tmin = np.inf #intersection distance
				for k in range(len(objects)): #iterate over objects
					obj = objects[k] #the current object
					(hits, intersection, t, normal) = obj.hit(self.rays[i,j])
					if(hits and t < tmin):
						tmin = t
						#find lighting; send as objects list a list with all objects except the current one
						self.image[i,j] = lightSource.lighting(intersection, normal, np.delete(objects, k))
				
						
	
	def insonify(self, scenario):
		'''
		This function simulates the insonification of a medium defined by a list of objects. The insonification is carried out by a rectangular transducer with a 2D Gaussian directivity. 
		
		INPUT:
			scenario: object containing all the simulation parameters
			
		OUTPUT:
			Ascan: A-scan of length NT
		
		'''
		self.image = np.zeros((self.nv, self.nh))
		self.tof = np.ones((self.nv, self.nh))*np.inf
		self.directions = np.zeros((self.nv,self.nh,3))
		self.source = np.zeros((self.nv, self.nh, 3))
		
		objects = scenario.objects
		#check if each ray hits an object and keep the smallest hit distance
		for i in range(self.nv):
			for j in range(self.nh):
				ray = self.rays[i,j]
				for obj in objects:
					(hits, p, t, n) = obj.hit(ray)
					if(hits and t < self.tof[i,j]):
						self.tof[i,j] = t
						self.image[i,j] = -ray.direction.dot(n)
						self.directions[i,j,:] = ray.direction
						self.source[i,j,:] = ray.source

						
		#convert from one way distance to round trip time
		self.tof = 2*self.tof/scenario.c
		#apply directivity
		self.image = np.multiply(self.image, self.directivity)
		
		#reference time vector
		time = np.arange(scenario.NT)/scenario.fs
		#all contributions will be added up onto the Ascan vector, mimicking a discrete integral over the transducer surface
		Ascan = np.zeros((scenario.NT))
		#Ascan1 = np.zeros((scenario.NT))
		#Ascan2 = np.zeros((scenario.NT))
		for i in range(self.nv):
			for j in range(self.nh):
				if self.tof[i,j] < np.inf:
					t_axis = time - self.tof[i,j]
					carrier = np.cos(2*np.pi*scenario.fc*t_axis + scenario.phi)
					#envelope = np.exp(-scenario.bw_factor*np.power(t_axis,2))
					envelope = 2/((np.sqrt(3*0.5))*np.pi**0.008)*(1-scenario.bw_factor*((t_axis)/0.5)**2)*np.exp(-scenario.bw_factor*(t_axis)**2/(2*0.5**2))
					sig = np.multiply(carrier, envelope)*self.image[i,j]
					#sig = envelope * self.image[i, j]
					Ascan = Ascan + sig

		#set the maximum possible amplitude to 1 by dividing by the number of rays
		Ascan = Ascan/(self.nh + self.nv)
		return Ascan#,self.directivity,self.image,self.throughPoints,self.directions,self.source,self.focus