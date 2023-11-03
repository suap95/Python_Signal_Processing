import numpy as np
import functools

class Rectangle:
	'''
	Plane geometry class that defines a 2D rectangular plane in 3D space. Rays can be tested for intersection with planes, and normals can be calculated at any point on the sphere. This class is intended to be used as a target for rays, and so results are given based on the assumption that inner products w.r.t. normals must be negative in order for "hits" to register. It is also assumed that everything comes in proper numpy format.
	'''

	def __init__(self, center, azimuth, coelevation, height, width):
		'''
		constructor that builds an imaging plane out of a center point, an azimuth, a coelevation, a height (y-tangent), and a width (x-tangent)
		
		INPUT:
			center: center point of the plane 
			azimuth: azimuth in x-y plane (x axis = 0) in radians
			coelevation: coelevation from z axis in radians
			height: length along local y-axis in meters
			width: length along local x-axis in meters
		
		'''
		self.center = center
		self.azimuth = azimuth
		self.coelevation = coelevation
		self.height = height
		self.width = width

		#plane normal
		self.normal = np.array([
			np.sin(coelevation)*np.cos(azimuth), 
			np.sin(coelevation)*np.sin(azimuth), 
			np.cos(coelevation)])

		#plane's basis vectors (tangent to the plane)
		self.tany = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
		self.tanx = np.cross(self.tany, self.normal) 
		self.basis = np.zeros((2,3))
		self.basis[0,:] = self.tanx
		self.basis[1,:] = self.tany

	@functools.lru_cache(maxsize=128)
	def hit(self, ray):
		'''
		function that determines whether a ray hits a plane instance
		
		INPUT:
			ray: a ray object
			
		OUTPUT:
			intersects: boolean w/ True value if ray intersects sphere
			intersection: point of intersection in 3D space
			t: distance between ray source and point of intersection
			normal: plane's normal
		'''
		intersects = False
		intersection = np.ones(3)*np.inf
		t = np.inf
		normal = self.normal
		
		l0 = ray.source
		l = ray.direction
		p0 = self.center
		
		#discriminant

		if np.abs(l.dot(normal)) > 1e-15: #if ray NOT ortho. to normal
			t = (p0-l0).dot(normal) / l.dot(normal)
			if (t < 0) or (l.dot(normal) >= 0):
				#if the ray would have to point in the opposite direction or 
				#it hits the wrong side of the object 
				t = np.inf
			else:
				intersection = l0 + l*t
				#compute where on the plane's 2D surface the intersection lies
				xy = np.abs(self.basis.dot(intersection-self.center))
				if(xy[0] > self.width/2 or xy[1] > self.height/2):
					#the point doesn't lie within the plane bounds
					t = np.inf
					intersection = np.ones(3)*np.inf
				else:
					#t is positive, the ray hits the correct side of the plane
					#and the intersection point lies within the plane bounds
					intersects = True
				
		return (intersects, intersection, t, normal)
			
