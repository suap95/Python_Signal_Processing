import numpy as np
import numpy.linalg as nlin

class Sphere:
	'''
	Sphere geometry class for ray-based imaging. This class allows the definition of a sphere object. Rays can be tested for intersection with spheres, and normals can be calculated at any point on the sphere. This class is intended to be used as a target for rays, and so results are given based on the assumption that inner products w.r.t. normals must be negative in order for "hits" to register. It is also assumed that everything comes in proper numpy format.
	'''

	def __init__(self, center, radius):
		'''
		constructor with center and radius of sphere
		'''
		self.center = center
		self.radius = radius


	def hit(self, ray):
		'''
		function that determines whether a ray hits a sphere instance

		INPUT:
			ray: a ray object

		OUTPUT:
			intersects: boolean w/ True value if ray intersects sphere
			intersection: point of intersection in 3D space
			t: distance between ray source and point of intersection
			normal: sphere's normal at point of intersection

		'''
		d = ray.direction
		v = ray.source - self.center #temporary vector for simplicity
		#find the two possible intersection points
		intersects = False #this flag tracks ray impact on sphere
		normal = np.zeros(3) #generic normal
		intersection = np.ones(3)*np.inf #intersection point
		t = np.inf #distance from ray source

		#discriminant of intersection calculation
		discriminant = v.dot(d)**2 - (v.dot(v) - self.radius**2)
		#discriminant = d.dot(v) #** 2  # - (v.dot(v) - self.radius**2)
	
		if(discriminant > 0): #if real root
			t = np.zeros(2) #ray length parameter
			t[0] = -v.dot(d) + np.sqrt(discriminant)
			t[1] = -v.dot(d) - np.sqrt(discriminant)
			t[t < 0] = np.inf #set negative values to infinity
			t = np.amin(t) #keep the smallest value (first inters.)
			
			#if a valid intersection point is found:
			if(t < np.inf):
				intersects = True
				intersection = ray.source + t*d
				normal = intersection - self.center
				normal = normal/nlin.norm(normal) #normalize
		return (intersects, intersection, t, normal)

