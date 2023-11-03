import numpy as np
import numpy.linalg as nlin
import sys


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
        d = d.reshape((len(ray.direction) * len(ray.direction),3))
        v = ray.source - self.center  # temporary vector for simplicity
        v1 = v.reshape((3,1))
        # find the two possible intersection points
        intersects = False  # this flag tracks ray impact on sphere
        normal = np.zeros(3)  # generic normal
        intersection = np.ones(3) * np.inf  # intersection point
        t = np.inf  # distance from ray source
        intersects1 = np.zeros(len(ray.direction) * len(ray.direction))
        intersection = np.zeros((len(ray.direction) * len(ray.direction),3))
        # discriminant of intersection calculation
        discriminant = d.dot(v1) ** 2 - (v.dot(v) - self.radius ** 2)
        # discriminant = d.dot(v) #** 2  # - (v.dot(v) - self.radius**2)
        real_root_indices = np.where(discriminant > 0)
        real_root_indices1 = np.array(real_root_indices[0])
        size = real_root_indices[0].size
        t1 = np.zeros((2,len(ray.direction) * len(ray.direction),1))
        t1[0,real_root_indices1[:],:] = -d[real_root_indices1[:],:].dot(v1) + np.sqrt(discriminant[real_root_indices1[:]])
        t1[1,real_root_indices1[:],:] = -d[real_root_indices1[:],:].dot(v1) - np.sqrt(discriminant[real_root_indices1[:]])

        indices1 = np.where(t1[0, :, :] < 0)
        indices2 = np.where(t1[1, :, :] < 0)
        if indices1[0]:
            t1[0, indices1[0], :] = np.inf
        if indices2[0]:
            t1[1, indices2[0], :] = np.inf

        min_values = np.amin(t1,axis=0)
        min_values = min_values.reshape((len(ray.direction) * len(ray.direction),1))
        intersection[real_root_indices1[:],:] = ray.source + min_values[real_root_indices1[:]] * d[real_root_indices1[:],:]
        normal = intersection - self.center
        norm_normal = nlin.norm(normal,axis=1)
        norm_normal = norm_normal.reshape(len(ray.direction)*len(ray.direction),1)
        normal = normal / norm_normal

        intersects1[real_root_indices1[:]] = True

        #min_values2 = np.where(t1[0,:,:] < t1[1,:,:])
        #t1[t1 < 0] = np.inf
        #t1 = np.amin(t1)
        #t2 = np.where(t1 < np.inf)
        '''
        if (discriminant > 0):  # if real root
            t = np.zeros(2)  # ray length parameter
            t[0] = -v.dot(d) + np.sqrt(discriminant)
            t[1] = -v.dot(d) - np.sqrt(discriminant)
            t[t < 0] = np.inf  # set negative values to infinity
            t = np.amin(t)  # keep the smallest value (first inters.)

            # if a valid intersection point is found:
            if (t < np.inf):
                intersects = True
                intersection = ray.source + t * d
                normal = intersection - self.center
                normal = normal / nlin.norm(normal)  # normalize
        '''
        return (intersects1, intersection, min_values, normal)

