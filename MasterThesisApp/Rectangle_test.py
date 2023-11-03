import numpy as np
import itertools
from timeit import default_timer as timer
import numexpr as ne

class Rectangle_test:
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

        # plane normal
        self.normal = np.array([
            np.sin(coelevation) * np.cos(azimuth),
            np.sin(coelevation) * np.sin(azimuth),
            np.cos(coelevation)])

        # plane's basis vectors (tangent to the plane)
        self.tany = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
        self.tanx = np.cross(self.tany, self.normal)
        self.basis = np.zeros((2, 3))
        self.basis[0, :] = self.tanx
        self.basis[1, :] = self.tany

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
        #intersects = np.zeros((1,181,181))
        #intersection1 = []
        s = timer()
        intersects1 = np.zeros(len(ray.direction)*len(ray.direction))
        #intersection = np.ones((3,181,181)) * np.inf
        #tof = np.ones((1,181,181)) * np.inf
        #t = np.inf
        l1 = np.arange(len(ray.direction))
        l2 = np.arange(len(ray.direction))

        normal = self.normal
        normal = normal.reshape((3,1))
        #start = timer()
        #results_test = np.array([ray.source for i, j in itertools.product(l1, l2)])
        temp = ray.source
        temp = temp.reshape((1,3))
        results_test = np.repeat(temp,len(ray.direction)*len(ray.direction),axis=0)
        #results_test = np.repeat(results_test,181,axis=0)
        #results_test1 = np.array([ray.direction[i, j] for i, j in itertools.product(l1, l2)])
        results_test1 = ray.direction[:, :]
        results_test1 = results_test1.reshape((len(ray.direction)*len(ray.direction),3))
        e = timer()
        #results_test = np.array(results_test)
        #results_test1 = np.array(results_test1)
        #results_test2 = results_test1.dot(normal)
        #gemm = get_blas_funcs("gemm",[results_test1, normal])
        temp_res = results_test1.dot(normal)
        results_test3 = ne.evaluate('abs(temp_res)')
        #results_test3 = abs(results_test1.dot(normal))
        #g = gemm(1, results_test1, normal)
        #results_test3 = abs(g)
        results_test5 = np.where(results_test3 > 1e-15)
        indices = results_test5[0]
        #sorted_sources = results_test[indices[:],:]
        #sorted_directions = results_test1[indices[:],:]
        cen = self.center
        cen = cen.reshape((1,3))
        #sorted_directions1 = sorted_directions.dot(normal)
        res_temp = results_test[indices[:], :]
        cen1 = ne.evaluate('cen - res_temp')
        #tof_s = (cen - results_test[indices[:],:]).dot(normal)/results_test1[indices[:],:].dot(normal)
        tof_s = (cen1.dot(normal)/results_test1[indices[:], :].dot(normal))
        tof_index = np.where(tof_s < 0)
        #results_test2_index = np.where(results_test1[indices[:],:].dot(normal) >= 0)
        common_index = np.array(np.where(np.logical_or(tof_index, np.where(results_test1[indices[:],:].dot(normal) >= 0)))[1])
        #common_index = np.array(common_index[1])
        if np.any(common_index):
            tof_s[common_index[:]] = np.inf

        intersection1 = ne.evaluate('results_test + results_test1*tof_s')
        xy1 = self.basis.dot((intersection1 - cen).transpose())
        xy1 = ne.evaluate('abs(xy1)')
        e1 = timer()
        #xy1 = np.abs(self.basis.dot((results_test + results_test1*tof_s - cen).transpose()))
        #out_index = np.where((xy1[0,:] > self.width / 2) | (xy1[1,:] > self.height / 2))
        w = self.width / 2
        h = self.height / 2
        #w1 = xy1[0, :]
        #h1 = xy1[1, :]
        #out_index_w = np.where(w1 > w)
        #out_index_h = np.where(h1 > h)
        #common_index1 = np.logical_or(out_index_w, out_index_h)
        #tof_s[out_index_w[:]] = np.inf
        tof_s[np.where(xy1[0, :] > w)[:]] = np.inf
        tof_s[np.where(xy1[1, :] > h)[:]] = np.inf
        intersection1[np.where(xy1[0, :] > w)[:],:] = np.ones(3) * np.inf

        #in_index_w = np.where(w1 < w)
        #in_index_h = np.where(h1 < h)
        common_index2 = np.intersect1d(np.where(xy1[0, :] < w), np.where(xy1[1, :] < h))
        intersects1[common_index2[:]] = True
        #end = timer()
        #print(end-start)
        e2 = timer()
        #print('Inside hit : '+str(e2-e1)+str(e1-e))
        return (intersects1, intersection1, tof_s, normal)

