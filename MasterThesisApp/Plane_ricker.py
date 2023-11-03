import numpy as np
import itertools
import numpy.linalg as nlin
import numpy.matlib as nmat
from Ray import Ray
# from numba import jit,float32,vectorize
from timeit import default_timer as timer
import math
import functools
# import numba
import numexpr as ne
from scipy.linalg import get_blas_funcs
import os

os.environ['NUMEXPR_MAX_THREADS'] = '20'
os.environ['NUMEXPR_NUM_THREADS'] = '10'
os.environ["MKL_NUM_THREADS"] = "1"


class Plane_ricker:
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

        # focus of the imaging plane, placed behind the plane's center
        self.focus = center - np.array([
            distance * np.sin(coelevation) * np.cos(azimuth),
            distance * np.sin(coelevation) * np.sin(azimuth),
            distance * np.cos(coelevation)])

        # imaging plane's normal vector
        self.normal = self.center - self.focus
        self.normal = self.normal / distance

        # imaging plane's basis vectors (tangent to the plane)
        self.tany = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
        self.tanx = np.cross(self.tany, self.normal)

    @functools.lru_cache(maxsize=90000)
    def prepareImagingPlane(self):
        '''
        this function prepares the imaging plane by calculating the center of each pixel (called the throughPoint here) and using that information to create a ray for each pixel. it uses only information given in the constructor. the reason it was split into a separate function is that a light sink imaging plane and a light source plane are essentially the same, but these extra rays aren't needed for the light source plane.

        INPUT:
            none

        OUTPUT:
            none

        '''

        self.image = np.zeros((self.nv, self.nh))

        # imaging plane's center coordinate per pixel, ordered from upper left corner [0,0] to lower right corner [nv-1, nh-1]

        vertOffset = (self.nv - 1) / 2 * self.vres * self.tany
        horzOffset = -(self.nh - 1) / 2 * self.hres * self.tanx
        origin00 = self.center + vertOffset + horzOffset
        # start = timer()
        # ordered as row, col, x, y, z
        self.throughPoints = np.zeros((self.nv, self.nh, 3))

        for row in range(self.nv):
            # pixel center adjusted for this row
            rowpix = origin00 - row * self.vres * self.tany
            for col in range(self.nh):
                # final pixel center
                # pix = rowpix + col*self.hres*self.tanx
                pix = rowpix + col * self.hres * self.tanx
                self.throughPoints[row, col, :] = pix

        '''
        for row,col in itertools.product(self.rows,self.cols):
            pix = self.rowp[:,row] + col * self.hres * self.tanx
            self.throughPoints[row,col,:] = pix
        '''
        # row_temp = self.rowp.reshape((self.nv,1,3))
        # col_temp = self.hres*tanx*(col1)
        # col_temp = col_temp.reshape((1,self.nh,3))
        # self.throughPoints = row_temp + col_temp

        # end = timer()
        # print(end - start)
        # create rays that go from the source to the center of each pixel
        # self.rays = np.array([ [ Ray(self.focus, self.throughPoints[row, col, :] - self.focus) for col in range(self.nh) ] for row in range(self.nv) ])
        # temp_focus = self.focus
        temp_focus = self.focus.reshape((1, 1, 3))
        thr = self.throughPoints
        t1 = ne.evaluate('thr - temp_focus')
        self.rays = Ray(self.focus, t1)
        # start1 = timer()
        # create another set of through points from the perspective of the focal point; this will be useful later on when exploiting homogeneous coords
        # note that the z axis is negative to preserve right-handedness
        v_axis = self.vres * (np.arange(self.nv) - (self.nv - 1) / 2)[::-1] / (-self.distance)
        h_axis = self.hres * (np.arange(self.nh) - (self.nh - 1) / 2) / (-self.distance)
        self.homogeneous = np.zeros((self.nv, self.nh, 3))
        self.homogeneous[:, :, 0] = nmat.repmat(h_axis, self.nv, 1)  # x coord
        self.homogeneous[:, :, 1] = nmat.repmat(v_axis[:, np.newaxis], 1, self.nh)  # y coord
        self.homogeneous[:, :, 2] = 1

        # compute the directivity: since the rays and their directions are precomputed already, the directivities can also be computed and stored. This follows from the directivity function depending only on the azimuth and coelevation, and not from the actual coordinate at which the rays intersect with objects. Furthermore, the calculation is much easier using homogeneous coordinates.
        # the transducer radiating point is now at (0,0,0), while the center of the imaging plane through which the rays pass is at (0,0,1). The directivity then depends only on the coordinates of the through-points in homogeneous coordinates.
        # numerator = np.power(self.homogeneous[:,:,0],2) + np.power(self.homogeneous[:,:,1],2)
        numerator = np.square(self.homogeneous[:, :, 0]) + np.square(self.homogeneous[:, :, 1])
        # self.directivity = np.exp(-numerator/np.tan(self.opening)**2)
        # end1 = timer()
        self.directivity = np.exp(-numerator / np.square(np.tan(self.opening)))

    # end2 = timer()
    # print(end1-start1)

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

        direction = point - self.focus  # ray direction
        dist = nlin.norm(self.focus - point)  # distance to target point
        ray = Ray(self.focus, direction)  # light source ray
        light = 0

        # only test if the ray points against the normal, i.e. if the illumination ray hits the point without going through the object
        if (direction.dot(normal) < 0):
            objDist = np.inf
            # find the distance to each object
            for obj in objects:
                (hits, intersection, t, nor) = obj.hit(ray)
                if (hits and t < objDist):
                    objDist = t

            if (objDist < dist):  # if a closer object was found
                return light

            # if we've made it to this point, no objects are in the way and the source illuminates the correct side of the object. we have to find which pixel of the lightPattern plane the ray is going through

            # first, find the length of the light ray from the source to the light pattern plane
            l = (self.center - ray.source).dot(self.normal) / ray.direction.dot(self.normal)
            intersect = ray.source + l * ray.direction

            # this point is given by a linear combination of a z-pointing vector and the tangent vector. these two are orthogonal and can be used to form perform a change of basis.
            basis = np.zeros((2, 3))
            basis[0, :] = self.tanx  # x component
            basis[1, :] = self.tany  # y component taken in the neg direction due to how values are stored in matrices
            coords = basis.dot(intersect)
            col = int(coords[0] / self.hres + (self.nh - 1) / 2)
            row = int(coords[1] / self.vres + (self.nv - 1) / 2)

            if (col < 0 or row < 0 or col >= self.nh or row >= self.nv):
                return light

            light = self.lightPattern[row, col] * np.abs(direction.dot(normal) / nlin.norm(direction))

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

        # for each pixel in the imaging plane, cast a ray and find which object it hits. then, find the lighting.
        for i in range(self.nv):  # row
            for j in range(self.nh):  # col
                tmin = np.inf  # intersection distance
                for k in range(len(objects)):  # iterate over objects
                    obj = objects[k]  # the current object
                    (hits, intersection, t, normal) = obj.hit(self.rays[i, j])
                    if (hits and t < tmin):
                        tmin = t
                        # find lighting; send as objects list a list with all objects except the current one
                        self.image[i, j] = lightSource.lighting(intersection, normal, np.delete(objects, k))

    @functools.lru_cache(maxsize=8000)
    def insonify(self, scenario):
        '''
        This function simulates the insonification of a medium defined by a list of objects. The insonification is carried out by a rectangular transducer with a 2D Gaussian directivity.

        INPUT:
            scenario: object containing all the simulation parameters

        OUTPUT:
            Ascan: A-scan of length NT

        '''
        self.image = np.zeros((self.nv, self.nh))
        self.tof = np.ones((self.nv, self.nh)) * np.inf
        tof_temp = self.tof.reshape((self.nv * self.nh, 1))
        image_temp = self.image.reshape((self.nv * self.nh, 1))
        # l1 = np.arange(self.nv)
        # l2 = np.arange(self.nh)
        objects = scenario.objects
        # check if each ray hits an object and keep the smallest hit distance
        # directions = np.array([self.rays.direction[i,j] for i, j in itertools.product(l1, l2)])
        # directions = self.rays.direction[:,:]
        directions = self.rays.direction[:, :].reshape((self.nv * self.nh, 3))
        '''
        start = timer()
        hits1, p1, t1, n1 = objects[0].hit(self.rays)
        hits2, p2, t2, n2 = objects[1].hit(self.rays)
        end = timer()
        common_index = np.intersect1d(np.where(hits1 == 1), np.where(t1 < tof_temp))
        tof_temp[common_index[:]] = t1[common_index[:]]
        image_temp[common_index[:]] = (-1) * directions[common_index[:], :].dot(n1)
        common_index1 = np.intersect1d(np.where(hits2 == 1), np.where(t2 < tof_temp))
        tof_temp[common_index1[:]] = t2[common_index1[:]]
        image_temp[common_index1[:]] = (-1) * directions[common_index1[:], :].dot(n2)
        self.image = image_temp.reshape((self.nv, self.nh))
        self.tof = tof_temp.reshape((self.nv, self.nh))
        end1 = timer()
        print('Without Loop : '+str(end-start) + str(end1-end))
        '''

        # start1 = timer()
        for obj in objects:
            # s = timer()
            hits1, p1, t1, n1 = obj.hit(self.rays)
            # hits3 = np.where(hits1 == 1)
            # t3 = np.where(t1 < tof_temp)
            # s1 = timer()
            common_index = np.intersect1d(np.where(hits1 == 1), np.where(t1 < tof_temp))
            tof_temp[common_index[:]] = t1[common_index[:]]
            # temp_dir = (-1)*directions[common_index[:],:]
            # image_temp[common_index[:]] = ne.evaluate('temp_dir*n1')
            common_index = np.array(common_index)
            counts = common_index.size
            for i in range(counts):
                di = directions[common_index[i], :]
                nor = n1[common_index[i], :].transpose()
                image_temp[common_index[i]] = (-1) * di.dot(nor)
            # image_temp[common_index[:]] = (-1)*directions[common_index[:], :].dot(n1)
            self.image = image_temp.reshape((self.nv, self.nh))
            self.tof = tof_temp.reshape((self.nv, self.nh))
        # e = timer()
        # print('Inside objects loop : '+str(e-s1) + str(s1-s))
        # end2 = timer()
        # print('Objects :' + str(end2 - start1))
        # convert from one way distance to round trip time
        self.tof = 2 * self.tof / scenario.c
        # apply directivity
        # self.image = np.multiply(self.image, self.directivity)
        dire = self.directivity
        ima = self.image
        self.image = ne.evaluate('ima*dire')

        # self.image = self.opt_multiply(self.image, self.directivity)
        # reference time vector
        time = np.arange(scenario.NT) / scenario.fs
        # all contributions will be added up onto the Ascan vector, mimicking a discrete integral over the transducer surface
        # Ascan = np.zeros((scenario.NT))

        tof_less_inf = np.where(self.tof[:] < np.inf)
        # tof_less_inf = ne.evaluate('where(self.tof[:] < np.inf)')
        tof_less_inf1 = tof_less_inf[0]
        tof_less_inf2 = tof_less_inf[1]
        # set the maximum possible amplitude to 1 by dividing by the number of rays
        # temp_tof = self.tof[tof_less_inf1[:], tof_less_inf2[:]]
        temp_tof1 = self.tof[tof_less_inf1[:], tof_less_inf2[:]].reshape((1, len(tof_less_inf1[:] * len(tof_less_inf2[:]))))
        # temp_image = self.image[tof_less_inf1[:], tof_less_inf2[:]]
        temp_image1 = self.image[tof_less_inf1[:], tof_less_inf2[:]].reshape((1, len(tof_less_inf1[:] * len(tof_less_inf2[:]))))
        time = time.reshape((scenario.NT, 1))
        # end = timer()
        # z = self.opt_subtract(time,temp_tof1)
        # z2 = opt1_subtract(time, temp_tof1)
        # start = timer()
        # z = time - temp_tof1
        # start5 = timer()
        # z1 = ne.evaluate('time - temp_tof1')
        # start4 = timer()
        # print(start5-start,start4-start5)
        # carrier = np.cos(2 * np.pi * scenario.fc * (time - temp_tof1) + scenario.phi)
        fc1 = scenario.fc
        phi1 = scenario.phi
        pi = np.pi
        # carrier = 2 * np.pi * scenario.fc * z1 + scenario.phi
        # carrier = ne.evaluate('2*pi*fc1*z1+phi1')
        # start1 = timer()
        # print(start1-start)
        # carrier = self.ident_parallel(2 * np.pi * scenario.fc * z + scenario.phi)
        # z2 = np.square(z1)
        # z2 = ne.evaluate('z1**2')
        # temp = np.multiply(-scenario.bw_factor, np.square(z1))
        bw = scenario.bw_factor
        some_const = 0.008
        sigma = 0.5
        amp = 2 / ((np.sqrt(3 * sigma)) * np.pi ** some_const)
        # temp = ne.evaluate('bw*z2')
        start2 = timer()
        # print(start2 - start)
        # envelope = np.exp(temp)
        # envelope = amp * (1 - bw * ((time - temp_tof1) / sigma) ** 2) * np.exp(-bw * (time - temp_tof1) ** 2 / (2 * sigma ** 2))
        # envelope2 = ne.evaluate('cos(2*pi*fc1*(time - temp_tof1)+phi1)*exp(bw*(time - temp_tof1)**2)*temp_image1')
        envelope2 = ne.evaluate('cos(2*pi*fc1*(time - temp_tof1)+phi1)*amp*(1 - bw * ((time - temp_tof1) / sigma) ** 2) * exp(-bw * (time - temp_tof1) ** 2 / (2 * sigma ** 2))*temp_image1')
        # envelope1 = ne.evaluate('')
        start3 = timer()
        # print(start3-start2)
        # temp3 = np.multiply(envelope2, envelope1)
        # temp1 = np.multiply(envelope2, envelope1) * temp_image1
        # temp1 = ne.evaluate('envelope2*temp_image1')
        # print(start3-start2)
        return ne.evaluate('sum(envelope2,1)') / (self.nh + self.nv)
    # some_val = np.multiply(carrier,envelope1)
    # start4 = timer()
    # print(start4-start3,start3-start2,start2-start1,start1-start)
    # z1 = self.opt_multiply(carrier,enevelope)
    # z1 = opt1_multiply(carrier, enevelope, temp_image1)
    # return np.sum(np.multiply(np.cos(2 * np.pi * scenario.fc * np.subtract(time,temp_tof1) + scenario.phi), np.exp(np.multiply(-scenario.bw_factor,np.square(np.subtract(time,temp_tof1))))) * temp_image1,axis=1)/(self.nh + self.nv)
    # return np.sum(np.multiply(envelope2, envelope1) * temp_image1, axis=1) / (self.nh + self.nv)
    # return s/(self.nh + self.nv)
    # return np.sum(z1*temp_image1, axis=1) / (self.nh + self.nv)
    # Ascan = np.sum(np.cos(2 * np.pi * scenario.fc * (time - temp_tof1) + scenario.phi)*np.exp(-scenario.bw_factor * np.square((time - temp_tof1))) * temp_image1,axis=1) / (self.nh + self.nv)
    # end1 = timer()
    # print('Carrier :' + str(end1-end))
    # Ascan = sig#/(self.nh + self.nv)
    # return Ascan

