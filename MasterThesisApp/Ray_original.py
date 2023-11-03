import numpy.linalg as nlin


class Ray:
    '''
    Ray class for ray-based imaging. This class allows the definition of a ray object. Rays are defined by a source coordinate in 3D space and a direction, i.e. by 2 vectors. It is assumed that all vectors and quantities come in proper numpy format.
    '''

    def __init__(self, source, direction):
        '''
        constructor with source and direction of ray
        '''

        self.source = source  # ray source
        # normalized ray direction

        self.direction = direction / nlin.norm(direction)