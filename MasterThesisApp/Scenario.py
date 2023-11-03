
class Scenario:
	'''
	Class containing all of the simulation parameters. This way, they only have to be defined once, and the transducer can be redefined independently. It contains a list of objects with a "hit" function, the number of samples, sampling frequency, and the pulse parameters (for a Gaussian pulse shape).
	'''
	
	def __init__(self, objects, c, NT, fs, bw_factor, fc, phi,no_symbols,symbol_time):
		'''
		prepare the scenario by storing all physical parameters
		
		INPUT:
			objects: np array of objects with a "hit" function that determines if the object is intersected by a given ray
			c: speed of sound in medium
			NT: number of time domain samples
			fs: sampling frequency in Hz
			bw_factor: Gabor function bandwidth factor in Hz^2
			fc: center frequency of Gabor function in Hz
			phi: phase of the pulse shape in radians
		'''
		
		self.objects = objects
		self.c = c
		self.NT = NT
		self.fs = fs
		self.bw_factor = bw_factor
		self.fc = fc
		self.phi = phi
		self.no_symbols = no_symbols
		self.symbol_time = symbol_time