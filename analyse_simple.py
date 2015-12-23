from warp import *
#from opmd_viewer import OpenPMDTimeSeries
from collections import defaultdict
from json import dumps,load
import warpplot_jlv as wp
SENTINEL  = float("inf")
###Exceptions###
def IndexIsEmpty(Exception):
	if not index:
		pass
		
def write(data1,data2,filename):
	f=open(filename,'w')
	m=len(data1)
	for i in range(0,m):
		f.write("{}	\t {}\n".format(data1[i],data2[i]))
	f.close()

class Fields:
	def __init__(self, filenum, dsets, gamma, timeSeries,instant):
		self.gamma 		= gamma
		if self.gamma == 1:
			
			self.ts     = timeSeries
			self.instant= instant
			print "You are requesting information at %gs" %(self.ts.t[instant])
		else:
			self.filenum	=	filenum
			self.dsets		=	dsets
			self.PD    		= wp.PlotData(self.dsets,self.filenum)
	
	def dim(self):
		return shape(self.getEz())
	
	def getEx(self):
		##By default, the axis is transposed
		if self.gamma == 1:
			Ex,infoEx = self.ts.get_field( t=self.ts.t[instant],  field='E', coord='x' )
		else:
			Ex =self.PD.fieldMap('Ex')
		return Ex
		
	def getEz(self):
		##By default, the axis is transposed
		if self.gamma == 1:
			Ez,infoEz = self.ts.get_field( t=self.ts.t[instant],  field='E', coord='z' )
		else:
			Ez =self.PD.fieldMap('Ez')
		return Ez
	
	def getEy(self):
		##By default, the axis is transposed
		if self.gamma == 1:
			Ey,infoEy = self.ts.get_field( t=self.ts.t[instant],  field='E', coord='y' )
		else:
			Ey =self.PD.fieldMap('Ey')
		return Ey
	
	def field_on_axis_Ex(self):
		m,l=shape(self.getEx())
		return array(self.getEx()[m/2,:]).flatten()
	
	def field_on_axis_Ez(self):
		m,l=shape(self.getEz())
		return array(self.getEz()[m/2,:]).flatten()
		
	def field_on_axis_Ey(self):
		m,l=shape(self.getEy())
		return array(self.getEy()[m/2,:]).flatten()
	
	def getAttribute(self,dataset='ne',attr=['x_lim','z_lim']):
		"""getting the box attributes, in x and in z
		have to make it more generic next time
		"""
		box_attr = []
		for i in attr: box_attr.append(self.PD.getAttribute(dataset,i))
		return box_attr
	
	def zValues(self):
		m,l = self.dim()
		if self.gamma == 1:
			Ez,info_Ez = self.ts.get_field( t=self.ts.t[instant],  field='E', coord='z' )
			zmin = info_Ez[0]
			zmax = info_Ez[1]
			
		else:
			zmin=self.getAttribute()[1][0]
			zmax=self.getAttribute()[1][1]		
		z   = linspace(zmin,zmax,l)
		return z
		
	def xValues(self):
		m,l = self.dim()
		if self.gamma == 1:
			Ez,info_Ez = self.ts.get_field( t=self.ts.t[instant],  field='E', coord='z' )
			xmin = info_Ez[2]
			xmax = info_Ez[3]
		else:
			xmin=self.getAttribute()[0][0]
			xmax=self.getAttribute()[0][1]
		x= linspace(xmin,xmax,m)
		return x
	
	def maxfield(self):
		"""
		return the position of max laser field starting from z>0
		"""
		z=self.zValues()
		window=max(z)-min(z)
		i_z= where(z>(0.8*window+min(z)))
		Ey = self.field_on_axis_Ey()
		new_z=take(z,i_z[0])
	
		new_Ey = take(Ey,i_z[0])
		for j in range(0,len(new_z)):
			if new_z[j]>0.:
				i=argmax(abs(new_Ey))
		return new_z[i],z

	def bucket (self):
		"""
		determining bucket by change of sign in ez field
		"""
		arg_z,z=self.maxfield()
		ii=where(z<arg_z)
		
		ez_filtered=savitzky_golay(self.field_on_axis_Ez(), 51, 3)
		#winon(0)
		#plg(ez_filtered/max(ez_filtered),z,color='red')
		#plg(self.field_on_axis_Ez()/max(self.field_on_axis_Ez()),z)
		ez=take(ez_filtered,ii[0])
		
		z=take(z,ii[0])
		root_zero=find_root(ez,z)	
		lrz=len(root_zero)
		k=-1
	        j=0

		buckets=[[] for i in xrange(lrz/2)]
		
		for i in range(lrz-1,-1,-1):

			if (j%2)==0:
				j=0
				k+=1
  			##we want the smaller number to be at the right
				i-=1
			else:
				i+=1
			buckets[k].append(root_zero[i])
			j+=1
		
		return buckets
	
	def bucket_drawing(self):
		buckets=self.bucket()
		buckets=array(buckets).flatten()
		zero_buckets=zeros(len(buckets))*5

		ppgeneric(zero_buckets,buckets,marker='\5',color='red')
		
class Particles:
	def __init__(self, filenum,dsets, gamma, timeSeries,instant,species,l_ebeam=0, file_ebeam = None):
		self.gamma 		= gamma
		self.l_ebeam = l_ebeam
		if self.gamma == 1:
			self.ts     = timeSeries
			self.instant= instant
			self.species=species
		else:
			self.filenum	=	filenum
			self.dsets		=	dsets
			#self.PD    		= wp.PlotData(self.dsets,self.filenum)
		if self.l_ebeam:
			tmp = PR.PR(file_ebeam)
			self.beamzstations = tmp.beamzstations
			self.tbarstations = tmp.tbarstations
			self.xrmsstations = tmp.xrmsstations
			self.xprmsstations = tmp.xprmsstations
			self.xemitnstations = tmp.xemitnstations
			self.ekstations = tmp.ekstations
			self.xbarstations =tmp.xbarstations
			self.xpbarstations = tmp.xpbarstations
			try:
				self.xxpstations   = tmp.xxpstations
				self.xs   = tmp.xs
				self.ys   = tmp.ys
				self.zs   = tmp.zs
				self.uxs   = tmp.uxs
				self.uys   = tmp.uys
				self.uzs   = tmp.uzs
				self.ws    = tmp.ws
			except KeyError:
				self.xxpstations  =zeros(len(self.beamzstations))
				self.xs   = 0.
				self.ys   = 0.
				self.zs   = 0.
				self.uxs   = 0.
				self.uys   = 0.
				self.uzs   = 0.
				self.ws    = 0.
		
	def getxbeam(self,index):
		if self.gamma == 1 : 
			x_beam = self.ts.get_particle(var_list=['x'], iteration=self.instant, species=self.species )[0]
			x_beam=array(x_beam)*1.e-6
		else:
				x_beam = self.dsets[self.filenum]['x'][0][:]
		try:
				x_beam=take(x_beam,index)
		except IndexIsEmpty:	
				x_beam=[]
				print "No particle above the gamma threshold is detected."
		return x_beam
		
	def getzbeam(self,index):
		if self.gamma == 1 : 
			z_beam = self.ts.get_particle(var_list=['z'], iteration=self.instant, species=self.species )[0]
			z_beam =array(z_beam)*1.e-6
				
		else:
			z_beam = self.dsets[self.filenum]['z'][0][:]
		try:
				z_beam=take(z_beam,index)
		except IndexIsEmpty:	
				z_beam=[]
				print "No particle above the gamma threshold is detected."	
		return z_beam
	
	def gettbeam(self,index):
		t_beam = self.dsets[self.filenum]['t'][0][:]
		try:
				t_beam=take(t_beam,index)
		except IndexIsEmpty:	
				t_beam=[]
				print "No particle above the gamma threshold is detected."	
		return t_beam
	
	def getgamma(self):
		if self.gamma ==1 :
			gamma_beam = sqrt(1.+(self.getuxbeam()**2+self.getuybeam()**2+self.getuzbeam()**2)/clight**2)	
		else:
			gamma_beam = self.dsets[self.filenum]['gamma'][0][:]
		return gamma_beam
		
	def getfilteredgamma(self,index):
		gamma_beam = self.getgamma()
		try:
				gamma_beam=take(gamma_beam,index)
		except IndexIsEmpty:	
				print "No particle above the gamma threshold is detected."	
		return gamma_beam
		
	def getwbeam(self,index):
		if self.gamma == 1 : 
			w_beam = self.ts.get_particle(var_list=['w'], iteration=self.instant, species=self.species )[0]
		else:
			w_beam = self.dsets[self.filenum]['w'][0][:]
		try:
				w_beam=take(w_beam,index)
		except IndexIsEmpty:	
				w_beam=[]
				print "No particle above the gamma threshold is detected."
		return w_beam
	
	def getuxbeam(self):
		if self.gamma == 1 : 
			ux_beam = self.ts.get_particle(var_list=['ux'], iteration=self.instant, species=self.species )[0]
		else:
			ux_beam = self.dsets[self.filenum]['vx'][0][:]*self.dsets[self.filenum]['gamma'][0][:]
		return ux_beam
	
	def getuybeam(self):
		if self.gamma == 1 : 
			uy_beam = self.ts.get_particle(var_list=['uy'], iteration=self.instant, species=self.species )[0]

		else:
			uy_beam = self.dsets[self.filenum]['vy'][0][:]*self.dsets[self.filenum]['gamma'][0][:]
		return uy_beam
	
	def getuzbeam(self):
		if self.gamma == 1 : 
			uz_beam = self.ts.get_particle(var_list=['uz'], iteration=self.instant, species=self.species )[0]
		else:
			uz_beam = self.dsets[self.filenum]['vz'][0][:]*self.dsets[self.filenum]['gamma'][0][:]
		return uz_beam
		
	def getemittance(self,variable='x'):

		if self.l_ebeam :
			ii=where(self.beamzstations<=Lplasma_lab)
			
			try: 
				emit = 	self.xemitnstations[ii[0][-1]]
			except IndexError:
				emit = self.xemitnstations[-1]			

		#else:
			#if self.gamma>1:
				#emit = self.PD.getEmittance(variable)
			#else:
				#emit = 0.0

		return emit
		
	def getfiltereduxbeam(self,index):
		ux_beam = self.getuxbeam()
		try:
				ux_beam=take(ux_beam,index)
		except IndexIsEmpty:	
				print "No particle above the gamma threshold is detected."	
		
		return ux_beam
	
	def getfilteredtbeam(self,index):
		t_beam = self.gettbeam()
		try:
				t_beam=take(t_beam,index)
		except IndexIsEmpty:	
				print "No particle above the gamma threshold is detected."	
		
		return t_beam
	
	def getfiltereduybeam(self,index):
		uy_beam = self.getuybeam()
		try:
				uy_beam=take(uy_beam,index)
		except IndexIsEmpty:	
				print "No particle above the gamma threshold is detected."	
		return uy_beam
	
	def getfiltereduzbeam(self,index):
		uz_beam = self.getuzbeam()
		try:
				uz_beam=take(uz_beam,index)
		except IndexIsEmpty:	
				print "No particle above the gamma threshold is detected."	
		return uz_beam
	
	def xRMS_JLV(self):
		if self.l_ebeam:
			ii=where(self.beamzstations<=Lplasma_lab)
			#print self.beamzstations[ii[0]][-1], self.beamzstations[ii[0]]
			try: 
				x_rms= 	self.xrmsstations[ii[0][-1]]
			except IndexError:
				x_rms = self.xrmsstations[-1]			
		else:
			x_rms=0.0
		return x_rms
		
	def xbar_JLV(self):
		if self.l_ebeam:
			ii=where(self.beamzstations<=Lplasma_lab)
			#print self.beamzstations[ii[0]][-1], self.beamzstations[ii[0]]
			try: 
				x_bar= 	self.xbarstations[ii[0][-1]]
			except IndexError:
				x_bar = self.xbarstations[-1]			
		else:
			x_bar=0.0
		return x_bar
	
	def xbeam_JLV(self):
		return self.xs
		
	def ybeam_JLV(self):
		return self.ys
		
	def zbeam_JLV(self):
		return self.zs
		
	def uxbeam_JLV(self):
		return self.uxs/clight
		
	def uybeam_JLV(self):
		return self.uys/clight
		
	def uzbeam_JLV(self):
		return self.uzs/clight
		
	def wbeam_JLV(self):
		return self.ws
		
	def xpbar_JLV(self):
		if self.l_ebeam:
			ii=where(self.beamzstations<=Lplasma_lab)
			#print self.beamzstations[ii[0]][-1], self.beamzstations[ii[0]]
			try: 
				xp_bar= 	self.xpbarstations[ii[0][-1]]
			except IndexError:
				xp_bar = self.xpbarstations[-1]			
		else:
			xp_bar=0.0
		return xp_bar
	
	def xxp_JLV(self):
		if self.l_ebeam:
			ii=where(self.beamzstations<=Lplasma_lab)
			#print self.beamzstations[ii[0]][-1], self.beamzstations[ii[0]]
			try: 
				xxp= 	self.xxpstations[ii[0][-1]]
			except IndexError:
				xxp = self.xxpstations[-1]			
		else:
			xxp=0.0
		return xxp
	
	def uxRMS_JLV(self):
		if self.l_ebeam:
			ii=where(self.beamzstations<=Lplasma_lab)
			beam  =  self.beamzstations[ii[0]]
		
			try: 
				ux_rms= 	self.xprmsstations[ii[0][-1]]
			
			except IndexError:
				ux_rms = self.xprmsstations[-1]			
		else:
			ux_rms=0.0

		return ux_rms
	
	def avEnergy_JLV(self):
		if self.l_ebeam:
			ii=where(self.beamzstations<=Lplasma_lab)
			try: 
				avEnergy= 	self.ekstations[ii[0][-1]]
			except IndexError:
				avEnergy = self.ekstations[-1]		
		else:
			avEnergy=0.
		return avEnergy*1e-6
		
		
	def filter(self,gamma_threshold=[],ROI=[]):	
		if not gamma_threshold:
			gamma_threshold.append(0.)
		if len(gamma_threshold)==1:
			gamma_threshold.append(SENTINEL)
			
		if self.gamma == 1 : 
			z_beam = self.ts.get_particle(var_list=['z'], iteration=self.instant, species=self.species )[0]
			z_beam =array(z_beam)*1.e-6
				
		else:
			z_beam = self.dsets[self.filenum]['z'][0][:]
		if not ROI:
			index=where((self.getgamma()>=gamma_threshold[0]) & (self.getgamma()<=gamma_threshold[1]) )
			print "No filtering in z"
		else:
			index=where((((self.getgamma()>=gamma_threshold) & (self.getgamma()<=gamma_threshold[1])) & ((z_beam>=ROI[0]) & (z_beam<=ROI[1]))))
		return index[0]	


def diff (data, z_data):
	return (data[:-1]-data[1:])/(max(data)*(z_data[1]-z_data[0]))
	
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = abs(int(window_size))
        order = abs(int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + abs(y[-half_window-1:-1][::-1] - y[-1])
    y = concatenate((firstvals, y, lastvals))
    return convolve( m[::-1], y, mode='valid')
	
def find_root(ez,z):

    #displaying the sign of the value
    l=len(ez)
    s=sign(ez)
    ii=[]
    for i in range (0,l-1):
    	if (s[i+1]+s[i]==0):##when the sum of the signs ==0, that means we hit a 0
		ii.append(i)
  	
    root_zero=take(z,ii).tolist()
    lrz=len(root_zero)
    ## if there's only one root found, there should be an end to it, we consider the min z
		## as the the limit
		
		#insert a z value at the first index
    if lrz==1:
    	root_zero.insert(0,min(z))
    ##if length of root is not pair, we remove the first value
    if len(root_zero)%2!=0:
    	root_zero=delete(root_zero,0)
    return root_zero
    
def gamma2energy(gamma_beam):
	try:
		energy=[i*0.511 for i in gamma_beam]
	except TypeError:
		energy=0
	return energy

def beam_charge(z_beam,w_beam):
		"""calculate the charge in the ROI (region of interest)"""
		try:
			charge=sum(w_beam)*echarge*10**12
			#ave_charge=average(charge)
		except TypeError:
			charge=0.
			print "No particles are detected"

		return charge
		
def beam_numParticles(z_beam):
	try:
		numPart= len(z_beam)
	except TypeError:
		numPart = 0.
		print "No particles are detected"
		
	return numPart
	
def beam_energy(gamma_beam,z_beam,w_beam,l_fwhm):
	energy = gamma2energy(gamma_beam)
	try:
			n_energy,energy = histogram(energy,bins=80,weights=w_beam)
	 		n_energy*=echarge*10**6  ##for MeV
	 		energy=delete(energy,0)
	except TypeError:
			energy=0.
	 		n_energy =0.
	 		print "No particles are detected"

	return n_energy,energy

def beam_statistics(gamma_beam,z_beam,w_beam,l_fwhm): 	
	n_energy,energy=beam_energy(gamma_beam,z_beam,w_beam,l_fwhm)
	try:
		average_energy=average(energy,weights = n_energy)
		variance = average((energy-average_energy)**2, weights=n_energy)
		std = sqrt(variance)
		eSpread=std/average_energy
	except ZeroDivisionError:
		average_energy = 0
		eSpread = 0
	#energy_spread = delta_EsE(energy,n_energy,l_fwhm) 
	return average_energy, eSpread

def emittance_calc(x,ux,w):

  """Calculation on emittance based on statistical approach in J. Buon (LAL) Beam phase space ande Emittance
  We first calculate the covariance, and the emittance is epsilon=sqrt(det(covariance matrix))
  Covariance is calculated with the weighted variance based on 
  http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
  """
  try:
  
  	w_x=average(x)
  	w_ux=average(ux)
  
  	ii=where(w!=0.)
  	nz_w=take(w,ii[0])
  	xux=sum(x*ux)/len(x)
  	variance_x=var(x)
  	variance_ux=var(ux)
  	covariance_xux=xux-w_x*w_ux
  
  	xuxw=[[variance_x,covariance_xux],[covariance_xux,variance_ux]]
  	emittance_w=sqrt(linalg.det(xuxw))
  	#print "PL",variance_x,variance_ux,covariance_xux,emittance_w
  	
		#variance_x=sum(w*(x-w_x)**2)/((len(nz_w)-1)*sum(w)/len(nz_w))
		#variance_ux=sum(w*(ux-w_ux)**2)/((len(nz_w)-1)*sum(w)/len(nz_w))
		#covariance_xux=sum(w*(ux-w_ux)*(x-w_x))/((len(nz_w)-1)*sum(w)/len(nz_w))

  except ZeroDivisionError:
  	emittance_w=0
  return emittance_w
	
def z_value_rms(env_left,env_right,e_rms,z_env_left,z_env_right):
    ind_left=where(env_left<e_rms)
    ind_right=where(env_right<e_rms)
    return env_left[ind_left[0][-1]], env_left[ind_left[0][-1]+1], env_right[ind_right[0][0]-1],env_right[ind_right[0][0]],\
    z_env_left[ind_left[0][-1]],  z_env_left[ind_left[0][-1]+1],  z_env_right[ind_right[0][0]-1], z_env_right[ind_right[0][0]]

def bilinear_interpolation(v1,v2,z1,z2,e_rms):
  """return the adjusted Energy level, calculated by bilinear interpolation"""
  dist=abs(v1-v2)

  z_new=z1+(1./dist)*(abs(v1-e_rms)*abs(z2-z1))

  return z_new

def get_rms(variable):
	
	return std(variable)
	
def delta_EsE(z_spectre,spectre,l_fwhm):
  #dist=ones(len(spectre))*(z_spectre[1]-z_spectre[0])
  rms=sqrt(sum((array(spectre))**2)/len(spectre))
  
  if l_fwhm:
    rms=1.359556*rms
  #print "rms %f, mean %f" %(rms,mean(spectre))
  E_max=max(spectre)
  ind=argmax(spectre)
  z_spectre_max=z_spectre[ind]

  spectre_left=spectre[0:ind]
  z_spectre_left=z_spectre[0:ind]
  spectre_right=spectre[ind+1::]
  z_spectre_right=z_spectre[ind+1::]
  indl, indll, indr, indrr, zindl, zindll, zindr, zindrr=z_value_rms(spectre_left,spectre_right,rms,z_spectre_left,z_spectre_right)
  z_spectre_left_rms=bilinear_interpolation(indl,indll,zindl,zindll,rms)

  z_spectre_right_rms=bilinear_interpolation(indr,indrr,zindr,zindrr,rms)
  delta_E=abs(z_spectre_left_rms-z_spectre_right_rms)
  delta_EE=delta_E/z_spectre_max

  #print "delta : %f " %delta_EE
  #print "left_rms: %f, right_rms: %f" %(z_spectre_left_rms,z_spectre_right_rms)

  return delta_EE


def myPlot(g,res,frame=0):

	winon(frame)
	ptitles("gamma=%d,nzplambda=%d" %(g,res))
	plsys(4)
	plg(ux_beam,x_beam)	
	ptitles("Ux-X","X","Ux")
	limits(min(x_beam),1.1*max(x_beam),min(ux_beam),max(ux_beam))
	plsys(5)
	plp(en,z_beam,marker='\1',color="red",msize=4)
	limits(min(z),max(z),0.5*max(en),1.1*max(en))
	ptitles("","z(m)","Beam Energy (MeV)")
	plsys(6)
	plp(x_beam,z_beam,marker='\1')
	limits(min(z),max(z),min(x_beam),1.1*max(x_beam))
	ptitles("X-Z plot","z(m)","x(m)")
	plsys(3)
	plg(Ez,z)
	limits(min(z),max(z),min(Ez),1.1*max(Ez))
	ptitles("Ez-field","z(m)")
	
def beam_variables(F,P,gamma_threshold, bucket=False):
	if bucket:
		buckets= F.bucket()
	else:	
		buckets=[]
	index=P.filter(gamma_threshold,buckets[0])
	x_beam  = P.getxbeam(index)
	z_beam  = P.getzbeam(index)
	w_beam  = P.getwbeam(index)
	gamma_beam = P.getfilteredgamma(index)
	ux_beam = P.getfiltereduxbeam(index)
	uz_beam = P.getfiltereduzbeam(index)
	t_beam  = P.gettbeam(index)
	
	return x_beam, z_beam, ux_beam, uz_beam, gamma_beam, w_beam, t_beam
	
def u2v(u,gamma):
	return u/gamma

def centralize(var_x,var_v,dt):
	return var_x+var_v*dt
	
def tree(): return defaultdict(tree)

def add(t, path):
  for node in path:
    t = t[node]
    
def dicts(t): return {k: dicts(t[k]) for k in t}    
#Initialisation

datadict = {'field_lab': ['Ey', 'Ez'],
					'elec_lab': ['ne', 'phase_space', 'phase_space_low'],
					'beam_lab': ['gamma', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'w','t']}
runid = 'warp_script'

subfolder = 'data'
file_name=[]

l_write=1
l_fwhm = 1
l_DESY =0
Lplasma_lab_all=500e-6
corr_data_instant=9
lambda_laser = 0.8e-6
freq_frame = 200

##Create a folder for the analysis result
path="/Volumes/WSIM4/boosted_frame/test_diag_edison/"
path_analysis= path + "Analysis_Data/"

try: 
	os.makedirs(path_analysis)
except OSError:
	if not os.path.isdir(path_analysis):
		raise

#default values

gammaBoost=[10]
analysis = tree()
num_folders_gamma = len(gammaBoost)
file_analysis  = path+"/analysis_res.txt"

file_variation = "self_injection_100_particles_highest_Nx"
fileselection = range(1, 11)
a_fileselection = [range(1,6), range(1,11), range(1,21)]
file_json = path+ "json/analysis_%s.json" %file_variation
#analysisTree=Tree()
l_plot=0
iCount = 0
stride = 1 

dsets = None
timeSeries=None
instant = 0
ins_particle   = 0
gammaBoost = 5
resolution=32
subtext = ""
folder=path+"gamma_test/gamma%d_%s/gamma%d_nzplambda%d%s/" %(gammaBoost,file_variation, gammaBoost,resolution,subtext)
species = 'beam'
file_ebeam = folder+"ebeamstations.pdb"
filePresent= os.path.isfile(file_ebeam)

if not filePresent:
	file_ebeam=None

Energy_threshold = 50
gamma_threshold = [Energy_threshold/0.511,]  ##have to be in agreement with the input script #Defining the filtered indices

latency=0e-6
l_fwhm=1

count =0

i = input('Choose the file number: ')

Lplasma_lab=(i+1)*Lplasma_lab_all/len(fileselection)-latency
print "You have chosen file %d, corresponding to Lplasma= %g" %(i, Lplasma_lab)
subfolder = "%sdata" %folder
data = wp.Data(runid = runid, subfolder = subfolder, fileselection = fileselection, datadict=datadict)
files = data.readFiles()
dsets = data.readDatasets()

F  = Fields(i, dsets, gammaBoost, timeSeries, instant)
P  = Particles(i,dsets, gammaBoost, timeSeries, ins_particle, species, filePresent, file_ebeam )

#1st beam
# -- calculating DESY quantities
x_beam, z_beam, ux_beam, uz_beam, gamma_beam, w_beam, t_beam = beam_variables(F,P,[50, 100./0.511],True)	
vx_beam = u2v(ux_beam,gamma_beam)
vz_beam = u2v(uz_beam,gamma_beam)

#Collapsing to the average time
t0      = ave(t_beam)
dt  		= t0-t_beam
new_tbeam = t_beam+dt
new_zbeam = centralize(z_beam,vz_beam,dt)
new_xbeam = centralize(x_beam,vx_beam,dt)
en = gamma2energy(gamma_beam)
new_emitx = emittance_calc(new_xbeam,ux_beam/clight,w_beam)
beamStat = beam_statistics(gamma_beam,z_beam,w_beam,l_fwhm)

#2nd beam
x_beam2, z_beam2, ux_beam2, uz_beam2, gamma_beam2, w_beam2, t_beam2 = beam_variables(F,P,[100./0.511],True)
vx_beam2 = u2v(ux_beam2,gamma_beam2)
vz_beam2 = u2v(uz_beam2,gamma_beam2)
t0      = ave(t_beam2)
dt  		= t0-t_beam2
new_tbeam2 = t_beam2+dt
new_zbeam2 = centralize(z_beam2,vz_beam2,dt)
new_xbeam2 = centralize(x_beam2,vx_beam2,dt)
en2 = gamma2energy(gamma_beam2)
new_emitx2 = emittance_calc(new_xbeam2,ux_beam2/clight,w_beam2)
beamStat2 = beam_statistics(gamma_beam2,z_beam2,w_beam2,l_fwhm)

#3rd beam take in everything
x_beam3, z_beam3, ux_beam3, uz_beam3, gamma_beam3, w_beam3, t_beam3 = beam_variables(F,P)
vx_beam3 = u2v(ux_beam3,gamma_beam3)
vz_beam3 = u2v(uz_beam3,gamma_beam3)
t0      = ave(t_beam3)
dt  		= t0-t_beam3
new_tbeam3 = t_beam3+dt
new_zbeam3 = centralize(z_beam3,vz_beam3,dt)
new_xbeam3 = centralize(x_beam3,vx_beam3,dt)
en3 = gamma2energy(gamma_beam3)
new_emitx3 = emittance_calc(new_xbeam3,ux_beam3/clight,w_beam3)
beamStat3 = beam_statistics(gamma_beam3,z_beam3,w_beam3,l_fwhm)

# -- calculating JLV quantities
x_beam_jlv = P.xbeam_JLV()
y_beam_jlv = P.ybeam_JLV()
z_beam_jlv = P.zbeam_JLV()
ux_beam_jlv = P.uxbeam_JLV()
uy_beam_jlv = P.uybeam_JLV()
uz_beam_jlv = P.uzbeam_JLV()
w_beam_jlv  = P.wbeam_JLV()
emit_jlv_calc  = emittance_calc(x_beam_jlv,ux_beam_jlv,w_beam_jlv)
var_x_jlv=(P.xRMS_JLV()**2-P.xbar_JLV()**2)
var_ux_jlv=(P.uxRMS_JLV()**2-P.xpbar_JLV()**2)
cov_jlv = P.xxp_JLV()-P.xbar_JLV()*P.xpbar_JLV()


def myPlot():
	plsys(3)
	ppg(ux_beam/clight, x_beam)
	ppg(ux_beam/clight, new_xbeam, color="red",msize=10)
	ppg(ux_beam2/clight, x_beam2)
	ppg(ux_beam2/clight, new_xbeam2, color="blue",msize=10)
	ptitles("Collapsed","x[m]","ux[m]")

	plsys(4)
	ppg(uz_beam/clight, z_beam)
	ppg(uz_beam/clight, new_zbeam, color="red",msize=10)
	ppg(uz_beam2/clight, z_beam2)
	ppg(uz_beam2/clight, new_zbeam2, color="blue",msize=10)
	ptitles("Collapsed","z[m]","uz[m]")
	
	plsys(5)

	ppg(ux_beam3/clight, x_beam3)
	ppg(ux_beam3/clight, new_xbeam3, color="black",msize=10)
	ptitles("Collapsed","x[m]","ux[m]")
	
	plsys(6)
	ppg(uz_beam3/clight, z_beam3)
	ppg(uz_beam3/clight, new_zbeam3, color="black",msize=10)
	ptitles("Collapsed","z[m]","uz[m]")

print "JLV",P.getemittance(),emittance_calc(x_beam,ux_beam/clight,w_beam),new_emitx


