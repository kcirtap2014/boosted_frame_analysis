from warp import *

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
  
file_variation = "external_injection_a0_2"
#file_variation = "test_trans_emit"
file_variation = "external_injection_a0_2_small_beam"
file_variation = "imposed_Ex"
file_variation = "jlv_diag_test"
file_variation = "self_injection_4_4part_highest_Nx_8cores"
snapshots=[0.2,0.5,1.0]
l_snapshots = 0
path = "/Volumes/WSIM4/boosted_frame/test_diag_edison/"

path_beam = path + "Beam_%s_plot" %file_variation

try: 
	os.makedirs(path_beam)
except OSError:
	if not os.path.isdir(path_beam):
		raise

gamma,nzplambda = getdatafromtextfile(path+'Analysis_Data/gamma_nzplambda_%s.txt' %(file_variation),dims=[2,None])
z,emitX         = getdatafromtextfile(path+'Analysis_Data/emitX_%s.txt' %(file_variation),dims=[2,None])
z,charge         = getdatafromtextfile(path+'Analysis_Data/charge_%s.txt' %(file_variation),dims=[2,None])
z,num_part     = getdatafromtextfile(path+'Analysis_Data/num_part_%s.txt' %(file_variation),dims=[2,None])
zs,emitxs    = getdatafromtextfile(path+'Analysis_Data/emit_middle_point_%s.txt'%(file_variation),dims=[2,None])
z_beam,uz_beam  = getdatafromtextfile(path+'Analysis_Data/z_uz_beam_%s.txt'%(file_variation),dims=[2,None])
x_beam,ux_beam  = getdatafromtextfile(path+'Analysis_Data/x_ux_beam_%s.txt'%(file_variation),dims=[2,None])
z,x_rms         = getdatafromtextfile(path+'Analysis_Data/x_rms_%s.txt'%(file_variation),dims=[2,None])
z,w_ux          = getdatafromtextfile(path+'Analysis_Data/w_ux_%s.txt'%(file_variation),dims=[2,None])
z,w_x           = getdatafromtextfile(path+'Analysis_Data/w_x_%s.txt'%(file_variation),dims=[2,None])
z,ux_rms        = getdatafromtextfile(path+'Analysis_Data/ux_rms_%s.txt'%(file_variation),dims=[2,None])
z,avEn          = getdatafromtextfile(path+'Analysis_Data/avEn%s.txt'%(file_variation),dims=[2,None])
x_middle,ux_middle = getdatafromtextfile(path+'Analysis_Data/xsuxs_%s.txt'%(file_variation),dims=[2,None])
os.chdir(path_beam)

winon(0)
plsys(3)
plg(x_rms*1e6,z,color='red')	
plp(x_rms*1e6, z, marker='\4',color="red")
ptitles("","z (um)","X_RMS (mm)")

plsys(4)
plg(charge*1e-3,z,color='red')	
plp(charge*1e-3, z, marker='\4',color="red")
ptitles("","z (um)","Charge (nC)")

plsys(5)
plg(emitX*1e6,z,color='red')	
plp(emitX*1e6,z, marker='\4',color='red')	

ptitles("","z (um)","emitX (mm.mrad)")

plsys(6)
plg(avEn, z, color='red')
plp(avEn, z, marker='\4',color="red")

ptitles("avEn","z(m)","E(MeV)")


pdf("evolution_case_gamma_%d_nzplambda_%d_a_20_beamstations" %(gamma,nzplambda))

if False:
	winon(1)

	plsys(5)
	#plp(t_beam,z_beam)	
	#ptitles("","z (m)","t")
	#limits(-3e-6,3e-6,-4.5,4.5)
	plg(x_bar_jlv_all, z_all*1e6, color="green")
	plg(w_x,z,color='red')	
	plp(w_x, z, marker='\4',color="red")
	
	ptitles("","z (um)","xbar")
	
	plsys(6)
	plg(charge_jlv_all, z_all*1e6, color="green")
	plg(num_part,z,color='red')	
	plp(num_part, z, marker='\4',color="red")
	plg(charge_jlv, z, color="blue")
	plp(charge_jlv, z, marker='\2',color="blue")
	ptitles("","z (um)","Num of Particles")


	plsys(3)
	plg(ux_bar_jlv_all, z_all*1e6, color="green")
	plg(w_ux,z,color='red')	
	plp(w_ux, z, marker='\4',color="red")
	plg(ux_bar_jlv, z, color="blue")
	plp(ux_bar_jlv, z, marker='\2',color="blue")
	#plp(ux_bar_jlv_all_smoothed, z_all*1e6, marker='\2',color="magenta")

	#limits(min(z), max(z), 0.9*min(ux_bar_jlv),1.1*max(ux_bar_jlv_all))

	ptitles("","z (um)","UX_bar")
	plsys(4)
	plg(xxp_jlv_all, z_all*1e6, color="green")
	#plg(e,z,color='red')	
	plg(xxp,z,color='red')	
	plp(xxp, z, marker='\4',color="red")
	plg(xxp_jlv, z, color="blue")
	plp(xxp_jlv, z, marker='\2',color="blue")
	#plp(xxp_jlv_all_smoothed, z_all*1e6, marker='\2',color="magenta")
	
	#
	emit_mid = emittance_calc(x_middle,ux_middle/clight,ones(shape(x_middle)[0]))
	emit_beam_jlv= emittance_calc(x_beam_jlv,ux_beam_jlv,ones(shape(x_beam_jlv)[0]))
	emit_beam= emittance_calc(x_beam,ux_beam/clight,ones(shape(x_beam_jlv)[0]))

	print emit_mid, emit_beam_jlv, emit_beam
	ptitles("","z (um)","XXP")

	#ptitles("","z (m)","emitX")
	pdf("evolution_case_gamma_%d_nzplambda_%d_b_20_beamstations" %(gamma,nzplambda))

