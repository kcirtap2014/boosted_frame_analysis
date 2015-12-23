from warp import *
file_variation = "external_injection_a0_2"
#file_variation = "test_trans_emit"
file_variation = "external_injection_a0_2_small_beam"
file_variation = "test_t_depos"
snapshots=[0.2,0.5,1.0]
l_snapshots = 0
path = "/Volumes/WSIM4/boosted_frame/test_diag_edison/"

path_beam = path + "Beam_%s_plot" %file_variation

try: 
	os.makedirs(path)
except OSError:
	if not os.path.isdir(path):
		raise

if l_snapshots:
	color=['blue','red','green','black']
	linestyles = ['-', '--', '-.', ':']
	markers= ['+','*','#']
	fig0,((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2, ncols=2)
	fig1,((ax5,ax6),(ax7,ax8))=plt.subplots(nrows=2, ncols=2)

	for index, i in enumerate(snapshots):
		subtext = "dtcoef%.1f" %i
		gamma,nzplambda = getdatafromtextfile('./Analysis_Data/gamma_nzplambda_%s_%s.txt' %(file_variation,subtext),dims=[2,None])
		z,emitX         = getdatafromtextfile('./Analysis_Data/emitX_%s_%s.txt' %(file_variation,subtext),dims=[2,None])

		z,emitX_jlv     = getdatafromtextfile('./Analysis_Data/emitX_jlv_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,emitX_jlv2    = getdatafromtextfile('./Analysis_Data/emitX_jlv2_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z_all,emitX_jlv_all    = getdatafromtextfile('./Analysis_Data/emitX_jlv_all_%s_%s.txt'%(file_variation,subtext),dims=[2,None])

		x_beam,ux_beam  = getdatafromtextfile('./Analysis_Data/x_ux_beam_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,x_rms         = getdatafromtextfile('./Analysis_Data/x_rms_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,x_bar_jlv     = getdatafromtextfile('./Analysis_Data/x_bar_jlv_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,xxp_jlv       = getdatafromtextfile('./Analysis_Data/xxp_jlv_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,xxp           = getdatafromtextfile('./Analysis_Data/xxp_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z_all,xxp_jlv_all = getdatafromtextfile('./Analysis_Data/xxp_jlv_all_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		#t_beam,z_beam   = getdatafromtextfile('./Analysis_Data/t_z_beam_%s_snapshot_%d.txt'%(file_variation,i),dims=[2,None])
		z_all,x_bar_jlv_all = getdatafromtextfile('./Analysis_Data/x_bar_jlv_all_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,w_ux          = getdatafromtextfile('./Analysis_Data/w_ux_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,w_x           = getdatafromtextfile('./Analysis_Data/w_x_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,x_rms_jlv     = getdatafromtextfile('./Analysis_Data/x_rms_jlv_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z_all,x_rms_jlv_all = getdatafromtextfile('./Analysis_Data/x_rms_jlv_all_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,ux_rms_jlv    = getdatafromtextfile('./Analysis_Data/ux_rms_jlv_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,ux_rms        = getdatafromtextfile('./Analysis_Data/ux_rms_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,ux_bar_jlv    = getdatafromtextfile('./Analysis_Data/ux_bar_jlv_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z_all,ux_bar_jlv_all = getdatafromtextfile('./Analysis_Data/ux_bar_jlv_all_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,avEn_jlv      = getdatafromtextfile('./Analysis_Data/avEn_jlv_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z,avEn          = getdatafromtextfile('./Analysis_Data/avEn%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		z_all,avEn_jlv_all = getdatafromtextfile('./Analysis_Data/avEn_jlv_all_%s_%s.txt'%(file_variation,subtext),dims=[2,None])
		
		
		ax1.set_xlabel('z(um)')
		ax1.set_ylabel('X_RMS')
		ax1.plot(z,x_rms*1e6,color=color[index],linestyle=linestyles[0])
		ax1.scatter(z,x_rms*1e6,color=color[index],marker=markers[0])
		ax1.plot(z,x_rms_jlv*1e6,color=color[index],linestyle=linestyles[1])
		#ax1.scatter(z,x_rms_jlv,color=color[index],marker=markers[1])
		ax1.plot(z_all*1e6,x_rms_jlv_all*1e6,color=color[index],linestyle=linestyles[2])
		plt.tight_layout()
		
		ax2.set_xlabel('z(um)')
		ax2.set_ylabel('UX_RMS')
		ax2.plot(z,ux_rms,color=color[index],linestyle=linestyles[0])
		ax2.scatter(z,ux_rms,color=color[index],marker=markers[0])
		ax2.plot(z,ux_rms_jlv,color=color[index],linestyle=linestyles[1])
		#ax2.scatter(z,ux_rms_jlv,color=color[index],marker=markers[1])
		#ax1.plot(z_all*1e6,ux_rms_jlv_all,color=color[index],linestyles=linestyles[2])
		plt.tight_layout()
		
		ax3.set_xlabel('z(um)')
		ax3.set_ylabel('emitX')
		ax3.plot(z,emitX,color=color[index],linestyle=linestyles[0])
		ax3.scatter(z,emitX,color=color[index],marker=markers[0])
		ax3.plot(z,emitX_jlv,color=color[index],linestyle=linestyles[1])
		#ax3.scatter(z,emitX_jlv,color=color[index],marker=markers[1])
		ax3.plot(z_all*1e6,emitX_jlv_all,color=color[index],linestyle=linestyles[2])
		plt.tight_layout()
		
		ax4.set_xlabel('z(um)')
		ax4.set_ylabel('Average Energy')
		ax4.plot(z,avEn,color=color[index],linestyle=linestyles[0])
		ax4.scatter(z,avEn,color=color[index],marker=markers[0])
		ax4.plot(z,avEn_jlv,color=color[index],linestyle=linestyles[1])
		#ax4.scatter(z,avEn_jlv,color=color[index],marker=markers[1])
		ax4.plot(z_all*1e6,avEn_jlv_all,color=color[index],linestyle=linestyles[2])
		plt.tight_layout()
		
		ax5.set_xlabel('z(um)')
		ax5.set_ylabel('X_bar')
		ax5.plot(z,w_x,color=color[index],linestyle=linestyles[0])
		ax5.scatter(z,w_x,color=color[index],marker=markers[0])
		ax5.plot(z,x_bar_jlv,color=color[index],linestyle=linestyles[1])
		#ax1.scatter(z,x_rms_jlv,color=color[index],marker=markers[1])
		ax5.plot(z_all*1e6,x_bar_jlv_all,color=color[index],linestyle=linestyles[2])
		plt.tight_layout()
		
		ax6.set_xlabel('z(um)')
		ax6.set_ylabel('UX_bar')
		ax6.plot(z,w_ux,color=color[index],linestyle=linestyles[0])
		ax6.scatter(z,w_ux,color=color[index],marker=markers[0])
		ax6.plot(z,ux_bar_jlv,color=color[index],linestyle=linestyles[1])
		#ax2.scatter(z,ux_rms_jlv,color=color[index],marker=markers[1])
		ax1.plot(z_all*1e6,ux_bar_jlv_all,color=color[index],linestyle=linestyles[2])
		plt.tight_layout()
		
		ax7.set_xlabel('z(um)')
		ax7.set_ylabel('XXP')
		ax7.plot(z,xxp,color=color[index],linestyle=linestyles[0])
		ax7.scatter(z,xxp,color=color[index],marker=markers[0])
		ax7.plot(z,xxp_jlv,color=color[index],linestyle=linestyles[1])
		#ax3.scatter(z,emitX_jlv,color=color[index],marker=markers[1])
		ax7.plot(z_all*1e6,xxp_jlv_all,color=color[index],linestyle=linestyles[2])
		plt.tight_layout()
		
		ax8.set_xlabel('x(um)')
		ax8.set_ylabel('Ux')
		#ax8.plot(x_beam,ux_beam,color=color[index],linestyle=linestyles[0])
		ax8.scatter(x_beam,ux_beam,color=color[index],marker=markers[0])
		#ax8.plot(z,avEn_jlv,color=color[index],linestyle=linestyles[1])
		#ax4.scatter(z,avEn_jlv,color=color[index],marker=markers[1])
		#ax8.plot(z_all*1e6,avEn_jlv_all,color=color[index],linestyle=linestyles[2])
		plt.tight_layout()
		
	os.chdir(path)	
	plt.show()
		

else:
	gamma,nzplambda = getdatafromtextfile('./Analysis_Data/gamma_nzplambda_%s.txt' %(file_variation),dims=[2,None])
	z,emitX         = getdatafromtextfile('./Analysis_Data/emitX_%s.txt' %(file_variation),dims=[2,None])

	z,emitX_jlv     = getdatafromtextfile('./Analysis_Data/emitX_jlv_%s.txt'%(file_variation),dims=[2,None])
	z,emitX_jlv2    = getdatafromtextfile('./Analysis_Data/emitX_jlv2_%s.txt'%(file_variation),dims=[2,None])
	z_all,emitX_jlv_all    = getdatafromtextfile('./Analysis_Data/emitX_jlv_all_%s.txt'%(file_variation),dims=[2,None])
	z_beam,uz_beam  = getdatafromtextfile('./Analysis_Data/z_uz_beam_%s.txt'%(file_variation),dims=[2,None])
	x_beam,ux_beam  = getdatafromtextfile('./Analysis_Data/x_ux_beam_%s.txt'%(file_variation),dims=[2,None])
	z,x_rms         = getdatafromtextfile('./Analysis_Data/x_rms_%s.txt'%(file_variation),dims=[2,None])
	z,x_bar_jlv     = getdatafromtextfile('./Analysis_Data/x_bar_jlv_%s.txt'%(file_variation),dims=[2,None])
	z,xxp_jlv       = getdatafromtextfile('./Analysis_Data/xxp_jlv_%s.txt'%(file_variation),dims=[2,None])
	z,xxp           = getdatafromtextfile('./Analysis_Data/xxp_%s.txt'%(file_variation),dims=[2,None])
	z_all,xxp_jlv_all = getdatafromtextfile('./Analysis_Data/xxp_jlv_all_%s.txt'%(file_variation),dims=[2,None])
	t_beam,z_beam   = getdatafromtextfile('./Analysis_Data/t_z_beam_%s.txt'%(file_variation),dims=[2,None])
	z_all,x_bar_jlv_all = getdatafromtextfile('./Analysis_Data/x_bar_jlv_all_%s.txt'%(file_variation),dims=[2,None])
	z,w_ux          = getdatafromtextfile('./Analysis_Data/w_ux_%s.txt'%(file_variation),dims=[2,None])
	z,w_x           = getdatafromtextfile('./Analysis_Data/w_x_%s.txt'%(file_variation),dims=[2,None])
	z,x_rms_jlv     = getdatafromtextfile('./Analysis_Data/x_rms_jlv_%s.txt'%(file_variation),dims=[2,None])
	z_all,x_rms_jlv_all = getdatafromtextfile('./Analysis_Data/x_rms_jlv_all_%s.txt'%(file_variation),dims=[2,None])
	z,ux_rms_jlv    = getdatafromtextfile('./Analysis_Data/ux_rms_jlv_%s.txt'%(file_variation),dims=[2,None])
	z,ux_rms        = getdatafromtextfile('./Analysis_Data/ux_rms_%s.txt'%(file_variation),dims=[2,None])
	z,ux_bar_jlv    = getdatafromtextfile('./Analysis_Data/ux_bar_jlv_%s.txt'%(file_variation),dims=[2,None])
	z_all,ux_bar_jlv_all = getdatafromtextfile('./Analysis_Data/ux_bar_jlv_all_%s.txt'%(file_variation),dims=[2,None])
	z,avEn_jlv      = getdatafromtextfile('./Analysis_Data/avEn_jlv_%s.txt'%(file_variation),dims=[2,None])
	z,avEn          = getdatafromtextfile('./Analysis_Data/avEn%s.txt'%(file_variation),dims=[2,None])
	z_all,avEn_jlv_all = getdatafromtextfile('./Analysis_Data/avEn_jlv_all_%s.txt'%(file_variation),dims=[2,None])


		
	os.chdir(path)
	winon(0)
	plsys(3)
	plg(x_rms,z,color='red')	
	plp(x_rms, z, marker='\4',color="red")
	plg(x_rms_jlv, z, color="blue")
	plp(x_rms_jlv, z, marker='\2',color="blue")
	plg(x_rms_jlv_all, z_all*1e6, color="green")

	ptitles("","z (um)","X_RMS")
	plsys(4)
	plg(ux_rms,z,color='red')	
	plp(ux_rms, z, marker='\4',color="red")
	plg(ux_rms_jlv, z, color="blue")
	plp(ux_rms_jlv, z, marker='\2',color="blue")

	ptitles("","z (um)","UX_RMS")
	plsys(5)
	plg(emitX,z,color='red')	
	plp(emitX, z, marker='\4',color="red")
	plg(emitX_jlv, z, color="blue")
	plp(emitX_jlv, z, marker='\2',color="blue")
	plg(emitX_jlv_all, z_all*1e6, color="green")
	plg(emitX_jlv_all, z_all*1e6, marker='\3',color="green")
	ptitles("","z (um)","emitX")

	plsys(6)
	plg(w_x,z,color='red')	
	plp(w_x, z, marker='\4',color="red")
	plg(x_bar_jlv, z, color="blue")
	plp(x_bar_jlv, z, marker='\2',color="blue")
	plg(x_bar_jlv_all, z_all*1e6, color="green")
	#plg(avEn,z,color='red')	
	#plp(avEn, z, marker='\4',color="red")
	#plg(avEn_jlv, z, color="blue")
	#plp(avEn_jlv, z, marker='\2',color="blue")
	#plg(avEn_jlv_all, z_all*1e6, color="green")
	#ptitles("","z (um)","Average Energy")

	pdf("evolution_case_gamma_%d_nzplambda_%d_440_stations" %(gamma,nzplambda))

	winon(1)

	plsys(5)
	plp(ux_beam,x_beam)	
	#plp(t_beam,z_beam)	
	#ptitles("","z (m)","t")
	#limits(-3e-6,3e-6,-4.5,4.5)
	

	#ptitles("","z (um)","X_bar")
	plsys(3)
	plg(w_ux,z,color='red')	
	plp(w_ux, z, marker='\4',color="red")
	plg(ux_bar_jlv, z, color="blue")
	plp(ux_bar_jlv, z, marker='\2',color="blue")
	plg(ux_bar_jlv_all, z_all*1e6, color="green")

	ptitles("","z (um)","UX_bar")
	plsys(4)
	#plg(e,z,color='red')	
	plg(xxp,z,color='red')	
	plp(xxp, z, marker='\4',color="red")
	plg(xxp_jlv, z, color="blue")
	plp(xxp_jlv, z, marker='\2',color="blue")
	plg(xxp_jlv_all, z_all*1e6, color="green")

	ptitles("","z (um)","XXP")

	plsys(6)
	plp(ux_beam,x_beam)	
	#limits(-3e-6,3e-6,-4.5,4.5)

	ptitles("","x (m)","ux_beam")
	pdf("evolution_case_gamma_%d_nzplambda_%d_440_stations" %(gamma,nzplambda))

