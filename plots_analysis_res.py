from warp import *
import matplotlib as plt
import json
from pprint import pprint
import os

l_charge=0
l_emittance=0
l_Desy=0
l_energy=0
l_rms = 0
l_runtime=0

commandNotUnderstood=True
while commandNotUnderstood:
	print "Please choose the following: \n 1 : l_charge\n 2 : l_emittance \n 3 : l_energy \n 4 : l_rms \n 5 : l_runtime"
	nb = input('Choose a number: ')

	if nb ==1:
		l_charge=1
		commandNotUnderstood=False
	elif nb ==2:
		l_emittance=1
		commandNotUnderstood=False
	elif nb ==3:
		l_energy=1
		commandNotUnderstood=False
	elif nb ==4:
		l_rms=1
		commandNotUnderstood=False
	elif nb ==5:
		l_runtime=1
		commandNotUnderstood=False
	else:
		commandNotUnderstood=True
		print "Command not understood, please choose the following: \n 1 : l_charge\n 2 : l_emittance \n 3 : l_energy \n 4 : l_rms \n 5 : l_runtime"
		nb = input('Choose a number: ')
		
gammaBoost=[5,10]
SENTINEL=float("inf")
subtext = subtext2 = "self_injection_21dec"
subtext = subtext2 = "self_injection_100_particles_highest_Nx"
#subtext2 = "external_injection_a0_2_small_beam_new"

#subtext= "test_t_depos"
#subtext2="test_t_depos_interpolated"

path = "/Volumes/WSIM4/boosted_frame/test_diag_edison/"
file_json = path+"json/analysis_%s.json" %subtext
path_beam = path + "Beam_%s_plot" %subtext

try: 
	os.makedirs(path_beam)
except OSError:
	if not os.path.isdir(path_beam):
		raise

def limitmax(max_old,max):
	if max_old<max:
		return max
	else:
		return max_old

def limitmin(min_old,min):
	if min_old>min:
		return min
	else:
		return min_old
		
with open(file_json) as data_file:    
    data = json.load(data_file)


if True:
	os.chdir(path_beam)
	color=['blue','red','green','black']
	maxemitX=0
	maxemitZ=0
	maxavEnergy =0
	maxeSpread=0
	maxcharge=0
	maxnumPart=0
	maxrunTime=0
	maxxrms=0
	maxuxrms=0
	maxzrms=0
	maxuxrms=0
	maxuzrms=0
	
	minemitX=SENTINEL
	minemitZ=SENTINEL
	minavEnergy =SENTINEL
	mineSpread=SENTINEL
	mincharge=SENTINEL
	minnumPart=SENTINEL
	minrunTime=SENTINEL
	minxrms=SENTINEL
	minuxrms=SENTINEL
	minzrms=SENTINEL
	minuxrms=SENTINEL
	minuzrms=SENTINEL
	for i,igamma in enumerate(gammaBoost):
		charge=[]
		emitX=[]
		emitZ=[]
		emitX_desy=[]
		emitZ_desy=[]
		avEnergy=[]
		eSpread=[]
		res=[]
		runTime=[]
		numPart=[]
		x_rms=[]
		z_rms=[]
		ux_rms=[]
		uz_rms=[]
		emitX_jlv= []
		x_rms_jlv=[]
		ux_rms_jlv=[]
		avEnergy_jlv=[]
		new_emitX=[]
		charge_jlv=[]
		if igamma==1:
			resolution=[16,24,32]
		else:
			resolution=[16,24,32,40,48]#,64]
			#resolution=[16]
		for j, jres in enumerate(resolution):
		
			charge.append(data[i]['resolution'][j]['charge'])
			charge_jlv.append(data[i]['resolution'][j]['charge_jlv'])
			emitX.append(data[i]['resolution'][j]['emitX'])
			emitZ.append(data[i]['resolution'][j]['emitZ'])
			x_rms.append(data[i]['resolution'][j]['x_rms'])
			x_rms_jlv.append(data[i]['resolution'][j]['x_rms_jlv'])
			ux_rms_jlv.append(data[i]['resolution'][j]['ux_rms_jlv'])
			z_rms.append(data[i]['resolution'][j]['z_rms'])
			ux_rms.append(data[i]['resolution'][j]['ux_rms'])
			uz_rms.append(data[i]['resolution'][j]['uz_rms'])
			emitX_desy.append(data[i]['resolution'][j]['emitX_desy'])
			new_emitX.append(data[i]['resolution'][j]['new_emitX'])
			emitZ_desy.append(data[i]['resolution'][j]['emitZ_desy'])
			emitX_jlv.append(data[i]['resolution'][j]['emitX_jlv'])
			avEnergy.append(data[i]['resolution'][j]['avEnergy'])
			avEnergy_jlv.append(data[i]['resolution'][j]['avEnergy_jlv'])
			eSpread.append(data[i]['resolution'][j]['eSpread'])
			numPart.append(data[i]['resolution'][j]['numPart'])
			runTime.append(data[i]['resolution'][j]['runTime'])
			res.append(jres)
		#winon(0)
		if l_charge:
			plsys(9)

			plg(charge,res,color=color[i])
			plp(charge,res,marker='\4',color=color[i])
			
			ptitles("Charge (pC/m)","nzplambda")
			maxcharge = limitmax(maxcharge, max(charge))
			mincharge = limitmin(mincharge, min(charge))
			
			limits(min(res)-1,max(res)+1,0.9*mincharge,1.1*maxcharge)
			plsys(10)
			plg(numPart,res,color=color[i])
			plp(numPart,res,marker='\4',color=color[i])
			plg(charge_jlv,res,color=color[i],linetype="dashdot")
			plp(charge_jlv,res,marker='\3',color=color[i])
			ptitles("Number of particles","nzplambda")
			maxnumPart = limitmax(maxnumPart, max(numPart))
			minnumPart = limitmin(minnumPart, min(numPart))
			limits(min(res)-1,max(res)+1,0.9*minnumPart,1.1*maxnumPart)
			#logxy(0,1)
			pdf("beam_charge_%s" %(subtext2))
		
		if l_rms:
			plsys(9)
			plg(x_rms,res,color=color[i])
			plp(x_rms,res,marker='\4',color=color[i])
			plg(x_rms_jlv,res,color=color[i],linetype="dashdot")
			plp(x_rms_jlv,res,marker='\3',color=color[i])
			ptitles("x_rms","nzplambda")
			maxxrms = limitmax(maxxrms, max(x_rms))
			minxrms = limitmin(minxrms, min(x_rms))
			limits(min(res)-1,max(res)+1,0.9*minxrms,1.2*maxxrms)
			
			plsys(10)
			plg(ux_rms,res,color=color[i])
			plp(ux_rms,res,marker='\4',color=color[i])
			plg(ux_rms_jlv,res,color=color[i],linetype="dashdot")
			plp(ux_rms_jlv,res,marker='\3',color=color[i])
			ptitles("ux_rms","nzplambda")
			maxuxrms = limitmax(maxuxrms, max(ux_rms))
			minuxrms = limitmin(minuxrms, min(ux_rms))
			limits(min(res)-1,max(res)+1,0.9*minuxrms,1.2*maxuxrms)
			#limits(min(res)-1,max(res)+1,0.9*minxrms,1.1*maxxrms)
			
			if False:
				plsys(5)
				plg(z_rms,res,color=color[i])
				plp(z_rms,res,marker='\4',color=color[i])
				ptitles("z_rms","nzplambda")
				maxzrms = limitmax(maxzrms, max(z_rms))
				minzrms = limitmin(minzrms, min(z_rms))
				#limits(min(res)-1,max(res)+1,0.9*minzrms,1.1*maxzrms)
			
			
				plsys(6)
				plg(uz_rms,res,color=color[i])
				plp(uz_rms,res,marker='\4',color=color[i])
				ptitles("uz_rms","nzplambda")
				maxuzrms = limitmax(maxuzrms, max(uz_rms))	
				minuzrms = limitmin(minuzrms, min(uz_rms))
				#limits(min(res)-1,max(res)+1,0.9*minuzrms,1.1*maxuzrms)
			
			pdf("beam_rms_%s" %(subtext2))
			
		if l_emittance:
		
			plsys(1)
			plg(emitX,res,color=color[i])
			plp(emitX,res,marker='\4',color=color[i])
			plg(emitX_jlv,res,color=color[i], linetype="dashdot")
			plp(emitX_jlv,res,marker='\3',color=color[i])
			ptitles("Emittance in X","nzplambda","emit X (mm.mrad)")
			limits(10,70, 0.,0.15)
			
			if l_Desy:
				plg(emitX_desy,res,color=color[i],linetype="dashdot")
				plp(emitX_desy,res,marker='\4',color=color[i])
			else:
				maxemitX = limitmax(maxemitX, max(emitX))
				minemitX = limitmin(minemitX, min(emitX))
				limits(min(res)-1,max(res)+1,0,1.1*maxemitX)
			if False:
				plsys(10)
				plg(emitZ,res,color=color[i])
				plp(emitZ,res,marker='\4',color=color[i])
				ptitles("Emittance in Z","nzplambda","emit Z (mm.mrad)" )
				if l_Desy:
					plg(emitZ_desy,res,color=color[i],linetype="dashdot")
					plp(emitZ_desy,res,marker='\4',color=color[i])
				else:
					maxemitZ = limitmax(maxemitZ, max(emitZ))
					minemitZ = limitmin(minemitZ, min(emitZ))
					limits(min(res)-1,max(res)+1,0,1.1*maxemitZ)
			pdf("beam_emittance_%s" %(subtext2))

		if l_energy:

			plsys(9)
			plg(avEnergy,res,color=color[i])
			plp(avEnergy,res,marker='\4',color=color[i])
			plg(avEnergy_jlv,res,color=color[i],linetype="dashdot")
			plp(avEnergy_jlv,res,marker='\3',color=color[i])
			ptitles("Average Energy","nzplambda","Energy (MeV)")
			maxavEnergy = limitmax(maxavEnergy, max(avEnergy))
			minavEnergy = limitmin(minavEnergy, min(avEnergy))
			limits(min(res)-1,max(res)+1,0.9*minavEnergy,1.1*maxavEnergy)

			plsys(10)
			plg(eSpread,res,color=color[i])
			plp(eSpread,res,marker='\4',color=color[i])
			ptitles("Energy spread","nzplambda","deltaE/E ")
			maxeSpread = limitmax(maxeSpread, max(eSpread))
			mineSpread = limitmin(mineSpread, min(eSpread))
			limits(min(res)-1,max(res)+1,0.9*mineSpread,1.1*maxeSpread)
			pdf("beam_energy_%s"%(subtext2))
			
		if l_runtime:
			plg(runTime,res,color=color[i])
			plp(runTime,res,marker='\4',color=color[i])
			ptitles("Run Time","nzplambda","time (s)")
			maxrunTime = limitmax(maxrunTime, max(runTime))
			minrunTime = limitmin(minrunTime, min(runTime))
			logxy(0,1)
			limits(min(res)-1,max(res)+1,0.9*minrunTime,1.1*maxrunTime)
