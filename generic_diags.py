from warp import *
# --- define shortcut
ppg=ppgeneric

def writeTime():
	if me ==0:
		file="TotalRunTime.txt"
	
		f=open(file,'w')
		f.write("%gs" %runtime)
		f.close()

def writeNum():
    if me ==0:
        file="ParamNum.txt"
    
        f=open(file,'w')
        f.write(", ".join(str(x) for x in param_nu))
        f.close()

def write_ptcl():
    """
      Write the particles data to a pickle file every hist_freq.
      The data is gathered on the first processor, which writes it to the disk.
    """

    if top.it%hist_freq == 0 :

        # Selection in energy
        uz_cutoff = 1.*clight # 0.5 MeV
        uz = elec.getuz()
        selec = (uz > uz_cutoff)

        # Get the data
        z = elec.getz()[selec]
        x = elec.getx()[selec]
        y= elec.gety()[selec]
        ux = elec.getux()[selec]/clight
        ux = elec.getuy()[selec]/clight
        uz = uz[selec]/clight
        w=elec.getw()[selec]
        if me == 0 :

            # Write the results in a file
            fname = 'elec%06d.pdb' % top.it
            f = PW.PW(fname)

            # Selected quantities
            f.t = top.time
            f.x = x
            f.z = z
            f.y = y
            f.ux = ux
            f.uy = uy
            f.uz = uz
            f.w = w

            f.close()

def write_ptcl_hyd_ions():
    """
      Write the particles data to a pickle file every hist_freq.
      The data is gathered on the first processor, which writes it to the disk.
    """

    if top.it%hist_freq == 0 :

        # Selection in energy
        uz_cutoff = 1.*clight # 0.5 MeV
        uz = hyd_ions[1].getuz()
        selec = (uz > uz_cutoff)

        # Get the data
        z = hyd_ions[1].getz()[selec]
        x = hyd_ions[1].getx()[selec]
        y = hyd_ions[1].gety()[selec]
        ux = hyd_ions[1].getux()[selec]/clight
        uy = hyd_ions[1].getuy()[selec]/clight
        uz = uz[selec]/clight
        ex = hyd_ions[1].getex()[selec]
        ey = hyd_ions[1].getey()[selec]
        ez = hyd_ions[1].getez()[selec]
        bx = hyd_ions[1].getbx()[selec]
        by = hyd_ions[1].getby()[selec]
        bz = hyd_ions[1].getbz()[selec]
        w=hyd_ions[1].getw()[selec]
        ssnum= hyd_ions[1].getpid()[selec]
        if me == 0 :

            # Write the results in a file
            fname = 'H_ions%06d.pdb' % top.it
            f = PW.PW(fname)

            # Selected quantities
            f.t = top.time
            f.x = x
            f.y = y
            f.z = z
            f.ux = ux
            f.uy = uy
            f.uz = uz
            f.w = w
            f.ex = ex
            f.ey = ey
            f.ez = ez
            f.bx = bx
            f.by = by
            f.bz = bz
            f.ssnum = ssnum
            f.close()

def write_ptcl_hyd():
    """
      Write the particles data to a pickle file every hist_freq.
      The data is gathered on the first processor, which writes it to the disk.
    """

    if top.it%hist_freq == 0 :
        
        # Selection in energy
        uz_cutoff = 1.*clight # 0.5 MeV
        uz = hyd_elec[1].getuz()
        selec = (uz > uz_cutoff)
    
        # Get the data
        z = hyd_elec[1].getz()[selec]
        x = hyd_elec[1].getx()[selec]
        y = hyd_elec[1].gety()[selec]
        ux = hyd_elec[1].getux()[selec]/clight
        uy = hyd_elec[1].getuy()[selec]/clight
        uz = uz[selec]/clight
	ex = hyd_elec[1].getex()[selec]
        ey = hyd_elec[1].getey()[selec]
        ez = hyd_elec[1].getez()[selec]
        bx = hyd_elec[1].getbx()[selec]
        by = hyd_elec[1].getby()[selec]
        bz = hyd_elec[1].getbz()[selec]
        w=hyd_elec[1].getw()[selec]
        ssnum= hyd_elec[1].getpid()[selec]
        if me == 0 :
        
            # Write the results in a file
            fname = 'H%06d.pdb' % top.it
            f = PW.PW(fname)
    
            # Selected quantities
            f.t = top.time
            f.x = x
            f.y = y
            f.z = z
            f.ux = ux
            f.uy = uy
            f.uz = uz
            f.w = w
            f.ex = ex
            f.ey = ey
            f.ez = ez
            f.bx = bx
            f.by = by
            f.bz = bz
            f.ssnum = ssnum
            f.close()

def write_ptcl_N4():
    """
      Write the particles data to a pickle file every hist_freq.
      The data is gathered on the first processor, which writes it to the disk.
    """

    if top.it%hist_freq == 0 :

        # Selection in energy
        uz_cutoff = 1.*clight # 0.5 MeV
        uz = nit_elec[4].getuz()
        selec = (uz > uz_cutoff)

        # Get the data
        z = nit_elec[4].getz()[selec]
        x = nit_elec[4].getx()[selec]
        y = nit_elec[4].gety()[selec]
        ux = nit_elec[4].getux()[selec]/clight
        uy = nit_elec[4].getuy()[selec]/clight
        uz = uz[selec]/clight
        ex = nit_elec[4].getex()[selec]
        ey = nit_elec[4].getey()[selec]
        ez = nit_elec[4].getez()[selec]
        bx = nit_elec[4].getbx()[selec]
        by = nit_elec[4].getby()[selec]
        bz = nit_elec[4].getbz()[selec]
        w  = nit_elec[4].getw()[selec]
        if me == 0 :

            # Write the results in a file
            fname = 'N4%06d.pdb' % top.it
            f = PW.PW(fname)

            # Selected quantities
            f.t = top.time
            f.x = x
            f.y = y
            f.z = z
            f.ux = ux
            f.uy = uy
            f.uz = uz
            f.ex = ex
            f.ey = ey
            f.ez = ez
            f.bx = bx
            f.by = by
            f.bz = bz
            f.w = w

            f.close()

def write_ptcl_N6():
    """
      Write the particles data to a pickle file every hist_freq.
      The data is gathered on the first processor, which writes it to the disk.
    """

    if top.it%hist_freq == 0 :

        # Selection in energy
        #uz_cutoff = 1.*clight # 0.5 MeV
        uz = nit_elec[6].getuz()
        #selec = (uz > uz_cutoff)

        # Get the data
        z = nit_elec[6].getz()
        x = nit_elec[6].getx()
        y = nit_elec[6].gety()
        ux = nit_elec[6].getux()/clight
        uy = nit_elec[6].getuy()/clight
        uz = uz/clight
        ex = nit_elec[6].getex()
        ey = nit_elec[6].getey()
        ez = nit_elec[6].getez()#[selec]
        bx = nit_elec[6].getbx()#[selec]
        by = nit_elec[6].getby()#[selec]
        bz = nit_elec[6].getbz()#[selec]
        ssnum= nit_elec[6].getpid()#[selec]
        w=nit_elec[6].getw()#[selec]
        if me == 0 :

            # Write the results in a file
            fname = 'N6%06d.pdb' % top.it
            f = PW.PW(fname)

            # Selected quantities
            f.t = top.time
            f.x = x
            f.z = z
            f.y = y
            f.ux = ux
            f.uy =uy
            f.uz = uz
            f.ex = ex
            f.ey = ey
            f.ez = ez
            f.bx = bx
            f.by = by
            f.bz = bz
            f.ssnum = ssnum 
            f.w = w

            f.close()

def write_ptcl_N7():
    """
      Write the particles data to a pickle file every hist_freq.
      The data is gathered on the first processor, which writes it to the disk.
    """

    if top.it%hist_freq == 0 :

        # Selection in energy
        #uz_cutoff = 1.*clight # 0.5 MeV
        uz = nit_elec[7].getuz()
        #selec = (uz > uz_cutoff)

        # Get the data
        z = nit_elec[7].getz()#[selec]
        x = nit_elec[7].getx()#[selec]
        y = nit_elec[7].gety()#[selec]
        ux = nit_elec[7].getux()/clight
        uy = nit_elec[7].getuy()/clight
        uz = uz/clight
        ex = nit_elec[7].getex()#[selec]
        ey = nit_elec[7].getey()#[selec]
        ez = nit_elec[7].getez()#[selec]
        bx = nit_elec[7].getbx()#[selec]
        by = nit_elec[7].getby()#[selec]
        bz = nit_elec[7].getbz()#[selec]
        ssnum= nit_elec[7].getpid()#[selec]
        w=nit_elec[7].getw()#[selec]
        if me == 0 :

            # Write the results in a file
            fname = 'N7%06d.pdb' % top.it
            f = PW.PW(fname)

            # Selected quantities
            f.t = top.time
            f.x = x
            f.z = z
            f.y =y
            f.ux = ux
            f.uy = uy
            f.uz = uz
            f.ex = ex
            f.ey = ey
            f.ez = ez
            f.bx = bx
            f.by = by
            f.bz = bz
            f.ssnum =ssnum
            f.w = w

            f.close()

def write_ptcl_N5():
    """
      Write the particles data to a pickle file every hist_freq.
      The data is gathered on the first processor, which writes it to the disk.
    """

    if top.it%hist_freq == 0 :

        # Selection in energy
        uz_cutoff = 1.*clight # 0.5 MeV
        uz = nit_elec[5].getuz()
        selec = (uz > uz_cutoff)

        # Get the data
        z = nit_elec[5].getz()[selec]
        x = nit_elec[5].getx()[selec]
        y = nit_elec[5].gety()[selec]
        ux = nit_elec[5].getux()[selec]/clight
        uy = nit_elec[5].getuy()[selec]/clight
        uz = uz[selec]/clight
        ex = nit_elec[5].getex()[selec]
        ey = nit_elec[5].getey()[selec]
        ez = nit_elec[5].getez()[selec]
        bx = nit_elec[5].getbx()[selec]
        by = nit_elec[5].getby()[selec]
        bz = nit_elec[5].getbz()[selec]
        ssnum= nit_elec[5].getpid()[selec]
        w= nit_elec[5].getw()[selec]
        if me == 0 :

            # Write the results in a file
            fname = 'N5%06d.pdb' % top.it
            f = PW.PW(fname)

            # Selected quantities
            f.t = top.time
            f.x = x
            f.z = z
            f.y = y
            f.ux = ux
            f.uy = uy
            f.uz = uz
            f.ex = ex
            f.ey = ey
            f.ez = ez
            f.bx = bx
            f.by = by
            f.bz = bz
            f.w = w
            f.ssnum=ssnum

            f.close()


def write_fields() :
    """
    Writes a pickle (.pdb) file containing the fields ex, ey, ez, by
    every hist_freq timesteps.
    The pickle also contains the corresponding coordinates x and z,
    and the corresponding extent
    The shape and arrange of the fields is such that it can be directly
    plotted with matplotlib's imshow (non need to transpose it,
    or reverse one dimension)
    """

    if top.it%hist_freq == 0 and dim=='2d' :
    
        fname = 'fields%06d.pdb' % top.it
        if l_circ:
            dens = elec.get_density()
            densN5 = nit_elec[5].get_density()
            densN6 = nit_elec[6].get_density()
            densN7 = nit_elec[7].get_density()
            ex = em.pfex(direction=1, output=True, show=False)
            ey = em.pfey(direction=1, output=True, show=False)
            ez = em.pfez(direction=1, output=True, show=False)
            #bx = em.pfbx(direction=1, output=True, show=False)
            #by = em.pfby(direction=1, output=True, show=False)
            #bz = em.pfbz(direction=1, output=True, show=False)
            #if em.l_getrho :
            #rho = em.pfrho(direction=1, output=True, show=False)
        else:
            dens = elec.get_density()
            ex = em.gatherex()
            ey = em.gatherey()
            ez = em.gatherez()
            rho = em.pfrho(direction=1, output=True, show=False)            

        if me == 0 :
            f = PW.PW(fname)

            f.x = w3d.xmesh
            # Add the values of x below the axis
            if em.l_2drz == 1 :
                f.x = hstack( (-f.x[:0:-1], f.x ) )
            f.z = w3d.zmesh + top.zgrid
            f.extent = array([ w3d.zmmin + top.zgrid,
                               w3d.zmmax + top.zgrid,
                               -w3d.xmmax, w3d.xmmax ])
            f.dens = dens
            f.dens5 = densN5
            f.dens6 = densN6
            f.dens7 = densN7
            f.ex = ex
            f.ey = ey
            f.ez = ez
            #f.bx = bx
            #f.by = by
            #f.bz = bz
            #if em.l_getrho :
            #f.rho = rho

            f.close()

def labdata(z,t,ux,uy,uz,gi,uzfrm):
    if me==0 and l_verbose:print 'enter labdata'
    np = shape(z)[0]
    gammafrm = sqrt(1.+uzfrm**2/clight**2)
    zpr = gammafrm*z-uzfrm*t
    tpr = gammafrm*t-uzfrm*z/clight**2
    setu_in_uzboosted_frame3d(np,ux,uy,uz,gi,uzfrm,gammafrm)
    if me==0 and l_verbose:print 'exit labdata'
    return zpr,tpr,ux,uy,uz
  
def ppz(xscale,msize,view,titles) :
    """
    plots the positions of beam on the Z axis
    """
    z=getz()
    if me==0:
        ppg(z*0,z*xscale,color=red,msize=msize,view=view,titles=titles)
    try:
        z=getz(pgroup=bf.pgroup)
        if me==0:
            ppg(z*0,z*xscale,color=blue,msize=msize,view=view,titles=titles)
    except:
        pass

def pzx(msize=1,color=red,titles=1,xscale=1.,yscale=1.,view=1):
    """
    plots ZX projection of e- beam
    """
    if not l_beam:return
    ppzx(msize=msize,color=color,titles=titles,xscale=xscale,yscale=yscale,view=view)
    try:
        ppzx(pgroup=bf.pgroup,color=blue,msize=msize,titles=titles,xscale=xscale,yscale=yscale,view=view)
    except:
        pass
      
def pzy(msize=1,color=red,titles=1,xscale=1.,yscale=1.,view=1):
    """
    plots ZY projection of e- beam
    """
    if not l_beam:return
    ppzy(msize=msize,color=color,titles=titles,xscale=xscale,yscale=yscale,view=view)
    try:
        ppzy(pgroup=bf.pgroup,color=blue,msize=msize,titles=titles,xscale=xscale,yscale=yscale,view=view)
    except:
        pass
      
def pxy(msize=1,color=red,titles=1,xscale=1.,yscale=1.,view=1):
    """
    plots XY projection of e- beam
    """
    if not l_beam:return
    ppxy(msize=msize,color=color,titles=titles,xscale=xscale,yscale=yscale,view=view)
    try:
        ppxy(pgroup=bf.pgroup,color=blue,msize=msize,titles=titles,xscale=xscale,yscale=yscale,view=view)
    except:
        pass
      
def pzxex(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    if dim=='1d':
        em.pfex(l_transpose=dim<>'1d',direction=1,view=view,titles=titles)
        ppz(xscale,msize,view,titles)
    elif circ_m > 0 :
        em.pfex(direction=1,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzx(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    else:
        em.pfex(l_transpose=dim<>'1d',direction=1,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzx(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    
def pzxey(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    if dim=='1d':
        em.pfey(l_transpose=dim<>'1d',direction=1,view=view,titles=titles)
        ppz(xscale,msize,view,titles)
    elif circ_m > 0 :  
        em.pfey( direction=1,view=view,titles=titles,
                 xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzx(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    else:  
        em.pfey( l_transpose=dim<>'1d',direction=1,view=view,titles=titles,xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzx(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    
def pzxez(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    if dim=='1d':
        em.pfez(l_transpose=dim<>'1d',direction=1,view=view,titles=titles)
        ppz(xscale,msize,view,titles)
    elif circ_m > 0 :
        em.pfez( direction=1,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzx(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    else :
        em.pfez( l_transpose=dim<>'1d',direction=1,view=view,titles=titles,xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzx(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
      
def pzyex(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    if dim=='1d':
        em.pfex(l_transpose=dim<>'1d',direction=0,view=view,titles=titles)
        ppz(xscale,msize,view,titles)
    elif circ_m > 0 :
        em.pfex(direction=0,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    else:
        em.pfex(l_transpose=dim<>'1d',direction=0,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    
def pzyey(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    if dim=='1d':
        em.pfey(l_transpose=dim<>'1d',direction=0,view=view,titles=titles)
        ppz(xscale,msize,view,titles)
    elif circ_m > 0 :
        em.pfey(direction=0,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    else:
        em.pfey(l_transpose=dim<>'1d',direction=0,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    
def pzyez(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    if dim=='1d':
        em.pfez(l_transpose=dim<>'1d',direction=0,view=view,titles=titles)
        ppz(xscale,msize,view,titles)
    elif circ_m > 0 :
        em.pfez(direction=0,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    else:
        em.pfez(l_transpose=dim<>'1d',direction=0,view=view,titles=titles,
                xscale=xscale,yscale=yscale,gridscale=gridscale)
        pzy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
      
def pxyex(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    em.pfex(l_transpose=1,direction=2,view=view,titles=titles,xscale=xscale,yscale=yscale,gridscale=gridscale)
    pxy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
  
def pxyey(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    em.pfey(l_transpose=1,direction=2,view=view,titles=titles,xscale=xscale,yscale=yscale,gridscale=gridscale)
    pxy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
  
def pxyez(msize=1,titles=1,xscale=1.,yscale=1.,view=1,gridscale=1.):
    em.pfez(l_transpose=1,direction=2,view=view,titles=titles,xscale=xscale,yscale=yscale,gridscale=gridscale)
    pxy(msize=msize,titles=0,view=view,xscale=xscale,yscale=yscale)
    
zstart0lab=0.
dzstations=Lplasma_lab/nzstations
beamzstations = zstart0lab+arange(0.,Lplasma_lab,dzstations) # list of diag stations z-locations in lab frame

ekstations = zeros(shape(beamzstations),'d')
ppzstations = zeros(shape(beamzstations),'d')
xbarstations = zeros(shape(beamzstations),'d')
xpbarstations = zeros(shape(beamzstations),'d')
xsqstations = zeros(shape(beamzstations),'d')
xpsqstations = zeros(shape(beamzstations),'d')
xxpstations = zeros(shape(beamzstations),'d')
if dim == "3d":
    ybarstations = zeros(shape(beamzstations),'d')
    ypbarstations = zeros(shape(beamzstations),'d')
    ysqstations = zeros(shape(beamzstations),'d')
    ypsqstations = zeros(shape(beamzstations),'d')
    yypstations = zeros(shape(beamzstations),'d')
tbarstations = zeros(shape(beamzstations),'d')
tsqstations = zeros(shape(beamzstations),'d')
ekstationstime = zeros(shape(beamzstations),'d')
ekstationscnt = zeros(shape(beamzstations),'d')
ekstationscnt2 = zeros(shape(beamzstations),'d')
npz = 1001
pzbeamstations = (200.e6/dfact*arange(npz))/(npz-1)
pzstations = zeros((shape(beamzstations)[0],npz),'d')

top.zoldpid=nextpid()

def save_elec():
	global timestart
	if (top.it %50==0):
            fname = 'ptcl%06d.pdb' % top.it
            uzfrm=-betafrm*gammafrm*clight
            z=getz(gather=0,bcast=0).copy()
            xlab = getx(gather=0,bcast=0).copy()
            ylab = gety(gather=0,bcast=0).copy()
            zlab,tlab,uxlab,uylab,uzlab = labdata(z,
                                        top.time,
                                        getux(gather=0,bcast=0).copy(),
                                        getuy(gather=0,bcast=0).copy(),
                                        getuz(gather=0,bcast=0).copy(),
                                        getgaminv(gather=0,bcast=0).copy(),
                                        uzfrm=uzfrm)
            if me ==0:
                f = PW.PW(fname)
                f.x = xlab
                f.y = ylab
                f.z = zlab
                f.ux = uxlab
                f.uy = uylab
                f.uz = uzlab
                f.tlab= tlab
                f.z_boosted =  getz(gather=0,bcast=0).copy()
                f.uz_boosted =  getuz(gather=0,bcast=0).copy()
                f.close()
  
xs=AppendableArray()
ys=AppendableArray()
zs=AppendableArray()
uxs=AppendableArray()
uys=AppendableArray()
uzs=AppendableArray()
ws =AppendableArray()

def updatebeamstations():
    global timestart
#  if top.it%10<>0:return
    if me==0 and l_verbose:print 'enter updatebeamstations'
    # --- compute beta*gamma*c
    uzfrm=-betafrm*gammafrm*clight
    # --- get nb particles on each CPU
    np = getn(gather=0,bcast=0) 
    if np>0:
        # --- get z on each CPU
        z=getz(gather=0,bcast=0).copy()
        zold=getpid(id=top.zoldpid-1,gather=0,bcast=0)
        zoldlab = gammafrm*zold-uzfrm*(top.time-top.dt)
        # --- get z, time and velocities in lab frame
        zlab,tlab,uxlab,uylab,uzlab = labdata(z,
                                              top.time,
                                              getux(gather=0,bcast=0).copy(),
                                              getuy(gather=0,bcast=0).copy(),
                                              getuz(gather=0,bcast=0).copy(),
                                              getgaminv(gather=0,bcast=0).copy(),
                                              uzfrm=uzfrm)
                                          
        w = abs(zlab-zoldlab)/dzstations
        # --- get x,y on each CPU
        x = getx(gather=0,bcast=0).copy()
        y = gety(gather=0,bcast=0).copy()
        ii= compress(((zoldlab<Lplasma_lab/2) & (zlab>Lplasma_lab/2)), arange(np))
      	
        if len(ii)>0:
        	xs.append(take(x,ii))
        	ys.append(take(y,ii))
        	zs.append(take(zlab,ii))
        	uxs.append(take(uxlab,ii))
        	uys.append(take(uylab,ii))
        	uzs.append(take(uzlab,ii))
        	#ws.append(take(w,ii))
        		
        # --- compute gamma in lab frame
        myglab = sqrt(1.+(uxlab**2+uylab**2+uzlab**2)/clight**2)
        # --- compute kinetic energy in lab frame
        mykelab = beam.sm*(myglab-1.)*clight**2/echarge
        # --- defines cutoffs if particle selection is ON
        if l_pselect:
            # --- set threshold on radius
            XYcutoff = E_BEAM_RADIUS*5.
            # --- set threshold on longitudinal velocity
            UZcutoff = 0.95*E_BEAM_GAMMA*E_BEAM_BETA*clight
            # --- set threshold on energy
            KEcutoff = 50.e6 # eV
            XYcutoff = None
            UZcutoff = None
            #KEcutoff = None
      			
        else:
            XYcutoff = None
            UZcutoff = None
            KEcutoff  = None
        if XYcutoff is not None:
            # --- select particle based on radius
            if dim=="3d":
                r2 = x*x+y*y
                XYcutoff2 = XYcutoff**2
                ii = compress(r2<XYcutoff2,arange(np))
            else:
                ii = compress(abs(x)<XYcutoff,arange(np))
            # --- get # of selected particles
            np = len(ii)
            # --- get weight, position, time, velocity and energy of selected particles
            w = take(w,ii)
            x = take(x,ii)
            y = take(y,ii)
            zlab = take(zlab,ii)
            tlab = take(tlab,ii)
            uxlab = take(uxlab,ii)
            uylab = take(uylab,ii)
            uzlab = take(uzlab,ii)
            mykelab = take(mykelab,ii)
        if UZcutoff is not None:
            # --- select particle based on longitudinal velocity
            ii = compress(uzlab>UZcutoff,arange(np))
            # --- get # of selected particles
            np = len(ii)
            # --- get weight, position, time, velocity and energy of selected particles
            w = take(w,ii)
            x = take(x,ii)
            y = take(y,ii)
            zlab = take(zlab,ii)
            tlab = take(tlab,ii)
            uxlab = take(uxlab,ii)
            uylab = take(uylab,ii)
            uzlab = take(uzlab,ii)
            mykelab = take(mykelab,ii)
        if KEcutoff is not None:
            # --- select particle based on energy
            ii = compress(mykelab>KEcutoff,arange(np))
            # --- get # of selected particles
            np = len(ii)
            # --- get weight, position, time, velocity and energy of selected particles
            w = take(w,ii)
            x = take(x,ii)
            y = take(y,ii)
            zlab = take(zlab,ii)
            tlab = take(tlab,ii)
            uxlab = take(uxlab,ii)
            uylab = take(uylab,ii)
            uzlab = take(uzlab,ii)
            mykelab = take(mykelab,ii)
        if np>0:
            xplab = uxlab/clight # normalized (gamma*beta*xp)
            yplab = uylab/clight # normalized (gamma*beta*yp) 
            nz = shape(ekstations)[0]
            deposgrid1dw(1,np,zlab,mykelab,w,nz-1,ekstations,ekstationscnt,beamzstations[0],beamzstations[-1])
            deposgrid1dw(1,np,zlab,beam.sm*uzlab*clight/echarge,w,nz-1,ppzstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            deposgrid1dw(1,np,zlab,x,w,nz-1,xbarstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            deposgrid1dw(1,np,zlab,x**2,w,nz-1,xsqstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            deposgrid1dw(1,np,zlab,xplab,w,nz-1,xpbarstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            deposgrid1dw(1,np,zlab,xplab**2,w,nz-1,xpsqstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            deposgrid1dw(1,np,zlab,x*xplab,w,nz-1,xxpstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            if dim == "3d":
                deposgrid1dw(1,np,zlab,y,w,nz-1,ybarstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
                deposgrid1dw(1,np,zlab,y**2,w,nz-1,ysqstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
                deposgrid1dw(1,np,zlab,yplab,w,nz-1,ypbarstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
                deposgrid1dw(1,np,zlab,yplab**2,w,nz-1,ypsqstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
                deposgrid1dw(1,np,zlab,y*yplab,w,nz-1,yypstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            deposgrid1dw(1,np,zlab,tlab,w,nz-1,tbarstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            deposgrid1dw(1,np,zlab,tlab**2,w,nz-1,tsqstations,ekstationscnt2,beamzstations[0],beamzstations[-1])
            setgrid2dw(np,zlab,uzlab*beam.sm*clight/echarge,w,nz-1,npz-1,pzstations,
                        beamzstations[0],beamzstations[-1],pzbeamstations[0],pzbeamstations[-1])
    if top.it%hist_freq==0:
        savebeamstations()
    if me==0 and l_verbose:print 'exit updatebeamstations'
  

def savebeamstations():
    if me==0 and l_verbose:print 'enter savebeamstations'
    pnums = parallelsum(ekstationscnt)
    pnum = where(pnums==0.,1.,pnums)
    ekst = parallelsum(ekstations)/pnum
    ppzst = parallelsum(ppzstations)/pnum
    xbar = parallelsum(xbarstations)/pnum
    xsq = parallelsum(xsqstations)/pnum
    xpbar = parallelsum(xpbarstations)/pnum
    xpsq = parallelsum(xpsqstations)/pnum
    xxp = parallelsum(xxpstations)/pnum
    if dim == "3d":
        ybar = parallelsum(ybarstations)/pnum
        ysq = parallelsum(ysqstations)/pnum
        ypbar = parallelsum(ypbarstations)/pnum
        ypsq = parallelsum(ypsqstations)/pnum
        yyp = parallelsum(yypstations)/pnum
    tbar = parallelsum(tbarstations)/pnum
    tsq = parallelsum(tsqstations)/pnum
    wti  = parallelsum(ekstationstime)/pnum
    pzst = parallelsum(pzstations)#*beam.sw
    if me==0:
        os.system('mv -f ebeamstations.pdb ebeamstationsold.pdb')
        f = PW.PW('ebeamstations.pdb')
        f.ekstations = ekst
        f.ppzstations = ppzst
        f.xbarstations = xbar
        f.xrmsstations = sqrt(xsq-xbar**2)
        f.xpbarstations = xpbar
        f.xprmsstations = sqrt(xpsq-xpbar**2)
        f.xxpstations   = xxp
        f.xemitnstations = sqrt((xsq-xbar*xbar)*(xpsq-xpbar*xpbar)-(xxp-xbar*xpbar)**2)
        if dim == "3d":
            f.ybarstations = ybar
            f.yrmsstations = sqrt(ysq)
            f.ypbarstations = ypbar
            f.yprmsstations = sqrt(ypsq)
            f.yemitnstations = sqrt((ysq-ybar*ybar)*(ypsq-ypbar*ypbar)-(yyp-ybar*ypbar)**2)
        f.tbarstations = tbar
        f.trmsstations = sqrt(tsq-tbar*tbar)
        f.ekstationstime = wti
        f.pzstations = pzst
        f.beamzstations = beamzstations-zstart0lab
        f.pzbeamstations = pzbeamstations
        f.pnumstations = pnums
        f.nx = w3d.nx
        f.ny = w3d.ny
        f.nz = w3d.nz
        f.time = top.time
        f.dt = top.dt
        f.it = top.it
        f.stencil=stencil
        f.dim=dim
        f.xs =xs[...]
        f.ys = ys[...]
        f.zs= zs[...]
        f.uxs =uxs[...]
        f.uys = uys[...]
        f.uzs= uzs[...]
        f.ws = ws[...]
        f.close()
        os.system('rm -f ebeamstationsold.pdb')
    if me==0 and l_verbose:print 'exit savebeamstations'
     
def plke(view=1):
    global kelab,pxlab,pylab,pzlab,zhlab
    if me==0 and l_verbose:print 'enter plke'
    ekcnt = parallelsum(ekstationscnt)
    ekcnt = where(ekcnt==0.,1.,ekcnt)
    ekst = parallelsum(ekstations)/ekcnt
    if me==0:
        plsys(view)
        pla(ekst*1.e-6,beamzstations*1.e3,color=red)
        ptitles('Energy (MeV)','Z (mm)','')
    if me==0 and l_verbose:print 'exit plke'
   
if nzfieldstations>0:
    zstations = arange(0.,Lplasma_lab,Lplasma_lab/nzfieldstations) # list of field diag stations z-locations in lab frame
    exstations = []
    eystations = []
    ezstations = []
    bxstations = []
    bystations = []
    bzstations = []
    tstations = []
    for i in range(shape(zstations)[0]):
        exstations.append(AppendableArray(typecode='d'))
        eystations.append(AppendableArray(typecode='d'))
        ezstations.append(AppendableArray(typecode='d'))
        bxstations.append(AppendableArray(typecode='d'))
        bystations.append(AppendableArray(typecode='d'))
        bzstations.append(AppendableArray(typecode='d'))
        tstations.append(AppendableArray(typecode='d'))
    
    def updateebstations():
        #  --- routine for accumulating EM field value at z locations (array zstations)
        global em,hist_freq
        if me==0 and l_verbose:print 'enter updateebstations'
        # --- compute z in calculation (boosted) frame
        zcalc = zstations/gammafrm-betafrm*clight*top.time
      
        # --- select z that are within grid range
        n = shape(zcalc)[0]
        ilist = compress((zcalc>=(w3d.zmmin+top.zgrid)) & (zcalc<(w3d.zmmax+top.zgrid)),arange(n))
        zcalc = take(zcalc,ilist)
        n = shape(zcalc)[0]
        if n==0:return
      
        # --- gather EM fields in calculation frame
        x=zcalc*0.
        y=zcalc*0.
        ex,ey,ez,bx,by,bz = em.getfieldsfrompositions(x,y,zcalc)
      
        if me==0:
            uxf = 0.
            uyf = 0.
            uzf = -gammafrm*betafrm*clight
            gammaf = gammafrm
            # --- convert EM fields to lab frame
            seteb_in_boosted_frame(n,ex,ey,ez,bx,by,bz,uxf,uyf,uzf,gammaf)
            # --- compute time in lab frame
            t = gammafrm*(top.time+betafrm*zcalc/clight)
            # --- store field and time values in appendable arrays
            for j,i in enumerate(ilist):
                tstations[i].append(t[j])
                exstations[i].append(ex[j])
                eystations[i].append(ey[j])
                ezstations[i].append(ez[j])
                bxstations[i].append(bx[j])
                bystations[i].append(by[j])
                bzstations[i].append(bz[j])
        
            # --- save data every hist_freq time steps
            if top.it%hist_freq==0:
                saveebstations()
        if me==0 and l_verbose:print 'exit updateebstations'
      
    def  saveebstations():
        if me>0:return
        if me==0 and l_verbose:print 'enter saveebstations'
        fname='ebhist.pdb'
        os.system('mv -f ebhist.pdb ebhistold.pdb')
        f=PW.PW(fname)
        ex = []
        ey = []
        ez = []
        bx = []
        by = []
        bz = []
        t = []
        for i in range(len(exstations)):
            ex.append(exstations[i][:])
            ey.append(eystations[i][:])
            ez.append(ezstations[i][:])
            bx.append(bxstations[i][:])
            by.append(bystations[i][:])
            bz.append(bzstations[i][:])
            t.append(tstations[i][:])
        f.ex=ex
        f.ey=ey
        f.ez=ez
        f.bx=bx
        f.by=by
        f.bz=bz
        f.t=t
        f.z=zstations
      
        f.close()
        os.system('rm -f ebhistold.pdb')
        if me==0 and l_verbose:print 'exit saveebstations'
