################################
# Warp plotting script alpha...
# M. Kirchen, S. Jalas
################################

import matplotlib

matplotlib.use('Agg')

import os
import os.path
import sys
from pylab import *
import numpy as np
import h5py

import cubehelix
import shiftcmap
import constants as cte
import time

from matplotlib import cm
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LightSource

import argparse
import copy
import logging

from matplotlib import ticker

rcparams = open('rcparams', 'r')
rc = mpl.rc_params_from_file('rcparams')
rcParams.update(rc)

#########################
# SCROLL DOWN for Setup!!
#########################

#class
class Cmap:
    def __init__(self, cmap=None):

        cmap = str(cmap)

        try:
            self.cmap = plt.get_cmap(cmap)
            print "Loaded Cmap: %s" % (str(cmap))
        except:
            self.cmap = None
        return

    def getMplCmap(self, cmap=None):

        cmap = str(cmap)
        self.cmap = plt.get_cmap(cmap)

    def alphaCmap(self):

        self.cmap._init()
        alphas = np.abs(np.linspace(-pi, pi, self.cmap.N))
        alphas = (-cos(alphas) + 1) / 2

        # alphas = np.abs(np.linspace(-1.0, 1.0, self.cmap.N))
        # mask1 = alphas[:] < 0.5
        # mask2 = alphas[:] > -0.5
        # mask_total = mask1 & mask2
        # alphas[mask_total] = 0
        self.cmap._lut[:-3, -1] = alphas

    def alphaCmapzero(self):

        self.cmap._init()
        alphas = np.abs(np.linspace(0, 1, self.cmap.N))
        #alphas = (-cos(alphas/2) + 1) / 2

        # alphas = np.abs(np.linspace(-1.0, 1.0, self.cmap.N))
        # mask1 = alphas[:] < 0.5
        # mask2 = alphas[:] > -0.5
        # mask_total = mask1 & mask2
        # alphas[mask_total] = 0
        self.cmap._lut[:-3, -1] = alphas

    def cubehelixCmap(self, **args):
        self.cmap = cubehelix.cmap(**args)

    def shiftCmap(self, **args):

        newcmap = shiftcmap.shiftedColorMap(self.cmap, **args)
        self.getMplCmap(cmap='shiftedcmap')

class Data:
    def __init__(self, runid=None, subfolder=None, fileselection=None, datadict=None):
				
        self.runid = runid

        if subfolder is not None:
            self.subfolder = str(subfolder)
            print "Loading files from subfolder: %s" % (str(subfolder))
        else:
            self.subfolder = '.'
        self.mainfolder = '%s' % (self.subfolder)
        print "Loading files from folder: %s" % self.mainfolder
        self.datadict = datadict
        print "Loading datadict: %s" % datadict
        self.fileselection = fileselection
        print "Selecting files: %s" % fileselection
        self.files = None
        self.dset = None

    def readFiles(self):

        print 'Reading files...'
        self.files = np.empty([len(self.fileselection), len(self.datadict)], dtype='object')
        for i, filenumber in enumerate(self.fileselection):
            for j, datatype in enumerate(self.datadict):
                path = '%s/%s/%s-%s-%04d.h5' % (self.mainfolder, datatype, self.runid, datatype, filenumber)
                print 'Opening File:' + path
                self.files[i][j] = [h5py.File(path, 'r'), datatype, filenumber]
        print 'Reading files completed!'
        return self.files

    def readDatasets(self):

        print 'Reading datasets...'
        self.dset = np.empty(len(self.files), dtype='object')
        for i, datatypeset in enumerate(self.files):
            self.dset[i] = {}
            for datatype in datatypeset:
                for dataset in self.datadict[datatype[1]]:
                    attributes = self.readAttributes(datatype[0][dataset])
                    self.dset[i][dataset] = datatype[0][dataset], datatype[1], datatype[2], attributes
        print 'Reading datasets completed!'
        return self.dset

    def readAttributes(self, dataset=None):

        attributes = {}
        for attribute in dataset.attrs:
            attributes[attribute] = dataset.attrs[attribute]
        return attributes

class Plot:
    def __init__(self, ax=gca(), datasets=None, filenumber=None):

        self.i = filenumber
        self.dset = datasets[self.i]
        self.plot = None
        self.ax = ax

    def colorPlot(self, dataset=None, cmap=None, alpha=1., limits=None):

        X,Y = np.mgrid[:dataset.shape[0],:dataset.shape[1]]
        # dataset = dataset/max(dataset.flatten())
        # self.plot = self.ax.plot_surface(-X,-Y,dataset, rstride=10, cstride=10 , cmap=cmap, linewidth=0, antialiased=False)
        self.plot = self.ax.imshow(dataset, aspect='auto', interpolation='bicubic', alpha=alpha, cmap=cmap,extent=limits)

        return self.plot

    def hexbinPlot(self, dataset=None, grids=80, alpha=1.0, limits=None):
        if len(dataset[0]) == 0:
            dataset = [[-1], [-1], [0]]
        self.plot = self.ax.hexbin(x=dataset[0], y=dataset[1], gridsize=grids, alpha=alpha, extent=limits,
                                   cmap=plt.get_cmap('bone_r'), bins='log')

        return self.plot

    def scatterPlot(self, dataset=None, s=13, alpha=1.0):

        self.plot = self.ax.scatter(dataset[0], dataset[1], c=dataset[2], alpha=alpha, s=s, cmap=plt.get_cmap('PuRd'), linewidths=0)
        return self.plot

    def potentialPlot(self, ):

        return

    def phasespacePlot(self, ):

        return

    def energyHistogram(self, ):

        return

    def linePlot(self, ydataset=None, xdataset=None, color=None, dashed=False):
        if xdataset is None:
            self.plot = self.ax.plot(ydataset, linewidth=0.5, color=color)
        else:
            self.plot = self.ax.plot(xdataset, ydataset, linewidth=0.5, color=color)
        return self.plot

    def dephasingPoint(self):

        return

    def horizontalLine(self, transposition=0, xmin=None, xmax=None):

        if xmin is None: xmin = self.ax.get_xlim()[0]
        if xmax is None: xmin = self.ax.get_xlim()[1]
        self.plot = self.ax.hlines(transposition, xmin, xmax, color='gray', linestyle='--', alpha=0.7)

        return self.plot

    def verticalLine(self, zposition=None, ymin=None, ymax=None):

        if ymin is None: ymin = self.ax.get_xlim()[0]
        if ymax is None: ymin = self.ax.get_xlim()[1]
        self.plot = self.ax.vlines(zposition, ymin, ymax, color='w', linestyle='--', alpha=0.7)

        return self.plot

class PlotData:
    def __init__(self, datasets=None, filenumber=None):
        self.i = filenumber
        self.dset = datasets[self.i]

    def fieldMap(self, dataset='ne', slice='x', sliceposition=0, slicehalf=True, delta_sliceposition=0):

        if slicehalf:
            if slice == 'x': sliceposition = int(shape(self.dset[dataset][0])[1] / 2.)
            if slice == 'y': sliceposition = int(shape(self.dset[dataset][0])[0] / 2.)
            if slice == 'z': sliceposition = int(shape(self.dset[dataset][0])[2] / 2.)
        else:
            sliceposition = sliceposition
        if slice == 'x': fieldmapdata = self.dset[dataset][0][:, sliceposition + delta_sliceposition, :]
        if slice == 'y': fieldmapdata = self.dset[dataset][0][sliceposition + delta_sliceposition, :, :]
        if slice == 'z': fieldmapdata = self.dset[dataset][0][:, :, sliceposition + delta_sliceposition]
        return fieldmapdata

    def beamScatter(self, xvalue='z', yvalue='x', zvalue=None):

        x = self.dset[xvalue][0][:]
        y = self.dset[yvalue][0][:]
        if zvalue is None:
            z = u'k'
        else:
            z = self.dset[zvalue][0][:]
        beamdata = np.array([x, y, z])
        return beamdata

    def longSlice(self, dataset='ne', slice='x', slicehalf=True, sliceposition=0, delta_sliceposition=0,
                  lineoutposition=0, delta_lineoutposition=0, lineoutcenter=True):

        if lineoutcenter:
            if slice == 'x': lineoutposition = int(shape(self.dset[dataset][0])[0] / 2.)
            if slice == 'y': lineoutposition = int(shape(self.dset[dataset][0])[1] / 2.)
        else:
            lineoutposition = lineoutposition

        longslicedata = self.fieldMap(dataset=dataset, slice=slice, slicehalf=slicehalf, sliceposition=sliceposition,
                                      delta_sliceposition=delta_sliceposition)[lineoutposition + delta_lineoutposition,
                        :]

        return longslicedata

    def transSlice(self, dataset='ne', slice='x', slicehalf=True, sliceposition=0, delta_sliceposition=0,
                   lineoutposition=0, delta_lineoutposition=0):

        zposition = lineoutposition + delta_lineoutposition
        transslicedata = self.fieldMap(dataset=dataset, slice=slice, slicehalf=slicehalf, sliceposition=sliceposition,
                                       delta_sliceposition=delta_sliceposition)[:, zposition]

        return transslicedata

    def getAttribute(self, dataset='ne', attr='x_lim'):
        if attr == 'shape':
            attribute = shape(self.dset[dataset][0])
        if attr == 'time':
            attribute = self.dset[dataset][3]['time']
        if attr == 'x_lim':
            attribute = [self.dset[dataset][3]['xmin'], self.dset[dataset][3]['xmax'], shape(self.dset[dataset][0])[0]]
        if attr == 'y_lim':
            attribute = [self.dset[dataset][3]['ymin'], self.dset[dataset][3]['ymax'], shape(self.dset[dataset][0])[1]]
        if attr == 'z_lim':
            attribute = [self.dset[dataset][3]['zmin'], self.dset[dataset][3]['zmax'], shape(self.dset[dataset][0])[2]]

        return attribute

    def getDephasingPoint(self):

        fieldslice = self.longSlice(dataset='Ez', slice='x')
        zlim = self.getAttribute(dataset='Ez', attr='z_lim')
        deps = []
        asign = np.sign(fieldslice)
        a = asign[0]

        for i in range(len(asign)):
            b = asign[i]
            if a < b:
                deps.append(1)
            else:
                deps.append(0)
            a = asign[i]

        deppoints = np.linspace(zlim[0], zlim[1], len(deps))
        deppoints = np.multiply(deps, deppoints)
        deppoints = np.delete(deppoints, np.where(deppoints == 0))

        return deppoints

    def getPulseDuration(self, mode='fwhm', dataset='Ex'):

        if mode == 'sigma':
            integr = np.trapz
            env = self.getLaserEnvelope()
            field = self.longSlice(dataset=dataset)
            axisdata = self.getAttribute(dataset=dataset, attr='z_lim')

            z = linspace(axisdata[0], axisdata[1], len(env))
            span = axisdata[1] - axisdata[0]
            dz = field[1] - field[0]
            intensity = env ** 2
            # normalization
            koef = cte.c ** 2 * integr(intensity, dx=dz)
            koefi = 1 / koef
            mean_time = cte.c * koefi * integr(z * intensity, dx=dz)

            variance = koefi * 8 * np.log(2) * integr((z - cte.c * mean_time) ** 2 * intensity, dx=dz)
            pulseduration = np.sqrt(variance)

        if mode == 'fwhm':
            env = self.getLaserEnvelope()
            intensity = env ** 2
            axisdata = self.getAttribute(dataset=dataset, attr='z_lim')
            z = linspace(axisdata[0], axisdata[1], len(env))

            half_max = max(intensity) / 2.
            # find when function crosses line half_max (when sign of diff flips)
            # take the 'derivative' of signum(half_max - Y[])
            d = sign(half_max - array(intensity[0:-1])) - sign(half_max - array(intensity[1:]))
            # plot(X,d) #if you are interested
            # find the left and right most indexes
            left_idx = find(d > 0)[0]
            right_idx = find(d < 0)[-1]
            pulseduration = (z[right_idx] - z[left_idx]) / cte.c  # return the difference (full width)

        return pulseduration

    def geta0(self, dataset='Ex'):

        field = self.longSlice(dataset=dataset)
        omega = self.getMeanFreq(dataset=dataset)

        a0_positive = max(field) * cte.e / (cte.m_e * cte.c * omega)
        a0_negative = - min(field) * cte.e / (cte.m_e * cte.c * omega)

        a0 = (a0_positive + a0_negative) / 2

        return a0

    def getLaserWaist(self, dataset='Ex', mode='sigma', slice='x', turbomode=True):

        if turbomode is True:
            field = self.fieldMap(dataset=dataset, slice=slice)
            slicemax = []
            for i in range(shape(field)[0]):
                slice = field[i, :]
                peak = (max(slice) - min(slice)) / 2.
                slicemax.append(peak)
        else:
            env = self.getLaserEnvelope2D(dataset=dataset, slice=slice)
            slicemax = []
            for i in range(shape(env)[0]):
                slice = env[i, :]
                slicemax.append(max(slice))

        if mode == 'sigma':
            integr = np.trapz
            axisdata = self.getAttribute(dataset=dataset, attr='x_lim')

            x = linspace(axisdata[0], axisdata[1], len(slicemax))
            span = axisdata[1] - axisdata[0]
            dx = span / len(slicemax)

            variance = (1 / integr(slicemax, dx=dx)) * integr(x ** 2 * slicemax, dx=dx)
            laser_waist = sqrt(2)*np.sqrt(variance)  # gaussian sigma

        if mode == 'fwhm':
            axisdata = self.getAttribute(dataset=dataset, attr='x_lim')
            x = linspace(axisdata[0], axisdata[1], len(slicemax)*100)
            intdata = np.power(slicemax, 1)
            intensity = np.interp(np.linspace(0, len(intdata), len(intdata)*100), range(len(intdata)), intdata)


            half_max = max(intensity) / 2.
            # find when function crosses line half_max (when sign of diff flips)
            # take the 'derivative' of signum(half_max - Y[])
            d = sign(half_max - array(intensity[0:-1])) - sign(half_max - array(intensity[1:]))
            # plot(X,d) #if you are interested
            # find the left and right most indexes
            left_idx = find(d > 0)[0]
            right_idx = find(d < 0)[-1]
            laser_waist = (x[right_idx] - x[left_idx])# / 2.35  # return the difference (full width)

        return laser_waist

    def getLaserEnvelope2D(self, dataset='Ex', slice='x', window_bounds_percent=40, downsamplerate=10):

        size_x, size_y, size_z = self.getAttribute(dataset=dataset, attr='shape')

        if slice == 'x': size = size_x
        if slice == 'y': size = size_y
        lineoutposition = -size / 2

        size /= downsamplerate

        env2d = np.zeros((size, size_z))

        for ind in range(size):
            env2d[ind, :] = self.getLaserEnvelope(dataset=dataset, slice=slice, lineoutposition=lineoutposition,
                                                  window_bounds_percent=window_bounds_percent)
            # env2d[ind, :] = self.getLaserEnvelope(dataset=dataset, slice=slice, lineoutposition = lineoutposition, window_bounds_percent =  window_bounds_percent)

            lineoutposition += 1 * downsamplerate

        return env2d

    def getMeanFreq(self, dataset='Ex', slice='x'):

        """
        See A. Beck et al. / Nuclear Instruments and Methods in Physics Research A 740 (2014) 67-73
        Equation (5)
                w      _              _
                 0    /  infin       /  infin             2
        <w(x)> = -----  |        r dr  |        w |a(x,r,w)|  dw
               2    _/  0          _/  0
              W (x)
        """
        integr = np.trapz

        data = self.longSlice(dataset=dataset, slice=slice)
        zlim = self.getAttribute(dataset=dataset, attr='z_lim')

        z_min = zlim[0]
        z_max = zlim[1]

        span = z_max - z_min
        N = data.size

        data_FT = np.fft.fft(data)
        data_freq = 2 * np.pi * np.fft.fftfreq(N, span / N)

        # taking intensity
        data_FT = (abs(data_FT[:N / 2])) ** 2
        data_freq = data_freq[:N / 2]

        # normalization
        dx = data_freq[1] - data_freq[0]
        koef = integr(data_FT, dx=dx)

        # mean calculation
        mean = cte.c * (1 / koef) * integr(data_freq * data_FT, dx=dx)

        return mean

    def getLaserEnvelope(self, dataset='Ex', slice='x', lineoutposition=0, window_bounds_percent=40):
        E = self.longSlice(dataset=dataset, slice=slice, lineoutposition=lineoutposition)
        size = E.size

        freqs = np.fft.fftfreq(size)  # , dz
        E_FT = np.fft.fft(E)

        central_freq_ind = np.argmax(E_FT[:size / 2])  # only values with negative frequency
        central_freq = freqs[central_freq_ind]

        left_bound_freq = central_freq * window_bounds_percent / 100.
        left_bound_freq_ind = np.argmin(abs(freqs - left_bound_freq))

        win_half_width_ind = central_freq_ind - left_bound_freq_ind

        E_FT_cut = np.zeros(size, dtype=np.complex)
        E_FT_cut[size / 2 - win_half_width_ind:size / 2 + win_half_width_ind] = E_FT[
                                                                                central_freq_ind - win_half_width_ind:central_freq_ind + win_half_width_ind]

        E_envelope = np.abs(np.fft.ifft(np.fft.fftshift(2 * E_FT_cut)))

        return E_envelope

    def getWignerTransform(self, dataset='Ex', slice='x', downsampling=10, **params):
        """
      See Trebino, R: Frequency Resolved Optical Gating: The measurements of Ultrashort Laser Pulses: year 2000: formula 5.2

      Uses laserEnvelope algorithm that takes window_bounds_percent as a parameter - consider correct value for this parameter.

      Obtained envelope is normalized by L2 norm then it is used for wigner transform.

      Resulting wigner transform is not taken to power of two.

      """
        window_bounds_percent = params['window_bounds_percent']
        E = self.fieldMap(dataset=dataset, slice=slice, sliceposition=0)

        res_x, t_res = E.shape
        t_res = int(round(t_res / downsampling))+1

        axisdata = self.getAttribute(dataset=dataset, attr='z_lim')

        t_min = axisdata[0] / cte.c
        t_max = axisdata[1] / cte.c
        t_range = t_max - t_min
        dt = t_range / t_res

        nyquist = np.pi * t_res / t_range

        omega = np.linspace(-nyquist, nyquist, t_res)
        # take slice in the middle of x

        E = self.longSlice(dataset=dataset, slice=slice)

        E = E[::downsampling]

        E_env = self.getLaserEnvelope(dataset=dataset, slice=slice, window_bounds_percent=window_bounds_percent)
        E_env = E_env[::downsampling]
        E_env /= np.sqrt(np.trapz(E_env ** 2, dx=dt))

        E_shift = np.zeros(t_res)

        wigner = np.zeros((t_res * 2, t_res))

        for i in range(t_res * 2):

            itau = i % t_res

            if i < t_res:

                E_shift[:itau] = E_env[t_res - itau: t_res]

                E_shift[itau:] = 0  # possibility to make faster by nullifying only one value

            else:

                E_shift[itau:] = E_env[: t_res - itau]

                E_shift[:itau] = 0

            EE = E * E_shift ** 2

            fftwigner = np.fft.fft(EE)

            wigner[i, :] = np.abs(fftwigner) ** 2

        return np.flipud(np.rot90(wigner[:, t_res / 2:])), [0, t_range, 0, nyquist]

    def getEmittance(self, dim='x'):

          """
          Calculates normalized emittance. Assumes x, vx in SI units.
          Approximation - taken mean of gamma to get momentum
          Calculates emmittance of all particles in beam data.

          Returns
          -------
          Float : Emittance in [mm mrad] units
          """

          beamdata = self.beamScatter(xvalue=dim, yvalue='v'+dim, zvalue='gamma')
          x=beamdata[0]
          vx=beamdata[1]
          gamma=beamdata[2]

          mean_gamma = np.mean(gamma)
          mean_xsq = np.mean(x**2)
          mean_pxsq = np.mean(vx**2) * cte.m_e**2 * mean_gamma**2
          mean_xpx = np.mean(x * vx) * cte.m_e * mean_gamma
          emit_n = 1/cte.m_e/cte.c * np.sqrt(mean_xsq * mean_pxsq - mean_xpx**2)

          return emit_n*1e6

class StaticPlot:
    def __init__(self, ax=gca(), datasets=None, filenumber=None):

        self.dset = datasets

        self.plotdata = PlotData(datasets=self.dset, filenumber=filenumber)

        self.style = Style(ax=ax, datasets=datasets)

        self.plot = Plot(ax=ax, datasets=datasets)

        self.ax = ax

    def densityPlot(self, filenumber=None, slice='x', **args):

        density = self.plotdata.fieldMap(dataset='ne', slice=slice)

        translim = self.plotdata.getAttribute(dataset='ne', attr=slice + '_lim')

        zlim = self.plotdata.getAttribute(dataset='ne', attr='z_lim')

        cmap = Cmap()

        # cmap.cubehelixCmap(reverse=False, start=0.18, rot=-0.65, gamma=0.6, sat=1, maxLight=2.0)

        cmap.cubehelixCmap(reverse=True, start=0, rot=-0.19, gamma=1.0, sat=0.50, maxLight=1.0)
        # cmap.alphaCmapzero()

        ls = LightSource(azdeg=-75, altdeg=65)

        shaded = ls.shade(density, plt.cm.binary)

        graphic = self.plot.colorPlot(dataset=density, cmap=cmap.cmap, limits=zlim[0:2] + translim[0:2], **args)

        # self.plot.colorPlot(dataset=shaded, limits=zlim[0:2] + translim[0:2], alpha=0.05, **args)

        self.style.axis_ticks()

        self.style.setLabel(xlabel='z', ylabel=slice)
        self.style.setTitle(label='density')

        return graphic

    def laserPlot(self, dataset='Ex', filenumber=None, slice='x', **args):

        laser = self.plotdata.fieldMap(dataset=dataset, slice=slice)

        translim = self.plotdata.getAttribute(dataset=dataset, attr=slice + '_lim')

        zlim = self.plotdata.getAttribute(dataset=dataset, attr='z_lim')

        cmap = Cmap()

        cmap.getMplCmap(cmap='gray')

        cmap.alphaCmap()

        graphic = self.plot.colorPlot(dataset=laser, cmap=cmap.cmap, alpha=0.6, limits=zlim[0:2] + translim[0:2], **args)

        self.style.axis_ticks()

        self.style.setLabel(xlabel='z', ylabel=slice)
        self.style.setTitle(label='laser')

        return graphic

    def fieldPlot(self, dataset='Ez', filenumber=None, slice='x', colormap='coolwarm', alphacmap=True, shiftcmap=True,
                  alpha=0.7, **args):

        field = self.plotdata.fieldMap(dataset=dataset, slice=slice)

        translim = self.plotdata.getAttribute(dataset=dataset, attr=slice + '_lim')

        zlim = self.plotdata.getAttribute(dataset=dataset, attr='z_lim')
        vmin = field.min()
        vmax = field.max()

        cmap = Cmap()
        cmap.getMplCmap(cmap=colormap)
        if alphacmap is True: cmap.alphaCmap()
        if shiftcmap is True: cmap.shiftCmap(midpoint=1 - vmax / (vmax + abs(vmin)))

        graphic = self.plot.colorPlot(dataset=field, cmap=cmap.cmap, alpha=alpha, limits=zlim[0:2] + translim[0:2],
                                      **args)

        self.style.axis_ticks()
        self.style.setLabel(xlabel='z', ylabel=slice)
        self.style.setTitle(label=dataset)

        return graphic

    def laserEnvelope2dPlot(self, dataset='Ex', filenumber=None, slice='x', **args):

        env2d = self.plotdata.getLaserEnvelope2D(dataset=dataset, slice=slice, window_bounds_percent=40)

        translim = self.plotdata.getAttribute(dataset=dataset, attr=slice + '_lim')

        zlim = self.plotdata.getAttribute(dataset=dataset, attr='z_lim')

        env2d = env2d / max(env2d)

        cmap = Cmap()
        cmap.getMplCmap(cmap='bone_r')

        graphic = self.plot.colorPlot(dataset=env2d, cmap=cmap.cmap, alpha=0.8, limits=zlim[0:2] + translim[0:2],
                                      **args)

        self.style.axis_ticks()
        # self.style.setLabel(label='envelopemap')
        # self.style.setTitle(label='envelope2d')

        return graphic

    def beamScatter(self, filenumber=None, s=4, alpha=0.4):

        beam = self.plotdata.beamScatter(zvalue='gamma')
        graphic = self.plot.scatterPlot(dataset=beam, alpha=alpha, s=s)
        return graphic

    def longLineoutPlot(self, filenumber=None, color='steelblue', dataset='Ez', slice='x', twinx=False,
                        horizontal_line=False, **args):

        zlim = self.plotdata.getAttribute(dataset=dataset, attr='z_lim')

        lineoutslice = self.plotdata.longSlice(dataset=dataset, slice=slice)

        axis = np.linspace(zlim[0], zlim[1], zlim[2])

        graphic = self.plot.linePlot(ydataset=lineoutslice, xdataset=axis, color=color)

        if horizontal_line:
            self.plot.horizontalLine(transposition=0)

        ylim = max([max(lineoutslice), -min(lineoutslice)])

        self.ax.set_ylim([-ylim * 1.05, ylim * 1.05])
        if dataset == 'ne': self.ax.set_ylim([0, ylim * 1.05])

        self.ax.set_xlim(zlim[0:2])

        self.style.axis_ticks(twinx=twinx)

        self.style.setLabel(xlabel='z', ylabel=dataset)
        self.style.setTitle(label=dataset)

        return graphic

    def diffLongLineoutPlot(self, filenumber=None, color='steelblue', dataset='Ez', slice='x', transshift=0,
                            twinx=False, horizontal_line=False, **args):

        zlim = self.plotdata.getAttribute(dataset=dataset, attr='z_lim')
        translim = self.plotdata.getAttribute(dataset=dataset, attr=slice + '_lim')

        lineoutslice0 = self.plotdata.longSlice(dataset=dataset, slice=slice, delta_lineoutposition=transshift)
        lineoutslice1 = self.plotdata.longSlice(dataset=dataset, slice=slice, delta_lineoutposition=transshift + 1)
        difflineout = lineoutslice1 - lineoutslice0

        delta_x = (translim[1] - translim[0]) / translim[2]

        lineoutslice = difflineout / delta_x

        axis = np.linspace(zlim[0], zlim[1], zlim[2])
        graphic = self.plot.linePlot(ydataset=lineoutslice, xdataset=axis, color=color)

        if horizontal_line:
            self.plot.horizontalLine(transposition=0)
        ylim = max([max(lineoutslice), -min(lineoutslice)])

        self.ax.set_ylim([-ylim * 1.05, ylim * 1.05])
        self.ax.set_xlim(zlim[0:2])
        self.style.axis_ticks(twinx=twinx)
        self.style.setLabel(xlabel='z', ylabel='d' + dataset)
        self.style.setTitle(label='d' + dataset)

        return graphic

    def potentialPlot(self, ):

        return

    def phasespacePlot(self, x='z', y='gamma', hexmode=False, **args):
        phasespdata = self.plotdata.beamScatter(xvalue=x, yvalue=y, zvalue=None)

        try:
            xylims = [min(phasespdata[0]) - abs(min(phasespdata[0]) / 10),
                      max(phasespdata[0]) + abs(max(phasespdata[0]) / 10),
                      min(phasespdata[1]) - abs(min(phasespdata[1]) / 10),
                      max(phasespdata[1]) + abs(max(phasespdata[1]) / 10)]
        except:
            xylims = [-1, 1, -1, 1]

        if hexmode is True:
            graphic = self.plot.hexbinPlot(dataset=phasespdata, limits=(xylims[0], xylims[1], xylims[2], xylims[3]))
        else:
            graphic = self.plot.scatterPlot(dataset=phasespdata)
        self.style.axis_ticks()
        self.style.setLabel(xlabel=x, ylabel=y)
        self.style.setTitle(label='phasespace')
        self.ax.set_xlim(xylims[0:2])
        self.ax.set_ylim(xylims[2:])

        return graphic

    def energyHistogram(self, ):

        return

    def dephasingPoint(self, filenumber=None):

        deppoints = self.plotdata.getDephasingPoint()
        graphic = self.plot.verticalLine(zposition=deppoints)

        return graphic

    def wignerTransform(self, filenumber=None, central_wavelength_nm=800, frequency_range_percent=None,
                        time_range_fs=None, downsampling=2, plot_intensity_lineout=True):

        wigner, bounds = self.plotdata.getWignerTransform(window_bounds_percent=40, downsampling=downsampling)
        cmap = Cmap()
        cmap.getMplCmap(cmap='bone_r')

        maxi, maxj = np.unravel_index(wigner.argmax(), wigner.shape)
        bound_temp = bounds[0]
        bounds[0] = -(bounds[1] - bounds[1] / wigner.shape[1] * maxj)
        bounds[1] = -(bound_temp - bounds[1] / wigner.shape[1] * maxj)

        central_freq_shift = 2.35 * 10 ** 15 * 800 / central_wavelength_nm

        bounds[2] -= central_freq_shift
        bounds[3] -= central_freq_shift

        ls = LightSource(azdeg=-75, altdeg=65)
        shaded = ls.shade(wigner[:, ::-1], plt.cm.binary)

        graphic = self.plot.colorPlot(dataset=wigner[:, ::-1], cmap=cmap.cmap,
                                      limits=(bounds[0], bounds[1], bounds[2], bounds[3]))

        # self.plot.colorPlot(dataset=shaded, alpha=0.05, limits=(bounds[0], bounds[1], bounds[2], bounds[3]))

        if time_range_fs is not None:
            xlim = (-time_range_fs * 10 ** -15, time_range_fs * 10 ** -15)
            self.ax.set_xlim(xlim)

        if frequency_range_percent is not None:
            ylim = (
                -central_freq_shift * frequency_range_percent / 100, central_freq_shift * frequency_range_percent / 100)
            self.ax.set_ylim(ylim)

        self.style.axis_ticks()
        self.style.setLabel(xlabel='t', ylabel='wigner')
        self.style.setTitle(label='wigner')

        if plot_intensity_lineout:
            intensity = sum(wigner[:, ::-1], axis=0)
            twinx_ax = self.ax.twinx()
            plot_twinx = Plot(ax=twinx_ax, datasets=self.dset)
            style_twinx = Style(ax=twinx_ax, datasets=self.dset)
            plot_twinx.linePlot(ydataset=intensity / max(intensity),
                                xdataset=linspace(bounds[0], bounds[1], shape(wigner)[1]), color='black', dashed=True)

            if time_range_fs is not None:
                xlim = (-time_range_fs * 10 ** -15, time_range_fs * 10 ** -15)
                twinx_ax.set_xlim(xlim)
            ylim = (0, 1.18)
            twinx_ax.set_ylim(ylim)
            style_twinx.axis_ticks(twinx=True)
            style_twinx.setLabel(xlabel='t', ylabel='wigner_int')

        return graphic

class Style:
    def __init__(self, ax=gca(), datasets=None, filenumber=None):

        self.i = filenumber
        self.dset = datasets[self.i]
        self.ax = ax
        self.title = None

    def axis_ticks(self, twinx=False):

        # Set linewidth of spines (thickness of axis)
        self.ax.spines['bottom'].set_linewidth(0.7)
        self.ax.spines['top'].set_linewidth(0.7)
        self.ax.spines['left'].set_linewidth(0.7)
        self.ax.spines['right'].set_linewidth(0.7)

        if twinx is False:
            self.ax.tick_params(axis='x', pad=1, direction='out', length=2.5, width=0.5, right='on', left='off',
                                top='off', labelright='on', labelleft='off')

            self.ax.tick_params(axis='y', pad=7, direction='out', length=2.5, width=0.5, right='off', left='on',
                                top='off', labelright='off', labelleft='on')
        else:
            self.ax.tick_params(axis='x', pad=1, direction='out', length=2.5, width=0.5, right='on', left='off',
                                top='off', labelright='on', labelleft='off')

            self.ax.tick_params(axis='y', pad=7, direction='out', length=2.5, width=0.5, right='on', left='off',
                                top='off', labelright='on', labelleft='off')

    def generateTexformatter(self, exp, int=False):

        def formatter(x, pos):
            xform = x * 10 ** exp
            if int is False:
                return r'$%s$' % xform
            else:
                return r'$%0.0f$' % xform

        return formatter

    def texaxis(self, expx=0, expy=0, intx=False, inty=False):

        self.ax.xaxis.set_major_formatter(FuncFormatter(self.generateTexformatter(expx, int=intx)))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self.generateTexformatter(expy, int=inty)))

    def setLabel(self, xlabel='z', ylabel='ne'):

        labeldict = {'x': [r'$\mathrm{x}\,(\mu \mathrm{m})$', 6, True],
                     'y': [r'$\mathrm{y}\,(\mu \mathrm{m})$', 6, True],
                     'z': [r'$\mathrm{z}\,(\mu \mathrm{m})$', 6, True],
                     't': [r'$\mathrm{t}\,(\mathrm{fs})$', 15, True],
                     'ne': [r'$n_e\,(10^{24}\mathrm{cm^{-3}})$', -24, False],
                     'Ex': [r'$\mathrm{E_x}\,(\mathrm{TV/m})$', -12, True],
                     'Ey': [r'$\mathrm{E_y}\,(\mathrm{GV/m})$', -9, True],
                     'Ez': [r'$\mathrm{E_z}\,(\mathrm{GV/m})$', -9, True],
                     'dne': [r'$n_e\,(10^{24}\mathrm{cm^{-4}})$', -24, False],
                     'dEx': [r'$\partial_r \, \mathrm{E_x}\,(\mathrm{PV/m^2})$', -15, True],
                     'dEy': [r'$\partial_r \, \mathrm{E_y}\,(\mathrm{PV/m^2})$', -15, True],
                     'dEz': [r'$\partial_r \, \mathrm{E_z}\,(\mathrm{PV/m^2})$', -15, True],
                     'wigner': [r'$\mathrm{\omega - \omega_0}\,(10^{14} \mathrm{rad/s})$', -14, True],
                     'wigner_int': [r'$\mathrm{Intensity}\,(\mathrm{a.u.})$', 0, False],
                     'a0': [r'$a_0$', 0, False],
                     'meanfreq': [r'$\mathrm{\omega}\,(10^{14}\mathrm{rad/s})$', -14, False],
                     'gamma': [r'$\gamma$', 0, False],
                     'duration': [r'$\tau\,(\mathrm{fs})$', 15, False],
                     'waist': [r'$w\,(\mu \mathrm{m})$', 6, False]}

        self.texaxis(expx=labeldict[xlabel][1], expy=labeldict[ylabel][1], intx=labeldict[xlabel][2],
                     inty=labeldict[ylabel][2])
        self.ax.set_xlabel(labeldict[xlabel][0])
        self.ax.set_ylabel(labeldict[ylabel][0])

    def setTitle(self, label='wigner', removetitle=False):

        if self.title is not None:
            self.title.remove()
        try:
            titledict = {'wigner': r'$\mathrm{WIGNER\,TRANSFORMATION}$',
                         'phasespace': r'$\mathrm{PHASESPACE}$',
                         'Ez': r'$\mathrm{LONG.\,FIELD}$',
                         'Ex': r'$\mathrm{TRANS.\,FIELD}$',
                         'Ey': r'$\mathrm{TRANS.\,FIELD}$',
                         'ne': r'$\mathrm{DENSITY}$',
                         'dEz': r'$\mathrm{LONG.\,FIELD \, DERIVATIVE}$',
                         'dEx': r'$\mathrm{TRANS.\,FIELD \, DERIVATIVE}$',
                         'dEy': r'$\mathrm{TRANS.\,FIELD \, DERIVATIVE}$',
                         'laser': r'$\mathrm{LASER}$',
                         'density': r'$\mathrm{PLASMA \, DENSITY}$',
                         'special': r'$\mathrm{DENSITY \,+\,FIELDS}$',
                         'duration': r'$\mathrm{PULSE \,DURATION}$',
                         'meanfreq': r'$\mathrm{MEAN \,LASER\, FREQUENCY}$',
                         'a0': r'$\mathrm{NORMALIZED \,VECTOR\, POTENTIAL}$',
                         'waist': r'$\mathrm{LASER\, WAIST}$'}

            if not removetitle:
                self.title = self.setAnnotation(text=titledict[label], xy=(0.5, 0.9))
        except:
            self.title = self.setAnnotation(text=label, xy=(0.5, 0.9))

    def setAnnotation(self, text='', xy=(0, 0)):

        annotation = self.ax.annotate(text, xy=xy, va="center", ha="center", xycoords="axes fraction",
                                      bbox=dict(boxstyle='round,pad=0.3', fc='1', lw=0.2, alpha=0.4), clip_on=False)

        return annotation

class TemporalPlot:
    def __init__(self, ax=gca(), datasets=None):

        self.dset = datasets
        self.i = range(len(datasets))
        self.plotdata = []
        for ii in self.i:
            plotdatainst = PlotData(datasets=self.dset, filenumber=ii)
            self.plotdata.append(plotdatainst)
        self.style = Style(ax=ax, datasets=datasets)
        self.plot = Plot(ax=ax, datasets=datasets)
        self.ax = ax

    def laserWaist(self, dataset='Ex', mode='fwhm', slice='x'):

        plotdata = []
        pos = []

        for i in self.i:
            data = self.plotdata[i].getLaserWaist(dataset=dataset, mode=mode, slice=slice)
            plotdata.append(data)

            lim = self.plotdata[i].getAttribute(dataset=dataset, attr='z_lim')
            pos.append(lim[1])

        self.plot.linePlot(ydataset=plotdata, xdataset=pos, color='black')

        self.style.axis_ticks()
        self.style.setLabel(xlabel='z', ylabel='waist')
        self.style.setTitle(label='waist')

    def pulseDuration(self, dataset='Ex'):

        plotdata = []
        pos = []

        for i in self.i:
            data = self.plotdata[i].getPulseDuration(mode='fwhm', dataset=dataset)
            plotdata.append(data)

            lim = self.plotdata[i].getAttribute(dataset=dataset, attr='z_lim')
            pos.append(lim[1])

        self.plot.linePlot(ydataset=plotdata, xdataset=pos, color='black')

        self.style.axis_ticks()
        self.style.setLabel(xlabel='z', ylabel='duration')
        self.style.setTitle(label='duration')

    def laserFrequency(self, dataset='Ex', slice='x'):

        plotdata = []
        pos = []

        for i in self.i:
            data = self.plotdata[i].getMeanFreq(dataset=dataset, slice=slice)
            plotdata.append(data)

            lim = self.plotdata[i].getAttribute(dataset=dataset, attr='z_lim')
            pos.append(lim[1])

        self.plot.linePlot(ydataset=plotdata, xdataset=pos, color='black')

        self.style.axis_ticks()
        self.style.setLabel(xlabel='z', ylabel='meanfreq')
        self.style.setTitle(label='meanfreq')

    def lasera0(self, dataset='Ex'):
        plotdata = []
        pos = []

        for i in self.i:
            data = self.plotdata[i].geta0(dataset=dataset)
            plotdata.append(data)

            lim = self.plotdata[i].getAttribute(dataset=dataset, attr='z_lim')
            pos.append(lim[1])

        self.plot.linePlot(ydataset=plotdata, xdataset=pos, color='black')

        self.style.axis_ticks()
        self.style.setLabel(xlabel='z', ylabel='a0')
        self.style.setTitle(label='a0')

    def plasmaWavelength(self, ):

        return

    def phaseVelocity(self, ):

        return

    def beamPosition(self, ):

        return

    def beamEnergy(self, ):

        return

    def beamLength(self, ):

        return

    def beamEmittance(self, ):

        return

    def beamDivergence(self, ):

        return

    def beamCharge(self, ):

        return

###################################################################

def saveFig(filename=None, directory=None, dpi=600, l_pdf=False):
    if not directory == None:

        if not os.path.exists(directory):
            os.makedirs(directory)

    filename = str(filename)

    if not directory == None: filename = str(directory) + '/' + filename

    if l_pdf == True:

        filename += '.pdf'



    else:

        filename += '.png'

    plt.savefig(filename, dpi=dpi)


def main():

	################
	# Change here...
	################


	###################################################################
	# Settings... 
	###################################################################

	# Define Datadict
	datadict = {'field_lab': ['Ey', 'Ez'],
							'elec_lab': ['ne', 'phase_space', 'phase_space_low'],
							'beam_lab': ['gamma', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'w']}

	runid = 'warp_script'

	subfolder = 'data'

	fileselection = range(1, 11)

	###################################################################
	# Read data... 
	###################################################################

	# Read Datasets...
	data = Data(runid = runid, subfolder = subfolder, fileselection = fileselection, datadict=datadict)


	files = data.readFiles()

	dsets = data.readDatasets()


	#from warp import *
	#setup()
	#for i in range(len(dsets)):
	#  Ez=dsets[i]['Ez'][0].value[:,0,:]
	#  ppg(Ez);fma()

	#raise
	


	###################################################################
	# Create temporal plots... 
	###################################################################

	# Temporals...
	# create figure...
	fig = plt.figure(figsize=(16.54, 11.69), dpi=600)
	fig.suptitle(r'$\mathbf{Simulation \, Results}$' '\n' r'$\mathrm{%s \,-\, Temporal \, Plots}$'%(data.runid.replace('_','\,')), x=0.05, y=0.95, horizontalalignment='left', verticalalignment='bottom')

	##########################
	# Pulse Duration
	##########################
	if False:
		ax7 = fig.add_axes([0.10, 0.40, 0.2, 0.2])
		plot7 = TemporalPlot(ax=ax7, datasets=data.dset)
		plot7.pulseDuration()

		##########################
		# Laser Mean Frequency
		##########################
		ax8 = fig.add_axes([0.1, 0.7, 0.2, 0.2])
		plot8 = TemporalPlot(ax=ax8, datasets=data.dset)
		plot8.laserFrequency()

		##########################
		# Laser a0
		##########################
		ax9 = fig.add_axes([0.4, 0.7, 0.2, 0.2])
		plot9 = TemporalPlot(ax=ax9, datasets=data.dset)
		plot9.lasera0()

		##########################
		# Laser Waist
		##########################
		ax10 = fig.add_axes([0.7, 0.7, 0.2, 0.2])
		plot10 = TemporalPlot(ax=ax10, datasets=data.dset)
		plot10.laserWaist(mode='sigma')

		##########################
		# Save temporal plot...
		##########################
		directory = data.runid + '-plots'
		filename = data.runid + '-TempralPlots'
		saveFig(filename=filename, directory=directory, dpi=600, l_pdf=False)

	print 'Temprals DONE!'
	clf()


	###################################################################
	# Create static plots... 
	###################################################################

	for i, dset in enumerate(dsets): #loop over snapshots...

			#create figure
			fig = plt.figure(figsize=(16.54, 11.69), dpi=600)
			fig.suptitle(r'$\mathbf{Simulation \, Results}$' '\n' r'$\mathrm{%s \, - \, %s}$'%(data.runid.replace('_','\,'), dsets[i]['ne'][2]), x=0.05, y=0.95, horizontalalignment='left', verticalalignment='bottom')
		
			##########################
			# Beam + Density + Long. Field + Laser Field
			##########################
			ax1 = fig.add_axes([0.10, 0.40, 0.5, 0.5])

			plot1 = StaticPlot(ax=ax1, datasets=data.dset, filenumber=i)
		
			beamscatter = plot1.beamScatter()

			densityplot = plot1.densityPlot()
			cax = fig.add_axes([0.60, 0.40, 0.01, 0.5])
			colorbar(densityplot, cax=cax, format=FuncFormatter(plot1.style.generateTexformatter(exp=-24)))

			fieldplot = plot1.fieldPlot(dataset='Ez')
			cax2 = fig.add_axes([0.10, 0.895, 0.25, 0.005])
			colorbar(fieldplot, cax=cax2, format=FuncFormatter(plot1.style.generateTexformatter(exp=-24)), ticks=[],
							 orientation='horizontal')

			laserplot = plot1.laserPlot(dataset='Ey')
			cax3 = fig.add_axes([0.35, 0.895, 0.25, 0.005])
			colorbar(laserplot, cax=cax3, format=FuncFormatter(plot1.style.generateTexformatter(exp=-24)), ticks=[],
							 orientation='horizontal')

			plot1.style.setTitle(label='special')

			##########################
			# Long. E-Field Lineout
			##########################
			ax2 = fig.add_axes([0.70, 0.40, 0.2, 0.2])

			plot2 = StaticPlot(ax=ax2, datasets=data.dset, filenumber=i)
			plot2.longLineoutPlot(dataset='Ez', horizontal_line=True, color='black')

			##########################
			# Trans Field Lineout + Trans. Focusing Field Lineout
			##########################
			ax3 = fig.add_axes([0.70, 0.70, 0.2, 0.2])

			plot3 = StaticPlot(ax=ax3, datasets=data.dset, filenumber=i)
			plot3.longLineoutPlot(dataset='Ey', color='black')
			plot3.style.setTitle(label='$\mathrm{TRANS. \, FIELD \,+\, DERIVATIVE}$')

			ax4 = ax3.twinx()

			plot4 = StaticPlot(ax=ax4, datasets=data.dset, filenumber=i)
			plot4.diffLongLineoutPlot(dataset='Ey', twinx=True, color='lightsteelblue', horizontal_line=False)
			plot4.style.setTitle(removetitle=True)

			##########################
			# Wigner Transformation
			##########################
			#ax5 = fig.add_axes([0.70, 0.10, 0.2, 0.2])

			#plot5 = StaticPlot(ax=ax5, datasets=data.dset, filenumber=i)
			#plot5.wignerTransform(central_wavelength_nm=800, frequency_range_percent=30, time_range_fs=100, downsampling=4,
			#                      plot_intensity_lineout=True)

			##########################
			# Phasespace of injected beam
			##########################
			ax6 = fig.add_axes([0.40, 0.10, 0.2, 0.2])

			plot6 = StaticPlot(ax=ax6, datasets=data.dset, filenumber=i)
			plot6.phasespacePlot(x='z', y='gamma', hexmode=True)
			# hcax = fig.add_axes([0.60,0.40,0.01,0.2])
			# colorbar(hex, cax = hcax, format = FuncFormatter(plot1.style.generateTexformatter(exp = 1)))

			##########################
			# Density Lineout
			##########################
			ax7 = fig.add_axes([0.10, 0.10, 0.2, 0.2])

			plot7 = StaticPlot(ax=ax7, datasets=data.dset, filenumber=i)
			plot7.longLineoutPlot(dataset='ne', color='black')

			##########################
			# Save static plot...
			##########################

			directory = data.runid + '-plots'

			filename = data.runid + '-' + 'specialPlot' + '-' + str(dsets[i]['ne'][2])

			saveFig(filename=filename, directory=directory, dpi=600, l_pdf=False)

			clf()

			print 'Plot Nr. ' + str(i + 1) + ' completed!'



