import numpy as np
from tables import *
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits as pyfits 
import math
from scipy.interpolate import splrep, splev
from astropy.stats import sigma_clip
import numpy.polynomial.polynomial as poly
import scipy.signal
from scipy import signal,interpolate
import argparse
from scipy.optimize import curve_fit
import collections as collections
from PyAstronomy import funcFit as fuf
from specutils.io import read_fits
from astropy.convolution import Trapezoid1DKernel
import os
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.gridspec as gridspec


def fit_splines(x,y,xdata,nknots):

	# Luego el arreglo de puntos: 
	knots = np.arange(x[1],x[len(x)-1],(x[len(x)-1]-x[1])/np.double(nknots)) 
	idx_knots = (np.arange(1,len(x)-1,(len(x)-2)/np.double(nknots))).astype('int') 
	knots = x[idx_knots] 

	tck = splrep(x,y,t=knots) 
	fit = splev(xdata,tck) 
	return fit

def clean_lines(wav,flux,spectype):
	if spectype == 'A':	####A or B type 
		mask0 = wav > 6860
		mask1 = ((wav >6557)&(wav <6584))
		mask2 = ((wav>5875)&(wav<5884))
		mask3 = ((wav>4333)&(wav<4353))
		mask4 = ((wav>4850)&(wav<4870))
		mask5 = ((wav>4090)&(wav<4110))
		#mask6 = ((wav>3875)&(wav<3900))
		#mask7 = ((wav>3920)&(wav<3945))
		#mask8 = ((wav>3955)&(wav<3980))
		mask9 = (wav<3850)
		mask10 = ((wav>4854)&(wav<4870))
		mask11 = ((wav>4470)&(wav<4486))
		mask12 = ((wav>4336)&(wav<4360))
		mask13 = ((wav>4550)&(wav<4551.5))
		mask = mask0 + mask1+mask2+mask3+mask4+mask5+mask9+mask10+mask11+mask12+mask13
		#mask = mask6 + mask7 + mask8 
		
	
	elif spectype == 'F' or spectype == 'eF' :	#### F type

		mask1 = ((wav > 6557) & (wav < 6570))
		mask2 = ((wav > 6488) & (wav < 6500))
		mask3 = ((wav > 5887) & (wav < 5900))
		mask4 = ((wav > 4860) & (wav < 4865))
		#mask5 = ((wav > 4470) & (wav < 4486))
		mask5 = ((wav>4550)&(wav<4551.5))

		#mask6 = ((wav > 4336) & (wav < 4360))
		mask7 = ((wav > 4335) & (wav < 4350))
		#mask8 = ((wav > 4090) & (wav < 4110))
		mask9 = ((wav > 3966) & (wav < 3975))
		#mask10 = ((wav > 3875) & (wav < 3900))
		#mask11 = ((wav > 3955) & (wav < 3980))
		mask12 = ((wav > 3925) & (wav < 3941))
		#mask13 = (wav < 3850)
		mask13 = (wav < 3550)
		mask = mask1+mask2+mask4+mask5+mask7+mask9+mask12+mask13

	return ~mask

def clean_lines_deconv(wav,flux,spectype):
	if spectype == 'A':	####A or B type 
		mask0 = (wav > 6860)
		mask1 = ((wav >6550)&(wav <6584))
		mask2 = ((wav>5875)&(wav<5884))
		mask3 = ((wav>4333)&(wav<4353))
		mask4 = ((wav>4850)&(wav<4870))
		mask5 = ((wav>4090)&(wav<4110))
		#mask6 = ((wav>3875)&(wav<3900))
		#mask7 = ((wav>3920)&(wav<3945))
		#mask8 = ((wav>3955)&(wav<3980))
		mask9 = (wav<3850)
		mask10 = ((wav>4854)&(wav<4870))
		mask11 = ((wav>4470)&(wav<4486))
		mask12 = ((wav>4336)&(wav<4360))
		mask = mask0+mask1+mask2+mask3+mask4+mask5+mask9+mask10+mask11+mask12
		#mask = mask6 + mask7 + mask8 
		
	
	elif spectype == 'F':	#### F type

		mask1 = ((wav > 6557) & (wav < 6572))
		mask2 = ((wav > 6488) & (wav < 6500))
		mask3 = ((wav > 5887) & (wav < 5900))
		mask4 = ((wav > 4857) & (wav < 4865))
		#mask5 = ((wav > 4470) & (wav < 4486))
		mask6 = ((wav > 4336) & (wav < 4360))
		mask7 = ((wav > 4333) & (wav < 4353))
		#mask8 = ((wav > 4090) & (wav < 4110))
		mask9 = ((wav > 3963) & (wav < 3975))
		#mask10 = ((wav > 3875) & (wav < 3900))
		#mask11 = ((wav > 3955) & (wav < 3980))
		mask12 = ((wav > 3925) & (wav < 3941))
		mask13 = (wav < 3850)

		mask = mask1+mask2+mask4+mask7+mask12+mask13
		#mask = mask + mask5 + mask8 + mask10 + mask11

	elif spectype == 'eF':	#### F type con emision

		#mask1 = ((wav > 6552) & (wav < 6572))
		#mask1 = ((wav > 6545) & (wav < 6600))
		mask1 = ((wav > 6545) & (wav < 6600))

		mask2 = ((wav > 6488) & (wav < 6500))
		mask3 = ((wav > 5887) & (wav < 5900))
		mask4 = ((wav > 4835) & (wav < 4898))
		#mask4 = ((wav > 4855) & (wav < 4872))

		#mask5 = ((wav > 4470) & (wav < 4486))
		mask6 = ((wav > 4336) & (wav < 4360))
		mask7 = ((wav > 4333) & (wav < 4353))
		mask8 = ((wav > 4080) & (wav < 4130))
		mask9 = ((wav > 3958) & (wav < 3988))
		#mask10 = ((wav > 3875) & (wav < 3900))
		#mask11 = ((wav > 3955) & (wav < 3980))
		mask12 = ((wav > 3925) & (wav < 3941))
		mask13 = (wav < 3850)

		##RECORTAR EMISION
		mask14 = ((wav > 6760) & (wav < 6765))
		mask15 = ((wav > 6662) & (wav < 6674))
		mask16 = ((wav > 6637) & (wav < 6655))
		mask17 = ((wav > 6107) & (wav < 6112))
		mask18 = ((wav > 5961) & (wav < 5963))
		mask19 = ((wav > 5899) & (wav < 5904))
		mask20 = ((wav > 5844) & (wav < 5846))
		mask21 = ((wav > 5830) & (wav < 5832))
		mask22 = ((wav > 5785) & (wav < 5787))
		mask23 = ((wav > 5772) & (wav < 5773))
		mask24 = ((wav > 5696) & (wav < 5698))
		mask25 = ((wav > 5643.5) & (wav < 5644.5))
		mask26 = ((wav > 5630) & (wav < 5632))
		mask27 = ((wav > 5578.5) & (wav < 5579.5))
		mask28 = ((wav > 5429) & (wav < 5430))
		mask29 = ((wav > 5110) & (wav < 5114))
		mask30 = ((wav > 5097) & (wav < 5099.5))
		mask31 = ((wav > 4917) & (wav < 4918))
		mask32 = ((wav > 4902.5) & (wav < 4903.5))
		mask33 = ((wav > 4833) & (wav < 4836))    ###AJUSTE GRUESO
		mask34 = ((wav > 4838.3) & (wav < 4839))    ###AJUSTE GRUESO
		mask35 = ((wav > 4797) & (wav < 4798))
		mask36 = ((wav > 4761.5) & (wav < 4762))
		mask37 = ((wav > 4725) & (wav < 4726))
		mask38 = ((wav > 4720.5) & (wav < 4721.5))
		mask39 = ((wav > 4615.5) & (wav < 4616.5))
		mask40 = ((wav > 4610.5) & (wav < 4611.5))
		mask41 = ((wav > 4568.5) & (wav < 4569.5))
		mask42 = ((wav > 4277.5) & (wav < 4280))


		mask_lines = mask1+mask2+mask3+mask4+mask6+mask7+mask9+mask12+mask13+mask8
		mask_emiss = mask14+mask15+mask16+mask17+mask18+mask19+mask20+mask21+mask22+mask23+mask24+mask25+mask26+mask27+mask28+mask29+mask30+mask31+mask32+mask33+mask34+mask35+mask36+mask37+mask38+mask39+mask40+mask41+mask42
		#mask = mask + mask5 + mask8 + mask10 + mask11
		mask = mask_lines+mask_emiss

	return wav[~mask],flux[~mask]


def clean_strong_lines(mw,sc,mode=1):
	if mode==1:
		#""""
		I = np.where((mw>6520)&(mw<6600))[0]
		sc[I] = 1.
		I = np.where((mw>5888)&(mw<5897))[0]
		sc[I] = 1.
		I = np.where((mw>4310)&(mw<4360))[0]
		sc[I] = 1.
		I = np.where((mw>4840)&(mw<4880))[0]
		sc[I] = 1.
		I = np.where((mw>4070)&(mw<4130))[0]
		sc[I] = 1.
		I = np.where((mw>3875)&(mw<3900))[0]
		sc[I] = 1.
		I = np.where((mw>3920)&(mw<3945))[0]
		sc[I] = 1.
		I = np.where((mw>3955)&(mw<3980))[0]
		sc[I] = 1.
		I = np.where(mw<3850)[0]
		sc[I] = 1.
		#"""
	if mode==2:
		#""""
		I = np.where((mw>6550)&(mw<6570))[0]
		sc[I] = 1.
		I = np.where((mw>5888)&(mw<5897))[0]
		sc[I] = 1.
		I = np.where((mw>4320)&(mw<4350))[0]
		sc[I] = 1.
		I = np.where((mw>4850)&(mw<4870))[0]
		sc[I] = 1.
		I = np.where((mw>4090)&(mw<4110))[0]
		sc[I] = 1.
		I = np.where((mw>3875)&(mw<3900))[0]
		sc[I] = 1.
		I = np.where((mw>3920)&(mw<3945))[0]
		sc[I] = 1.
		I = np.where((mw>3955)&(mw<3980))[0]
		sc[I] = 1.
		I = np.where(mw<3850)[0]
		sc[I] = 1.
		#"""
	return sc


def clean_lines_deconv_ones(wav,flux,spectype,med):
	if spectype == 'A':	####A or B type
 		I = np.where((wav>6520)&(wav<6600))[0]
		flux[I] = med
		I = np.where(wav > 6860)[0]
		flux[I] = med
		I = np.where((wav >6550)&(wav <6584))[0]
		flux[I] = med
		I = np.where((wav>5875)&(wav<5884))[0]
		flux[I] = med
		I = np.where((wav>4333)&(wav<4353))[0]
		flux[I] = med
		I = np.where((wav>4850)&(wav<4870))[0]
		flux[I] = med
		I = np.where((wav>4090)&(wav<4120))[0]
		flux[I] = med
		I = np.where(wav<3850)[0]
		flux[I] = med
		I  = np.where((wav>4854)&(wav<4870))[0]
		flux[I] = med
		I  = np.where((wav>4470)&(wav<4486))[0]
		flux[I] = med
		I  = np.where((wav>4336)&(wav<4360))[0]
		flux[I] = med
		
	
	elif spectype == 'F':	#### F type

		I = np.where((wav > 6530) & (wav < 6600))[0]
		flux[I] = med
		I = np.where((wav > 6488) & (wav < 6500))[0]
		flux[I] = med
		I = np.where((wav > 5887) & (wav < 5900))[0]
		flux[I] = med
		I = np.where((wav > 4816.5) & (wav < 4817.8))[0]
		flux[I] = med
		I = np.where((wav > 4840) & (wav < 4890))[0]
		flux[I] = med
		I = np.where((wav > 4336) & (wav < 4360))[0]
		flux[I] = med
		I = np.where((wav > 4333) & (wav < 4353))[0]
		flux[I] = med
		I = np.where((wav > 3963) & (wav < 3975))[0]
		flux[I] = med
		I = np.where((wav > 3925) & (wav < 3941))[0]
		flux[I] = med
		I = np.where(wav < 3850)[0]
		flux[I] = med
		if instrument == 'feros':
			I = np.where((wav>4090)&(wav<4120))[0]
			flux[I] = med
	####NO LO HE HECHO uwu
	elif spectype == 'eF':	#### F type con emision


		I = np.where((wav > 6530) & (wav < 6600))[0]
		flux[I] = med
		I = np.where((wav > 6488) & (wav < 6500))[0]
		flux[I] = med
		I = np.where((wav > 5887) & (wav < 5900))[0]
		flux[I] = med
		I = np.where((wav > 4816.5) & (wav < 4817.8))[0]
		flux[I] = med
		I = np.where((wav > 4840) & (wav < 4890))[0]
		flux[I] = med
		I = np.where((wav > 4336) & (wav < 4360))[0]
		flux[I] = med
		I = np.where((wav > 4333) & (wav < 4353))[0]
		flux[I] = med
		I = np.where((wav > 3963) & (wav < 3975))[0]
		flux[I] = med
		I = np.where((wav > 3925) & (wav < 3941))[0]
		flux[I] = med
		I = np.where(wav < 3850)[0]
		flux[I] = med

		##RECORTAR EMISION
		I = np.where((wav > 3925) & (wav < 3941))[0]
		flux[I] = med
		I = np.where((wav > 6760) & (wav < 6765))[0]
		flux[I] = med
		I = np.where((wav > 6662) & (wav < 6674))[0]
		flux[I] = med
		I = np.where((wav > 6637) & (wav < 6655))[0]
		flux[I] = med
		I = np.where((wav > 6107) & (wav < 6112))[0]
		flux[I] = med
		I = np.where((wav > 5961) & (wav < 5963))[0]
		flux[I] = med
		I = np.where((wav > 5899) & (wav < 5904))[0]
		flux[I] = med
		I = np.where((wav > 5844) & (wav < 5846))[0]
		flux[I] = med
		I = np.where((wav > 5830) & (wav < 5832))[0]
		flux[I] = med
		I = np.where((wav > 5785) & (wav < 5787))[0]
		flux[I] = med
		I = np.where((wav > 5772) & (wav < 5773))[0]
		flux[I] = med
		I = np.where((wav > 5696) & (wav < 5698))[0]
		flux[I] = med
		I = np.where((wav > 5643.5) & (wav < 5644.5))[0]
		flux[I] = med
		I = np.where((wav > 5630) & (wav < 5632))[0]
		flux[I] = med
		I = np.where((wav > 5578.5) & (wav < 5579.5))[0]
		flux[I] = med
		I = np.where((wav > 5429) & (wav < 5430))[0]
		flux[I] = med
		I = np.where((wav > 5110) & (wav < 5114))[0]
		flux[I] = med
		I = np.where((wav > 5097) & (wav < 5099.5))[0]
		flux[I] = med
		I = np.where((wav > 4917) & (wav < 4918))[0]
		flux[I] = med
		I = np.where((wav > 4902.5) & (wav < 4903.5))[0]
		flux[I] = med
		I = np.where((wav > 4833) & (wav < 4836))[0]
		flux[I] = med ###AJUSTE GRUESO
		I = np.where((wav > 4838.3) & (wav < 4839))[0]
		flux[I] = med    ###AJUSTE GRUESO
		I = np.where((wav > 4797) & (wav < 4798))[0]
		flux[I] = med
		I = np.where((wav > 4761.5) & (wav < 4762))[0]
		flux[I] = med
		I = np.where((wav > 4725) & (wav < 4726))[0]
		flux[I] = med
		I = np.where((wav > 4720.5) & (wav < 4721.5))[0]
		flux[I] = med
		I = np.where((wav > 4615.5) & (wav < 4616.5))[0]
		flux[I] = med
		I = np.where((wav > 4610.5) & (wav < 4611.5))[0]
		flux[I] = med
		I = np.where((wav > 4568.5) & (wav < 4569.5))[0]
		flux[I] = med
		I = np.where((wav > 4277.5) & (wav < 4280))[0]
		flux[I] = med


	maskaux = flux == med
	if plots != None:
		plt.plot(wav[maskaux],flux[maskaux],'ko',linewidth=2.5,alpha=.6)

	return wav, flux

def lines_to_one(wav,flux,spectype):

	I = np.where((mw>6520)&(mw<6600))[0]
	sc[I] = 1.
	I = np.where((mw>5888)&(mw<5897))[0]
	sc[I] = 1.
	I = np.where((mw>4310)&(mw<4360))[0]
	sc[I] = 1.
	I = np.where((mw>4070)&(mw<4130))[0]
	sc[I] = 1.
	I = np.where((mw>3875)&(mw<3900))[0]
	sc[I] = 1.
	I = np.where((mw>3920)&(mw<3945))[0]
	sc[I] = 1.
	I = np.where((mw>3955)&(mw<3980))[0]
	sc[I] = 1.
	I = np.where(mw<3850)[0]
	sc[I] = 1.

	return


c =299792.458 #kms

def n_Edlen(l):
    sigma = 1e4 / l
    sigma2 = sigma*sigma
    n = 1 + 1e-8 * (8342.13 + 2406030 / (130-sigma2) + 15997/(38.9-sigma2))
    return n

def ToAir(l):
    return (l / n_Edlen(l))

def ToVacuum(l):
    cond = 1
    l_prev = l.copy()
    while(cond):
        l_new = n_Edlen(l_prev) * l
        if (max(np.absolute(l_new - l_prev)) < 1e-10):
            cond = 0
        l_prev = l_new
    return l_prev

def shift_doppler(obs_wav,vel):
	return obs_wav/(vel*1e13/(c*1e13)+1)

def addcolsbyrow(array1,array2):
    auxtable = Table([[],[] ], names=('velocity','broadening'),dtype=(float,float))
    for i in np.arange(len(array1)):
        table = [array1[i], array2[i]]
        auxtable.add_row(table)
    return auxtable

def ordenar(array1,array2):
    zp = list(zip(array1,array2))
    sort = sorted(zp)
    array1,array2 = zip(*sort)
    array1 = np.array(array1)
    array2 = np.array(array2)

    mask = np.isnan(array2)
    array1 = array1[~mask]
    array2 = array2[~mask]
    
    mask = np.isnan(array1)
    array1 = array1[~mask]
    array2 = array2[~mask]
    

    return array1,array2

###Tomar un orden aproximarlo en los bordes, hacerle mascara para generar orden nuevo
def cont(f):
	# this function performs a continuum normalization
	x = np.arange(len(f))
	fo = f.copy()
	xo = x.copy()
	mf = scipy.signal.medfilt(f,31)
	c = np.polyfit(x,mf,3)

	while True:
		m = np.polyval(c,x)
		res = f - m
		I = np.where(res>0)[0]
		dev = np.median(res[I])
		J = np.where((res>-1*dev)&(res<4*dev))[0]
		#print len(J)
		K = np.where(res<-1*dev)[0]
		H = np.where(res>4*dev)[0]
		if (len(K)==0 and len(H) == 0) or len(J)<0.5*len(fo):
			break
		x,f = x[J],f[J]
		c = np.polyfit(x,f,3)
	return np.polyval(c,xo)


def rlineal(im,swav,sflux):
	lineal_flux = []
	lineal_wav = []

	for i in np.arange(len(im[0,:,0])-3):
		wav = im[0,i+3,:]
		flux = im[1,i+3,:]

		auxswav = swav.copy()
		auxsflux = sflux.copy()	

		mask = ( auxswav < np.max(wav)) & (auxswav > np.min(wav) )
		auxswav = auxswav[mask]
		auxsflux = auxsflux[mask]
	
		nflux = flux/cont(flux)

		tck = interpolate.splrep(auxswav,auxsflux,k=3)

		iflux = interpolate.splev(wav,tck)

		auxsflux /= np.add.reduce(auxsflux)
		nflux /= np.add.reduce(nflux)
		iflux /= np.add.reduce(iflux)
		
		dif = np.median(auxsflux) - np.median(iflux)
		lineal_wav.append(np.array(wav))
		lineal_flux.append(np.array(iflux)) 

	return lineal_wav,lineal_flux



def linealizacion(im,linstep,mode,instrument):
	lineal_x = []
	lineal_y = []
	print '-> Comienza linealizacion...'
	for i in np.arange(len((im[0,:,0]))):	
		#print 'linealizacion orden: ',i
		#im[0,i,:] = shift_doppler(im[0,i,:],vels)

		wav = im[0,i,:][im[3,1,:] != 0]
		flux = im[3,i,:][im[3,1,:] != 0]

		if instrument != 'feros':
			if np.max(im[0,i,:]) <= 5500:
				#print 'esta condicion'
				mask = (wav < np.max(wav)-4) & (wav > np.min(wav) + 5) 
				step = 3201
				auxx = wav[mask]
				auxy = flux[mask]			
				#auxx = wav[150:-150]
				#auxy = flux[150:-150]
				#auxx = im[0,i,:][500:-250]
				#auxy = im[3,i,:][500:-250]			
			else:
				step = 3501
				mask = (wav < np.max(wav) -1) & (wav > np.min(wav) + 1) 
				auxx = wav[mask]
				auxy = flux[mask]
				#auxx = im[0,i,:][200:-200]
				#auxy = im[3,i,:][200:-200]

		else:
			if np.max(im[0,i,:]) <= 5500:
				#print 'esta condicion'
				mask = (wav < np.max(wav)-4) & (wav > np.min(wav) + 40) 
				step = 3201
				auxx = wav[mask]
				auxy = flux[mask]			
			else:
				step = 3501
				mask = (wav < np.max(wav) -1) & (wav > np.min(wav) + 1) 
				auxx = wav[mask]
				auxy = flux[mask]

		minim = math.ceil(np.min(auxx))		#aprox al minimo entero
		maxi = math.floor(np.max(auxx))		#aprox al maximo entero

		x = auxx[(auxx < maxi) & (auxx > minim)]	#recorta la con la aprox
		y = auxy[(auxx < maxi) & (auxx > minim)]	
		if len(x) == 0:
			continue

		xdata,step = np.linspace(minim,maxi,(maxi-minim)*linstep+1.,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.125
		#xdata,step = np.linspace(minim,maxi,(maxi-minim)*8.+1,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.125
	 	#xdata,step = np.linspace(minim,maxi,(maxi-minim)*100.+1,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.001
		
	 	if mode == 'spline':
			fit = fit_splines(x,y,xdata,3101)	#Equiespacio los espectros
			#fit = fit_splines(x,y,xdata,901)	#Equiespacio los espectros


		elif mode == 'lineal':	
		 	tck = interpolate.splrep(x,y,k=3)
			fit = interpolate.splev(xdata,tck)
		else:
			print 'Error en opcion mode: elija spline o lineal'
			raise

		mask = (xdata < np.max(xdata) -3) & (xdata> np.min(xdata) + 3) 
			
		xdata = xdata[mask]		#Le quito los extremos del fit, que se alejan mucho
		fit = fit[mask]

		#ESTOS DOS PLOTS
		if plots != None:		
			plt.plot(x,y,'bo')
			plt.plot(xdata,fit,'go')

		lineal_x.append(np.array(xdata))
		lineal_y.append(np.array(fit)) 

	if plots != None:		
		plt.show()
	#raise
	print '-> Termina linealizacion'

	return lineal_x,lineal_y



def linealizacion_tres(spectra_list,linstep,mode):
	lineal_x = []
	lineal_y = []
	print '-> Comienza linealizacion...'

	for i in np.arange(len(spectra_list)):
		wav = spectra_list[i].wavelength
		flux = spectra_list[i].flux	
		wav = shift_doppler(wav,vel)
		wav = np.asarray(wav)
		flux = np.asarray(flux)

		if np.max(wav) >= 6500:
			continue
		if np.max(wav) <= 4700:
			print 'esta condicion en', np.max(wav)
			mask = (wav < np.max(wav)-8) & (wav > np.min(wav) + 10) 
			step = 3201

			auxx = wav[mask]
			auxy = flux[mask]			
		else:
			step = 3501
			mask = (wav < np.max(wav) -6) & (wav > np.min(wav) + 5) 

			auxx = wav[mask]
			auxy = flux[mask]
		
		if np.max(wav) >= 6500:
			continue
		if np.max(wav) <= 3900:
			continue
		minim = math.ceil(np.min(auxx))		#aprox al minimo entero
		maxi = math.floor(np.max(auxx))		#aprox al maximo entero

		x = auxx[(auxx < maxi) & (auxx > minim)]	#recorta la con la aprox
		y = auxy[(auxx < maxi) & (auxx > minim)]	
		if len(x) == 0:
			continue

		xdata,step = np.linspace(minim,maxi,(maxi-minim)*linstep+1.,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.125
		#xdata,step = np.linspace(minim,maxi,(maxi-minim)*8.+1,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.125
	 	#xdata,step = np.linspace(minim,maxi,(maxi-minim)*100.+1,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.001
		
	 	if mode == 'spline':
			fit = fit_splines(x,y,xdata,3101)	#Equiespacio los espectros
			#fit = fit_splines(x,y,xdata,901)	#Equiespacio los espectros


		elif mode == 'lineal':	
		 	tck = interpolate.splrep(x,y,k=3)
			fit = interpolate.splev(xdata,tck)
		else:
			print 'Error en opcion mode: elija spline o lineal'
			raise

		mask = (xdata < np.max(xdata)-4) & (xdata > np.min(xdata) + 4) 
		xdata = xdata[mask]
		fit = fit[mask]
		if plots != None:		
			plt.plot(x,y,'bo')
			plt.plot(xdata,fit,'go')

		lineal_x.append(np.asarray(xdata))
		lineal_y.append(np.asarray(fit)) 
	
	if plots != None:		
		plt.show()

	print '-> Termina linealizacion'

	return lineal_x,lineal_y

def linealizacion_chiron(im,linstep,mode,instrument):
	lineal_x = []
	lineal_y = []
	print '-> Comienza linealizacion...'

	for i in np.arange(len((im[:,0,0]))):	

		if np.min(im[i,:,0]) > 6860:
			continue


		wav = im[i,:,0]
		flux = im[i,:,1]

		if np.max(im[i,:,0]) <= 5500:
			#print 'esta condicion'
			mask = (wav < np.max(wav)-3) & (wav > np.min(wav) + 4) 
			step = 3201
			auxx = wav[mask]
			auxy = flux[mask]			
	
		else:
			step = 3501
			mask = (wav < np.max(wav) -1) & (wav > np.min(wav) + 1) 
			auxx = wav[mask]
			auxy = flux[mask]

		minim = math.ceil(np.min(auxx))		#aprox al minimo entero
		maxi = math.floor(np.max(auxx))		#aprox al maximo entero

		x = auxx[(auxx < maxi) & (auxx > minim)]	#recorta la con la aprox
		y = auxy[(auxx < maxi) & (auxx > minim)]	
		#print 'largos',(len(x)),len(y)
		if len(x) == 0:
			continue

		xdata,step = np.linspace(minim,maxi,(maxi-minim)*linstep+1.,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.125
		#xdata,step = np.linspace(minim,maxi,(maxi-minim)*8.+1,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.125
	 	#xdata,step = np.linspace(minim,maxi,(maxi-minim)*100.+1,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.001
		
	 	if mode == 'spline':
			fit = fit_splines(x,y,xdata,3101)	#Equiespacio los espectros
			#fit = fit_splines(x,y,xdata,901)	#Equiespacio los espectros


		elif mode == 'lineal':	
		 	tck = interpolate.splrep(x,y,k=3)
			fit = interpolate.splev(xdata,tck)
		else:
			print 'Error en opcion mode: elija spline o lineal'
			raise

		mask = (xdata < np.max(xdata) -2) & (xdata> np.min(xdata) + 2) 
			
		xdata = xdata[mask]		#Le quito los extremos del fit, que se alejan mucho
		fit = fit[mask]

		#ESTOS DOS PLOTS
		if plots != None:		
			plt.plot(x,y,'bo')
			plt.plot(xdata,fit,'go')

		lineal_x.append(np.array(xdata))
		lineal_y.append(np.array(fit)) 

	if plots != None:		
		plt.show()
	#raise
	print '-> Termina linealizacion'

	return lineal_x,lineal_y

def fit_cleanlines_iter(lineal_x,lineal_y,grado,n):
	waux = []
	faux = []
	ay = []
	fit_pol = []
	auxx = []
	auxy = []

	#m = 0
	#mask = clean_lines(lineal_x,lineal_y,spectype)
	####tengo errores, si hay erroroes solo ajustar lineal en linealx,linealy, sin borrar lineas ... 
	#if len(lineal_y[mask]) == 0:
	#	print 'error'
	#	return np.ones(len(lineal_x))

	for j in np.arange(n):
		if j == 0:
			mask = clean_lines(lineal_x,lineal_y,spectype)
			auxx.append(lineal_x[mask])
			auxy.append(lineal_y[mask])

			#if len(auxx[0]) == 0:

			#	continue
			if len(auxx[j]) == 0:
				plt.plot(lineal_x,lineal_y)
				plt.plot(lineal_x[mask],lineal_y[mask],'r')
				plt.show()

			coefs = poly.polyfit(auxx[j], auxy[j], grado)
			fit = poly.polyval(lineal_x, coefs)
			fit_pol.append(fit)
			ay.append(lineal_y/fit_pol[j])
		else:
			#ay = waux.copy()
			#print j
			#if len(auxx[0]) == 0 :
			#	continue
			#print 'largo',len(waux[j-1])
			#if (len (waux[j-1]) == 0):
			#	ay.append(np.ones(len(lineal_y)))
			#	continue
			#else:
			#plt.plot()
			#print 'waux', len(waux[j-1])
			#print np.min(waux[j-1])
			if len(waux[j-1]) < 5:
				print 'Elimino un orden completo al limpiar lineas. Cambiar *clean_lines*'
				raise
			coefs = poly.polyfit(waux[j-1], faux[j-1], grado)
			fit = poly.polyval(lineal_x, coefs)
			fit_pol.append(fit)				
			ay.append(lineal_y/fit)

		clip = sigma_clip(ay[-1], sigma=2.0)    #aplico sigma clip
		waux1 = np.asarray(lineal_x)[~clip.mask]
		faux1 = np.asarray(lineal_y)[~clip.mask]   
		mask = clean_lines(waux1,faux1,spectype)
		#waux1 = waux1[mask]
		#faux1 = faux1[mask]
		waux.append(waux1[mask])    
		faux.append(faux1[mask])   			

	if plots != None:
		plt.plot(waux[-1],faux[-1])

	#return lineal_y/fit_pol[-1]
	return fit_pol[-1]

def normalizacion(lineal_x,lineal_y):
	auxx = []
	auxy = []
	norm_y = []
	norm_x = []

	#recorro cada orden 
	for i in np.arange(len(lineal_x)):
		fitlineal_y = fit_cleanlines_iter (lineal_x[i],lineal_y[i],1,5)
		
		norm_y.append(np.array(lineal_y[i]/fitlineal_y))
		norm_x.append(np.array(lineal_x[i]))				

	return norm_x,norm_y

def merge(lineal_x,lineal_y):
	#raise
	raw_lineal_x = np.asarray(lineal_x).copy()
	raw_lineal_y = np.asarray(lineal_y).copy()
	wav = []
	flux = []
	auxx = []
	auxy = []
	if instrument == 'chiron':
		gradopol = 5
	else:
		gradopol = 2
	#Suma coincidencias de ordenes
	for i in np.arange(len(lineal_x)-1):
		if instrument == 'chiron':
			mask1 = lineal_x[i+1] <= np.max(lineal_x[i])   #mascara, [i+1] es el orden anterior del [i]. Donde coincide [i+1] con  
			mask2 = lineal_x[i] >= np.min(lineal_x[i+1]) 			
		else:
			mask1 = lineal_x[i+1] >= np.min(lineal_x[i])   #mascara, [i+1] es el orden anterior del [i]. Donde coincide [i+1] con  
			mask2 = lineal_x[i] <= np.max(lineal_x[i+1]) 
		#print 'No hay coincidencias rango', np.min(lineal_x[i]),'-', np.max(lineal_x[i]), 'y rango ', np.min(lineal_x[i+1]),'-', np.max(lineal_x[i+1])
		#Orden siguiente no se solapa
		#print 'nonceromask', np.count_nonzero(mask1)

		#SI NO EXISTEN COICIDENCIAS. Todo mask1 es falso
		if np.count_nonzero(mask1) == 0:
			print 'No hay coincidencias rango', np.min(lineal_x[i]),'-', np.max(lineal_x[i]), 'y rango ', np.min(lineal_x[i+1]),'-', np.max(lineal_x[i+1])
			lineal_fity = fit_cleanlines_iter(lineal_x[i],lineal_y[i],gradopol,5)
			lineal_fity2_aux = fit_cleanlines_iter(lineal_x[i+1],lineal_y[i+1],gradopol,5)
			lineal_fity2_aux = lineal_y[i+1]/lineal_fity2_aux
			if plots != None:
				plt.plot(lineal_x[i],lineal_y[i],alpha=.3)
				plt.plot(lineal_x[i],lineal_fity)		

			lineal_fity = lineal_y[i]/lineal_fity
			wav = np.append(wav,lineal_x[i])
			flux = np.append(flux, lineal_fity)

			#print 'minimo y maxi', np.max(lineal_x[i]), np.min(lineal_x[i+1])
			paso = np.round(lineal_x[i][1]-lineal_x[i][0],6)
			aux = (np.min(lineal_x[i+1]) - np.max(lineal_x[i]))/paso
			inter_x = []
			for j in np.arange(aux-1):

				newx = np.max(lineal_x[i])+paso*(j+1.)
				inter_x.append(newx)
		
			junto_aux_x = np.append(lineal_x[i][-1],lineal_x[i+1][0])
			junto_aux_y = np.append(lineal_fity[-1],lineal_fity2_aux[0])

			#median_aux = np.median(fit_cleanlines_iter(junto_aux_x,junto_aux_y,gradopol,5))
			#inter_y = np.ones(len(inter_x))*median_aux
			#tck = interpolate.splrep(junto_aux_x,junto_aux_y,k=3)
			#inter_y = interpolate.splev(inter_x,tck)
			
			coefs = poly.polyfit(junto_aux_x,junto_aux_y, 1)
			inter_y = poly.polyval(inter_x, coefs)

			if plots != None:
				plt.plot(inter_x,inter_y)

			wav = np.append(wav,inter_x)
			flux = np.append(flux, inter_y)	

			#juntox = np.append(lineal_x[i],np.append(inter_x,lineal_x[i+1]))
			#juntoy = np.append(lineal_fity,np.append(inter_y,lineal_fity2_aux))

		else:
			print 'Coinciden los ordenes ', i, 'y', i+1, ' entre ', np.min(lineal_x[i]),'-', np.max(lineal_x[i]), 'y rango ', np.min(lineal_x[i+1]),'-', np.max(lineal_x[i+1])
			#print len(lineal_y[i+1]), len(mask1),  len(lineal_y[i]), len (mask2)
			sumy = (lineal_y[i+1][mask1] + lineal_y[i][mask2])  #guardo la suma de los ordenes
			sumx = lineal_x[i+1][mask1]

			lineal_x[i+1] = lineal_x[i+1][mask1==False]  #recorto el orden anterior
			lineal_y[i+1] = lineal_y[i+1][mask1==False]  

			lineal_x[i] = lineal_x[i][mask2==False]  #recorto el orden
			lineal_y[i] = lineal_y[i][mask2==False]  #recorto el orden

			if plots != None:
				plt.plot(lineal_x[i+1],lineal_y[i+1], alpha=.3)

			
			
			lineal_fity = fit_cleanlines_iter(lineal_x[i],lineal_y[i],gradopol,5)
			if plots != None:
				plt.plot(lineal_x[i],lineal_fity)

			lineal_fity = lineal_y[i]/lineal_fity
			wav = np.append(wav,lineal_x[i])
			flux = np.append(flux, lineal_fity)		

			if len(sumx[clean_lines(sumx,sumy,spectype)]) <= 1:
				print 'el orden', i, 'no se solapa con el',i+1
			elif len(sumy[clean_lines(sumx,sumy,spectype)]) <= 1:
				print 'el orden', i, 'no se solapa con el',i+1
				#print 'oli'
			elif len(sumx) <= 1:
				print 'el orden', i, 'no se solapa con el',i+1
			else:
				#print 'Se solapa orden',i
				lineal_fitysum = fit_cleanlines_iter(sumx,sumy,2,2)
				if plots != None:		
					plt.plot(sumx,sumy,alpha=.3)

					plt.plot(sumx,lineal_fitysum,'k')

				lineal_fitysum = sumy/lineal_fitysum
				wav = np.append(wav, sumx)
				flux = np.append(flux,lineal_fitysum)

	if plots != None:
		plt.show()
	#raise
	return wav,flux

def merge_chiron(lineal_x,lineal_y):
	#raise
	raw_lineal_x = np.asarray(lineal_x).copy()
	raw_lineal_y = np.asarray(lineal_y).copy()
	wav = []
	flux = []
	auxx = []
	auxy = []
	if instrument == 'chiron':
		gradopol = 5
	else:
		gradopol = 2
	#Suma coincidencias de ordenes
	for i in np.arange(len(lineal_x)-1):
		mask1 = lineal_x[i+1] >= np.min(lineal_x[i])   #mascara, [i+1] es el orden anterior del [i]. Donde coincide [i+1] con  
		mask2 = lineal_x[i] <= np.max(lineal_x[i+1]) 
		#print 'No hay coincidencias rango', np.min(lineal_x[i]),'-', np.max(lineal_x[i]), 'y rango ', np.min(lineal_x[i+1]),'-', np.max(lineal_x[i+1])

		##si no hay coincidencias
		if np.max(lineal_x[i]) >=  np.min(lineal_x[i+1]):
			print 'No hay coincidencias rango', np.min(lineal_x[i]),'-', np.max(lineal_x[i]), 'y rango ', np.min(lineal_x[i+1]),'-', np.max(lineal_x[i+1])
			lineal_fity = fit_cleanlines_iter(lineal_x[i],lineal_y[i],gradopol,5)
			if plots != None:
				plt.plot(lineal_x[i],lineal_y[i],alpha=.3)
				plt.plot(lineal_x[i],lineal_fity)		

			lineal_fity = lineal_y[i]/lineal_fity
			wav = np.append(wav,lineal_x[i])
			flux = np.append(flux, lineal_fity)

		else:

			sumy = (lineal_y[i+1][mask1] + lineal_y[i][mask2])  #guardo la suma de los ordenes
			sumx = lineal_x[i+1][mask1]

			lineal_x[i+1] = lineal_x[i+1][mask1==False]  #recorto el orden anterior
			lineal_y[i+1] = lineal_y[i+1][mask1==False]  

			lineal_x[i] = lineal_x[i][mask2==False]  #recorto el orden
			lineal_y[i] = lineal_y[i][mask2==False]  #recorto el orden

			if plots != None:
				plt.plot(lineal_x[i+1],lineal_y[i+1], alpha=.3)

			
			lineal_fity = fit_cleanlines_iter(lineal_x[i],lineal_y[i],gradopol,5)
			if plots != None:
				plt.plot(lineal_x[i],lineal_fity)

			lineal_fity = lineal_y[i]/lineal_fity
			wav = np.append(wav,lineal_x[i])
			flux = np.append(flux, lineal_fity)		

			if len(sumx[clean_lines(sumx,sumy,spectype)]) <= 1:
				print 'el orden', i, 'no se solapa con el',i+1
				print 'OLO'
			elif len(sumx) <= 1:
				print 'el orden', i, 'no se solapa con el',i+1
			else:
				print 'No se solapa orden',i
				lineal_fitysum = fit_cleanlines_iter(sumx,sumy,2,2)
				if plots != None:		
					plt.plot(sumx,sumy,alpha=.3)

					plt.plot(sumx,lineal_fitysum,'k')

				lineal_fitysum = sumy/lineal_fitysum
				wav = np.append(wav, sumx)
				flux = np.append(flux,lineal_fitysum)

	if plots != None:
		plt.show()
	raise
	return wav,flux

def merge_test(lineal_x,lineal_y):
	#raise
	raw_lineal_x = np.asarray(lineal_x).copy()
	raw_lineal_y = np.asarray(lineal_y).copy()
	wav = []
	flux = []
	auxx = []
	auxy = []
	#Suma coincidencias de ordenes
	for i in np.arange(len(lineal_x)-1):
		mask1 = lineal_x[i+1] >= np.min(lineal_x[i])   #mascara, [i+1] es el orden anterior del [i]. Donde coincide [i+1] con  
		mask2 = lineal_x[i] <= np.max(lineal_x[i+1]) 

		if len(lineal_y[i+1][mask1]) == len (lineal_y[i][mask2]):  ###aplicar solo si los ordenes se solapan

			sumy = (lineal_y[i+1][mask1] + lineal_y[i][mask2])  #guardo la suma de los ordenes
			sumx = lineal_x[i+1][mask1]

			lineal_x[i+1] = lineal_x[i+1][mask1==False]  #recorto el orden anterior
			lineal_y[i+1] = lineal_y[i+1][mask1==False]  

			lineal_x[i] = lineal_x[i][mask2==False]  #recorto el orden
			lineal_y[i] = lineal_y[i][mask2==False]  #recorto el orden

		else:
			print 'el orden', i, 'no se solapa con el',i+1

		if len(sumx[clean_lines(sumx,sumy,spectype)]) <= 1:
			#print 'oli'
			continue
		if len(sumy[clean_lines(sumx,sumy,spectype)]) <= 1:
			continue
			#print 'oli'
		if len(sumx) <= 1:
			continue

		lineal_fity = fit_cleanlines_iter(lineal_x[i],lineal_y[i],2,5)
		lineal_fitysum = fit_cleanlines_iter(sumx,sumy,2,2)
		
		lineal_fity = lineal_y[i]/lineal_fity
		lineal_fitysum = sumy/lineal_fitysum
				##ESTE PLOT
		plt.plot(sumx,lineal_fitysum,'k')
		plt.plot(lineal_x[i],lineal_fity)
		wav = np.append(wav,lineal_x[i])
		wav = np.append(wav, sumx)
		flux = np.append(flux, lineal_fity)
		flux = np.append(flux,lineal_fitysum)

	plt.show()
	return wav,flux

def clean_outliers(wav,flux):
	#cleanflux = clean_lines(wav,flux)
	clip = sigma_clip(flux, sigma=7.0)    #aplico sigma clip
	wav = np.asarray(wav)[~clip.mask]
	flux = np.asarray(flux)[~clip.mask]   
	return wav,flux

def gauss(x,a,x0,sigma):
	p = [a, x0, sigma]
	return p[0]* np.exp(-((x-p[1])/p[2])**2)

def ajuste_gauss(x,y):
	mean = sum(x * y) /sum(y)
	sigma = np.sqrt(sum(y * (x - mean)**2.)/ sum(y))
	#dif = (np.max(x) + np.min(x))/2.
	#mask = (x > (dif/10.*2)) & (x < (dif/10.*8))
	print len(y)/10.
	maxi = np.max(y[int(len(y)/10.):-int(len(y)/10.)])
	p0 = [maxi,mean, (x[1]-x[0])*10.]
	fit, tmp = curve_fit(gauss,x,y,p0=p0)
	xnew = np.linspace(x[0],x[-1],len(x))
	gausfit = gauss(x,fit[0],fit[1],fit[2])

	return gausfit

def wavstovel(wav):
	mid = ((np.max(wav)+np.min(wav))/2.)
	return ((wav-mid)/mid)*(c), mid

def resamplewav(wav):
	mid = (np.max(wav)-np.min(wav))/2.
	vels = np.linspace(-100.,100.,2e2)

	return  mid / (1 - (vels/c))

def veltowav(vel,wav0):
	wav = ((vel/c)*wav0)+wav0
	return wav

def generatevelarray(wav,flux,step):
	mid = ((np.max(wav)+np.min(wav))/2.)
	nvel,mid = wavtovel(wav)
	print nvel[1] - nvel[0]
	plt.plot(nvel,flux)
	plt.show()

	wavi = veltowav(nvel,mid)

	plt.plot(wavi,flux,'bo')
	plt.plot(wav,flux,'ro')	
	plt.show()
	
	print (nvel[1] - nvel[0])/10.
	xvel = np.linspace(np.min(nvel),np.max(nvel), len(nvel)*3. )
	tck = interpolate.splrep(nvel,flux,k=3)
	fluxvel = interpolate.splev(xvel,tck)	
	
	plt.plot(xvel,fluxvel,'ro')
	plt.plot(nvel,flux,'bo')
	plt.show()
	raise

	tck = interpolate.splrep(nvel,flux,k=3)
	fluxvel = interpolate.splev(xvel,tck)

	mid = ((np.max(wav)+np.min(wav))/2.)
	nwav = veltowav(xvel, mid)

	return x,yflux,xvel


def veldeconvolution2(w,nf,wav,smod):

	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	nn = 10*int(len(nf)/(w[-1]-w[0]))
	
	# Now I generate the matrix of the template spectrum
	mat = []
	i=0

	midwav = []
	stepvel = []
	j = 0
	vels = np.linspace(-velslim,velslim,velstep)
	while i < len(nmod)-nn:

		#print i
		#vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		#genero arreglo de vel a partir de las wav, guardo wav centrales
		#synwav_samplevel = resamplewav(w[i:i+nn+1])
		mid = np.median(w[i:i+nn+1])
		#mid = (np.max(w[i:i+nn+1]) + np.min(w[i:i+nn+1]))/2.
		
		new_wav = mid / (1. - (vels/c))
		#genero un vector de la matriz de deconvolucion
		#linealizo

		#interpolacion antigua
		#tck = interpolate.splrep(w[i:i+nn+1],nmod[i:i+nn+1],k=1)
		#new_flux = interpolate.splev(new_wav,tck)


		lineal_interp = interp1d(w[i:i+nn+1],nmod[i:i+nn+1])
		new_flux = lineal_interp(new_wav)

		if 	np.isnan(new_flux).sum() != 0 :

			j +=1
			new_flux = np.ones(len(new_flux))

		vec = new_flux.copy()
		if len(mat) == 0:
			mat = vec
		else:	
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = nf[int(0.5*nn):-(int(0.5*nn))]

	####longitud de onda de sampleo del sintetico ..... el y y el ww deberia estar en longitud de onda que ocupo para llevar a velocidades (wav central)
	## hacer interpolacion lineal ... en vez de splines 
	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	return np.array(vels),np.array(A)


def veldeconvolution3(w,nf,wav,smod):

	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	nn = 6*int(len(nf)/(w[-1]-w[0]))
	
	# Now I generate the matrix of the template spectrum
	mat = []
	i=0

	midwav = []
	stepvel = []
	print 'step',len(w[i:i+nn+1])

	#velsfix = np.linspace(-100,100,200)
	velsfix = np.linspace(-velslim,velslim,velstep)

	auxwav = []
	while i < len(nmod)-nn:
		#print i
		#vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		#genero arreglo de vel a partir de las wav, guardo wav centrales
		#synwav_samplevel = resamplewav(w[i:i+nn+1])
		mid = np.median(w[i:i+nn+1])
		#mid = (np.max(w[i:i+nn+1]) + np.min(w[i:i+nn+1]))/2.
		
		vels,midwav = wavstovel(w[i:i+nn+1])
		auxwav.append(midwav)
		#genero un vector de la matriz de deconvolucion
		#linealizo
		tck = interpolate.splrep(vels,nmod[i:i+nn+1],k=3)
		new_flux = interpolate.splev(velsfix,tck)
		if 	np.isnan(new_flux).sum() != 0:

			#print np.isnan(new_flux).sum()
			#print np.isnan(new_wav).sum()
			#plt.plot(new_wav,new_flux,'bo')
			#plt.show()
			return np.asarray([0]),np.asarray([0])

		#new_flux, new_wav = ordenar(new_wav,new_flux)				
		#plt.plot(w[i:i+nn+1],nmod[i:i+nn+1],'ro')
		#plt.plot(new_wav,new_flux,'bo')
		#plt.show()

		vec = new_flux.copy()
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = nf[int(0.5*nn):-(int(0.5*nn))]

	#midwav = np.linspace(np.min(auxwav),np.max(auxwav), len(auxwav) )
	#tck = interpolate.splrep(ww,y,k=3)
	#midflux = interpolate.splev(midwav,tck)


	#plt.plot(w,nf,'ro')
	#plt.plot(xvel,fluxvel,'bo')
	#plt.show()
	#raise
	# we dont need reshape

	####longitud de onda de sampleo del sintetico ..... el y y el ww deberia estar en longitud de onda que ocupo para llevar a velocidades (wav central)
	## hacer interpolacion lineal ... en vez de splines 
	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
		#print A # this is the kernel
	#print B
	#print C
	#print D

	#x = np.linspace(-len(A)/(xvel[-1]-xvel[0])/2.,len(A)/(xvel[-1]-xvel[0])/2., len(A))
	#x = np.linspace(-len(A)/np.min(stepvel)/2.,len(A)/np.min(stepvel)/2., len(A))

	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	#plt.plot(x,A)
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')

	#raise
	#plt.show()
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	#plt.plot(vels,A)
	#plt.show()

	#plt.plot(w,nf)
	#plt.plot(w,nf,'ro')

	#plt.plot(w,nmod)
	#plt.plot(w,nmod,'go')

	#plt.show()
	return np.array(velsfix),np.array(A)



def veldeconvolution(w,nf,wav,smod):

	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	####esta todo sampleado a w

	#mid = ((np.max(w)+np.min(w))/2.)

	#nvel,_ = wavtovel(w)
	#plt.plot(nvel,nf)
	#plt.show()	
	
	#xvel = np.linspace(np.min(nvel),np.max(nvel), len(nvel) )
	#tck = interpolate.splrep(nvel,nf,k=3)
	#fluxvel = interpolate.splev(xvel,tck)	

	#xvel = np.linspace(np.min(mid),np.max(mid), len(mid) )
	#tck = interpolate.splrep(mid,nf,k=3)
	#fluxvel = interpolate.splev(nmid,tck)	

	nn = 2*int(len(nf)/(w[-1]-w[0]))


	#nn = 2*int(len(fluxvel)/(xvel[-1]-xvel[0]))	
	# Now I generate the matrix of the template spectrum
	mat = []
	i=0
	#print len(nmod)-nn 			#len(nmod)-nn es el largo que tendra el nuevo template
	#raise
	midwav = []
	stepvel = []
	while i < len(nmod)-nn:
		#print i
		#vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		#genero arreglo de vel a partir de las wav, guardo wav centrales
		synvel,mid = wavstovel(w[i:i+nn+1])
		midwav = np.append(midwav,mid)
		#genero un vector de la matriz de deconvolucion
		vecflux = nmod[i:i+nn+1]
		#linealizo
		linsynvel = np.linspace(np.min(synvel),np.max(synvel), len(synvel) )
		stepvel = np.append(stepvel,linsynvel[1]-linsynvel[0])
		#plt.plot(synvel,vecflux)
		#plt.show()
		tck = interpolate.splrep(synvel,vecflux,k=3)
		linsynfluxvel = interpolate.splev(linsynvel,tck)					
		
		vec = linsynfluxvel.copy()
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat
	xvel = np.linspace(np.min(midwav),np.max(midwav), len(midwav) )
	tck = interpolate.splrep(w,nf,k=3)
	fluxvel = interpolate.splev(xvel,tck)

	#plt.plot(w,nf,'ro')
	#plt.plot(xvel,fluxvel,'bo')
	#plt.show()
	print np.min(stepvel),np.max(stepvel)
	#raise
	# we dont need reshape
	ww = xvel.copy()
	y = fluxvel.copy()	
	####longitud de onda de sampleo del sintetico ..... el y y el ww deberia estar en longitud de onda que ocupo para llevar a velocidades (wav central)
	## hacer interpolacion lineal ... en vez de splines 
	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	#print A # this is the kernel
	#print B
	#print C
	#print D
	xvel,_ = wavstovel(xvel)

	#x = np.linspace(-len(A)/(xvel[-1]-xvel[0])/2.,len(A)/(xvel[-1]-xvel[0])/2., len(A))
	x = np.linspace(-len(A)/np.min(stepvel)/2.,len(A)/np.min(stepvel)/2., len(A))

	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	#plt.plot(x,A)
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	bini = xvel[1]-xvel[0]
	dif = (xvel[-1]+xvel[0])/2.
	print xvel
	print bini, dif
	#raise
	#plt.show()
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	#plt.plot(x,A)
	#plt.show()

	#plt.plot(w,nf)
	#plt.plot(w,nf,'ro')

	#plt.plot(w,nmod)
	#plt.plot(w,nmod,'go')

	#plt.show()
	return x,A

def rconvolution(w,nf,wav,smod):

	tck = interpolate.splrep(w,nf,k=3)
	inf = interpolate.splev(wav,tck)
	plt.plot(w,nf)
	plt.plot(w,nf,'ro')

	nn = 2*int(len(inf)/(w[-1]-w[0]))

	# Now I generate the matrix of the template spectrum
	mat = []
	i=0
	print len(smod)-nn 			#len(nmod)-nn es el largo que tendra el nuevo template
	#raise
	while i < len(smod)-nn:
		#print i
		vec = smod[i:i+nn+1]    #smod flux template interpolado al sampleo observado 
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	# and here I reshape the observed spectrum to match with the dimensions of the multiplication of the matrix with the kernel vector
	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = inf[int(0.5*nn):-(int(0.5*nn))]

	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	#print A # this is the kernel

	x = np.linspace(-len(A)/(w[-1]-w[0])/2.,len(A)/(w[-1]-w[0])/2., len(A))

	plt.plot(x,A)

	return x,A


def wav_deconvolution(w,nf,wav,smod):
	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)

	#I will consider that the kernel has a width of 2 amstrongs, and here I compute how many pixels do I need for that.
	
	nn = 6*int(len(nf)/(w[-1]-w[0]))

	# Now I generate the matrix of the template spectrum
	mat = []
	i=0
	print len(nmod)-nn 			#len(nmod)-nn es el largo que tendra el nuevo template
	#raise
	while i < len(nmod)-nn:
		#print i
		vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	# and here I reshape the observed spectrum to match with the dimensions of the multiplication of the matrix with the kernel vector
	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = nf[int(0.5*nn):-(int(0.5*nn))]
	print 'nf y w',len(nf), len(w)

	print 'len y y len mat', len(y), len(mat)

	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	#print A # this is the kernel
	x = np.linspace(-len(A)/2.,len(A)/2.,len(A)  )
	plt.xlabel('wav A')
	
	return x,A

def condicion_linea(wav):
	a = False
	if (np.min(wav_aux) < 6565.) & (np.max(wav_aux) > 6565):
		a = True
	if (np.min(wav_aux) < 4863.) & (np.max(wav_aux) > 4863):
		a = True
	if (np.min(wav_aux) < 4342.) & (np.max(wav_aux) > 4342):
		a = True
	if (np.min(wav_aux) < 5180.) & (np.max(wav_aux) > 5180):
		a = True
	if (np.min(wav_aux) < 5380.) & (np.max(wav_aux) > 5380):
		a = True
	if (np.min(wav_aux) < 5650.) & (np.max(wav_aux) > 5650):
		a = True
	if (np.min(wav_aux) < 5850.) & (np.max(wav_aux) > 5850):
		a = True
	if (np.min(wav_aux) < 6080.) & (np.max(wav_aux) > 6080):
		a = True
	if (np.min(wav_aux) < 6480.) & (np.max(wav_aux) > 6480):
		a = True

	return a 

def deconv_multiproc(data):
	w,nf,wav,smod = data 

	print 'rango de ', np.min(wav),' - ', np.max(w)

	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	nn = 10*int(len(nf)/(w[-1]-w[0]))
	
	# Now I generate the matrix of the template spectrum
	mat = []
	i=0

	midwav = []
	stepvel = []
	#print 'step',len(w[i:i+nn+1])
	j = 0
	### algunas config que sirven son: -100,100,250, con linstep = 95 u 80
	vels = np.linspace(-velslim,velslim,velstep)

	while i < len(nmod)-nn:

		#vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		#genero arreglo de vel a partir de las wav, guardo wav centrales
		#synwav_samplevel = resamplewav(w[i:i+nn+1])
		mid = np.median(w[i:i+nn+1])
		
		new_wav = mid / (1. - (vels/c))
		#genero un vector de la matriz de deconvolucion
		#linealizo
		#new interpolation
		lineal_interp = interp1d(w[i:i+nn+1],nmod[i:i+nn+1])
		new_flux = lineal_interp(new_wav)


		if 	np.isnan(new_flux).sum() != 0 :

			j +=1
			print ('se corta') 
			new_flux = np.ones(len(new_flux))

			#return np.asarray([0]),np.asarray([0])

		#new_flux, new_wav = ordenar(new_wav,new_flux)				

		vec = new_flux.copy()
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = nf[int(0.5*nn):-(int(0.5*nn))]

	# we dont need reshape

	####longitud de onda de sampleo del sintetico ..... el y y el ww deberia estar en longitud de onda que ocupo para llevar a velocidades (wav central)
	## hacer interpolacion lineal ... en vez de splines 
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
		#print A # this is the kernel

	if plots != None:

		plt.plot(vels,A)
		#titulo = str('rango de ' + np.min(wav) + ' - ' + np.max(wav))
		plt.title(titulo)
		plt.show()
	#raise
	return np.array(A)

def deconv_multiproc_newmatrix(data):
	w,nf,wav,smod = data 

	print 'rango de ', np.min(wav),' - ', np.max(w)

	#Resampleo espectro sintetico a sample observado
	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	nn = 10*int(len(nf)/(w[-1]-w[0]))
	
	# Now I generate the matrix of the template spectrum
	mat = []
	

	midwav = []
	stepvel = []
	#print 'step',len(w[i:i+nn+1])
	j = 0
	### algunas config que sirven son: -100,100,250, con linstep = 95 u 80
	vels = np.linspace(-velslim,velslim,velstep)


	i = 0

	# formamos matrix de convolucion (len(nmod)-nn+1)x(n-1)
	# llamaremos m = len(nmod) largo espectro observado (o)
	# donde las filas son o1 o2 ..on / o2 o3 ... on+1 etc  (o: elementos del espectro observado)
	# hasta o(m-nn+1) o(m-nn+2) ... o(m)
	# por lo que tendra (m-n+1) filas y n columnas
	# esta matriz esta convolucionada con kernel de longitud nn
	# como i parte en cero, restamos 1 al largo numerico
	while i <= len(nmod)-nn:

		#se resta uno pq nn es una posicion.. si n=1, en caso i=0 w[0:0] es solo un elemento asi q esta bien! 
		mid = np.median(w[i:i+nn-1])
		#print 'Step old', w[1]-w[0]
		new_wav = mid / (1. - (vels/c))
		#print 'Step new', new_wav[1]-new_wav[0]
		#new interpolation
		lineal_interp = interp1d(w[i:i+nn-1],nmod[i:i+nn-1])
		new_flux = lineal_interp(new_wav)
			
		#AQI puede haber error	
		if 	np.isnan(new_flux).sum() != 0 :
	
			#print np.isnan(new_flux).sum()
			#print np.isnan(new_wav).sum()

			j +=1
			print ('se corta') 
			new_flux = np.ones(len(new_flux))

		vec = new_flux.copy()
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	ww = w[int(0.5*nn)-1:-(int(0.5*nn))]
	y = nf[int(0.5*nn)-1:-(int(0.5*nn))]
	mid2 = np.median(ww)

	vels2 = np.linspace(-velslim,velslim,len(ww))
	new_ww = mid2 / (1. - (vels2/c))
	lineal_interp = interp1d(ww,y)
	new_y = lineal_interp(new_ww)

	# we dont need reshape

	#print len(y)
	#print len(new_y )

	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	A2,B2,C2,D2 = np.linalg.lstsq(mat, new_y, rcond=None)

	if plots != None:

		plt.plot(vels,A)
		plt.plot(vels,A2,'b')
		plt.show()

	return np.array(A)


def deconv_multiproc_newmatrix2(data):
	w, nf, wav, smod = data 

	print 'rango de ', np.min(wav),' - ', np.max(w)

	#Resampleo espectro sintetico a sample observado
	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	nn = 10*int(len(nf)/(w[-1]-w[0]))
	
	# Now I generate the matrix of the template spectrum
	mat = []
	

	midwav = []
	stepvel = []
	j = 0
	### algunas config que sirven son: -100,100,250, con linstep = 95 u 80
	vels = np.linspace(-velslim,velslim,velstep)

	wav_syn = w[int(nn/2)-1:-int(nn/2)]
	flux_syn = nmod[int(nn/2)-1:-int(nn/2)]

	print 'LONGITUDES', len(wav_syn), nn, len(nf)

	padding = np.zeros(nn - 1, nf.dtype)

	first_col = np.r_[flux_syn, padding]
	first_row = np.r_[flux_syn[0], padding]

	H = scipy.linalg.toeplitz(first_col, first_row)

	maux = np.matmul(H.transpose(),H)
	paux = np.matmul(np.linalg.inv(maux), H.transpose())
	baux = np.matmul(paux,nf)

	A,B,C,D = np.linalg.lstsq(H, nf , rcond=None)


	print baux
	#if plots != None:

		#plt.plot(w,nmod)
		#plt.plot(w,nmod,'go')
	plt.plot(np.arange(len(A))[20:-20],A[20:-20],'r')
	plt.plot(np.arange(len(baux))[20:-20], baux[20:-20],'b')
	titulo = str('rango de ' + str(np.min(w)) + ' - ' + str(np.max(w)))
	plt.title(titulo)
	plt.show()
	raise

	fig = plt.figure(constrained_layout=True,figsize=[22, 4])
	gs = gridspec.GridSpec(4, 1, figure=fig)
	ax = fig.add_subplot(gs[:3, 0])
	ax.plot(np.arange(len(A))[20:-20],A[20:-20], '-k', lw=1, zorder=-2)
			
	ax.set_xlabel("vels")
	ax.set_ylabel("broadening kernel")
			
	plt.title("broadening kernel - rms")
	#plt.show()
	ax2 = fig.add_subplot(gs[3, 0])
	ax2.plot(w,nf,'-b', lw=.5)
	ax2.plot(w,nmod,'-r')
	#ax2.set_xlim([-40,40])
	ax.set_xlabel("wav")
	ax.set_ylabel("flux")
	plt.title('spectra' )
	
	#ax3 = fig.add_subplot(gs[1, 1])
	#ax3.scatter(ph, f, c='gold', edgecolor='black', s=15, lw=.5)
	#period = period*2.
	#ph = (t - t0 + 0.5*period) % period - 0.5*period
	#ax3.scatter(ph, f, s=15, lw=.5)
		
			
	#fig.suptitle('%s'%name, fontsize=16)
	#plt.title('phased light curve' )
	#plt.savefig(name_deconv.split('.dat')[0] + '_meanrms'+str(rms1) + '.png')
	#plt.close()
	plt.show()
	#raise
	print 'Largo A', len(A)
	raise
	return np.array(A)



def vsin_kernel(wave_spec,flux_spec, vrot, epsilon):
	#wave_ = np.log(wave_spec) 
	#velo_ = np.linspace(wave_[0],wave_[-1],len(wave_)) 
	#flux_ = np.interp(velo_,wave_,flux_spec) 
	wave_ = np.log(wave_spec) 
	#wave_ = wave_spec.copy()
	velo_ = np.linspace(wave_[0],wave_[-1],len(wave_)) 
	flux_ = flux_spec.copy()
	#plt.plot(wave_spec,flux_spec,'bo')
	#plt.plot(wave_spec,flux_,'ro')
	#plt.show()
	dvelo = velo_[1]-velo_[0] 
	#vrot = vrot/(c*1e-3) 
	vrot_ = vrot/(c) 

	#-- compute the convolution kernel and normalise it 
	n = int(2*vrot_/dvelo) 
	velo_k = np.arange(n)*dvelo 
	velo_k -= velo_k[-1]/2. 
	y = 1 - (velo_k/vrot_)**2 # transformation of velocity 
	G = (2*(1-epsilon)*np.sqrt(y)+np.pi*epsilon/2.*y)/(np.pi*vrot_*(1-epsilon/3.0))  # the kernel 
	G /= G.sum() 
	#-- convolve the flux with the kernel 

	flux_conv = np.convolve(1-flux_,G,mode='same') 
	velo_ = np.arange(len(flux_conv))*dvelo+velo_[0] 
	wave_conv = np.exp(velo_) 

	#vrot = vrot
	#-- compute the convolution kernel and normalise it 
	#n = int(2*vrot/dvelo) 
	#velo_k = np.arange(n)*dvelo 
	#velo_k -= velo_k[-1]/2. 

	return G,velo_k

def broadGaussFast(x, y, sigma, edgeHandling=None, maxsig=None):
    """
    Apply Gaussian broadening. 
    This function broadens the given data using a Gaussian
    kernel.
    Parameters
    ----------
    x, y : arrays
        The abscissa and ordinate of the data.
    sigma : float
        The width (i.e., standard deviation) of the Gaussian
        profile used in the convolution.
    edgeHandling : string, {None, "firstlast"}, optional
        Determines the way edges will be handled. If None,
        nothing will be done about it. If set to "firstlast",
        the spectrum will be extended by using the first and
        last value at the start or end. Note that this is
        not necessarily appropriate. The default is None.
    maxsig : float, optional
        The extent of the broadening kernel in terms of
        standard deviations. By default, the Gaussian broadening
        kernel will be extended over the entire given spectrum,
        which can cause slow evaluation in the case of large spectra.
        A reasonable choice could, e.g., be five.  
    Returns
    -------
    Broadened data : array
        The input data convolved with the Gaussian
        kernel.
    """
    # Check whether x-axis is linear
    dxs = (x[1:] - x[0:-1])*0.1

    #if abs(max(dxs) - min(dxs)) > np.mean(dxs) * 1e-6:
    #	print ('wavs no son equidistantes!')
    #    raise
    if maxsig is None:
        lx = len(x)
    else:
        lx = int(((sigma * maxsig) / dxs[0]) * 2.0) + 1
    # To preserve the position of spectral lines, the broadening function
    # must be centered at N//2 - (1-N%2) = N//2 + N%2 - 1
    nx = (np.arange(lx, dtype=np.int) - sum(divmod(lx, 2)) + 1) * dxs[0]
    gf = fuf.GaussFit1d()
    gf["A"] = 1.0
    gf["sig"] = sigma
    e = gf.evaluate(nx)
    # This step ensured that the
    e /= np.sum(e)

    '''if edgeHandling == "firstlast":
                    nf = len(y)
                    y = np.concatenate((np.ones(nf) * y[0], y, np.ones(nf) * y[-1]))
                    result = np.convolve(y, e, mode="same")[nf:-nf]
                elif edgeHandling is None:
                    result = np.convolve(y, e, mode="same")
                else:
                    raise(PE.PyAValError("Invalid value for `edgeHandling`: " + str(edgeHandling),
                                         where="broadGaussFast",
                                         solution="Choose either 'firstlast' or None"))'''
    return e,nx



def instrBroadGaussFast(wvl, flux, resolution, edgeHandling=None, fullout=False, maxsig=None):
    """
    Apply Gaussian instrumental broadening. 
    This function broadens a spectrum assuming a Gaussian
    kernel. The width of the kernel is determined by the
    resolution. In particular, the function will determine
    the mean wavelength and set the Full Width at Half
    Maximum (FWHM) of the Gaussian to
    (mean wavelength)/resolution. 
    Parameters
    ----------
    wvl : array
        The wavelength
    flux : array
        The spectrum
    resolution : int
        The spectral resolution.
    edgeHandling : string, {None, "firstlast"}, optional
        Determines the way edges will be handled. If None,
        nothing will be done about it. If set to "firstlast",
        the spectrum will be extended by using the first and
        last value at the start or end. Note that this is
        not necessarily appropriate. The default is None.
    fullout : boolean, optional
        If True, also the FWHM of the Gaussian will be returned.
    maxsig : float, optional
        The extent of the broadening kernel in terms of
        standard deviations. By default, the Gaussian broadening
        kernel will be extended over the entire given spectrum,
        which can cause slow evaluation in the case of large spectra.
        A reasonable choice could, e.g., be five.  
    Returns
    -------
    Broadened spectrum : array
        The input spectrum convolved with a Gaussian
        kernel.
    FWHM : float, optional
        The Full Width at Half Maximum (FWHM) of the
        used Gaussian kernel.
    """
    # Check whether wvl axis is linear
    dwls = wvl[1:] - wvl[0:-1]
    #if abs(max(dwls) - min(dwls)) > np.mean(dwls) * 1e-6:
    #	print ('error: Las longitudes de onda deberian estar equiespaciadas')
    #    raise

    meanWvl = np.mean(wvl)
    fwhm = 1.0 / float(resolution) * meanWvl
    sigma = fwhm / (2.0 * np.sqrt(2. * np.log(2.)))

    e,x = broadGaussFast(
        wvl, flux, sigma, edgeHandling=edgeHandling, maxsig=maxsig)

    if not fullout:
        return result
    else:
        return e,x, fwhm




 ####leer IRAF spectra
def nonlinearwave(nwave, specstr, verbose=False):
    """Compute non-linear wavelengths from multispec string
    
    Returns wavelength array and dispersion fields.
    Raises a ValueError if it can't understand the dispersion string.
    """

    fields = specstr.split()
    if int(fields[2]) != 2:
        raise ValueError('Not nonlinear dispersion: dtype=' + fields[2])
    if len(fields) < 12:
        raise ValueError('Bad spectrum format (only %d fields)' % len(fields))
    wt = float(fields[9])
    w0 = float(fields[10])
    ftype = int(fields[11])
    if ftype == 3:

        # cubic spline

        if len(fields) < 15:
            raise ValueError('Bad spline format (only %d fields)' % len(fields))
        npieces = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            print 'Dispersion is order-%d cubic spline' % npieces
        if len(fields) != 15 + npieces + 3:
            raise ValueError('Bad order-%d spline format (%d fields)' % (npieces, len(fields)))
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        s = (np.arange(nwave, dtype=float) + 1 - pmin) / (pmax - pmin) * npieces
        j = s.astype(int).clip(0, npieces - 1)
        a = (j + 1) - s
        b = s - j
        x0 = a ** 3
        x1 = 1 + 3 * a * (1 + a * b)
        x2 = 1 + 3 * b * (1 + a * b)
        x3 = b ** 3
        wave = coeff[j] * x0 + coeff[j + 1] * x1 + coeff[j + 2] * x2 + coeff[j + 3] * x3

    elif ftype == 1 or ftype == 2:

        # chebyshev or legendre polynomial
        # legendre not tested yet

        if len(fields) < 15:
            raise ValueError('Bad polynomial format (only %d fields)' % len(fields))
        order = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            if ftype == 1:
                print 'Dispersion is order-%d Chebyshev polynomial' % order
            else:
                print 'Dispersion is order-%d Legendre polynomial (NEEDS TEST)' % order
        if len(fields) != 15 + order:
            # raise ValueError('Bad order-%d polynomial format (%d fields)' % (order, len(fields)))
            if verbose:
                print 'Bad order-%d polynomial format (%d fields)' % (order, len(fields))
                print "Changing order from %i to %i" % (order, len(fields) - 15)
            order = len(fields) - 15
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        pmiddle = (pmax + pmin) / 2
        prange = pmax - pmin
        x = (np.arange(nwave, dtype=float) + 1 - pmiddle) / (prange / 2)
        p0 = np.ones(nwave, dtype=float)
        p1 = x
        wave = p0 * coeff[0] + p1 * coeff[1]
        for i in range(2, order):
            if ftype == 1:
                # chebyshev
                p2 = 2 * x * p1 - p0
            else:
                # legendre
                p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
            wave = wave + p2 * coeff[i]
            p0 = p1
            p1 = p2

    else:
        raise ValueError('Cannot handle dispersion function of type %d' % ftype)

    return wave, fields

def apodiz_trap(wav,flux,percent):
	trap = np.ones(len(wav))
	corte = int(percent*len(trap)/100.)
	#trap[0:corte] = trap[0:corte] / trap[0:corte]
	trap[0:corte] = np.linspace(0,1,len(trap[0:corte]))
	trap[-corte:-1] = np.linspace(1,0,len(trap[-corte:-1]))
	trap[-1] = 0.

	return ((flux-1.)*trap) +1.

def readmultispec(fitsfile, reform=True, quiet=False):
    """Read IRAF echelle spectrum in multispec format from a FITS file
    
    Can read most multispec formats including linear, log, cubic spline,
    Chebyshev or Legendre dispersion spectra
    
    If reform is true, a single spectrum dimensioned 4,1,NWAVE is returned
    as 4,NWAVE (this is the default.)  If reform is false, it is returned as
    a 3-D array.
    """

    fh = pyfits.open(fitsfile)
    try:
        header = fh[0].header
        flux = fh[0].data
    finally:
        fh.close()
    temp = flux.shape
    nwave = temp[-1]
    if len(temp) == 1:
        nspec = 1
    else:
        nspec = temp[-2]

    # first try linear dispersion
    try:
        crval1 = header['crval1']
        crpix1 = header['crpix1']
        cd1_1 = header['cd1_1']
        ctype1 = header['ctype1']
        if ctype1.strip() == 'LINEAR':
            wavelen = np.zeros((nspec, nwave), dtype=float)
            ww = (np.arange(nwave, dtype=float) + 1 - crpix1) * cd1_1 + crval1
            for i in range(nspec):
                wavelen[i, :] = ww
            # handle log spacing too
            dcflag = header.get('dc-flag', 0)
            if dcflag == 1:
                wavelen = 10.0 ** wavelen
                if not quiet:
                    print 'Dispersion is linear in log wavelength'
            elif dcflag == 0:
                if not quiet:
                    print 'Dispersion is linear'
            else:
                raise ValueError('Dispersion not linear or log (DC-FLAG=%s)' % dcflag)

            if nspec == 1 and reform:
                # get rid of unity dimensions
                flux = np.squeeze(flux)
                wavelen.shape = (nwave,)
            return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': None}
    except KeyError:
        pass

    # get wavelength parameters from multispec keywords
    try:
        wat2 = header['wat2_*']
        count = len(wat2)
    except KeyError:
        raise ValueError('Cannot decipher header, need either WAT2_ or CRVAL keywords')

    # concatenate them all together into one big string
    watstr = []
    for i in range(len(wat2)):
        # hack to fix the fact that older pyfits versions (< 3.1)
        # strip trailing blanks from string values in an apparently
        # irrecoverable way
        # v = wat2[i].value
        v = wat2[i]
        v = v + (" " * (68 - len(v)))  # restore trailing blanks
        watstr.append(v)
    watstr = ''.join(watstr)

    # find all the spec#="..." strings
    specstr = [''] * nspec
    for i in range(nspec):
        sname = 'spec' + str(i + 1)
        p1 = watstr.find(sname)
        p2 = watstr.find('"', p1)
        p3 = watstr.find('"', p2 + 1)
        if p1 < 0 or p1 < 0 or p3 < 0:
            raise ValueError('Cannot find ' + sname + ' in WAT2_* keyword')
        specstr[i] = watstr[p2 + 1:p3]

    wparms = np.zeros((nspec, 9), dtype=float)
    w1 = np.zeros(9, dtype=float)
    for i in range(nspec):
        w1 = np.asarray(specstr[i].split(), dtype=float)
        wparms[i, :] = w1[:9]
        if w1[2] == -1:
            raise ValueError('Spectrum %d has no wavelength calibration (type=%d)' %
                             (i + 1, w1[2]))
            # elif w1[6] != 0:
            #    raise ValueError('Spectrum %d has non-zero redshift (z=%f)' % (i+1,w1[6]))

    wavelen = np.zeros((nspec, nwave), dtype=float)
    wavefields = [None] * nspec
    for i in range(nspec):
        # if i in skipped_orders:
        #    continue
        verbose = (not quiet) and (i == 0)
        if wparms[i, 2] == 0 or wparms[i, 2] == 1:
            # simple linear or log spacing
            wavelen[i, :] = np.arange(nwave, dtype=float) * wparms[i, 4] + wparms[i, 3]
            if wparms[i, 2] == 1:
                wavelen[i, :] = 10.0 ** wavelen[i, :]
                if verbose:
                    print 'Dispersion is linear in log wavelength'
            elif verbose:
                print 'Dispersion is linear'
        else:
            # non-linear wavelengths
            wavelen[i, :], wavefields[i] = nonlinearwave(nwave, specstr[i],
                                                         verbose=verbose)
        wavelen *= 1.0 + wparms[i, 6]
        if verbose:
            print "Correcting for redshift: z=%f" % wparms[i, 6]
    if nspec == 1 and reform:
        # get rid of unity dimensions
        flux = np.squeeze(flux)
        wavelen.shape = (nwave,)
    return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': wavefields}




def deconvolution_process(wav,flux,wav_syn,flux_syn,rango,skip):
	#rango = 100.
	num = (np.max(wav)-np.min(wav))/rango
	num = np.floor(num)
	print num
	x = []
	y = []
	for i in np.arange(num-skip):
		i = i +skip
		#print 'rango numero',i
		print 'rango numero:',i ,np.min(wav)+rango*(i),'to', np.min(wav)+rango*(1+i)
		#selecciono rango de 100 A
		mask = (wav > np.min(wav)+rango*(i) ) & (wav < np.min(wav)+rango*(1+i) )
		mask_syn = (wav_syn > np.min(wav)+rango*(i) ) & (wav_syn < np.min(wav)+rango*(1+i) )
		wav_aux = wav[mask]
		flux_aux = flux[mask]
		flux_syn_aux = flux_syn[mask_syn]
		wav_syn_aux = wav_syn[mask_syn]
		
		#reviso si hay lineas gruesas en el rango seleccionado, si hay, lo salto
		#if condicion_linea(wav_aux,) == True:
		#	continue
		#else:
		#	print 'en este rango no hay lineas gruesas'

		if np.sum(clean_lines(wav_aux,flux_aux,spectype)) < 50:
			print 'muy pocos puntos', i
			continue
		#corrijo con ajuste lineal, itero 5 para encontrar el mejor fit
		fitspec = fit_cleanlines_iter(wav_aux,flux_aux,1,5)
		fitsyn = fit_cleanlines_iter(wav_syn_aux,flux_syn_aux,1,5)

		dif = np.median(fitspec)-np.median(fitsyn)
		flux_aux = flux_aux-dif 

		#elimino lineas anchas
		#mask = clean_lines(wav_aux,flux_aux,spectype)
		#wav_aux = wav_aux[mask]
		#flux_aux = flux_aux[mask] 
		#mask = clean_lines(wav_syn_aux,flux_syn_aux,spectype)
		#flux_syn_aux = flux_syn_aux[mask] 
		#wav_syn_aux = wav_syn_aux[mask]
		wav_aux,flux_aux = ordenar(wav_aux,flux_aux)
		wav_syn_aux,flux_syn_aux = ordenar(wav_syn_aux,flux_syn_aux)
		#print 'nans',np.sum((clean_lines(wav_aux,flux_aux,spectype)))
		#plt.plot(wav_aux,flux_aux)
		#plt.show()

		flux_aux = apodiz_trap(wav_aux,flux_aux,percent=20.)
		flux_syn_aux = apodiz_trap(wav_syn_aux,flux_syn_aux,percent=20.)

		#xi,yi = rconvolution(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
		if deconv == 'vel':
			xi,yi = veldeconvolution2(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
		elif deconv == 'wav':
			xi,yi = wav_deconvolution(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
		else:
			print 'error --decov: solo puede ser vel o wav'
			raise

		if len(xi) == 1 :
			continue
		#elimino deconvoluciones muy outliers
		#### defino arreglo velocidad. llevo los 2 a velocidades con el lambda central. Resampleo velocidades a velocidades preestablecidas
		#### veo que longitudes de onda tienen que ser con esas velocidades a ese lambda central 
		#### dopler a longitud de onda central obtengo arreglo de longitudes de onda. 
		#ajustar gaussina y ver que fit tiene mayor dispersion
		#gausfit = ajuste_gauss(xi,yi)
		#lim = int(len(yi)/10.*3)
		#short = np.std(yi[lim:-lim]/gausfit[lim:-lim])
		#plt.plot(xi,gausfit,'ro')
		#plt.plot(xi[lim:-lim],yi[lim:-lim]/gausfit[lim:-lim])
		#plt.plot(xi[mask],yi[mask])
		#plt.show()
		#short = np.std(yi[(xi < 0.2) & (xi > -0.2)])

		#print 'std',short

		#if short > 0.2:
		#	continue
		else:
			x.append(xi)
			y.append(yi)
			#plt.plot(xi,yi)
			#plt.show()

		#plot muestra mean de deconvolucion con i=8 rangos
		'''if i == 18:
			x = np.array(x)
			y = np.array(y)
			aux = []
			cont = np.arange(len(x))
			for i in np.arange(len(x[0])):
				clip = sigma_clip(y[:,i], sigma=1.5)    #aplico sigma clip
				aux = np.append(aux,np.asarray(cont)[clip.mask])

			auxx = collections.Counter(aux)
			print 'aux',aux 
			print 'auxx',auxx
			print 'comunes',auxx.most_common(3)
			print 'aux',auxx.most_common(3)[0]
			print 'aux',auxx.most_common(3)[0][0]

			raise'''
		'''if i == 8:
			aux = []
			print ('empieza contar largos')
			for i in np.arange(len(y)):
				aux = np.append(aux,len(y[i]))
			print ('empieza maximo')

			for j in np.arange(len(y)):
				print int(np.min(aux))
				y[j] = y[j][0:int(np.min(aux))]
			ymean = np.mean(y,axis=0)
			xmean = x[0].copy()
			print ('empieza condiciones')

			if (len(x[0]) > len(ymean)):
				xmean = xmean[0:len(ymean)]
			elif (len(x[0]) < len(ymean)):
				ymean = ymean[0:len(x[0])]

			print('plot')
			plt.plot(xmean,ymean)
			plt.show()
		'''
		#print 'termino'
	#plt.show()
	print "termino deconvolucion"
	return x,y


def clean_deconvolved(x,y):


	##las deconvoluciones tienen distinto largo (dif de 2 o 3 elementos)
	##los dejo de la misma dimension
	aux = []
	for i in np.arange(len(y)):
		aux = np.append(aux,len(y[i]))
	for j in np.arange(len(y)):
		y[j] = y[j][0:int(np.min(aux))]
		x[j] = x[j][0:int(np.min(aux))]

	ymean = np.median(y,axis=0)
	xmean = x[0].copy()

	print "plot mean"
	#plt.plot(xmean,ymean,'bo')

	x = np.array(x)
	y = np.array(y)
	aux = []
	cont = np.arange(len(x))
	#mask = (x[0] < 50.) & (x[0] >-50)
	'''for i in np.arange(len(x[0][mask])):
		if 
		clip = sigma_clip(y[:,i], sigma=1.5)    #aplico sigma clip
		aux = np.append(aux,np.asarray(cont)[clip.mask])
	auxx = collections.Counter(aux)
	print 'aux',aux 
	print 'auxx',auxx
	print 'comunes',auxx.most_common(3)
	print 'aux',auxx.most_common(3)[0]
	print 'aux',auxx.most_common(3)[0][0]'''

	####limpio primera deconv 
	n = len(x[0])

	for i in np.arange(len(x[0][int(n/3):-int(n/3)])):
		clip = sigma_clip(y[:,i+int(n/3)], sigma=1.5)    #aplico sigma clip
		aux = np.append(aux,np.asarray(cont)[clip.mask])
	auxx = collections.Counter(aux)
	#print 'aux',aux 
	#print 'auxx',auxx
	#print 'comunes',auxx.most_common(3)
	#print 'aux',auxx.most_common(3)[0]
	#print 'aux',auxx.most_common(3)[0][0]

	n = 4
	mask =  [True for i in range(len(x))] 
	#print mask
	for i in np.arange(len(x)):
		for j in np.arange(len(auxx.most_common(n))):
			if int(auxx.most_common(n)[j][0]) == i:
				mask[i] = False
	mask = np.array(mask)
	cleanx = x[mask]
	cleany = y[mask]

	#for i in np.arange(len(cleanx)):
		#plt.plot(cleanx[i][10:-10],cleany[i][10:-10],alpha=.3)
	#	plt.plot(cleanx[i],cleany[i],alpha=.7)

	#for i in np.arange(len(x[~mask])):
		#plt.plot(x[~mask][i][10:-10],y[~mask][i][10:-10],'k')
	#	plt.plot(x[~mask][i],y[~mask][i],'k',alpha=.5)

	#plt.xlim(-80,80)
	#plt.show()

	ycleanmean = np.median(cleany,axis=0)
	xcleanmean = cleanx[0].copy()

	print "plot mean"
	plt.close()
	plt.plot(xcleanmean,ycleanmean,'k')
	#plt.xlim(-80,80)
	#plt.show()



	####aplico trapezoide
	trap = None
	if trap != None:
		trapezoid_1D_kernel = Trapezoid1DKernel(2.0, slope=0.3)
		ytrap = []
		ytraplin = []
		xtraplin = []

		for i in np.arange(len(x)):
			auxy = np.convolve(y[i],trapezoid_1D_kernel,mode='same')
			ytrap.append(auxy)

			auxx = np.linspace(np.min(x[0]),np.max(x[0]),len(x[0])*5)
			tck = interpolate.splrep(x[i],ytrap[i],k=3)
			auxy2 = interpolate.splev(auxx,tck)
			ytraplin.append(auxy2)
			xtraplin.append(auxx)
			#print ytrap[i]

			#Primera Deconv
			#plt.plot(x[i],y[i],'k',alpha=.3)
			#Primera Deconv + trap
			#plt.plot(x[i],ytrap[i])



		#limpio trap 
		xtraplin = np.array(xtraplin)
		ytraplin = np.array(ytraplin)

		n = len(xtraplin[0])
		#print 'n', n
		for i in np.arange(len(xtraplin[0][int(n/3):-int(n/3)])):
			clip = sigma_clip(ytraplin[:,i+int(n/3)], sigma=1.5)    #aplico sigma clip
			aux = np.append(aux,np.asarray(cont)[clip.mask])
		auxx = collections.Counter(aux)

		n = 4
		mask =  [True for i in range(len(xtraplin))] 
		#print mask
		for i in np.arange(len(xtraplin)):
			for j in np.arange(len(auxx.most_common(n))):
				if int(auxx.most_common(n)[j][0]) == i:
					mask[i] = False
		mask = np.array(mask)

		xtraplin_clean = xtraplin[mask]
		ytraplin_clean = ytraplin[mask]

		#for i in np.arange(len(xtraplin_clean)):
			#plt.plot(cleanx[i][10:-10],cleany[i][10:-10],alpha=.3)
		#	plt.plot(xtraplin_clean[i],ytraplin_clean[i],alpha=.7)

		#for i in np.arange(len(xtraplin[~mask])):
			#plt.plot(x[~mask][i][10:-10],y[~mask][i][10:-10],'k')
		#	plt.plot(x[~mask][i],y[~mask][i],'k',alpha=.5)
		#plt.show()




		ytraplinclean_mean = np.median(ytraplin_clean,axis=0)
		xtraplinclean_mean = xtraplin_clean[0].copy()

		print "plot mean"
		plt.plot(xtraplinclean_mean,ytraplinclean_mean)




		ytrapmean = np.median(ytrap,axis=0)
		xmean = cleanx[0].copy()

		print "plot mean"

		ytraplinmean = np.median(ytraplin,axis=0)
		xtraplinmean = xtraplin[0].copy()

		print "plot mean"
		plt.plot(xtraplinmean,ytraplinmean,'r',alpha=.4)
		
		#plt.plot(xcleanmean,ycleanmean,'b')
	plt.xlim(-130,130)
	plt.savefig(base+'%s_step%s_rango%s_velstep%s_velslim%s_deconv.png'%( spec.split('/')[-1].split('.')[0], int(linstep), int(rango),int(velstep), int(velslim) ))

	if plots != None:
		plt.show()
	else:
		plt.close()



	print ('hacer columnas')
	if trap != None:
		table = addcolsbyrow(xtraplinclean_mean,ytraplinclean_mean)
		x = xtraplinclean_mean
		y = ytraplinclean_mean
	else:
		table = addcolsbyrow(xcleanmean,ycleanmean)
		x = xcleanmean
		y = ycleanmean
	print ('escribir archivo')
	#table = addcolsbyrow(xtraplinclean_mean,ytraplinclean_mean)
	table.write(base+'%s_step%s_rango%s_velstep%s_velslim%s_deconv.txt'%( spec.split('/')[-1].split('.')[0], int(linstep), int(rango),int(velstep), int(velslim)), format='ascii')
	
	return x,y

def generate_rangos(skip, wav,flux,wav_syn,flux_syn,num):

	data = []
	for i in np.arange(num-skip):
		i = i +skip
		print 'rango numero ',i, ' entre ', np.min(wav)+rango*(i),'to', np.min(wav)+rango*(1+i)
		#selecciono rango de 100 A
		mask = (wav > np.min(wav)+rango*(i) ) & (wav < np.min(wav)+rango*(1+i) )
		mask_syn = (wav_syn > np.min(wav)+rango*(i) ) & (wav_syn < np.min(wav)+rango*(1+i) )
		wav_aux = wav[mask]
		flux_aux = flux[mask]
		flux_syn_aux = flux_syn[mask_syn]
		wav_syn_aux = wav_syn[mask_syn]
		

		#plt.show()
		#plt.plot(wav_aux,flux_aux)
		#plt.plot(wav_syn_aux,flux_syn_aux)
		#plt.show()
		if np.sum(clean_lines(wav_aux,flux_aux,spectype)) < 40:
			print 'muy pocos puntos', i
			continue
		#corrijo con ajuste lineal, itero 5 para encontrar el mejor fit
		#print np.sum(clean_lines(wav_aux,flux_aux,spectype))

		fitspec = fit_cleanlines_iter(wav_aux,flux_aux,1,5)
		fitsyn = fit_cleanlines_iter(wav_syn_aux,flux_syn_aux,1,5)

		dif = np.median(fitspec)-np.median(fitsyn)
		flux_aux = flux_aux-dif 

		#elimino lineas anchas
		wav_aux,flux_aux = clean_lines_deconv_ones(wav_aux,flux_aux,spectype,np.median(fitspec))
		wav_syn_aux,flux_syn_aux  = clean_lines_deconv_ones(wav_syn_aux,flux_syn_aux,spectype,np.median(fitsyn))
		
		#raise
		if 	len(wav_aux) == 0:
			continue	
		wav_aux,flux_aux = ordenar(wav_aux,flux_aux)
		wav_syn_aux,flux_syn_aux = ordenar(wav_syn_aux,flux_syn_aux)
		#print 'nans',np.sum((clean_lines(wav_aux,flux_aux,spectype)))

		if plots != None:

			plt.plot(wav_aux,flux_aux)
			plt.plot(wav_syn_aux,flux_syn_aux)
			plt.show()

		flux_apod = apodiz_trap(wav_aux,flux_aux,percent=20.)
		flux_syn_apod = apodiz_trap(wav_syn_aux,flux_syn_aux,percent=20.)
		aux = []
		aux = [wav_aux, flux_apod, wav_syn_aux, flux_syn_apod]
		data.append(aux)
		#if i == 5:
		#	print len(data)
		#	print len(data[0])
			#plt.plot(data[0][0], data[0][1])
			#plt.show()

	return data


def clean_deconv2(vels, deconv):

	x = vels
	y = deconv

	#
	aux = []
	for i in np.arange(len(y)):
		aux = np.append(aux,len(y[i]))
	for j in np.arange(len(y)):
		y[j] = y[j][0:int(np.min(aux))]
		x = x[0:int(np.min(aux))]

	ymean = np.mean(y,axis=0)
	ymedian = np.median(y,axis=0)
	xmean = x.copy()

	#print "plot mean"
	#plt.plot(xmean,ymedian,'r')
	#plt.plot(xmean,ymean,'b')
	#plt.show()

	rms1 = np.sqrt(np.mean(ymean**2/len(ymean)))
	rms2 = np.sqrt(np.mean(ymedian**2/len(ymedian)))

	print 'rms all mean', rms1
	print 'rms median', rms2
	#raise

	x = np.array(x)
	y = np.array(y)
	aux = []
	cont = np.arange(len(y))
	#mask = (x[0] < 50.) & (x[0] >-50)
	'''for i in np.arange(len(x[0][mask])):
		if 
		clip = sigma_clip(y[:,i], sigma=1.5)    #aplico sigma clip
		aux = np.append(aux,np.asarray(cont)[clip.mask])
	auxx = collections.Counter(aux)
	print 'aux',aux 
	print 'auxx',auxx
	print 'comunes',auxx.most_common(3)
	print 'aux',auxx.most_common(3)[0]
	print 'aux',auxx.most_common(3)[0][0]'''

	####limpio primera deconv 
	n = len(x)
	#print y 
	#raise
	#print len(x[int(n/4):-int(n/4)])
	#print len(y)
	#print len(x)
	#a = [[0,1][2,3][4,5][6,7][8,9]]
	#print a[:,2 ]
	#raise
	#print y[:,int(n/4)]
	#print len(y[:,int(n/4)])

	for i in np.arange(len(x[int(n/3):-int(n/3)])):
		#clip = sigma_clip(y[:,i+int(n/4)], sigma=1.5)    #aplico sigma clip
		clip = sigma_clip(y[:,i+int(n/3)], sigma=1.5)    #aplico sigma clip

		aux = np.append(aux,np.asarray(cont)[clip.mask])
	auxx = collections.Counter(aux)
	#print 'aux',aux 
	#print 'auxx',auxx
	#print 'comunes',auxx.most_common(3)
	#print 'aux',auxx.most_common(3)[0]
	#print 'aux',auxx.most_common(3)[0][0]

	n = 4
	mask =  [True for i in range(len(y))] 
	#print mask
	for i in np.arange(len(y)):
		for j in np.arange(len(auxx.most_common(n))):
			if int(auxx.most_common(n)[j][0]) == i:
				mask[i] = False

	mask = np.array(mask)
	cleanx = x
	cleany = y[mask]
	
	if plots != None:
		for i in np.arange(len(cleany)):
			#plt.plot(cleanx[i][10:-10],cleany[i][10:-10],alpha=.3)
			plt.plot(cleanx,cleany[i],alpha=.7)

		for i in np.arange(len(y[~mask])):
			#plt.plot(x[~mask][i][10:-10],y[~mask][i][10:-10],'k')
			plt.plot(x,y[~mask][i],'k',alpha=.5)

		#plt.xlim(-80,80)
		plt.show()
		plt.close()
	
	ycleanmedian = np.median(cleany,axis=0)
	ycleanmean = np.mean(cleany,axis=0)
	xcleanmean = cleanx.copy()


	xcleanmean = xcleanmean[5:-5]
	ycleanmean = ycleanmean[5:-5]
	ycleanmedian = ycleanmedian[5:-5]
	#ycleanmean_gauss = ajuste_gauss(xcleanmean,ycleanmean)
	#ycleanmedian_gauss = ajuste_gauss(xcleanmean,ycleanmedian)

	#residuos = ycleanmean/ycleanmean_gauss
	#residuos_median = ycleanmedian/ycleanmedian_gauss

	#rms1 = np.sqrt(np.mean(residuos**2/len(residuos)))
	#rms2 = np.sqrt(np.mean(residuos_median**2/len(residuos_median)))
	
	#print 'rms all mean', rms1
	#print 'rms median', rms2
	#print "plot mean"

	'''fig = plt.figure(constrained_layout=True,figsize=[22, 4])
				gs = gridspec.GridSpec(4, 1, figure=fig)
				ax = fig.add_subplot(gs[:3, 0])
				ax.plot(xcleanmean,ycleanmean, '-k', lw=1, zorder=-2)
				ax.plot(xcleanmean,ycleanmean_gauss, '-r', lw=1, zorder=-2,alpha=.4)
			
				ax.set_xlabel("vels")
				ax.set_ylabel("broadening kernel")
			
				plt.title("broadening kernel - rms" + str(rms1))
				#plt.show()
				ax2 = fig.add_subplot(gs[3, 0])
				ax2.plot(xcleanmean[55:-55],ycleanmean[55:-55]/ycleanmean_gauss[55:-55],'-ko', lw=.5)
				#ax2.set_xlim([-40,40])
				ax.set_xlabel("wav")
				ax.set_ylabel("flux")
			
				plt.title('spectra' )
			
			
				#ax3 = fig.add_subplot(gs[1, 1])
				#ax3.scatter(ph, f, c='gold', edgecolor='black', s=15, lw=.5)
				#period = period*2.
				#ph = (t - t0 + 0.5*period) % period - 0.5*period
				#ax3.scatter(ph, f, s=15, lw=.5)
			
			
				#fig.suptitle('%s'%name, fontsize=16)
				#plt.title('phased light curve' )
				plt.savefig(name_deconv.split('.dat')[0] + '_meanrms'+str(rms1) + '.png')
				plt.close()'''
	#plt.savefig('%s.png'% args.TIC)
	#plt.savefig(new_base+image+'_fit/'+image+'_test_phased.png')
	#plt.show()


	##PLOT OLD
	#if plots != None:
	plt.plot(xcleanmean,ycleanmedian,'r')
	plt.plot(xcleanmean,ycleanmean,'b')
	plt.xlim(-velslim,velslim)
	#plt.savefig(name_deconv.split('.dat')[0] + '_meanrms'+str(rms1) + '.png')
	plt.savefig(name_deconv.split('.dat')[0]  + '.png')

	plt.close()
	#plt.show()
	#raise
	return xcleanmean,ycleanmean



def apodiz_trap(wav,flux,percent):
	trap = np.ones(len(wav))
	corte = int(percent*len(trap)/100.)
	#trap[0:corte] = trap[0:corte] / trap[0:corte]
	trap[0:corte] = np.linspace(0,1,len(trap[0:corte]))
	trap[-corte:-1] = np.linspace(1,0,len(trap[-corte:-1]))
	trap[-1] = 0.
	#for i in np.arange(len(trap[0:corte])):

		#hacer funcion para que en extremo divida por si misimo y dsp vaya a dividir por uno
	#mask = trap < 1
	#print len(trap[mask])
	#plt.plot(np.arange(len(trap)),trap)
	#plt.show()
	#plt.plot(wav,flux-1.,'k',alpha=.6)
	#plt.plot(wav,(flux-1.)*trap,'r')
	#plt.show()
	return ((flux-1.)*trap) +1.

def sort_files(RVs):
    from datetime import datetime
    # year, month, day, hour=0, minute=0, second=0, microsecond=0

    dates = list(map(lambda x: "-".join(x.split('.')[0].split('_')[2:]), RVs))
    #print dates 
    dates = list(map(lambda x: datetime(*np.int_(x.split("-"))), dates))
    #print(dates)
    zp = list(zip(dates, RVs))
    zp.sort()
    _, RVs = zip(*zp)
    return RVs

def sort_files_ceres(RVs):
	from datetime import datetime
	# year, month, day, hour=0, minute=0, second=0, microsecond=0
	dates = list(map(lambda x: "-".join(x.split('.fits')[0].split('_')[1:]), RVs))
	dates = list(map(lambda x: "-".join(x.split('-UT')), dates))
 
	#dates = list(map(lambda x: datetime(*np.int_(x.split("-"))), dates))
	#print(dates)

	zp = list(zip(dates, RVs))
	zp.sort()
	_, RVs = zip(*zp)
	return RVs

def list_files(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
       	if os.path.isfile(os.path.join(path, name)):
			if len(name.split('fits')) > 1:
				files.append(name)
    return files 


#ORDENA LAS FECHAS
#specs = list(filter(lambda x: ".fits" in x and os.path.isfile(x), os.listdir(folder)))
#print 'ORDEN', specs.sort(key=lambda spectra: spectra[0], reverse=True)

def leer_deconv(table):
	kernels = []
	for j in np.arange(len(table)):
		if j != 0:
			#print 'se cumple'
			kernels.append(np.array(aux))
		aux = []
		for i in np.arange(len(table[j])):
			#print table[1][i]
			aux.append(table[j][i])
	return kernels

parser = argparse.ArgumentParser(description='Linealizacion Merge y deconvolucion')
parser.add_argument('folder', help='directorio/')
parser.add_argument('--vel', type=float, default=-17.7710, help='velocidad orbita')
parser.add_argument('--rango', type=float, default=100.0, help='rango deconvolucionado')
parser.add_argument('--template', type=str, default='ap00t6250g50k0odfnew_sample0.005.out', help='nombre template')
#parser.add_argument('--bkg', type=str, default=None, help='Para estimar bkg, ingresar Yes')
parser.add_argument('--linstep', type=float, default=50, help='step para linealizacion')
parser.add_argument('--mode', type=str, default='spline', help='modo de linealizacion spline y lineal')
parser.add_argument('--spectype', type=str, default='F', help='Tipo espectral F o A')
parser.add_argument('--deconv', type=str, default='vel', help='tipo de deconvolucion, en wav o vel')
parser.add_argument('--instrument', type=str, default='feros', help='espectrografo utilizado')
parser.add_argument('--skip', type=int, default=1, help='skip orders')
parser.add_argument('--plots', type=str, default=None, help='hacer plots o no')
parser.add_argument('--ncpu', type=int, default=8, help='multiprocesing')
parser.add_argument('--velslim', type=float, default=100., help='limite de velocidades')
parser.add_argument('--velstep', type=float, default=200., help='datos totales de velocidades')
parser.add_argument('--combined', type=int, default=1, help='Cantidad de espectros a combinar. Utilice numeros enteros')

args = parser.parse_args()
folder  = args.folder
vel = args.vel
rango = args.rango
template = args.template
linstep = args.linstep
mode = args.mode

spectype = args.spectype
deconv = args.deconv
instrument = args.instrument
skip = args.skip
plots = args.plots
velslim = args.velslim
velstep = args.velstep
combined = args.combined

print 'Analizando los espectros del directorio', folder


###PRUEBA MATRIZZ
'''from scipy.linalg import solve_toeplitz, toeplitz

c = np.array([1,0,0,0,0,0,0,0])
r = np.array([1,2,3,4,5,6,7,8,9,10,0,0,0,0,0,0,0,0,0])
b = np.array([2,4,6,8,10,1,2,3])

obs = np.array([1,3,2,1,0,2,7,1,0,1,3,4,7,4,0,1,12,7,2])
syn = np.array([1,1,0,0,0,1,0,0,0,0,0,1,0,0,1])
nn = 5

#padding = np.zeros(b.shape[0] - 1, b.dtype)
padding = np.zeros(nn - 1, obs.dtype)

first_col = np.r_[syn, padding]
first_row = np.r_[syn[0], padding]

H = scipy.linalg.toeplitz(first_col, first_row)

print(repr(H))
print obs[2:-2]
o = obs[int(nn/2):-int(nn/2)]
print o
A,B,C,D = np.linalg.lstsq(H, obs , rcond=None)

print H.transpose()
print np.transpose(H)
m = np.matmul(H.transpose(),H)
p = np.matmul(np.linalg.inv(m), H.transpose())
print p
raise 
print np.linalg.inv(H)
raise'''
#H1 = toeplitz(syn)
#print(repr(H1))
#raise
#A1,B,C,D = np.linalg.lstsq(H1, o , rcond=None)

#plt.plot(np.arange(len(A)),A)
#plt.plot(np.arange(len(A1)),A1,'r')
#plt.show()

'''specs = list_files(folder)
olo = []
for i in np.arange(10):
	olo  = []
	for k in np.arange(8):
		olo.append(k)
	if i == 0 :
		mat = olo
	else:
		mat = np.vstack((mat,olo))
'''

specs = list_files(folder)
specs = sort_files_ceres(specs)

print specs

#for spec in specs:
#	hdulist = pyfits.open(folder+spec)

#	print hdulist[0].header['TEXP (s)']
#raise

if combined != None:

	names_deconv = []
	if int(combined) == 1:
		base = folder+'results_deconvolution/'
		if not os.path.exists(base):
			os.makedirs(base)
	else:
		base = folder+'results_deconvolution_combined'+str(combined)+'/'
		if not os.path.exists(base):
			os.makedirs(base)

	n = int(len(specs)/int(combined))

	for i in np.arange(n):
		names_deconv = []
		wavs = []
		fluxs = []
		name_info = '_rango'+ str(rango)+ '_linstep'+str(linstep) + '_velslim' + str(velslim)+'_velstep' + str(velstep)+'_deconv.dat'

		for j in np.arange(int(combined)):

			spec = specs[i*int(combined)+j]
			
			if spec.split('_sp.')[-1] == 'fits':
				name_deconv = spec.split('_sp.fits')[0]
			else:
				name_deconv = spec.split('.fits')[0]

			if j == 0:
				names_deconv.append(name_deconv)
			if j != 0:
				name_deconv = '_'.join(name_deconv.split('_')[1:])
				names_deconv.append(name_deconv)	

				if j == (int(combined)-1): 
					#RUTA Y NOMBRE DECONV
					for k in np.arange(len(names_deconv)):
						if k == 0:
							name_deconv = base+names_deconv[k]
						else:
							name_deconv = name_deconv +'+'+names_deconv[k]
						if k == (len(names_deconv)-1):
							name_deconv = name_deconv + name_info
			
			if int(combined) == 1:
				name_deconv = base + name_deconv + name_info

			#@@@@@ DESDE ACA @@@

			## Leer espectro
			spec = folder + spec

			if instrument == 'ceres' or 'feros':
				hdulist = pyfits.open(spec)
				hdulist.info()
				im = pyfits.getdata(spec)

				im[0,:,:] = shift_doppler(im[0,:,:],vel)
				#x = im [0,:,:]
				#y = im [1,:,:]

				lineal_x, lineal_y = linealizacion(im,linstep,mode,instrument)

			elif instrument == 'chiron':
				hdulist = pyfits.open(spec)
				hdulist.info()
				for a in hdulist[0].header:
					print a 
					print hdulist[0].header[a]

				im = pyfits.getdata(spec)
				
				print 'Tiempo de exp', hdulist[0].header['EXPTIME']
				#for i in np.arange(len(im[:,0,0])):
				#	plt.plot(im[i,:,0],im[i,:,1])
					#plt.plot(im[i,:,1])
				#plt.show()
				#raise

				im[:,:,0] = shift_doppler(im[:,:,0],vel)
				lineal_x, lineal_y = linealizacion_chiron(im,linstep,mode,instrument)



			elif instrument == 'tres':
				spectra_list = read_fits.read_fits_spectrum1d(spec)

				#spectra_list = readmultispec(spec)
				#wav = spectra_list['wavelen']
				#flux = spectra_list['flux']
				#wav = wav +15.
				#wav = shift_doppler(wav,vel)
				lineal_x, lineal_y = linealizacion_tres(spectra_list,linstep,mode)
				#lineal_x = spectra_list['wavelen']
				#lineal_y = spectra_list['flux']
			else:
				print 'corregir variable instrument'
				raise

			#leo template
			if template == 't06000_g+2.5_m05p00_hr.fits':
				hmod = pyfits.getheader(template)
				smod = pyfits.getdata(template)[0]
				flux_syn = pyfits.getdata(template)[0]
				wav_syn = np.arange(len(smod))*hmod['CDELT1']+hmod['CRVAL1']
				wav_syn = ToVacuum(wav_syn)

			elif template[:4] == 'ap00':

				data = np.loadtxt(template,usecols=[0,1],unpack=True)
				wav_syn = ToVacuum(data[0])
				flux_syn = data[1]


			else:
				print 'por favor corregir nombre template'
				raise


			print '-> Comenzo merge ... '
			wav,flux = merge(lineal_x,lineal_y)
			#ordeno y limpio clean_outliers
			wav,flux = ordenar(wav,flux)
			print '-> Termina merge ...'
			wavs.append(wav)
			fluxs.append(flux)

		if int(combined) == 1:
			fluxmean = fluxs[0]
		else:
			fluxmean = np.mean(fluxs, axis=0)

		#plt.plot(wav, fluxmean)
		#plt.plot(wavs[0],fluxs[0], 'k', alpha=.6)
		#plt.show()

		#hdulist = pyfits.open(spec)
		#hdulist.info()
		#print hdulist[0].header
		#for a in hdulist[0].header:
		#	print a 
		#print hdulist[0].header['TEXP (s)']
		#raise
			#print('Analizando: '+ str(spec))
			#print ('name_deconv',name_deconv)
	
		if os.path.exists(name_deconv):
			#print 'aqi '
			table = Table.read(name_deconv,format='ascii')
			kernels = leer_deconv(table)
			vels = np.linspace(-velslim,velslim,velstep)
			if plots != None:
				for i in np.arange(len(kernels)):
					plt.plot(vels,kernels[i],'k',alpha=.6)
				plt.show()
			
			xclean, yclean = clean_deconv2(vels, kernels)
			if plots != None:
				plt.plot(xclean,yclean)
				plt.show()
			x0 = 4.
			sigma = 0.3
			a = 1.
			ygauss = gauss(xclean,a,x0,sigma)
			#plt.plot(xclean,ygauss)
			#plt.plot(xclean,yclean)
			#plt.show()

			continue
		else:

			num = (np.max(wav)-np.min(wav))/rango
			num = np.floor(num)
			print num
			data = generate_rangos(skip, wav,fluxmean,wav_syn,flux_syn,num)
			print len(data)

			if deconv == 'vel':
				#deconv = Table(rows=Parallel(n_jobs=args.ncpu, verbose=0)(delayed(deconv_multiproc)(f) for f in tqdm(data))) 
				deconvolution = Table(rows=Parallel(n_jobs=args.ncpu, verbose=0)(delayed(deconv_multiproc_newmatrix)(f) for f in tqdm(data))) 

			elif deconv == 'wav':
				x,y = Table(rows=Parallel(n_jobs=args.ncpu, verbose=0)(delayed(wav_deconvolution_newmatrix2)(f) for f in tqdm(data)))
			else:
				print 'error --decov: solo puede ser vel o wav'
				raise

			print 'termino deconvolucion'

			print len(deconvolution)
			print deconvolution
			print name_deconv
			
			deconvolution.write(name_deconv,format='ascii')
			table = Table.read(name_deconv,format='ascii')
			kernels = leer_deconv(table)
			vels = np.linspace(-velslim,velslim,velstep)
			#for i in np.arange(len(kernels)):
			#	plt.plot(vels,kernels[i])
			#plt.show()


			xclean, yclean = clean_deconv2(vels, kernels)
	raise

raise

