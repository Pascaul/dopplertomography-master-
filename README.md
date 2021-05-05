# dopplertomography-master-
doppler tomography code 

Examples:

Para espectros de CHIRON
> python deconvolution.py ~/dopplertomography-master/TOI-812/CHIRON/ --deconv velocidad  --rango 200 --linstep 80  --template ap00t6750g40k0odfnew_sample0.005.out --instrument chiron --skip 0 --velslim 200 --velstep 251 --combined 1 --spectype F --ncpu 7 --addzeros 0  --vel -75 --mode spline

Para espectros de FEROS
>  python deconvolution.py ~/dopplertomography-master/TOI-812/FEROS/  --deconv velocidad  --rango 200 --linstep 60  --template ap00t6750g40k0odfnew_sample0.005.out --instrument feros --skip 0 --velslim 200 --velstep 251 --combined 1 --spectype F --ncpu 7 --addzeros 0  --vel -75 --mode spline

Para espectros de FEROS combinados
>  python deconvolution.py ~/dopplertomography-master/TOI-812/FEROS/  --deconv velocidad  --rango 200 --linstep 60  --template ap00t6750g40k0odfnew_sample0.005.out --instrument feros --skip 0 --velslim 200 --velstep 251 --combined 2 --spectype F --ncpu 7 --addzeros 0  --vel -75 --mode spline


