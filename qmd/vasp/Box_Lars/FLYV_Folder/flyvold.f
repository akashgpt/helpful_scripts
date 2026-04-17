	parameter (iflag=1)
	parameter (nmax=100000)
	real x(nmax)
	n = 0
	read(*,*) ratio
	do 1 i=1,nmax
	 read(*,*,end=10) x(i)
	 n = n + 1
1	continue
10	continue
        ibeg = n/ratio + 1
        nseg = n - ibeg + 1
c	ibeg = 1
c	nseg = n

        call flyv(x(ibeg),nseg,xm,sig,iflag)

	print*, xm,sig,nseg

	stop
	end
C****************************************
	subroutine flyv(x,n,xm,sig,iflag)

C  Flyvbjerg and Petersen, J. Chem. Phys., 91, 461, 1989 

c	real x(1000000),xp(1000000)
	real x(n),xp(n)
	fn = float(n)

	xm = 0.
	x2m = 0.
	do 11 i=1,n
	 xm = xm + x(i)
	 x2m = x2m + x(i)*x(i)
         print*, i,x(i),xm,x2m
11	continue
	xm = xm/float(n)
	x2m = x2m/float(n)
	if (iflag .eq. 0) return
	if (n .lt. 2) then
	 sig = 0.
	 return
	end if

	nrbin = log(fn)/log(2.) - 1
	print*, 'Number of re-binnings = ',nrbin,xm,n
	ci = (x2m - xm*xm)/float(n-1)
	faci = 1./sqrt(2.*float(n-1))
	dc = faci*ci
	irb = 0
	np = n

	cmax = -1.
	cold = 0.
	print 100, irb,np,ci,sqrt(ci),dc
	cmax = ci
	do 3 i=1,n
3	xp(i) = x(i)
	np = fn
	do 1 irb=1,nrbin
	 c = 0.
	 np = np/2
	 do 2 i=1,np
	  if (np .eq. 1) go to 2
	  fac = 1./sqrt(2.*float(np-1))
	  xp(i) = (xp(2*i) + xp(2*i-1))/2
	  c = c + (xp(i) - xm)*(xp(i) - xm)/(float(np)*float(np-1))
	  if (np .le. 4) print*, irb,np,xp(i),c
2	 continue
	 dc = fac*sqrt(c)
	 diff = sqrt(c) - sqrt(cold)
	 if (abs(diff) .lt. dc) then
	  print 100, irb,np,c,sqrt(c),dc,'*'
	  if (c .gt. cmax) then
	   cmax = c
	   dcmax = dc
	  end if
	 else
	  print 100, irb,np,c,sqrt(c),dc
	 end if
	 cold = c
1	continue
	sig = sqrt(cmax)
	print 200, sqrt(cmax),dcmax

	return
100	format(2i10,3f15.6,a5)
200	format(3f15.6,a5)
	end
