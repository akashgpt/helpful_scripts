PROGRAM flyv_master

IMPLICIT NONE 

	! parameter (iflag=1)
	INTEGER, PARAMETER	::	sp = selected_real_kind(6, 37)
	INTEGER, PARAMETER	::	dp = selected_real_kind(15, 307)
	INTEGER, PARAMETER	::	qp = selected_real_kind(33, 4931)
	INTEGER, PARAMETER	::	rvp = dp
	INTEGER, PARAMETER	::	ivp = dp
	INTEGER(KIND=ivp), PARAMETER		::	nmax = 100000
	REAL(KIND=rvp), DIMENSION(nmax)	:: x
	INTEGER(KIND=ivp)	:: read_err, n, ibeg, nseg
	REAL(KIND=rvp)	::	xm, sig

	n = 0
	read_err = 0
	DO WHILE (read_err == 0 .AND. n<=nmax)
		n = n + 1
		READ(*,*,IOSTAT=read_err) x(n)
	ENDDO

	ibeg = 1
	nseg = n-1

	CALL flyv(x(ibeg),nseg,xm,sig)

	print*, xm,sig,nseg

CONTAINS
! C****************************************
	SUBROUTINE flyv(x,n,xm,sig)

	! C  Flyvbjerg and Petersen, J. Chem. Phys., 91, 461, 1989 
	IMPLICIT NONE
	! c	real x(1000000),xp(1000000)
		INTEGER(KIND=ivp)	::	n
		REAL(KIND=rvp), DIMENSION(n)	:: x, xp
		REAL(KIND=rvp)	::	fn, xm, x2m, sig, nrbin
		REAL(KIND=rvp)	::	ci, faci, dc, cmax, cold, fac, c, diff, dcmax
		INTEGER(KIND=ivp)	::	irb, np, i

		fn = float(n)

		xm = 0.
		x2m = 0.
		DO i=1,n
			xm = xm + x(i)
			x2m = x2m + x(i)*x(i)
	! c	 print*, i,x(i),xm,x2m
		ENDDO

		xm = xm/float(n)
		x2m = x2m/float(n)
		! if (iflag .eq. 0) return
		IF (n .lt. 2) THEN
			sig = 0.
		ELSE
			nrbin = log(fn)/log(2.) - 1
		! c	print*, 'Number of re-binnings = ',nrbin,xm,n
			ci = (x2m - xm*xm)/float(n-1)
			faci = 1./sqrt(2.*float(n-1))
			dc = faci*ci
			irb = 0
			np = n

			cmax = -1.
			cold = 0.
		! c	print 100, irb,np,ci,sqrt(ci),dc
			cmax = ci

			xp = x

			np = fn

			DO irb=1,INT(nrbin)
				c = 0.
				np = np/2

				! i=1
				! DO WHILE (np /= 1 .AND. i<=np)
				! 	i=i+1
				! 	fac = 1./sqrt(2.*float(np-1))
				! 	xp(i) = (xp(2*i) + xp(2*i-1))/2
				! 	c = c + (xp(i) - xm)*(xp(i) - xm)/(float(np)*float(np-1))
				! ! c	  if (np .le. 4) print*, irb,np,xp(i),c
				! ENDDO

				IF (np == 1) THEN
				ELSE
					DO i = 1, np
						fac = 1./sqrt(2.*float(np-1))
						xp(i) = (xp(2*i) + xp(2*i-1))/2
						c = c + (xp(i) - xm)*(xp(i) - xm)/(float(np)*float(np-1))
					! c	  if (np .le. 4) print*, irb,np,xp(i),c
					ENDDO
				ENDIF

				dc = fac*sqrt(c)
				diff = sqrt(c) - sqrt(cold)
				if (abs(diff) .lt. dc) then
			! c	  print 100, irb,np,c,sqrt(c),dc,'*'
					if (c .gt. cmax) then
					cmax = c
					dcmax = dc
					end if
				else
			! ! c	  print 100, irb,np,c,sqrt(c),dc
				end if
				cold = c
			ENDDO
			sig = sqrt(cmax)
		! c	print 200, sqrt(cmax),dcmax

			! return
		! 100	format(2i10,3f15.6,a5)
		! 200	format(3f15.6,a5)
		ENDIF

	END SUBROUTINE flyv

END PROGRAM flyv_master
