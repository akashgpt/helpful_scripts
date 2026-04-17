#!/bin/bash
# Purpose:  To compute time-average values of thermodynamic quantites from NVT MD simulations using VASP.
# Quantities computed include internal energy, enthalpy, electronic entropy, pressure, heat capacity, thermal pressure coefficient, and Gruneisen parameter.
# Author: Lars Stixrude, Akash Gupta

eVK=11605 	# Convert electron volts to Kelvin
R=8.3143  	# The gas constant
bP=10     	# GPa to kbar
AV=0.6022 	# A^3 per atom to cm^3 per mol
eVJ=96.4869 	# eV to kJ per mol
eVG=160.21863	# GPa A^3 to eV
MG=1000.	# GPa to MPa

# Check to see that an input file name has been specified
# usage
if [ -z "$1" ]
then
    echo "Usage: $(basename $0) [OUTCAR name]"
    exit 1;
fi

# Optionally readin file: ratio
# which includes a value of the ratio that may differ from the default value of 5.
# ratio = Total number of steps / Number of transient steps.  i.e. ratio=5 assumes that the first 20 percent of the time series is transient.
ratio=4 #5
ratiofile="ratio"
if [ -f "$ratiofile" ]; then
    ratio=$(awk '{print $1}' ratio)
    echo "File ratio exists. Using specified value of ratio =" ${ratio}
else
    echo "File ratio does not exist. Using default value of ratio =" ${ratio}
fi

# Optionally readin file: temperature
# which includes a value of the temperature that may differ from that in the OUTCAR file 
# useful e.g. in case the temperature is so large that it overflows the format in the OUTCAR file: T>=100,000 K).
# Default value of temperature read from the OUTCAR file
# T=$(grep -a -m 1 "TEBEG  =" $1 | cut -c 1-18 | awk '{print $3}')
T=$(grep -a -m 1 "TEBEG =" $1 | awk '{print $3}')
temperaturefile="temperature"
if [ -f "$temperaturefile" ]; then
    T=$(awk '{print $1}' temperature)
    echo "File temperature exists. Using specified value of temperature =" $T
else
    echo "File temperature does not exist. Using OUTCAR value of temperature =" $T
fi

# Read N, V from OUTCAR
N=$(grep -a -m 1 "NIONS =" $1 | awk '{print $12}')
V=$(grep -a -m 1 "volume of cell" $1 | awk '{print $5}')

# Read atomic masses from OUTCAR and calculate mass density
awk '{print} /Ionic Valenz/ {exit}' $1 | grep -a -A1 "Mass of Ions in am" | grep -a -v Mass | fmt -1 | tail -n +3 > masses
awk '{print} /ions per type/ {exit}' $1 | grep -a "ions per type" | fmt -1 | tail -n +5 > numbers
rhom=$(paste masses numbers | awk -F' ' '{sum+=$1*$2;} END{print sum;}')
rhom=$(echo "scale=16;$rhom/$V/$AV" | bc)

# Following line is to speed up the grepping!
# sed -n '/Total+kin/, /total energy   ETOTAL/p' $1 > OUTCAR_part
sed -n '/external pressure/, /total energy   ETOTAL/p' $1 > OUTCAR_part

# Read energy from OUTCAR
grep -a "energy  without entropy=" OUTCAR_part | awk '{print $4,$4*$4}' > xelist1

# Read temperature from OUTCAR: format requires that this must be less than 100,000 K
grep -a "(temperature" OUTCAR_part | cut -c 57-64 | awk '{print $1,$1*$1}' > xtlist1

# Read pressure from OUTCAR
grep -a "Total+kin" OUTCAR_part | awk '{print ($2+$3+$4)/3.,($2+$3+$4)*($2+$3+$4)/9.}' > xplist1

# Read free energy from OUTCAR
grep -a "free  energy" OUTCAR_part | awk '{print $5,$5*$5}' > xflist1

# Read ETOTAL (conserved quantity) from OUTCAR
grep -a "total energy   ETOTAL" OUTCAR_part | awk '{print $5,$5*$5}' > xqlist1

# Read all Volume of cells from OUTCAR
grep -a "volume of cell :" OUTCAR_part | awk '{print $5,$5*$5}' > xvlist1
# delete first two lines
# sed -i '1,2d' xvlist1 # Not required given how OUTCAR_part is created

# Read external pressure ++ from OUTCAR
grep -a "external" OUTCAR_part | awk '{print $4,$4*$4}' > xeplist1
grep -a "kinetic pressure" OUTCAR_part | awk '{print $7,$7*$7}' > xkplist1
grep -a "Pullay stress" OUTCAR_part | awk '{print $9,$9*$9}' > xpslist1

# Read F (for free energy calculation) from OSZICAR -- for TI calculations (same as free energy from OUTCAR otherwise)
# or read even line from xflist1 into xflist2
# grep "F=" OSZICAR | awk '{print $7, $7*$7}' > xflist2

# if (grep "SCALED FREE ENERGIE" OUTCAR | wc -l) == 0.5 * (grep "free  energy" OUTCAR | wc -l), then read even lines from xflist1, ... into xflist2, ...
# Count the number of lines matching the two patterns
scaled_count=$(grep "SCALED FREE ENERGIE" OUTCAR | wc -l)
free_count=$(grep "free  energy" OUTCAR | wc -l)
half_free=$(echo "0.5 * $free_count" | bc)
# make half_free an integer
half_free=${half_free%.*}

# Define TI_mode: 1 if scaled_count equals half_free or half_free +/- 1 (in case issue with incomplete output in OUTCAR), 0 otherwise.
if [ "$scaled_count" -eq "$half_free" ] || [ "$scaled_count" -eq $((half_free + 1)) ] || [ "$scaled_count" -eq $((half_free - 1)) ]; then
    TI_mode=1
    # echo "TI_mode switched on."
else
    TI_mode=0
fi


echo "TI_mode is (peavg.sh): $TI_mode" #; scaled_count is: $scaled_count, free_count is: $free_count, half_free is: $half_free"
# print number of lines in xflist1, xqlist1, xelist1, xtlist1, xplist1
# wc -l xflist1 xqlist1 xelist1 xtlist1 xplist1

# If TI_mode is true (i.e. equals 1), then process the files.
if [ "$TI_mode" -eq 1 ]; then
    echo "Selecting odd instances of energies (where relevant) to not select SCALED values which are simply, SCALEE * <unscaled energy>"
    awk 'NR%2==1' xelist1 > xelist2 # only choosing odd lines
    # awk 'NR%2==0' xtlist1 > xtlist2
    # awk 'NR%2==0' xplist1 > xplist2
    awk 'NR%2==1' xflist1 > xflist2
    # awk 'NR%2==0' xqlist1 > xqlist2
fi


# xlist1 contains 21 columns: E, E^2, P, P^2, E*P, F, F^2, E-F, (E-F)^2, T, T^2, E_conserved, E_conserved^2, Vcell, Vcell^2, ExP, ExP^2, KP, KP^2, PS, PS^2
if [ "$TI_mode" -eq 1 ]; then
    paste xelist2 xplist1 xflist2 xtlist1 xqlist1 xvlist1 xeplist1 xkplist1 xpslist1 | awk '{print $1,$2,$3,$4,$1*$3,$5,$6,($1-$5),($1-$5)*($1-$5),$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18}' > xlist1
else
    paste xelist1 xplist1 xflist1 xtlist1 xqlist1 xvlist1 xeplist1 xkplist1 xpslist1 | awk '{print $1,$2,$3,$4,$1*$3,$5,$6,($1-$5),($1-$5)*($1-$5),$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18}' > xlist1
fi

nlines=$(awk -F, 'END{print NR}' xlist1)

if [ "$TI_mode" -eq 1 ]; then
    nelines=$(awk -F, 'END{print NR}' xelist2)
    nplines=$(awk -F, 'END{print NR}' xplist1)
    nflines=$(awk -F, 'END{print NR}' xflist2)
    neplines=$(awk -F, 'END{print NR}' xeplist1)
    nkplines=$(awk -F, 'END{print NR}' xkplist1)
    npslines=$(awk -F, 'END{print NR}' xpslist1)
    nvlines=$(awk -F, 'END{print NR}' xvlist1)
else
    nelines=$(awk -F, 'END{print NR}' xelist1)
    nplines=$(awk -F, 'END{print NR}' xplist1)
    nflines=$(awk -F, 'END{print NR}' xflist1)
    neplines=$(awk -F, 'END{print NR}' xeplist1)
    nkplines=$(awk -F, 'END{print NR}' xkplist1)
    npslines=$(awk -F, 'END{print NR}' xpslist1)
    nvlines=$(awk -F, 'END{print NR}' xvlist1)
fi

if [ $nelines -ne $nplines ]; then
    echo "WARNING: number of lines in energy and pressure are not equal" $nelines $nplines
fi
if [ $nelines -ne $nflines ]; then
    echo "WARNING: number of lines in energy and free energy are not equal" $nelines $nplines
fi
if [ $nplines -ne $nflines ]; then
    echo "WARNING: number of lines in free energy and pressure are not equal" $nelines $nplines
fi

# similarly for nplines, neplines, nkplines, npslines
if [ $nplines -ne $neplines ]; then
    echo "WARNING: number of lines in pressure and external pressure are not equal" $nplines $neplines
fi
if [ $nplines -ne $nkplines ]; then
    echo "WARNING: number of lines in pressure and kinetic pressure are not equal" $nplines $nkplines
fi
if [ $nplines -ne $npslines ]; then
    echo "WARNING: number of lines in pressure and pullay stress are not equal" $nplines $npslines
fi

if [ $nplines -ne $nvlines ]; then
    echo "WARNING: number of lines in pressure and volume are not equal" $nplines $nvlines
fi


# start=$[$nlines/$ratio+1]
start=$(awk -v n="$nlines" -v r="$ratio" 'BEGIN { printf "%d", (n / r) + 1 }')

# Optionally read file: chunkdata.txt which includes:
# first time step, last time step, time step interval
# Overwrite start and end time steps if file chunkdata.txt exists
chunkfile="chunkdata.txt"
if [ -f "$chunkfile" ]; then
    start=$(awk '{print $1}' chunkdata.txt)
    nlines=$(awk '{print $2}' chunkdata.txt)
    echo "File chunkdata.txt exists. Using specified values for start and end =" $[start],$[nlines]
else
    echo "File chunkdata.txt does not exist. Using default values of start and end =" $[start],$[nlines]
fi

# xlist2 contains the same information as xlist1, but parsed according to chunkdata.txt, or default set by ratio
sed -n "$start","$nlines"p xlist1 > xlist2


# Find values of the mean and the error in the mean of each quantity using correlated statistics (Flyvbjerg and Petersen, 1989)
read -r E  sE < <( awk '{print $1}' xlist2 | flyv | awk '{print $1,$2}')					# Internal Energy
read -r E2 sE2 < <(awk '{print $2}' xlist2 | flyv | awk '{print $1,$2}')					# Internal Energy squared
read -r P  sP < <( awk '{print $3}' xlist2 | flyv | awk '{print $1/'$bP',$2/'$bP'}')				# Pressure
read -r P2 sP2 < <(awk '{print $4}' xlist2 | flyv | awk '{print $1/'$bP'/'$bP',$2/'$bP'/'$bP'}')		# Pressure squared
read -r EP sEP < <(awk '{print $5}' xlist2 | flyv | awk '{print $1/'$bP',$2/'$bP'}')				# Internal Energy * Pressure
read -r F  sF < <( awk '{print $6}' xlist2 | flyv | awk '{print $1,$2}')					# Free Energy = Internal Energy - Temperature * Electronic Entropy
read -r S  sS < <( awk '{print $8}' xlist2 | flyv | awk '{print $1/'$T'*'$eVK'/'$N',$2/'$T'*'$eVK'/'$N'}')	# Electronic Entropy
read -r Tk sTk < <(awk '{print $10}' xlist2 | flyv | awk '{print $1,$2}')					# Temperature computed from the instantaneous kinetic energy
read -r Q  sQ < <( awk '{print $12}' xlist2 | flyv | awk '{print $1,$2}')					# Conserved quantity
read -r Vcell  sVcell < <( awk '{print $14}' xlist2 | flyv | awk '{print $1,$2}')					# Vcell is the cell volume in case it is changing over time
read -r ExP sExP < <(awk '{print $16}' xlist2 | flyv | awk '{print $1/'$bP',$2/'$bP'}')					# External pressure
read -r KP  sKP < <(awk '{print $18}' xlist2 | flyv | awk '{print $1/'$bP',$2/'$bP'}')					# Kinetic pressure
read -r PS  sPS < <(awk '{print $20}' xlist2 | flyv | awk '{print $1/'$bP',$2/'$bP'}')					# Pullay stress

# V=Vcell
# Vcell is the cell volume in case it is changing over time
Vcell_initial=$V
V=$Vcell

# Some other quantities 
M=$(echo "scale=16;$nlines-$start+1" | bc)		# Total number of time steps
rho=$(echo "scale=16;$N/$V" | bc)			# Number density
Vm=$(echo "scale=16;$V/$N*$AV" | bc)			# molar volume

drift=$(awk 'NR==1{f=$12} END{print ($12-f)/'$N'/'$M'*1000*1000}' xlist2)	# drift in the conserved quantity
Epk=$(echo "scale=16;$E+1.5*$N*$T/$eVK" | bc)					# internal energy plus kinetic energy
H=$(echo "scale=16;$E + $P*$V/$eVG + 1.5*$N*$T/$eVK" | bc)			# enthalpy 
sH=$(echo "scale=16;sqrt($sE^2+($sP*$V/$eVG)^2)" | bc)				# uncertainty in enthalpy 
EJ=$(echo "scale=16;$Epk*$eVJ/$N" | bc)						# internal energy plus kinetic energy
HJ=$(echo "scale=16;$H*$eVJ/$N" | bc)						# enthalpy 
sEJ=$(echo "scale=16;$sE*$eVJ/$N" | bc)						# uncertainty in internal energy plus kinetic energy
sHJ=$(echo "scale=16;$sH*$eVJ/$N" | bc)						# uncertainty in enthalpy
CV=$(echo "scale=16;($E2 - $E*$E)/($N*$T^2)*$eVK^2+1.5" | bc)			# heat capacity
sCV=$(echo "scale=16;sqrt($sE^2*($E2-($E)*($E)))/($N*$T^2)*$eVK^2" | bc)	# uncertainty in heat capacity (Allen and Tildesley Eq. 6.23)
aKT=$(echo "scale=16;$MG*($EP - $E*$P)/$T^2*$eVK + 1.*$R*$N/($V*$AV)" | bc)	# thermal pressure coefficient
saKT=$(echo "scale=16;sqrt(sqrt($sE^2/($E2-($E)*($E))*$sP^2/($P2-($P)*($P))))*$MG*($EP - ($E)*($P))/$T/$T*$eVK" | bc)	# uncertainty in thermal pressure coefficient
gam=$(echo "scale=16;1.0*$aKT/($CV*$R*$N)*($V*$AV)" | bc)			# Gruneisen parameter
sgam=$(echo "scale=16;1.0*$aKT/($CV*$R*$N)*($V*$AV)*sqrt(($saKT/$aKT)^2 + ($sCV/$CV)^2)" | bc)	# uncertainty in Gruneisen parameter
ecor=$(echo "scale=16;$sE^2/($E2-($E)*($E))*$M" | bc) 				# Correlation length in energy
pcor=$(echo "scale=16;$sP^2/($P2-($P)*($P))*$M" | bc) 				# Correlation length in pressure
var=$(echo "scale=16;$E2-($E)*($E)" | bc)
sig=$(echo "scale=16;sqrt($var)" | bc)
cubic_cell_size=$(echo "scale=16; e( l($Vcell) / 3 )" | bc -l)  # cubic cell size



#  Compute quantities, including those from fluctuations (heat capacity, thermal pressure coefficient, Gruneisen parameter)
#  For fluctuation formulae, see Allen and Tildesley (1989) pg. 51 et seq.

echo "Temperature = " $T "K" > analysis/peavg.out
echo "Computed temperature = " $Tk "+-" $sTk "K" >> analysis/peavg.out
echo "Number = " $N >> analysis/peavg.out
echo "Number density = " $rho "atom/A^3" >> analysis/peavg.out
echo "Mass density = " $rhom "g/cm^3" >> analysis/peavg.out
echo "Volume = " $Vm "cm^3/mol-atom" >> analysis/peavg.out
echo "Conserved Quantity = " $Q "eV, drift" $drift "meV/atom/ps" >> analysis/peavg.out
echo "Internal Energy = " $E "+-" $sE "eV" >> analysis/peavg.out
echo "Internal Energy (incl. kinetic) = " $Epk "+-" $sE "eV" >> analysis/peavg.out
echo "Enthalpy (incl. kinetic) = " $H "+-" $sH "eV" >> analysis/peavg.out
echo "E-TS_el = " $F "+-" $sF "eV" >> analysis/peavg.out
echo "Electronic entropy = " $S "+-" $sS "Nk_B" >> analysis/peavg.out
echo "Internal Energy (incl. kinetic) =" $EJ "+-" $sEJ "kJ/mol-atom" >> analysis/peavg.out
echo "Enthalpy (incl. kinetic) =" $HJ "+-" $sHJ "kJ/mol-atom" >> analysis/peavg.out
echo "Pressure = " $P "+-" $sP "GPa" >> analysis/peavg.out
echo "CV/Nk =" $CV "+-" $sCV >> analysis/peavg.out
echo "alpha*K_T =" $aKT "+-" $saKT "MPa/K" >> analysis/peavg.out
echo "Gruneisen parameter =" $gam "+-" $sgam >> analysis/peavg.out
echo "Time Steps =" $M "Ratio =" $ratio " E Correlation Length = " $ecor "P Correlation Length =" $pcor >> analysis/peavg.out
echo "Free energy (TOTEN) = " $F "+-" $sF "eV" >> analysis/peavg.out
echo "Volume of cell = " $Vcell "+-" $sVcell "A^3" "( initial: ${Vcell_initial} )" >> analysis/peavg.out
echo "Cubic cell size = " $cubic_cell_size "A" >> analysis/peavg.out
echo "External pressure = " $ExP "+-" $sExP "GPa" >> analysis/peavg.out
echo "Kinetic pressure = " $KP "+-" $sKP "GPa" >> analysis/peavg.out
echo "Pullay stress = " $PS "+-" $sPS "GPa" >> analysis/peavg.out
echo "Total time steps = " $nlines >> analysis/peavg.out

echo $T > analysis/peavg_numbers.out #0
echo $Tk >> analysis/peavg_numbers.out #1
echo $sTk >> analysis/peavg_numbers.out #2
echo $N >> analysis/peavg_numbers.out #3
echo $rho >> analysis/peavg_numbers.out #4
echo $rhom >> analysis/peavg_numbers.out #5
echo $Vm >> analysis/peavg_numbers.out #6
echo $Q >> analysis/peavg_numbers.out #7
echo $drift >> analysis/peavg_numbers.out #8
echo $E >> analysis/peavg_numbers.out #9
echo $sE >> analysis/peavg_numbers.out #10
echo $Epk >> analysis/peavg_numbers.out #11
echo $sE >> analysis/peavg_numbers.out #12
echo $H >> analysis/peavg_numbers.out #13
echo $sH >> analysis/peavg_numbers.out #14
echo $F >> analysis/peavg_numbers.out #15
echo $sF >> analysis/peavg_numbers.out #16
echo $S >> analysis/peavg_numbers.out #17
echo $sS >> analysis/peavg_numbers.out #18
echo $EJ >> analysis/peavg_numbers.out #19
echo $sEJ >> analysis/peavg_numbers.out #20
echo $HJ >> analysis/peavg_numbers.out #21
echo $sHJ >> analysis/peavg_numbers.out #22
echo $P >> analysis/peavg_numbers.out #23
echo $sP >> analysis/peavg_numbers.out #24
echo $CV >> analysis/peavg_numbers.out #25
echo $sCV >> analysis/peavg_numbers.out #26
echo $aKT >> analysis/peavg_numbers.out #27
echo $saKT >> analysis/peavg_numbers.out #28
echo $gam >> analysis/peavg_numbers.out #29
echo $sgam >> analysis/peavg_numbers.out #30
echo $nlines >> analysis/peavg_numbers.out #31 - different from above - this is the total tsteps completed
echo $ratio >> analysis/peavg_numbers.out #32
echo $ecor >> analysis/peavg_numbers.out #33
echo $pcor >> analysis/peavg_numbers.out #34
echo $F >> analysis/peavg_numbers.out #35
echo $sF >> analysis/peavg_numbers.out #36
echo $Vcell >> analysis/peavg_numbers.out #37
echo $sVcell >> analysis/peavg_numbers.out #38
echo $cubic_cell_size >> analysis/peavg_numbers.out #39
echo $ExP >> analysis/peavg_numbers.out #40
echo $sExP >> analysis/peavg_numbers.out #41
echo $KP >> analysis/peavg_numbers.out #42
echo $sKP >> analysis/peavg_numbers.out #43
echo $PS >> analysis/peavg_numbers.out #44
echo $sPS >> analysis/peavg_numbers.out #45
# echo $M $ratio $ecor $pcor >> analysis/peavg_numbers.out


# Cleanup
rm xplist1 xelist1 xflist1 xtlist1 xqlist1 xlist1 xlist2 masses numbers OUTCAR_part
rm -f xelist2 xflist2  
rm -f xvlist1
rm -f xeplist1 xkplist1 xpslist1
