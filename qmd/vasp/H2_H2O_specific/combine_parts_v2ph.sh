#!/bin/bash
# Purpose:  To combine sub-parts of a simulation
# For two-phase simulations

export TEST_DIR_MASTER=$PWD

echo "Workig on ${TEST_DIR} ..."


counter=1
cd "${TEST_DIR}a"/
echo ""
echo "------------------------"
echo "@ ${TEST_DIR}a"
source data_4_analysis.sh
cat OUTCAR > ../"${TEST_DIR}"/OUTCAR
cp CONTCAR ../"${TEST_DIR}"/CONTCAR
export TEST_DIR_versions=$counter


#after checking a0XXXa, now going for a0XXXb to a0XXXz
for letter in {b..z}
do
if [ -d "../${TEST_DIR}${letter}" ]; then
    (( counter+=1 ))
    cd ../"${TEST_DIR}${letter}"/
    echo ""
    echo "------------------------"
    echo "@ ${TEST_DIR}${letter} (${counter})"
    source data_4_analysis.sh
    cat OUTCAR >> ../"${TEST_DIR}"/OUTCAR
    cp CONTCAR ../"${TEST_DIR}"/CONTCAR
fi
done

#and now going for a0XXXaa to a0XXXaz
for letter in {a..z}
do
if [ -d "../${TEST_DIR}a${letter}" ]; then
    (( counter+=1 ))
    cd ../"${TEST_DIR}a${letter}"/
    echo ""
    echo "------------------------"
    echo "@ ${TEST_DIR}a${letter} (${counter})"
    source data_4_analysis.sh
    cat OUTCAR >> ../"${TEST_DIR}"/OUTCAR
    cp CONTCAR ../"${TEST_DIR}"/CONTCAR
fi
done

#and now going for a0XXXba to a0XXXbz
for letter in {a..z}
do
if [ -d "../${TEST_DIR}b${letter}" ]; then
    (( counter+=1 ))
    cd ../"${TEST_DIR}b${letter}"/
    echo ""
    echo "------------------------"
    echo "@ ${TEST_DIR}b${letter} (${counter})"
    source data_4_analysis.sh
    cat OUTCAR >> ../"${TEST_DIR}"/OUTCAR
    cp CONTCAR ../"${TEST_DIR}"/CONTCAR
fi
done

export TEST_DIR_versions=$counter

echo ""
echo "------------------------"
echo "@ ${TEST_DIR}"
cd "../${TEST_DIR}"/
source data_4_analysis.sh
rm OUTCAR

cd ..
mkdir -p "${TEST_DIR}_last"
mkdir -p "${TEST_DIR}_last/aDATCAR_inst"
mkdir -p "${TEST_DIR}_last/aDATCAR_avg"
cp "${TEST_DIR}/POSCAR" "${TEST_DIR}_last/"
cp "${TEST_DIR}/CONTCAR" "${TEST_DIR}_last/"
cp "${TEST_DIR}/INCAR" "${TEST_DIR}_last/"
cp "${TEST_DIR}/POTCAR" "${TEST_DIR}_last/"
cp "${TEST_DIR}/OSZICAR" "${TEST_DIR}_last/"
cp -r "${TEST_DIR}/analysis/" "${TEST_DIR}_last/"
cd "${TEST_DIR}_last/"
rm XDATCAR
cp *CAR aDATCAR_inst/
cp *CAR aDATCAR_avg/

cd $VASP_ANALYSIS
echo ""
#./combine_XDATCARs_gfortran.out #for debugging
./combine_XDATCARs.out
echo ""
./numden.out

cd $TEST_DIR_MASTER

echo "${TEST_DIR} ($TEST_DIR_versions) done"
