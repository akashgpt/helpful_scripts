#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

#take "file_prefix" as input after keyword "-file" or "-f"
# Create the parser
parser = argparse.ArgumentParser(description='Process file prefix.')

# Add the argument to take "-file" or "-f"
parser.add_argument('-file', '-f', type=str, default='dp_test', help='File prefix input (default: dp_test)')

# Parse the arguments
args = parser.parse_args()

# Access the file_prefix argument
file_prefix = args.file

eV_A3_to_GPa = 160.21766208 # Conversion factor from eV/A^3 to GPa
# print("File prefix: ", file_prefix)

def min_max(dat):
    MIN = np.min(np.min(dat,axis=0))
    MAX = np.max(np.max(dat,axis=0))
    return MIN,MAX

def rmse(org,pred): # same as dp_test l2err
    dif = pred - org
    return np.sqrt(np.mean(dif**2))

def err_plot_v2(e, f, v, filename='test_err'):
    print(f"Working on filename: {filename}")
    print("RMSE\t")
    print("N\t", len(e))
    print("e\t", rmse(e[:,0],e[:,1]), '\teV')
    print("f\t", rmse(f[:,:3],f[:,3:]), '\teV/Å')
    print("v\t", rmse(v[:,:9],v[:,9:]), '\tGPa')
    print("\n")

    # add these lines to the file MLMD_vs_DFT.csv after opening it in append mode
    # with open('MLMD_vs_DFT.csv', 'a') as f:
    #     f.write(f"RMSE values for {filename}\n")
    #     f.write("N\t" + str(len(e)) + "\n")
    #     f.write("e\t" + str(rmse(e[:,0],e[:,1])) + "\teV\n")
    #     f.write("f\t" + str(rmse(f[:,:3],f[:,3:])) + "\teV/Å\n")
    #     f.write("v\t" + str(rmse(v[:,:9],v[:,9:])) + "\teV\n")
    #     f.write("\n")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.2, hspace=0.15)

    ax[0].scatter(e[:,0],e[:,1],c='k',s=4, lw=0, alpha=0.5)
    ax[0].set_xlabel('DFT energy (eV)')
    ax[0].set_ylabel('MLP energy (eV)')

    for i in range(3):
        ax[1].scatter(f[:,i],f[:,i+3],c='k',s=4, lw=0, alpha=0.5)
    ax[1].set_xlabel('DFT force (eV ' + r'$\mathrm{\AA^{-1})}$')
    ax[1].set_ylabel('MLP force (eV ' + r'$\mathrm{\AA^{-1})}$')

    for i in range(9):
        ax[2].scatter(v[:,i],v[:,i+9],c='k',s=4, lw=0, alpha=0.5)
    ax[2].set_xlabel('DFT stress (GPa)')
    ax[2].set_ylabel('MLP stress (GPa)')
    # ax[2].set_xlabel('DFT stress (GPa)')
    # ax[2].set_ylabel('MLP stress (GPa)')

    ax[0].plot(min_max(e), min_max(e), c='C1', ls='--')
    ax[1].plot(min_max(f), min_max(f), c='C1', ls='--') 
    ax[2].plot(min_max(v), min_max(v), c='C1', ls='--')

    # print number of data points in super title
    fig.suptitle(f'N = {len(e)}')

    plt.savefig(filename + '.png', dpi=300)
    plt.show()


def err_per_atom_plot_v2(e, f, v, filename='test_err_per_atom'):

    e = e*1000 # convert eV to meV
    # v = v*1000 # convert eV to meV

    print("RMSE (per atom for e and v)\t")
    print("N\t", len(e))
    print("e\t", rmse(e[:,0],e[:,1]), '\tmeV/atom')
    print("f\t", rmse(f[:,:3],f[:,3:]), '\teV/Å')
    print("v\t", rmse(v[:,:9],v[:,9:]), '\tGPa/atom')
    print("\n")

    # add these lines to the file MLMD_vs_DFT.csv after opening it in append mode
    # with open('MLMD_vs_DFT.csv', 'a') as f:
    #     f.write(f"RMSE (per atom) values for {filename}\n")
    #     f.write("N\t" + str(len(e)) + "\n")
    #     f.write("e\t" + str(rmse(e[:,0],e[:,1])) + "\tmeV/atom\n")
    #     f.write("f\t" + str(rmse(f[:,:3],f[:,3:])) + "\teV/Å\n")
    #     f.write("v\t" + str(rmse(v[:,:9],v[:,9:])) + "\tmeV/atom\n")
    #     f.write("\n")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.2, hspace=0.15)

    ax[0].scatter(e[:,0],e[:,1],c='k',s=4, lw=0, alpha=0.5)
    ax[0].set_xlabel('DFT energy (meV/atom)')
    ax[0].set_ylabel('MLP energy (meV/atom)')

    for i in range(3):
        ax[1].scatter(f[:,i],f[:,i+3],c='k',s=4, lw=0, alpha=0.5)
    ax[1].set_xlabel('DFT force (eV ' + r'$\mathrm{\AA^{-1})}$')
    ax[1].set_ylabel('MLP force (eV ' + r'$\mathrm{\AA^{-1})}$')

    for i in range(9):
        ax[2].scatter(v[:,i],v[:,i+9],c='k',s=4, lw=0, alpha=0.5)
    # ax[2].set_xlabel('DFT stress (GPa)')
    # ax[2].set_ylabel('MLP stress (GPa)')
    ax[2].set_xlabel('DFT stress (GPa/atom')
    ax[2].set_ylabel('MLP stress (GPa/atom')

    ax[0].plot(min_max(e), min_max(e), c='C1', ls='--')
    ax[1].plot(min_max(f), min_max(f), c='C1', ls='--') 
    ax[2].plot(min_max(v), min_max(v), c='C1', ls='--')

    # print number of data points in super title
    fig.suptitle(f'N = {len(e)}')

    plt.savefig(filename + '.png', dpi=300)
    # plt.show()








def err_plot(e, f, v, filename='test_err'):
    print(f"Working on filename: {filename}")
    print("RMSE\t")
    print("N\t", len(e))
    print("e\t", rmse(e[:,0],e[:,1]), '\teV')
    print("f\t", rmse(f[:,:3],f[:,3:]), '\teV/Å')
    print("v\t", rmse(v[:,:9],v[:,9:]), '\teV')
    print("\n")

    # add these lines to the file MLMD_vs_DFT.csv after opening it in append mode
    # with open('MLMD_vs_DFT.csv', 'a') as f:
    #     f.write(f"RMSE values for {filename}\n")
    #     f.write("N\t" + str(len(e)) + "\n")
    #     f.write("e\t" + str(rmse(e[:,0],e[:,1])) + "\teV\n")
    #     f.write("f\t" + str(rmse(f[:,:3],f[:,3:])) + "\teV/Å\n")
    #     f.write("v\t" + str(rmse(v[:,:9],v[:,9:])) + "\teV\n")
    #     f.write("\n")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.2, hspace=0.15)

    ax[0].scatter(e[:,0],e[:,1],c='k',s=4, lw=0, alpha=0.5)
    ax[0].set_xlabel('DFT energy (eV)')
    ax[0].set_ylabel('MLP energy (eV)')

    for i in range(3):
        ax[1].scatter(f[:,i],f[:,i+3],c='k',s=4, lw=0, alpha=0.5)
    ax[1].set_xlabel('DFT force (eV ' + r'$\mathrm{\AA^{-1})}$')
    ax[1].set_ylabel('MLP force (eV ' + r'$\mathrm{\AA^{-1})}$')

    for i in range(9):
        ax[2].scatter(v[:,i],v[:,i+9],c='k',s=4, lw=0, alpha=0.5)
    ax[2].set_xlabel('DFT stress (eV)')
    ax[2].set_ylabel('MLP stress (eV)')
    # ax[2].set_xlabel('DFT stress (GPa)')
    # ax[2].set_ylabel('MLP stress (GPa)')

    ax[0].plot(min_max(e), min_max(e), c='C1', ls='--')
    ax[1].plot(min_max(f), min_max(f), c='C1', ls='--') 
    ax[2].plot(min_max(v), min_max(v), c='C1', ls='--')

    # print number of data points in super title
    fig.suptitle(f'N = {len(e)}')

    plt.savefig(filename + '.png', dpi=300)
    plt.show()




def err_per_atom_plot(e, f, v, filename='test_err_per_atom'):

    e = e*1000 # convert eV to meV
    # v = v*1000 # convert eV to meV

    print("RMSE (per atom for e and v)\t")
    print("N\t", len(e))
    print("e\t", rmse(e[:,0],e[:,1]), '\tmeV/atom')
    print("f\t", rmse(f[:,:3],f[:,3:]), '\teV/Å')
    print("v\t", rmse(v[:,:9],v[:,9:]), '\teV/atom')
    print("\n")

    # add these lines to the file MLMD_vs_DFT.csv after opening it in append mode
    # with open('MLMD_vs_DFT.csv', 'a') as f:
    #     f.write(f"RMSE (per atom) values for {filename}\n")
    #     f.write("N\t" + str(len(e)) + "\n")
    #     f.write("e\t" + str(rmse(e[:,0],e[:,1])) + "\tmeV/atom\n")
    #     f.write("f\t" + str(rmse(f[:,:3],f[:,3:])) + "\teV/Å\n")
    #     f.write("v\t" + str(rmse(v[:,:9],v[:,9:])) + "\tmeV/atom\n")
    #     f.write("\n")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.2, hspace=0.15)

    ax[0].scatter(e[:,0],e[:,1],c='k',s=4, lw=0, alpha=0.5)
    ax[0].set_xlabel('DFT energy (meV/atom)')
    ax[0].set_ylabel('MLP energy (meV/atom)')

    for i in range(3):
        ax[1].scatter(f[:,i],f[:,i+3],c='k',s=4, lw=0, alpha=0.5)
    ax[1].set_xlabel('DFT force (eV ' + r'$\mathrm{\AA^{-1})}$')
    ax[1].set_ylabel('MLP force (eV ' + r'$\mathrm{\AA^{-1})}$')

    for i in range(9):
        ax[2].scatter(v[:,i],v[:,i+9],c='k',s=4, lw=0, alpha=0.5)
    # ax[2].set_xlabel('DFT stress (GPa)')
    # ax[2].set_ylabel('MLP stress (GPa)')
    ax[2].set_xlabel('DFT stress (eV/atom')
    ax[2].set_ylabel('MLP stress (eV/atom')

    ax[0].plot(min_max(e), min_max(e), c='C1', ls='--')
    ax[1].plot(min_max(f), min_max(f), c='C1', ls='--') 
    ax[2].plot(min_max(v), min_max(v), c='C1', ls='--')

    # print number of data points in super title
    fig.suptitle(f'N = {len(e)}')

    plt.savefig(filename + '.png', dpi=300)
    # plt.show()










# create a new file MLMD_vs_DFT for reporting the RMSE values
# os.system(f"echo '~~~ RMSE values for MLMD vs DFT ~~~' > MLMD_vs_DFT")
# os.system(f"echo ' ' >> MLMD_vs_DFT")


# if file named file_prefix+'.e.out' and file_prefix+'.v_GPa.out' exists
if os.path.isfile(file_prefix+'.e.out') and os.path.isfile(file_prefix+'.v_GPa.out'):
    e=np.loadtxt(file_prefix+'.e.out')
    f=np.loadtxt(file_prefix+'.f.out')
    v=np.loadtxt(file_prefix+'.v_GPa.out')
    err_plot_v2(e, f, v, 'MLMD_vs_DFT_default')
else:
    e=np.loadtxt(file_prefix+'.e.out')
    f=np.loadtxt(file_prefix+'.f.out')
    v=np.loadtxt(file_prefix+'.v.out')
    err_plot(e, f, v, 'MLMD_vs_DFT_default')


if os.path.isfile(file_prefix+'.e.tr.out') and os.path.isfile(file_prefix+'.v.gpa.tr.out'):
    e=np.loadtxt(file_prefix+'.e.tr.out')
    f=np.loadtxt(file_prefix+'.f.tr.out')
    v=np.loadtxt(file_prefix+'.v.gpa.tr.out')
    err_plot_v2(e, f, v, 'MLMD_vs_DFT_tr')

if os.path.isfile(file_prefix+'.e.test.out') and os.path.isfile(file_prefix+'.v.gpa.test.out'):
    e=np.loadtxt(file_prefix+'.e.test.out')
    f=np.loadtxt(file_prefix+'.f.test.out')
    v=np.loadtxt(file_prefix+'.v.gpa.test.out')
    err_plot_v2(e, f, v, 'MLMD_vs_DFT_test')

# run if file named file_prefix+'.e_peratom.out' exists
if os.path.isfile(file_prefix+'.e_peratom.out') and os.path.isfile(file_prefix+'.v_GPa_peratom.out'):
    e_peratom=np.loadtxt(file_prefix+'.e_peratom.out')
    f=np.loadtxt(file_prefix+'.f.out')
    v_peratom=np.loadtxt(file_prefix+'.v_GPa_peratom.out')
    err_per_atom_plot_v2(e_peratom, f, v_peratom, 'MLMD_vs_DFT_per_atom')
else:
    e_peratom=np.loadtxt(file_prefix+'.e_peratom.out')
    f=np.loadtxt(file_prefix+'.f.out')
    v_peratom=np.loadtxt(file_prefix+'.v_peratom.out')
    err_per_atom_plot(e_peratom, f, v_peratom, 'MLMD_vs_DFT_per_atom')

