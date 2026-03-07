import numpy as np
import matplotlib.pyplot as plt
import plots
from pathlib import Path

directory_path = Path('./data') 
extension = '*.npz'

for file_path in directory_path.glob(extension):
    if file_path.is_file() and 'random' in file_path.name:
        random_name = './data/' + file_path.name
        identity_name = './data/' + file_path.name.replace('random_','identity_')
        continuum_name = './data/' + file_path.name.replace('random_','continuum_')
        tag = file_path.name[5:-4].replace('random_','')

        M = np.load(random_name)
        loss_random = M['loss']
        M = np.load(identity_name)
        loss_identity = M['loss']
        M = np.load(continuum_name)
        loss_continuum = M['loss']

        plt.figure()
        plt.plot(loss_random,label="Random Initialization")
        plt.plot(loss_identity,label="Identity Initialization")
        plt.plot(loss_continuum,label="Continuum Initialization")
        plt.xscale('log')
        plt.legend()
        plots.savefig('figs/loss_'+tag+'.pdf',grid=True,axis=True)


