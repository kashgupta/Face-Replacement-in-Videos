'''
  File name: feat_desc.py
  Author:
  Date created:
'''
import numpy as np
from utils import GaussianPDF_2D
from scipy import signal


def feat_desc(img, x, y):
  h = y.size
  padded = np.lib.pad(img, 20, 'constant', constant_values=(0))

  descs = np.zeros((64,h))

  G = GaussianPDF_2D(1, 1.4, 7, 7)
  img = signal.convolve(img, G, mode="same")
  padded = np.lib.pad(img, 20, 'constant', constant_values=(0))

  for i in range(0, h):
      full = padded[y[i]:y[i]+40,x[i]:x[i]+40]
      sampled = full[0:39:5,0:39:5]
      mu = np.mean(sampled)
      sd = np.std(sampled)
      norm = (sampled-mu)/sd
      lin = np.reshape(norm,[64,])
      descs[:,i] = lin
      
  return descs