import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time

from sys import argv, exit
from optparse import OptionParser


from utils.utils_auc import *

os.environ['PYTHONINSPECT'] = '1' 


if (len(argv) < 2) :
  print ("test_pdac_linear_fit_mc.py num_mc")
  print ('''
    -m [n1, 100]
  ''')

  exit (1)


parser = OptionParser()

parser.add_option("-m", "--n1", type = "int", dest="n1", \
    help="num of c1", default=100)


(options, args) = parser.parse_args()
num_mc = int (args[0])

ratings_fname = './kaggle/ratings_combined_stat.npy'

with open(ratings_fname, 'rb') as f:
    mean_dict = np.load(f, allow_pickle = True).item ()
    covar_dict = np.load(f, allow_pickle = True).item ()

# now simulating random ratings following distribution. 
n_classes = len (mean_dict)  
seed = 12345
seed = int (time.time ())
rng = np.random.default_rng (seed)
n1 = options.n1
samples_class =  np.ones (n_classes, dtype = np.int64) * n1
labels = np.repeat (    np.arange (n_classes) , repeats = samples_class) 

n_methods = 2
mauc_est = np.zeros ((num_mc, 1, n_methods))
mauc_std = np.zeros ((num_mc, n_methods, n_methods) ) 

stride = 500

start_time = time.time ()
for  imc in range (num_mc) : 

  ratings  = {}
  ratings_all = np.empty ( (0, n_classes * 2))
  total_samples = 0 
  for iclass in range ( n_classes ) :  
    m1 = mean_dict [iclass]  [0]
    cov1 = covar_dict [iclass]  
    ratings  [iclass] = rng.multivariate_normal (m1, cov1, size = samples_class [iclass] )
    ratings_all  = np.concatenate ( (ratings_all, ratings [iclass]), axis = 0)
    total_samples  +=  samples_class [iclass]

#  ratings_both = np.transpose  ( np.reshape (ratings_all, [total_samples,  n_classes, 2]), [2,0,1] ) 
  ratings_both =  np.stack ((ratings_all [:, 0:n_classes], ratings_all [:, n_classes:] ))
  a_mauc, b_covar, c_std, _ = fun_mauc_mrmc_jk (ratings_both, labels )
  mauc_est [imc] = a_mauc
  mauc_std [imc] = b_covar

  end_time = time.time () 
  if ((imc +1)% stride == 0) :
    print ('at imc {:6d}, time {:10.4g}'.format (imc+1, end_time -start_time))

    start_time = end_time
mean_auc_mc = np.mean (mauc_est [:, 0, :], axis = 0, keepdims=True) 
tmp =  (mauc_est [:, 0,:] - mean_auc_mc ) 
covar_mc = np.dot (np.transpose (tmp), tmp) / (num_mc - 1) 
val_b = np.sqrt ( covar_mc ) 
covar_jk_mean = np.mean (mauc_std, axis = 0)  
val_c = np.sqrt (  covar_jk_mean ) 

probability = 0.95
ellipse_mc = fun_covar_ellipse (probability, covar_mc, num_pts = 50 )  
ellipse_jk = fun_covar_ellipse (probability, covar_jk_mean, num_pts = 50 )  

print ('')
print ('mean_auc_mc' ) 
print ( mean_auc_mc [0] ) 
print ('')
print ('auc_mc_covar' ) 
print ( val_b ) 
print ('')
print ('jk_mean_auc_stdv' ) 
print ( val_c ) 
print ('')

fontsize = 16 
fig  = plt.figure( figsize=(8, 6))
ax = fig.add_subplot (1,1,1)
ax.plot (ellipse_mc [0] + mean_auc_mc [0,0], ellipse_mc[1] + mean_auc_mc [0,1], label = 'MC', 
                linestyle = 'dashed')
ax.plot (ellipse_jk [0] + mean_auc_mc [0,0], ellipse_jk[1] + mean_auc_mc [0,1], label = 'proposed')
ax.plot ( mean_auc_mc [0,0],  mean_auc_mc [0,1], marker = '+', markersize = 10, color = 'r')
ax.plot (np.array ([0., 1.]), np.array ([0., 1.]), linestyle = 'dotted',linewidth = 0.5)
ax.grid ()
ax.legend (fontsize = fontsize, loc = 'lower right')
ax.set_aspect ('equal')

ax.set_xlabel ('SVC MAUC (3-class ovo)', fontsize = fontsize)
ax.set_ylabel ('RF MAUC (3-class ovo)', fontsize = fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)

#ax.set_xlim ([0.65, 0.85])
#ax.set_ylim ([0.65, 0.85])
ax.set_xlim ([0.60, 0.90])
ax.set_ylim ([0.60, 0.90])
ax.yaxis.set_ticks (np.arange(0.60, 0.90+0.001, 0.1)  )
plt.show ( block = False  ) 

