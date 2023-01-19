import numpy as np
from scipy.stats import  chi2


def fun_covar_ellipse (probability, covar, num_pts = 50 )  :

  w, v = np.linalg.eigh (covar)

  quantile_sqrt = np.sqrt (chi2.ppf (probability , df = 2)  )
  angle = np.linspace ( 0, 2*np.pi, num = num_pts )
  cos_angle, sin_angle = np.cos (angle ), np.sin (angle)
  zx = quantile_sqrt * cos_angle
  zy = quantile_sqrt * sin_angle

  ellipse = np.dot ( v , np.dot ( np.diag (np.sqrt(w)),  np.stack (  (zx, zy)) ) )

  return ellipse

def fun_auc (ratings1, ratings2) :

  n1 = len(ratings1)
  n2 = len(ratings2)
  n1xn2 = n1 * n2

  r1mat, r2mat = np.meshgrid (ratings1, ratings2 )
  posmat12 = np.where (r1mat > r2mat, 1.0, 0.) + np.where ( r1mat == r2mat, 0.5, 0. ) 

  auc_slow = np.sum ( posmat12 )    /n1xn2

  comp1 = np.sum (posmat12, axis = 0) /n2
  comp2 = np.sum (posmat12, axis = 1) /n1
  auc_var = np.sum ((comp1 - auc_slow)**2/(n1-1)/n1) +np.sum ((comp2 - auc_slow)**2/(n2-1)/n2)
  auc_std = np.sqrt (auc_var)

  r12  = np.concatenate ( [ratings1, ratings2])
  idx12 = np.argsort (r12)
  r12_sorted = r12[idx12]
  rev_idx12 = np.argsort (idx12)
  auc_fast = (np.sum (rev_idx12[0: n1] )   - n1*(n1-1)/2) /n1xn2

  second_ord = np.sum ((posmat12  - auc_slow )  **2 ) / n1xn2 
  assert auc_fast == auc_slow

  return auc_fast, auc_slow,  auc_std, comp1, comp2, posmat12, second_ord



# ratings : numpy array of size (nsamples, nclasses) , concatenate all classes
# labels: numpy array of size (nsamples, ) , integer index of class membership
# nsamples = nsamples_class1 + \cdots + nsamples_class_K

def fun_mauc_mrmc_jk (y_score_all, labels ) :  

  if len (y_score_all.shape) == 2 : 
    y_score_all = y_score_all [np.newaxis, :]

  n_methods, total_samples, n_classes   = y_score_all.shape  

  ratings_class = {}
  samples_class = np.zeros ( n_classes, dtype = np.int64) 
  for iclass in range ( n_classes) :  
    ratings_class [iclass] = y_score_all [:, labels == iclass, :]
    samples_class [iclass] = np.sum (labels == iclass)

  col_index_array  = np.concatenate ( ( np.array ([0])  , np.cumsum (samples_class) ))
  auc_array = np.zeros  ((n_methods, n_classes, n_classes)) 
  auc_std_array = np.zeros  ((n_methods, n_classes, n_classes)) 

  k2 = n_classes**2
  M_jack = np.zeros ( (n_methods, k2, total_samples)) 
  for imethod in range (n_methods) : 

    for iclass in range (n_classes) : 
      other_classes = [i for i in range (n_classes)] 
      other_classes.remove (iclass)

      for jclass in other_classes : 
        r1 = ratings_class [iclass] [imethod, :, iclass ]
        r2 = ratings_class [jclass] [imethod, :, iclass ]
        _, auc1, auc1_std, comp1, comp2, posmat, _ = fun_auc (r1, r2)

        auc_array [imethod, iclass, jclass] = auc1
        auc_std_array [imethod, iclass, jclass] = auc1_std
        row_idx  = iclass * n_classes + jclass 
        col0 = col_index_array  [iclass]
        col1 = col_index_array  [iclass+1]

        M_jack [imethod, row_idx, col0:col1] = comp1 - auc1 

        col0 = col_index_array  [jclass]
        col1 = col_index_array  [jclass+1]
        M_jack [imethod, row_idx, col0:col1] = comp2 - auc1

  covar_all = np.zeros ( (n_methods, n_methods,  k2, k2))
  #covar12 = np.zeros ( (n_methods,  n_methods))

  for iclass in range (n_classes) : 
    n1 = samples_class [iclass]
    col0 = col_index_array [iclass]
    col1 = col_index_array [iclass+1]

    for imethod in range (n_methods)  : 
      for jmethod in range (n_methods) : 

        covar_all [imethod, jmethod]  += np.dot (M_jack [imethod, :, col0:col1] ,
                                     np.transpose (M_jack [jmethod, :, col0:col1])) / (n1* (n1-1))

  mauc_est = np.sum (auc_array, axis = (-1,-2) )  / (n_classes * (n_classes -1))
  mauc_covar = np.sum (covar_all, axis = (-1, -2) ) / ( n_classes * (n_classes-1))**2  
  mauc_std = np.sqrt  (mauc_covar)

  return mauc_est, mauc_covar, mauc_std, samples_class


