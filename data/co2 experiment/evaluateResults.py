# Evaluate results – 373K

with open('RESULTS_373.npy', 'rb') as RESULTS_FILE:
    Y = np.load(RESULTS_FILE, allow_pickle=True)
    Y_pred = np.load(RESULTS_FILE, allow_pickle=True)
    y_train = np.load(RESULTS_FILE, allow_pickle=True)
    y_test = np.load(RESULTS_FILE, allow_pickle=True)
    Y_pred_train = np.load(RESULTS_FILE, allow_pickle=True)
    Y_pred_test = np.load(RESULTS_FILE, allow_pickle=True)
    P_TRAIN_STD = np.load(RESULTS_FILE, allow_pickle=True)


print(mean_absolute_error(Y, Y_pred))
print(r2_score(Y, Y_pred))


print(mean_absolute_error(y_train, Y_pred_train))
print(r2_score(y_train, Y_pred_train))


print(mean_absolute_error(y_test, Y_pred_test))
print(r2_score(y_test, Y_pred_test))


fig, ax = plt.subplots(1, 2, figsize=(8,3), sharey=True)
ax[0].hist(y_test)
ax[0].set_xlabel('True Test', fontsize=14)
ax[1].hist(Y_pred_test)
ax[1].set_xlabel('Predicted Test', fontsize=14)

X_TRAIN = X_TRAIN_373K[:, 1:]
X_TEST = X_TEST_373K[:, 1:]
TRAINx_temp_MEAN = np.mean(X_TRAIN, axis=0)
TRAINx_temp_STD = np.std(X_TRAIN, axis=0)

# Normalize the training inputs/features/indepedents
TRAINx_temp = (X_TRAIN - TRAINx_temp_MEAN) / TRAINx_temp_STD

# Normalize, based on the testing inputs/features/indepedents
TESTx_temp = (X_TEST - TRAINx_temp_MEAN) / TRAINx_temp_STD

# X = np.vstack((TRAINx_temp, TESTx_temp)).T
X = np.vstack((TRAINx_temp, TESTx_temp)).T
Y = np.vstack((Y_TRAIN_373K, Y_TEST_373K))

X_train = X.T[:X_TRAIN.shape[0], :]; y_train = Y[:X_TRAIN.shape[0]]
X_test = X.T[X_TRAIN.shape[0]:, :]; y_test = Y[X_TRAIN.shape[0]:]

P_TRAIN = X_TRAIN_373K[:, 0]#.reshape(-1, 1)
P_TRAIN_MEAN = np.mean(P_TRAIN)
P_TRAIN_STD = np.std(P_TRAIN)
P_TEST = X_TEST_373K[:, 0]#.reshape(-1, 1)
P = P_TEST/P_TRAIN_STD


density = X_test[:, 0]; spg = X_test[:, 1]; volume = X_test[:, 2]; pld = X_test[:, 3]; lcd = X_test[:, 4]
void_frac = X_test[:, 5]; surf_area_m2g = X_test[:, 6]; surf_area_m2cm3 = X_test[:, 7]; ASA = X_test[:, 8];
AV = X_test[:, 9]; NASA = X_test[:, 10]; NAV = X_test[:, 11]; VolFrac = X_test[:, 12];
largest_free_sphere = X_test[:, 13]; largest_included_sphere = X_test[:, 14]
largest_included_sphere_free = X_test[:, 15]


# For 373 K


f1 = ((VolFrac)**2)/np.exp(largest_included_sphere_free)*((VolFrac)**2); beta1 = 0.266050507
f2 = 1*1/np.exp(volume); beta2 = 0.9116918015
f3 = np.exp(void_frac)/np.exp(largest_free_sphere)*1; beta3 = 1.4559584838
f4 = np.exp(NAV)/np.exp(largest_included_sphere)*1; beta4 = -1.0856496179
f5 = np.exp(density)*np.exp(largest_free_sphere)/((pld)**2); beta5 = 0.6402381483
f6 = ((surf_area_m2g)**3)*np.exp(largest_free_sphere)/(VolFrac); beta6 = -0.247674111
f7 = (pld)*((largest_included_sphere_free)**3)*(largest_included_sphere_free); beta7 = 0.0003715575
f8 = 1/np.exp(largest_included_sphere_free)*1; beta8 = -1.6479469314
f9 = np.exp(NASA)/np.exp(largest_included_sphere)*1; beta9 = -1.5995129966
f10 = 1/np.exp(NASA)/np.exp(density); beta10 = -1.3827478537
f11 = np.exp(VolFrac)/np.exp(lcd)*1; beta11 = 3.9368946984
f12 = np.exp(volume)/np.exp(AV)*1; beta12 = 6.0629210217
f13 = np.exp(surf_area_m2cm3)/np.exp(void_frac)/1; beta13 = -3.4863767559
f14 = np.exp(AV)*np.exp(NASA)/((largest_free_sphere)**2); beta14 = -0.4487744225
intercept = 0.41493005195881416

theta = beta1*((f1*P)/(1 + f1*P)) + beta2*((f2*P)/(1 + f2*P)) + beta3*((f3*P)/(1 + f3*P)) + beta4*((f4*P)/(1 + f4*P)) + beta5*((f5*P)/(1 + f5*P)) + beta6*((f6*P)/(1 + f6*P)) + beta7*((f7*P)/(1 + f7*P)) + beta8*((f8*P)/(1 + f8*P)) + beta9*((f9*P)/(1 + f9*P)) + beta10*((f10*P)/(1 + f10*P)) + beta11*((f11*P)/(1 + f11*P)) + beta12*((f12*P)/(1 + f12*P)) + beta13*((f13*P)/(1 + f13*P)) + beta14*((f14*P)/(1 + f14*P))


## For 323 K

f1 = np.exp(pld)/np.exp(largest_included_sphere_free)*1; beta1 = -3.4004723526
f2 = 1/((void_frac)**2)*np.exp(NAV); beta2 = -1.4303541263
f3 = 1/np.exp(largest_free_sphere)*1; beta3 = -0.8208028619
f4 = np.exp(largest_included_sphere)*np.exp(surf_area_m2cm3)/((NAV)**2); beta4 = -8.8552500676
f5 = 1/1*1; beta5 = 8.4895498826
f6 = ((volume)**2)*((VolFrac)**2)*((volume)**2); beta6 = 1.0724573644
f7 = np.exp(largest_free_sphere)*np.exp(volume)/np.exp(NAV); beta7 = 2.45584746
f8 = np.exp(VolFrac)/((largest_included_sphere)**3)*((largest_included_sphere)**3); beta8 = -7.2652270057
f9 = np.exp(spg)/np.exp(density)/np.exp(surf_area_m2g); beta9 = -0.3209063601
f10 = np.exp(largest_included_sphere)*np.exp(VolFrac)/((NAV)**2); beta10 = 6.8987542214
f11 = np.exp(void_frac)*np.exp(void_frac)*1; beta11 = 4.3803767918
f12 = np.exp(volume)/((surf_area_m2g)**2)/((void_frac)**2); beta12 = 2.0055834944
f13 = 1/((NAV)**2)*1; beta13 = 2.7863864423
f14 = np.exp(density)*np.exp(NASA)/np.exp(largest_free_sphere); beta14 = -0.5179514497
intercept = 0.0796867253629201

theta = beta1*((f1*P)/(1 + f1*P)) + beta2*((f2*P)/(1 + f2*P)) + beta3*((f3*P)/(1 + f3*P)) + beta4*((f4*P)/(1 + f4*P)) + beta5*((f5*P)/(1 + f5*P)) + beta6*((f6*P)/(1 + f6*P)) + beta7*((f7*P)/(1 + f7*P)) + beta8*((f8*P)/(1 + f8*P)) + beta9*((f9*P)/(1 + f9*P)) + beta10*((f10*P)/(1 + f10*P)) + beta11*((f11*P)/(1 + f11*P)) + beta12*((f12*P)/(1 + f12*P)) + beta13*((f13*P)/(1 + f13*P)) + beta14*((f14*P)/(1 + f14*P))


yt = theta + intercept


fig, ax = plt.subplots(1, 3, figsize=(10,3), sharey=True)
ax[0].hist(y_test)
ax[0].set_xlabel('True Test', fontsize=14)
ax[1].hist(Y_pred_test)
ax[1].set_xlabel('Predicted Test', fontsize=14)
ax[2].hist(yt)
ax[2].set_xlabel('WTF Predicted Test', fontsize=14)

## For 323 K

with open('RESULTS_323.npy', 'rb') as RESULTS_FILE:
    Y = np.load(RESULTS_FILE, allow_pickle=True)
    Y_pred = np.load(RESULTS_FILE, allow_pickle=True)
    y_train = np.load(RESULTS_FILE, allow_pickle=True)
    y_test = np.load(RESULTS_FILE, allow_pickle=True)
    Y_pred_train = np.load(RESULTS_FILE, allow_pickle=True)
    Y_pred_test = np.load(RESULTS_FILE, allow_pickle=True)
    P_TRAIN_STD = np.load(RESULTS_FILE, allow_pickle=True)


print(mean_absolute_error(Y, Y_pred))
print(r2_score(Y, Y_pred))


print(mean_absolute_error(y_train, Y_pred_train))
print(r2_score(y_train, Y_pred_train))


print(mean_absolute_error(y_test, Y_pred_test))
print(r2_score(y_test, Y_pred_test))


fig, ax = plt.subplots(1, 2, figsize=(8,3), sharey=True)
ax[0].hist(y_test)
ax[0].set_xlabel('True Test', fontsize=14)
ax[1].hist(Y_pred_test)
ax[1].set_xlabel('Predicted Test', fontsize=14)

X_TRAIN = X_TRAIN_323K[:, 1:]
X_TEST = X_TEST_323K[:, 1:]
TRAINx_temp_MEAN = np.mean(X_TRAIN, axis=0)
TRAINx_temp_STD = np.std(X_TRAIN, axis=0)

# Normalize the training inputs/features/indepedents
TRAINx_temp = (X_TRAIN - TRAINx_temp_MEAN) / TRAINx_temp_STD

# Normalize, based on the testing inputs/features/indepedents
TESTx_temp = (X_TEST - TRAINx_temp_MEAN) / TRAINx_temp_STD

# X = np.vstack((TRAINx_temp, TESTx_temp)).T
X = np.vstack((TRAINx_temp, TESTx_temp)).T
Y = np.vstack((Y_TRAIN_323K, Y_TEST_323K))

X_train = X.T[:X_TRAIN.shape[0], :]; y_train = Y[:X_TRAIN.shape[0]]
X_test = X.T[X_TRAIN.shape[0]:, :]; y_test = Y[X_TRAIN.shape[0]:]

P_TRAIN = X_TRAIN_323K[:, 0]#.reshape(-1, 1)
P_TRAIN_MEAN = np.mean(P_TRAIN)
P_TRAIN_STD = np.std(P_TRAIN)
P_TEST = X_TEST_323K[:, 0]#.reshape(-1, 1)
P = P_TEST/P_TRAIN_STD


density = X_test[:, 0]; spg = X_test[:, 1]; volume = X_test[:, 2]; pld = X_test[:, 3]; lcd = X_test[:, 4]
void_frac = X_test[:, 5]; surf_area_m2g = X_test[:, 6]; surf_area_m2cm3 = X_test[:, 7]; ASA = X_test[:, 8];
AV = X_test[:, 9]; NASA = X_test[:, 10]; NAV = X_test[:, 11]; VolFrac = X_test[:, 12];
largest_free_sphere = X_test[:, 13]; largest_included_sphere = X_test[:, 14]
largest_included_sphere_free = X_test[:, 15]


f1 = np.exp(pld)/np.exp(largest_included_sphere_free)*1; beta1 = -3.4004723526
f2 = 1/((void_frac)**2)*np.exp(NAV); beta2 = -1.4303541263
f3 = 1/np.exp(largest_free_sphere)*1; beta3 = -0.8208028619
f4 = np.exp(largest_included_sphere)*np.exp(surf_area_m2cm3)/((NAV)**2); beta4 = -8.8552500676
f5 = 1/1*1; beta5 = 8.4895498826
f6 = ((volume)**2)*((VolFrac)**2)*((volume)**2); beta6 = 1.0724573644
f7 = np.exp(largest_free_sphere)*np.exp(volume)/np.exp(NAV); beta7 = 2.45584746
f8 = np.exp(VolFrac)/((largest_included_sphere)**3)*((largest_included_sphere)**3); beta8 = -7.2652270057
f9 = np.exp(spg)/np.exp(density)/np.exp(surf_area_m2g); beta9 = -0.3209063601
f10 = np.exp(largest_included_sphere)*np.exp(VolFrac)/((NAV)**2); beta10 = 6.8987542214
f11 = np.exp(void_frac)*np.exp(void_frac)*1; beta11 = 4.3803767918
f12 = np.exp(volume)/((surf_area_m2g)**2)/((void_frac)**2); beta12 = 2.0055834944
f13 = 1/((NAV)**2)*1; beta13 = 2.7863864423
f14 = np.exp(density)*np.exp(NASA)/np.exp(largest_free_sphere); beta14 = -0.5179514497
intercept = 0.0796867253629201

theta = beta1*((f1*P)/(1 + f1*P)) + beta2*((f2*P)/(1 + f2*P)) + beta3*((f3*P)/(1 + f3*P)) + beta4*((f4*P)/(1 + f4*P)) + beta5*((f5*P)/(1 + f5*P)) + beta6*((f6*P)/(1 + f6*P)) + beta7*((f7*P)/(1 + f7*P)) + beta8*((f8*P)/(1 + f8*P)) + beta9*((f9*P)/(1 + f9*P)) + beta10*((f10*P)/(1 + f10*P)) + beta11*((f11*P)/(1 + f11*P)) + beta12*((f12*P)/(1 + f12*P)) + beta13*((f13*P)/(1 + f13*P)) + beta14*((f14*P)/(1 + f14*P))


yt = theta + intercept


fig, ax = plt.subplots(1, 3, figsize=(10,3), sharey=True)
ax[0].hist(y_test)
ax[0].set_xlabel('True Test', fontsize=14)
ax[1].hist(Y_pred_test)
ax[1].set_xlabel('Predicted Test', fontsize=14)
ax[2].hist(yt)
ax[2].set_xlabel('WTF Predicted Test', fontsize=14)

