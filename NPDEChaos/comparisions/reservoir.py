import numpy as np
import matplotlib.pyplot as plt
from julia import Main
from scipy import integrate


def non_lin_func(x):
    x[1::2] = np.square(x[1::2])
    return x

def ESN(data, trainLen, testLen, initLen, inSize, outSize, resSize, Win_scale, sparsity, spec_radius, reg, method, false_model = None, verbose = False, non_linear_func = True):
    if verbose:
        print("starting to set up reservoir")

    if method == 'ann':
        Win = np.zeros((resSize, inSize))
        for i in range(resSize):
            Win[i, np.random.randint(0, inSize)] = Win_scale * (np.random.rand(1) * 2 - 1)

    elif method == 'hybrid':
        Win = np.zeros((resSize, inSize * 2))
        for i in range(resSize):
            Win[i, np.random.randint(0, inSize * 2)] = Win_scale * (np.random.rand(1) * 2 - 1)
    # print("nonzero entries in Win:", np.mean(np.sum(np.where(Win != 0, 1, 0), axis = 1)))

    W = np.random.rand(resSize, resSize)
    # delete the fraction of connections given by sparsity:
    sparse_cond = np.random.rand(*W.shape) < sparsity
    W[sparse_cond] = 0
    W[np.logical_not(sparse_cond)] = 2 *(W[np.logical_not(sparse_cond)] - 0.5)
    # compute spectral radius:
    radius = np.max(np.abs(np.linalg.eigvals(W)))
    # rescale to get requested spectral radius:
    W = W * (spec_radius / radius)

    if method == 'ann':
        X = np.zeros((resSize, trainLen - initLen))
    elif method == 'hybrid':
        X = np.zeros((resSize + inSize, trainLen - initLen))
    # target:
    Yt = data[:, initLen + 1 : trainLen + 1]

    if verbose:
        print("Finished setting up random elements of the reservoir.")
    # run the reservoir with the data and collect X
    if non_linear_func:
        def nlf(x):
            return non_lin_func(x)
    else:
        def nlf(x):
            return x

    x = np.zeros((resSize,1))
    for t in range(trainLen):
        if verbose:
            if (t%25)==0:
                print(t,"/",trainLen," computing state vector")

        u = np.reshape(data[:, t], (inSize, 1))

        if method == 'ann':
            x = np.tanh( np.dot( Win, u)  + np.dot( W, x ) )
            if t >= initLen:
                X[:, t - initLen] = nlf(x[:,0])
        elif method == 'hybrid':
            uf =  np.reshape((integrate.odeint(false_model, data[:,t], [0.,dt], tfirst=True).T)[:, -1], (inSize,1))
            x = np.tanh( np.dot( Win, np.vstack((u, uf) ))  + np.dot( W, x ) )
            if t >= initLen:
                X[:resSize, t-initLen] = nlf(x[:,0])
                X[resSize : resSize + inSize, t-initLen] = uf[:,0]

    if verbose:
        print("Finished computing reservoir state vector.")
    # train Wout (ridge regression)


    X_T = X.T
    if method == 'ann':
        Wout = np.dot( np.dot(Yt,X_T), np.linalg.inv( np.dot(X,X_T) + reg * np.eye(resSize) ) )
    elif method == 'hybrid':
        Wout = np.dot( np.dot(Yt,X_T), np.linalg.inv( np.dot(X,X_T) + reg * np.eye(resSize + inSize) ) )

    if verbose:
        print("Finished training output weights.")

    # predict with the trained ESN
    Y = np.zeros((outSize,testLen))
    u = np.reshape(data[:, trainLen], (inSize, 1))
    for t in range(testLen):

        if method == 'ann':
            x = np.tanh( np.dot( Win, u)  + np.dot( W, x ) )
            # y = np.dot( Wout, np.vstack((1,u,x)) )
            y = np.dot( Wout, nlf(x))
            Y[:,t] = y[:,0]
            # generative mode:
            u = y
            ## predictive mode:
            # u = data[trainLen+t+1]
        elif method == 'hybrid':
            uf =  np.reshape((integrate.odeint(false_model, u[:,0], [0.,dt], tfirst=True).T)[:, -1], (inSize,1))
            x = np.tanh( np.dot( Win, np.vstack((u, uf)) )  + np.dot( W, x ) )
            y = np.dot( Wout, np.vstack((nlf(x), uf)) )
            Y[:,t] = y[:,0]
            u = y

    if verbose:
        print("Finished making predictions.")
    # define error metric as a fct of time
    diff = Y[:, :testLen] - data[:, trainLen + 1: trainLen + testLen + 1]
    if inSize == 1:
        error = np.sqrt(diff**2) / np.sqrt(data[:, trainLen + 1 : trainLen + testLen + 1]**2)
        error = error[:,0]
    elif inSize > 1:
        error = np.sqrt(np.dot(diff.T, diff)[np.diag_indices(testLen)]) / np.sqrt(np.mean(np.dot(data[:, trainLen + 1: trainLen + testLen+ 1].T, data[:, trainLen + 1: trainLen + testLen+ 1])[np.diag_indices(testLen)]))

    return Y, error, Wout

def CGLEf(t,y):
    u = np.zeros((y.size))
    u[:ndim] = np.dot(lap, y[:ndim] - (alpha*y[ndim:]))
    u[ndim:] = np.dot(lap, y[ndim:] + (alpha*y[:ndim]))
    return u

def KSf(t,y):
    return np.dot((-dx4),y) - np.multiply(y, np.dot(dx1, y))

eps = 0.001

def KSf_eps(t,y):
    return np.dot((-dx4),y) - (1+eps)*np.dot(dx2,y) - np.multiply(y, np.dot(dx1, y))


def KS_true(t,y):
    return np.dot((-dx4),y) - np.multiply(y, np.dot(dx1, y)) - np.dot(dx2,y)

a = 10.
b = 28.
c = 8.0/3.0
def l63(t,y):
    u = np.zeros(3)
    u[0] = -a * y[0] + a * y[1]
    u[1] = b * y[0] - y[1] - y[0] * y[2]
    u[2] = -c * y[2] + y[0] * y[1]
    return u

eps = 0.05
def l63f(t,y):
    u = np.zeros(3)
    u[0] = -a * y[0] + a * y[1]
    u[1] = (1 + eps) * b * y[0] - y[1] - y[0] * y[2]
    u[2] = -c * y[2] + y[0] * y[1]
    return u

print("Starting, setting up everything ....")
#mode = "cgle"
#mode = "ks"

mode = "ks"

samp = 1
dt = 0.1
#dt = 0.25
trainLen = int(25/ samp)

testLen = int(1000 / samp)
initLen = int(1 / samp)
tlen = 5000

print("Running Reservoir for:")
print(mode)
print("----")

if mode=="cgle":
    n = 50
    L = 75
    ndim = n*n
    dx = L/n

    Main.include("juliatopy.jl")
    lap = np.array(Main.LaplacianCGLE(n, dx))

    data = Main.compute_data()

    inSize = int(ndim*2)
    outSize = int(ndim*2)

elif mode=="ks":
    n = 64
    L = 35
    ndim = n
    dx = L/n

    inSize = int(ndim)
    outSize = int(ndim)

    Main.include("juliatopy_ks.jl")
    dx1 = np.array(Main.ks_fd_dx1(n, dx))
    dx2 = np.array(Main.ks_fd_dx2(n, dx))
    dx4 = np.array(Main.ks_fd_dx4(n, dx))

    data = Main.compute_ks_data()

elif mode=="l63":
    ndim = 3
    n = 3

    inSize = int(ndim)
    outSize = int(ndim)

    data = integrate.odeint(l63, [0.,1.,0.], np.arange(0.,(testLen+trainLen+2000)*dt,step=dt), tfirst=True).T
    data = data[:,1000:]


else:
    print('wrong mode')


alpha = 2.

#trainLen = int(10000 / samp)


############################################################################### comppare with prediction of wrong model
# generate the ESN reservoir

resSize = 20000
spec_radius = .4
#sparsity = .99
sparsity = .97
#Win_scale = .15
Win_scale = 0.5

reg = 1e-4  # regularization coefficient
method = 'ann'
# a = 0.3 # leaking rate+


# load CLGE from Julia




valid_time_ann = 0
valid_time_hybrid = 0
valid_time_model = 0

print("finished generating the data")
print("starting the reservoir computations")

N_runs = 1
for i in range(N_runs):
    print("run: %i/%i",i,N_runs)
    if mode=="ks":

        Y_model = (integrate.odeint(KSf, data[:,0], np.arange(0.,testLen*dt,step=dt), tfirst=True)).T
        diff_model = Y_model[:, :testLen] - data[:,:testLen]
        error_model = np.sqrt(np.dot(diff_model.T, diff_model)[np.diag_indices(testLen)]) / np.sqrt(np.mean(np.dot(data[:, :testLen].T, data[:,:testLen])[np.diag_indices(testLen)]))
        print("done wrong model...")


        Y_ann, error_ann, Wout_ann = ESN(data, trainLen, testLen, initLen, inSize, outSize, resSize, Win_scale, sparsity, spec_radius, reg, method = 'ann', false_model = KSf, verbose=True)
        print("done pure reservoir...")


        Y_hybrid, error_hybrid, Wout_hybrid = ESN(data, trainLen, testLen, initLen, inSize, outSize, resSize, Win_scale, sparsity, spec_radius, reg, method = 'hybrid', false_model = KSf, verbose=True)

    elif mode=="cgle":

        #Y_model = (integrate.odeint(CGLEf, data[:,0], np.arange(0.,testLen*dt,step=dt), tfirst=True)).T
        #diff_model = Y_model[:, :testLen] - data[:,:testLen]
        #error_model = np.sqrt(np.dot(diff_model.T, diff_model)[np.diag_indices(testLen)]) / np.sqrt(np.mean(np.dot(data[:, :testLen].T, data[:,:testLen])[np.diag_indices(testLen)]))
        #print("done wrong model...")

        Y_ann, error_ann, Wout_ann = ESN(data, trainLen, testLen, initLen, inSize, outSize, resSize, Win_scale, sparsity, spec_radius, reg, method = 'ann', false_model = CGLEf, verbose=True)

        print("done pure reservoir...")

        Y_hybrid, error_hybrid, Wout_hybrid = ESN(data, trainLen, testLen, initLen, inSize, outSize, resSize, Win_scale, sparsity, spec_radius, reg, method = 'hybrid', false_model = CGLEf, verbose=True)

    elif mode=="l63":

        #Y_model = (integrate.odeint(l63f, data[:,0], np.arange(0.,testLen*dt,step=dt), tfirst=True)).T
        #diff_model = Y_model[:, :testLen] - data[:,:testLen]
        #error_model = np.sqrt(np.dot(diff_model.T, diff_model)[np.diag_indices(testLen)]) / np.sqrt(np.mean(np.dot(data[:, :testLen].T, data[:,:testLen])[np.diag_indices(testLen)]))

        Y_ann, error_ann, Wout_ann = ESN(data, trainLen, testLen, initLen, inSize, outSize, resSize, Win_scale, sparsity, spec_radius, reg, method = 'ann', false_model = l63f, verbose=True)
        Y_hybrid, error_hybrid, Wout_hybrid = ESN(data, trainLen, testLen, initLen, inSize, outSize, resSize, Win_scale, sparsity, spec_radius, reg, method = 'hybrid', false_model = l63f, verbose=True)


    if mode=="clge":
        time_scale = dt * samp #/ 0.16724655
    elif mode=="l63":
        time_scale = dt * samp #/ .9056
    else:
        time_scale = dt * samp * 0.07


    valid_time_ann += np.arange(testLen)[np.where(error_ann > .4)][0] * time_scale
    valid_time_hybrid += np.arange(testLen)[np.where(error_hybrid > .4)][0] * time_scale
    #valid_time_model += np.arange(testLen)[np.where(error_model > .4)][0] * time_scale


valid_time_ann /= N_runs
valid_time_hybrid /= N_runs
#valid_time_model /= N_runs


print('valid time ann:', valid_time_ann)
print('valid time hybrid:', valid_time_hybrid)
#print('valid time model:', valid_time_model)


if mode=="clge":
    time_scale = dt * samp / 0.16724655
else:
    time_scale = dt * samp

# plot some signals
fig = plt.figure(figsize=(13,9))

ax = fig.add_subplot(411)

ax.plot(np.arange(trainLen + testLen) * time_scale, data[0, : trainLen + testLen], 'k' )
# ax.plot(np.arange(trainLen, trainLen + testLen) * time_scale ,  fm_pred[0], 'r' )
ax.plot(np.arange(trainLen + 1, trainLen + testLen + 1) * time_scale ,  Y_ann.T[:,0], 'b' )
ax.plot(np.arange(trainLen + 1, trainLen + testLen + 1) * time_scale ,  Y_hybrid.T[:,0], 'm' )
ax.axvline(x = trainLen * time_scale, color = 'k', ls = '-')
ax.axvline(x = trainLen * time_scale + valid_time_ann, color = 'k', ls = '-.')
ax.axvline(x = trainLen * time_scale + valid_time_hybrid, color = 'k', ls = '--')
ax.legend(['true model', 'ESN', 'Hybrid model'])
ax.set_ylabel(r'$x$')
ax.set_xlim(100, 135)

ax = fig.add_subplot(412)
ax.plot(np.arange(trainLen + testLen) * time_scale, data[1, : trainLen + testLen], 'k' )
# ax.plot(np.arange(trainLen, trainLen + testLen) * time_scale ,  fm_pred[1], 'r' )
ax.plot(np.arange(trainLen + 1, trainLen + testLen + 1) * time_scale ,  Y_ann.T[:,1], 'b' )
ax.plot(np.arange(trainLen + 1, trainLen + testLen + 1) * time_scale ,  Y_hybrid.T[:,1], 'm' )
ax.axvline(x = trainLen * time_scale, color = 'k', ls = '-')
ax.axvline(x = trainLen * time_scale + valid_time_ann, color = 'k', ls = '-.')
ax.axvline(x = trainLen * time_scale + valid_time_hybrid, color = 'k', ls = '--')
ax.set_ylabel(r'$y$')
ax.set_xlim(100, 135)

ax = fig.add_subplot(413)
ax.plot(np.arange(trainLen + testLen) * time_scale, data[2, : trainLen + testLen], 'k' )
# ax.plot(np.arange(trainLen, trainLen + testLen) * time_scale ,  fm_pred[2], 'r' )
ax.plot(np.arange(trainLen + 1, trainLen + testLen + 1) * time_scale ,  Y_ann.T[:,2], 'b' )
ax.plot(np.arange(trainLen + 1, trainLen + testLen + 1) * time_scale ,  Y_hybrid.T[:,2], 'm' )
ax.axvline(x = trainLen * time_scale, color = 'k', ls = '-')
ax.axvline(x = trainLen * time_scale + valid_time_ann, color = 'k', ls = '-.')
ax.axvline(x = trainLen * time_scale + valid_time_hybrid, color = 'k', ls = '--')
ax.set_ylabel(r'$z$')
ax.set_xlim(100, 135)

ax = fig.add_subplot(414)
ax.semilogy(np.arange(trainLen, trainLen + testLen) * time_scale, error_ann, 'b' )
ax.semilogy(np.arange(trainLen, trainLen + testLen) * time_scale, error_hybrid, 'm' )
ax.axhline(y = .4, color = 'k', ls = '--')
ax.set_ylabel(r'$error$')
ax.set_xlabel(r'$\lambda_{max}t$')
ax.set_xlim(100, 135)

fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot(111)
ax.plot(data[0, trainLen : trainLen + testLen], data[2, trainLen : trainLen + testLen], 'k-', label = 'target data')
# ax.plot(fm_pred[0], fm_pred[1], 'r' )
ax.plot(Y_hybrid.T[:,0], Y_hybrid.T[:,2], 'm-', label = 'hybrid model')
plt.show()
