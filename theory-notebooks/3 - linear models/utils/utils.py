import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
plt.style.use('deeplearning.mplstyle')

def plotTwoLogLosses():
    """ plots the logistic loss """
    fig,ax = plt.subplots(1,2,figsize=(6,3),sharey=True)
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    x = np.linspace(0.01,1-0.01,200)
    ax[0].plot(x,-np.log(x))
    ax[0].set_title("y = 1")
    ax[0].set_ylabel("loss")
    ax[0].set_xlabel(r"$h_{w}(x)$")
    ax[1].plot(x,-np.log(1-x))
    ax[1].set_title("y = 0")
    ax[1].set_xlabel(r"$h_{w}(x)$")
    ax[0].annotate("prediction \nmatches \ntarget ", xy= [1,0], xycoords='data',
                 xytext=[-10,30],textcoords='offset points', ha="right", va="center",
                   arrowprops={'arrowstyle': '->', 'color': dlorange, 'lw': 3},)
    ax[0].annotate("loss increases as prediction\n differs from target", xy= [0.1,-np.log(0.1)], xycoords='data',
                 xytext=[10,30],textcoords='offset points', ha="left", va="center",
                   arrowprops={'arrowstyle': '->', 'color': dlorange, 'lw': 3},)
    ax[1].annotate("prediction \nmatches \ntarget ", xy= [0,0], xycoords='data',
                 xytext=[10,30],textcoords='offset points', ha="left", va="center",
                   arrowprops={'arrowstyle': '->', 'color': dlorange, 'lw': 3},)
    ax[1].annotate("loss increases as prediction\n differs from target", xy= [0.9,-np.log(1-0.9)], xycoords='data',
                 xytext=[-10,30],textcoords='offset points', ha="right", va="center",
                   arrowprops={'arrowstyle': '->', 'color': dlorange, 'lw': 3},)
    plt.suptitle("Loss Curves for Two Categorical Target Values", fontsize=12)
    plt.tight_layout()
    plt.show()