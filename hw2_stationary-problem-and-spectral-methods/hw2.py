import os
import sys
import numpy
import matplotlib.animation
import matplotlib.pyplot as plt
import scipy.integrate

def E_0(n):
    return n + 0.5


def phi(N, x):
    coef = tuple(0 if i < N else 1 for i in range(N+1))
    return numpy.polynomial.hermite.hermval(x, c=coef) * numpy.exp(-(x**2) / 2) / (numpy.power(numpy.pi, 0.25) * numpy.sqrt(2**N * numpy.math.factorial(N)))


def calculate_H(size, lam, x):
    H = numpy.diag([E_0(n) for n in range(size)])

    Phi = [None for _ in range(size)]
    for i in range(size):
        Phi[i] = phi(i, x)

    for i in range(size):
        for j in range(size):
            H[i,j] += lam * numpy.trapz(Phi[i]*x*x*x*x*Phi[j], x)

    return H, numpy.array(Phi)


def psi(c, x):
    psi = numpy.zeros(len(x), dtype=complex)
    for i in range(len(c)):
        coef = tuple(0 if j < i else 1 for j in range(i+1))
        psi += c[i] / (numpy.power(numpy.pi, 0.25) * numpy.sqrt(2**i * numpy.math.factorial(i))) * \
            numpy.polynomial.hermite.hermval(x, c=coef) * numpy.exp(-(x**2) / 2)
    return psi


def calculate_psi(ncalc, nret, lam, x):
    H, Phi = calculate_H(ncalc, lam, x)
    eigval, eigvec = numpy.linalg.eigh(H)
    psis = []
    for i in range(nret):
        psis.append(psi(eigvec[:,i], x))
    return eigval[:nret], psis


def plot_energies(x, Es, psis, lam):
    fig = plt.figure(figsize=(16,9))
    plt.title("Excited eigenstates of anharmonic oscillator $V(x) = \\frac{{1}}{{2}}x^2 + \\lambda x^4$ for $\\lambda = {}$".format(numpy.round(lam, 2)), fontsize=18)

    plots = []
    for psi in psis:
        plots += plt.plot(x, numpy.abs(psi)**2, linewidth=2)

    plt.legend(plots, Es, title="$E$", loc="upper right")
    plt.grid(linestyle="dotted", alpha=0.5)
    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$||\\psi(x)||^2$", fontsize=16)
    
    fig.savefig("task1_lam={}.pdf".format(int(lam*10)), bbox_inches="tight")
    # plt.show()


def plot_per_lam(x, lam_to_Epsi):
    fig, axes = plt.subplots(2, 3, figsize=(16,9))
    fig.suptitle("Excited eigenstates of anharmonic oscillator $V(x) = \\frac{{1}}{{2}}x^2 + \\lambda x^4$")
    
    n = 0
    axes_to_plots = [[] for _ in range(6)]
    axes_to_energies = [[] for _ in range(6)]
    for lam in lam_to_Epsi:
        n = len(lam_to_Epsi[lam][1])
        for i in range(len(lam_to_Epsi[lam][0])):
            axes_to_plots[i] += axes[int(i/3), i%3].plot(x, numpy.abs(lam_to_Epsi[lam][1][i])**2, linewidth=0.5)
            axes_to_energies[i].append(lam_to_Epsi[lam][0][i])
            print(lam_to_Epsi[lam][0][i])

    print(n)
    for i in range(n):
        axes[int(i/3), i%3].set_title("$n$-th excited eigenstate for $n={}$".format(i))
        axes[int(i/3), i%3].set_xlabel("$x$")
        axes[int(i/3), i%3].set_ylabel("$||\\psi(x)||^2$")
        axes[int(i/3), i%3].grid(linestyle="dotted", alpha=0.5)
        axes[int(i/3), i%3].legend(axes_to_plots[i], numpy.round(axes_to_energies[i], 2), title="$E_{}$".format(i))

    fig.legend(axes_to_plots[0], numpy.round(sorted(lam_to_Epsi.keys()), 2), title="$\\lambda$", loc="right")
    fig.savefig("task1_states_per_n.pdf", bbox_inches="tight")


def task1():
    L = 5
    lams = [0.2*i for i in range(6)]
    ncalc = 16
    nret = 6
    h = 0.01
    x = numpy.linspace(-L, L, int(2*L/h) + 1)

    lam_to_Epsi = {}

    for lam in lams:
        print(lam)
        Es, psis = calculate_psi(ncalc, nret, lam, x)
        lam_to_Epsi[lam] = [Es, psis]
        plot_energies(x, Es, psis, lam)

    plot_per_lam(x, lam_to_Epsi)


def animate(x, states, filename, lam, h, interval_base=10):
    print("Rendering animation ...")
    fig, ax = plt.subplots()
    plt.suptitle("($\\lambda = {0}$)".format(lam))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\psi|^2$")
    
    def animate_func(frame):
        print("{0} | {1}".format(len(states), frame))
        ax.clear()
        ax.set_ylim(bottom=0, top=1)
        ax.plot(x, numpy.abs(states[frame])**2/(h**2))
    
    animation = matplotlib.animation.FuncAnimation(fig, animate_func, frames=len(states), interval=interval_base)
    animation.save(filename+".mp4")


def plot_animation(x, lam_psits, lams, tau, h):
    axes_to_plots = [[] for _ in range(6)]
    ts = numpy.array([20*i for i in range(6)])
    alphas = numpy.linspace(0.1, 1, len(ts))

    fig, axes = plt.subplots(2,3, figsize=(16,9))
    fig.suptitle("Time evolution of anharmonic oscillator $V(x) = \\frac{{1}}{{2}} x^2 + \\lambda x^4$", fontsize=16)
    
    for i in range(len(lams)):
        for ti, t in enumerate(ts):
            axes_to_plots[i] += axes[int(i/3), i%3].plot(x, numpy.abs(lam_psits[i][t])**2 *(h**2), 'b-', alpha=alphas[ti])

        axes[int(i/3), i%3].set_title("$\\lambda={}$".format(numpy.round(lams[i], 2)))
        axes[int(i/3), i%3].set_xlabel("$x$")
        axes[int(i/3), i%3].set_ylabel("$||\\psi(x)||^2$")
        axes[int(i/3), i%3].set_ylim(bottom=0, top=1.4)
        axes[int(i/3), i%3].grid(linestyle="dotted", alpha=0.5)
        
    fig.legend(axes_to_plots[0], ts*tau, title="$t = n\\tau$", loc="right")
    fig.savefig("task2_timeevolution.pdf", bbox_inches="tight")
    

def task2():
    L = 5
    lams = [0.2*i for i in range(6)]
    ncalc = 15
    nret = 6
    h = 0.01
    x = numpy.linspace(-L, L, int(2*L/h) + 1)
    tau = 0.01
    t1 = 10
    ts = numpy.linspace(0, t1, int(t1/tau) + 1)

    psi0 = phi(0, x)
    lam_psits = []

    for lam in lams:
        print(lam)
        lam_psits.append([])
        Es, psis = calculate_psi(ncalc, nret, lam, x)
        
        for t in ts:
            lam_psits[-1].append(numpy.zeros(len(x), dtype=complex))
            for n in range(nret):
                lam_psits[-1][-1] += numpy.dot(psis[n], psi0) * numpy.exp(-1j * Es[n] * t) * psis[n]

        lam_psits[-1] = numpy.array(lam_psits[-1])
    # plt.plot(x, numpy.abs(lam_psits[0][0])**2)
    # plt.show()

    # animate(x, lam_psits[-1], "animacija", lams[-1], h)
    plot_animation(x, lam_psits, lams, tau, h)



if __name__ == "__main__":
    os.chdir("hw2_stationary-problem-and-spectral-methods/images/")
    
    # task1()
    task2()