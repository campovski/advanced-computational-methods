import os
import sys
import numpy
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
    plt.ylabel("$\\psi(x)$", fontsize=16)
    
    fig.savefig("task1_lam={}.pdf".format(int(lam*10)), bbox_inches="tight")
    # plt.show()


def task1():
    L = 5
    lams = [0.1*i for i in range(6)]
    ncalc = 15
    nret = 5
    h = 0.01
    x = numpy.linspace(-L, L, int(2*L/h) + 1)
    for lam in lams:
        print(lam)
        Es, psis = calculate_psi(ncalc, nret, lam, x)
        plot_energies(x, Es, psis, lam)



if __name__ == "__main__":
    os.chdir("hw2_stationary-problem-and-spectral-methods/images/")
    
    task1()