import sys
import os
import numpy
import scipy.linalg
import matplotlib.animation
import matplotlib.pyplot as plt


def phi(N, x):
    coef = tuple(0 if i < N else 1 for i in range(N+1))
    return numpy.polynomial.hermite.hermval(x, c=coef) * numpy.exp(-(x**2) / 2) / (numpy.power(numpy.pi, 0.25) * numpy.sqrt(2**N * numpy.math.factorial(N)))


def V1D(lam, x):
    return x*x / 2 + lam * numpy.power(x, 4)


def V2D(lam, x, y):
    return (x*x + y*y) / 2 + lam * x*x*y*y


def finite_difference_step(psi, tau, h, V):
    return psi + 1j * tau * ((numpy.roll(psi,1) + numpy.roll(psi,-1) - 2 * psi) / (2*h**2) - V * psi)


def finite_propagator_step(psi, tau, h, K, V):
    hbar_psi = psi
    state = psi
    for k in range(1, K+1):
        hbar_psi = - (1j * tau * ((numpy.roll(hbar_psi,1) + numpy.roll(hbar_psi,-1) - 2 * hbar_psi) / (2*h**2) - V * hbar_psi))
        state += (-1j*tau)**k / numpy.math.factorial(k) * hbar_psi
    return state


def implicit_scheme_step(psi, tau, h, V):
    upper = - 1j * tau / (4*h*h) * numpy.ones(len(psi))
    upper[0] = 0
    lower = - 1j * tau / (4*h*h) * numpy.ones(len(psi))
    lower[-1] = 0
    diag = 1j * tau / 2 * V + 1 + 1j * tau / (2*h*h)

    ab = numpy.array([upper, diag, lower], dtype=complex)
    b = psi + 1j * tau / 4 * ((numpy.roll(psi,1) + numpy.roll(psi,-1) - 2 * psi) / (h**2) - 2 * V * psi)
    
    return scipy.linalg.solve_banded((1,1), ab, b)


def time_evolution(N, lam, x, h, tau, niter, method, K=None):
    if method == 1 and K is None:
        print("Specify K for finite propagator")
        sys.exit()

    Vm = V1D(lam, x)
    states = numpy.zeros((niter+1, len(Vm)), dtype=complex)

    states[0] = phi(N, x)
    for n in range(niter):
        states[n][0] = 0
        states[n][-1] = 0
        if method == 0:
            states[n+1] = finite_difference_step(states[n], tau, h, Vm)
        elif method == 1:
            states[n+1] = finite_propagator_step(states[n], tau, h, K, Vm)
        elif method == 2:
            states[n+1] = implicit_scheme_step(states[n], tau, h, Vm)

    return states


def animate(x, states, filename, lam, h, tau, N, interval_base=200, interval_slow=500, range_slow=None, frames_slow=20):
    print("Rendering animation ...")
    fig, ax = plt.subplots()
    plt.suptitle("Time evolution of anharmonic oscilator wave function\n($\\lambda = {0}, h = {1}, \\tau = {2}, N = {3}$)".format(lam, h, tau, N))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\psi|^2$")
    
    def animate_func(frame):
        ax.clear()
        ax.plot(x, numpy.abs(states[frame])**2)

    def animate_func2(frame):
        ax.clear()
        ax.plot(x, numpy.abs(states[frame+range_slow])**2)

    os.chdir("hw1_schroedinger-equation-spectrum-and-time-evolution/images/")
    
    animation = matplotlib.animation.FuncAnimation(fig, animate_func, frames=len(states), interval=interval_base)
    animation.save(filename+".html")

    if range_slow is not None:
        animation = matplotlib.animation.FuncAnimation(fig, animate_func2, frames=frames_slow, interval=interval_slow)
        animation.save(filename+"_slow.html")


def testing(method=2):
    if method == 0:  # tau < h**2
        N = 0
        lam = 0
        L = 10
        h = 0.1
        tau = 0.0099
        niter = 120
        x = numpy.linspace(-L, L, 2*L/h)

        states = time_evolution(N, lam, x, h, tau, niter, method)
        animate(x, states, "naloga1_finite_diff"+str(N)+","+str(lam)+","+str(h)+","+str(tau), lam, h, tau, N, range_slow=90)

    elif method == 1:  # tau < 2 * pi * h**2
        N = 0
        lam = 0
        L = 10
        h = 0.1
        tau = 0.0099 * 3.14
        niter = 400
        K = 10
        x = numpy.linspace(-L, L, 2*L/h)
        states = time_evolution(N, lam, x, h, tau, niter, method, K=K)
        animate(x, states, "naloga1_finite_propagator"+str(N)+","+str(lam)+","+str(h)+","+str(tau)+","+str(K), lam, h, tau, N, interval_base=20, range_slow=340, frames_slow=60)

    elif method == 2:
        N = 0
        lam = 0
        L = 10
        h = 0.1
        tau = 0.0099
        niter = 1000
        x = numpy.linspace(-L, L, 2*L/h)
        states = time_evolution(N, lam, x, h, tau, niter, method)
        animate(x, states, "naloga1_implicit_scheme"+str(N)+","+str(lam)+","+str(h)+","+str(tau), lam, h, tau, N, interval_base=5)


def task1():
    Ns = range(6)
    lams = numpy.linspace(0, 0.5, 3)
    L = 10
    h = 0.1
    tau = 0.0099
    niter = 1000
    x = numpy.linspace(-L, L, int(2*L/h) + 1)

    print("Generating states ...")
    N_to_lams_to_states = {}
    for N in Ns:
        N_to_lams_to_states[N] = {}
        for lam in lams:
            N_to_lams_to_states[N][lam] = time_evolution(N, lam, x, h, tau, niter, method=2)

    # # for lambda = 0 it's constant
    # fig, axes = plt.subplots(2, 3, figsize=(16,9))
    # fig.suptitle("Evolution for $\\lambda=0$ is time independent")
    # for N in Ns:
    #     i = N // 3
    #     j = N % 3
    #     axes[i, j].set_title("$N={}$".format(N))
    #     axes[i, j].set_ylim(bottom=0, top=1)
    #     axes[i, j].plot(x, numpy.abs(N_to_lams_to_states[N][0][0])**2)
    # fig.savefig("naloga1_N_lambda_0.pdf", bbox_inches="tight")
    # sys.exit()

    # plot each subplot different N
    print("Animating per N ...")
    fig, axes = plt.subplots(2, 3, figsize=(32,18))
    fig.suptitle("Time evolution")

    def animate_per_N(frame):
        print("\t{0} | {1}".format(niter, frame+1))
        for N in Ns:
            i = N // 3
            j = N % 3
            axes[i, j].clear()
            axes[i, j].set_title("$N = {}$".format(N))
            axes[i, j].set_ylim(bottom=0, top=1.5)
            for lam in lams:
                axes[i, j].plot(x, numpy.abs(N_to_lams_to_states[N][lam][frame])**2)

    animation = matplotlib.animation.FuncAnimation(fig, animate_per_N, frames=niter, interval=50)
    animation.save("naloga1_N.mp4")

    # plot each subplot different lambda
    print("Animating per lambda ...")
    fig, axes = plt.subplots(1, 3, figsize=(32,18))
    fig.suptitle("Time evolution")

    def animate_per_lam(frame):
        print("\t{0} | {1}".format(niter, frame+1))
        for lam in lams:
            i = int(lam * 4)
            axes[i].clear()
            axes[i].set_title("$\\lambda = {}$".format(lam))
            axes[i].set_ylim(bottom=0, top=1.5)
            for N in Ns:
                axes[i].plot(x, numpy.abs(N_to_lams_to_states[N][lam][frame])**2)

    animation = matplotlib.animation.FuncAnimation(fig, animate_per_lam, frames=niter, interval=50)
    animation.save("naloga1_lambda.mp4")

    # plot each subplot different N, lambda=0
    print("Animating different N, lambda = 0 ...")
    fig, axes = plt.subplots(2, 3, figsize=(32,18))
    fig.suptitle("Time evolution for $\\lambda=0$")

    def animate_per_N_fixed_lambda(frame):
        print("\t{0} | {1}".format(niter, frame+1))
        lam = 0
        for N in Ns:
            i = N // 3
            j = N % 3
            axes[i, j].clear()
            axes[i, j].set_title("$N = {}$".format(N))
            axes[i, j].set_ylim(bottom=0, top=1)
            axes[i, j].plot(x, numpy.abs(N_to_lams_to_states[N][lam][frame])**2)

    animation = matplotlib.animation.FuncAnimation(fig, animate_per_N_fixed_lambda, frames=niter, interval=5)
    animation.save("naloga1_N_lambda_0.mp4")


def time_evolution_changing_lambda(lams, x, h, tau, niter, L, a=5):
    interval = int(numpy.ceil(niter / len(lams)))
    
    print("Generating states ...")
    states = numpy.zeros((niter+1, len(x)), dtype=complex)
    states[0] = phi(0, x-a)
    for n in range(niter):
        Vm = V1D(lams[n//interval], x)
        states[n][0] = 0
        states[n][-1] = 0
        states[n+1] = implicit_scheme_step(states[n], tau, h, Vm)

    print("Animating TASK 2 ...")
    fig, ax = plt.subplots()
    fig.suptitle("Task 2: $N=0, a={0}, \\lambda \\in {1}$".format(a, lams))

    def animate(frame):
        print("Animating TASK 2\t{0} | {1}".format(niter, frame+1))
        ax.clear()
        ax.set_title("$\\lambda={}$".format(lams[frame//interval]))
        ax.set_ylim(bottom=0, top=1.5)
        ax.plot(x, numpy.abs(states[frame])**2)

    animation = matplotlib.animation.FuncAnimation(fig, animate, frames=niter, interval=5)
    animation.save("naloga2_L{0}_a{1}.mp4".format(L, a))


def task2(a):
    Ns = range(6)
    lams = numpy.linspace(0, 0.5, 11)
    lams = [0]
    L = 10
    h = 0.1
    tau = 0.0099
    niter = 10000
    x = numpy.linspace(-L, L, int(2*L/h) + 1)

    states = time_evolution_changing_lambda(lams, x, h, tau, niter, L, a=a)


def task2_extra():
    """
    Time of travel of Gaussian to 0 from a = L/2 with respect to L.
    """
    N = 0
    lam = 0
    Ls = numpy.array([2*L for L in range(1,23)])
    h = 0.01
    tau = 0.000099

    iterss = []

    for L in Ls:
        a = L // 2
        print(L)
        x = numpy.linspace(-L, L, int(2*L/h) + 1)
        # eps = int(0.1 * len(x))

        Vm = V1D(lam, x)
        state = phi(N, x-a)

        iters = 0
        while True:
            prob = numpy.abs(state)**2
            mid = int(2*L/h) // 2
            # if max(prob) in prob[mid-eps:mid+eps]:
            if numpy.argmax(prob) <= mid:
                print(iters)
                iterss.append(iters)
                break

            state[0] = 0
            state[-1] = 0
            state = implicit_scheme_step(state, tau, h, Vm)
            iters += 1

    fig = plt.figure()
    plt.title("Iterations of Gaussian travel to center")
    plt.xlabel("$L$")
    plt.ylabel("Time")
    plt.plot(Ls, tau*numpy.array(iterss))
    plt.show()
    fig.savefig("naloga2_iters_of_gaussian_travel.pdf", bbox_inches="tight")


def task2_extra2():
    """
    Time of travel of Gaussian to 0 for fixed L and different a.
    """
    N = 0
    lam = 0
    L = 10
    h = 0.001
    tau = 0.000099
    aa = numpy.array([0.25*a for a in range((L-1)*4)])
    x = numpy.linspace(-L, L, int(2*L/h) + 1)
    Vm = V1D(lam, x)
    # eps=int(0.1*len(x))

    iterss = []
    for a in aa:
        print(a)
        state = phi(N, x-a)

        iters = 0
        while True:
            prob = numpy.abs(state)**2
            mid = int(2*L/h) // 2
            # if max(prob) in prob[mid-eps:mid+eps]:
            if numpy.argmax(prob) <= mid:
                print(iters)
                iterss.append(iters)
                break

            state[0] = 0
            state[-1] = 0
            state = implicit_scheme_step(state, tau, h, Vm)
            iters += 1

    fig = plt.figure()
    plt.title("Iterations of Gaussian travel to center ($L={}$)".format(L))
    plt.xlabel("$a$")
    plt.ylabel("Time")
    plt.plot(aa, tau*numpy.array(iterss))
    plt.show()
    fig.savefig("naloga2_iters_of_gaussian_travel_fixedL={}.pdf".format(L), bbox_inches="tight")


def task2_extra3():
    N = 0
    lam = 0
    L = 10
    h = 0.01
    tau = 0.001
    aa = numpy.array([0, 0.1, 0.2, 0.3, 0.4] + [0.5*a for a in range(1,L-1)])
    x = numpy.linspace(-L, L, int(2*L/h) + 1)
    Vm = V1D(lam, x)
    niter = 20000

    errorss = []
    for a in aa:
        print(a)
        errors = []
        initial = numpy.abs(phi(N, x-a))**2
        initial[0] = 0
        initial[-1] = 0
        state = phi(N, x-a)
        for i in range(niter):
            diff = (numpy.abs(state)**2 - initial)
            errors.append(numpy.sqrt(diff.dot(diff)))
            state[0] = 0
            state[-1] = 0
            state = implicit_scheme_step(state, tau, h, Vm)
        errorss.append(errors)
    
    fig = plt.figure()
    plt.title("Distance between initial state and running state")
    plt.xlabel("Time")
    plt.ylabel("Distance ($||\\,|\psi(t=n\\tau)|^2 - |\\psi(0)|^2\\,||$)")
    plots = []
    for i in range(len(aa)):
        plots += plt.plot(tau*numpy.array(range(niter)), numpy.array(errorss[i]) / numpy.amax(errorss), linewidth=1)
    plt.legend(plots, aa, title="$a$", loc="upper right")
    fig.savefig("naloga2_distance.pdf", bbox_inches="tight")
    plt.show()


def task2_extra3a():
    N = 0
    lam = 0
    L = 10
    h = 0.01
    tau = 0.001
    a = 3
    x = numpy.linspace(-L, L, int(2*L/h) + 1)
    Vm = V1D(lam, x)
    niter = 20000

    states = numpy.zeros((niter+1, len(x)), dtype=complex)
    states[0] = phi(0, x-a)
    initial = numpy.abs(states[0])**2
    for n in range(niter):
        states[n][0] = 0
        states[n][-1] = 0
        states[n+1] = implicit_scheme_step(states[n], tau, h, Vm)

    print("Animating TASK 2 ...")
    fig, ax = plt.subplots()
    fig.suptitle("Time evolution ($a = 3$)")

    def animate(frame):
        print("Animating TASK 2\t{0} | {1}".format(niter, frame+1))
        diff = (numpy.abs(states[frame])**2 - initial)
        dist = numpy.sqrt(diff.dot(diff))
        ax.clear()
        ax.set_title("$d(\\psi(0), \\psi({0})) = {1}$".format(frame, numpy.round(dist, decimals=5)))
        ax.set_ylim(bottom=0, top=1)
        ax.plot(x, initial, "k-", linewidth=1)
        ax.plot(x, numpy.abs(states[frame])**2)

    animation = matplotlib.animation.FuncAnimation(fig, animate, frames=niter, interval=2)
    animation.save("naloga2_L{0}_a{1}.mp4".format(L, a))

    


def task3(a=5):
    Ns = range(6)
    lams = numpy.linspace(0, 0.5, 11)
    L = 10
    h = 0.1
    tau = 0.0099
    niter = 10000
    x = numpy.linspace(-L, L, int(2*L/h) + 1)
    

if __name__ == "__main__":
    os.chdir("hw1_schroedinger-equation-spectrum-and-time-evolution/images/")

    # task1()
    for a in [1, 3, 5]:
        task2(a)
    # task3()
    # task2_extra()
    # task2_extra2()
    task2_extra3a()