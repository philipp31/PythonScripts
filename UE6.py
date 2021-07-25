import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint

m, d, k, g = 0.5, 0.25, 1, 9.81
def twoMassDgl(t,y):
    return [y[1], (1/m)*(-d*y[1] - k*y[0] + m*g)]

sol = solve_ivp(twoMassDgl, (0,20), [0, 2], max_step=0.1)
plt.figure()
plt.plot(sol.t, sol.y[0], label="x(t)")
plt.xlabel("Zeit [s]")
plt.ylabel("Weg [m]")
plt.legend()
plt.grid()


Uo, L, R = 9, 0.1, 2
def RLSystem(y, t): # parameter müssen vertauscht sein
    y1p = (1/L)*(Uo-R*y[0])
    return y1p
#sol = solve_ivp(RLSystem, (0,1), [0], max_step=1/50)
t = np.linspace(0, 1, 50)
y0 = 0
sol = odeint(RLSystem, y0, t)# fkt, y0, zeitvektor
plt.figure()
plt.plot(t, sol, label="i(t)")
x = np.arange(0, 1, 1/50)
i = (Uo/R)*(1-np.exp(-R*x/L))
plt.plot(x, i, "--", label="Exakte Lsg i(t)")
plt.legend()
plt.xlabel("Zeit t / s")
plt.ylabel("Strom i / A")
plt.grid()
#plt.show()


def solveRLC(R):
    plt.figure()
    L, C = 0.1, 5*(10**-6)
    def dglSys(t,y):
        y1p = (1/L)*(-y[1]-R*y[0])
        y2p = (1/C)*(y[0])
        return [y1p, y2p]
    #y[0]=I , y[2] = U
    sol =  solve_ivp(dglSys, (0, 10), [3, 0])
    plt.subplot(211)
    plt.title("R=%5.2f"%R)
    plt.plot(sol.t, sol.y[1], label="Spannung")
    plt.legend()
    plt.subplot(212)
    plt.plot(sol.t, sol.y[0], label="Stromstärke")
    plt.legend()

def solveRLC2(R):
    L, C = 0.1, 5*(10**-6)
    y0 = [3, 0]
    plt.figure()
    def sys(t, y):
        y1p = y[1]
        y2p = (-1/L)*((y[0]/C)+R*y[1])
        return [y1p, y2p]
    sol = solve_ivp(sys, (0,5), y0)
    plt.plot(sol.t, sol.y[0], label="Strom")
    plt.legend()
    plt.plot()
"""solveRLC(0.1)
solveRLC2(0.1)
plt.show()"""

def DuffingOszillator(omega):
    #omega1, omega2 = 1.3, 1.4
    gamma, beta, alpha, delta = 0.3, -1, 1, 0.2
    y0=[0, 0]
    def sys(t,y):
        y1p = y[1]
        y2p = gamma*np.cos(omega*t) - delta*y[1] - beta*y[0] - alpha*(t**3)
        return [y1p, y2p]
    sol = solve_ivp(sys, (0,10), y0)
    plt.figure()
    plt.subplot(211)
    plt.title("Duffing Oszillator Anregung w=%5.2fHz"%omega)
    plt.plot(sol.t, sol.y[0])
    plt.xlabel("Zeit / t")
    plt.ylabel("Weg / m")
    plt.grid()
    plt.subplot(212)
    plt.plot(sol.y[0], sol.y[1])
    plt.xlabel("Weg / m")
    plt.ylabel("Geschw. / m/s")
    plt.tight_layout()
    plt.grid()
DuffingOszillator(1.3)
DuffingOszillator(1.4)
plt.show()

def solveCarSimulation(y0, v=1, L=0.2):
    c = 400000
    D = 0.2
    m = 1200
    d, Uo = 2*D*np.sqrt(c*m), 0.05
    omega = 2*np.pi*v/L
    def autoSys(t, y):
        u = Uo*np.sin(omega*t)
        uStrich = omega*Uo*np.cos(omega*t)
        return [y[1], -(1/m)*( c*(y[0]-u) + d*(y[1]-uStrich) )]
    sol = solve_ivp(autoSys, (0,10), y0, max_step=0.001)
    plt.figure()
    plt.subplot(211)
    plt.plot(sol.t, sol.y[0], label="")
    plt.ylabel("Amplitude x")
    plt.xlabel("Zeit t")
    plt.title("v = "+str(v))
    plt.subplot(212)
    plt.title("Underground")
    plt.plot(sol.t, Uo*np.sin(omega*sol.t))
    plt.tight_layout()

    file = open("C:\\Users\\philipp\\Desktop\\Car_Simul.csv", "w")
    file.write("Zeit, Amplitude\n")
    for i in range(len(sol.t)):
        file.write("%5.2f, %5.2f\n"%(sol.t[i], sol.y[0][i]))
    file.close()
"""
for i in range(5):
    solveCarSimulation([0,0], i, 2)"""
solveCarSimulation([0, 0])
plt.show()

