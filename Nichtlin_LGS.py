import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

def nichtLinearesGS(x_start):
    x = x_start[0]  # als erstes alle parameter initialisieren
    y = x_start[1]
    z = x_start[2]
    a = x_start[3]
# alle gleichungen müssen = 0 sein!!!!
    return [a-2, a*x+y, a*x**2 + y**2-0.5, a*z**3 + y**3]

x_start = [0, 0, 0, 0]
sol = op.fsolve(nichtLinearesGS, x_start)

print("Lösung des nichtlinearen LGS:\n", sol)
print("Prüfe wie gut unsere Lösung ist:")
print(nichtLinearesGS(sol))

x = np.linspace(0,4)
coef = [2, -1]
y = np.polyval(coef, x)
roots = np.roots(coef)
y_roots = np.zeros(len(roots))
plt.plot(x,y)
plt.plot(roots, y_roots, "*")
plt.show()