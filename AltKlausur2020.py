import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
import WordProcess as wp

# a)
x = np.arange(-3, 3, 0.1)
plt.figure()
beta = 1
for i in range(1, 11):
    Sigmoid = np.tanh(beta*x)
    lab="Sigmoid mit Beta="+str(beta)
    plt.plot(x, Sigmoid, "-o", label=lab)
    beta += 1

plt.plot(x, np.sign(x), "-o", label="Sign(x)")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

# b)
m1, m2 = 2, 10
k1, k2 = 100, 100
g, mu = 9.81, 0.2
def dgl_sys(t, y):
    y1p = y[1]
    y2p = (1/m1)*(-k1*y[0] + k2*(y[2]-y[0]) -mu*m1*g*np.tanh(1000*y[1]))
    y3p = y[3]
    y4p = (1/m2)*(-k2*(y[2]-y[0]) -mu*m2*g*np.tanh(1000*y[3]))
    return [y1p, y2p, y3p, y4p]
# x1(0)= 1 , x2(0)= -1
sol = int.solve_ivp(dgl_sys, (0, 15), [1, 0, -1, 0], method="Radau")
plt.figure()
plt.plot(sol.t, sol.y[0], label="x1(t)")
plt.plot(sol.t, sol.y[2], label="x2(t)")
plt.legend()
plt.grid()
plt.xlabel("t")
plt.ylabel("Auslenkung x(t)")
plt.title("System ohne Kraftanregung")

# c)
m1, m2 = 2, 10
k1, k2 = 100, 100
g, mu = 9.81, 0.2
freq = 0.2

def muCoef(t):
    if t <= 200:
        return 0.2
    if t >= 300:
        return 0.05
    return (t-200)*(-0.0015) + 0.2

def dgl_sys(t, y):
    y1p = y[1]
    y2p = (1/m1)*(-k1*y[0] + k2*(y[2]-y[0]) -muCoef(t)*m1*g*np.tanh(1000*y[1]) + np.sin(2*np.pi*freq*t))
    y3p = y[3]
    y4p = (1/m2)*(-k2*(y[2]-y[0]) -muCoef(t)*m2*g*np.tanh(1000*y[3]))
    return [y1p, y2p, y3p, y4p]
# x1(0)= 1 , x2(0)= -1
sol = int.solve_ivp(dgl_sys, (0, 500), [0, 0, 0, 0], method="Radau")
plt.figure()
plt.plot(sol.t, sol.y[0], label="x1(t)")
plt.plot(sol.t, sol.y[2], label="x2(t)")
plt.title("System mit Kraftanregung")
plt.legend()
plt.grid()
plt.xlabel("t")
plt.ylabel("Auslenkung x(t)")
plt.show()


##########################################
#A2
MainWords, WordCount = wp.WordCount("myText.txt", key=True)
print("")
print(type(MainWords))
print(MainWords.shape)

fig = plt.figure(figsize=(8, 8))
plt.xticks(rotation=45)
words = []
occurences = []
for i in range(5):
    words.append(MainWords[i])
    occurences.append(WordCount[i])

plt.title("5 Häufigksten Wörter im Text")
plt.xlabel("Wörter")
plt.ylabel("Häufigkeit")
plt.bar(words, occurences)
plt.tight_layout()
plt.show()
fig.savefig("Bar_Haufigkeitsanalyse.png")


#######################################################
#A3
#a)
A = np.array([[1, 1, -9, 3], [-5, 3, 8, -2], [-10, -5, 1, -8], [0, -4, 4, 5]])
A2 = np.array([[-3, 5, 10], [-2, 7, 14], [-3, 1, 2]])
b = np.array([11, 97, -19, -18])
b2 = np.array([34, 0, 95])

#b)
def LGScheck(A):
    det = np.linalg.det(A)
    if det != 0:
        return 1, det
    return 0, det
det1 = LGScheck(A)
print(det1)
det2 = LGScheck(A2)
if det1[0] == 0 or det2[0] == 0:
    print("Eine der Matrizen ist nicht regulär!")
print("Determinante der 1 Matrix:%5.2f, Determinante der 2 Matrix:%5.2f" %(det1[1], det2[1]))

#c)
def LGSsolve(A, b):
    try:
        sol = np.linalg.solve(A, b)
        return sol, True
    except:
        sol = np.linalg.lstsq(A, b)[0]
        print("Warnung System nicht eindeutig lösbar!")
    return sol, False

#d)
x1, sol1 = LGSsolve(A,b)
x2, sol2 = LGSsolve(A2, b2)
print("Lösung für A1: ", x1, ", konnte Sie eindeutig berechnet werden:" , sol1)
print("Lösung für A2: ", x2, ", konnte Sie eindeutig berechnet werden:" , sol2)








