import numpy as np
import matplotlib.pyplot as plt
import copy

def solveIntegral(a, b, fkt):
    N=4
    # Simpson Regel:
    x=[a]
    h = (b-a)/N
    for i in range(1, N+1):
        x.append(a+i*h)
    x.sort() # sortieren sonst sind werte verkehrt herums
    Q = (b-a)*((7/90)*fkt(a) + (32/90)*fkt(x[1]) + (12/90)*fkt(x[2]) + (32/90)*fkt(x[3]) + (7/90)*fkt(b))
    print("Integral mit Grenzen a=%5.2f, b=%5.2f hat Wert Q=%5.2f"%(x[0], x[N], Q), "\n\n")
    return Q


def f(x):
    return x*np.sin(x)

def solveGS(A,b):
    lastIndex = len(A)-1
    B = A
    if(lastIndex == 2):

        faktor = A[lastIndex][0] / A[0][0]
        A[lastIndex][:] -= A[0][:]*faktor # null in 3 zeile erzeugen
        b[lastIndex] -= b[0]*faktor

        faktor = A[1][0] / A[0][0]
        A[1][:] -= A[0][:] * faktor # null in 2 Zeile erzeugen
        b[1] -= b[0]*faktor

        faktor = A[lastIndex][1] / A[1][1]
        A[lastIndex][:] -= A[1][:] * faktor   # 2 null in 3 zeile erzeugen
        b[lastIndex] -= b[1]*faktor

        x = [0, 0, 0]
        #print("A:" , A, ", b: ", b)
        x[2] = b[2]/A[2][2]
        x[1] = 1/A[1][1]*(b[1] - A[1][2]*x[2])
        x[0] = 1/A[0][0]*(b[0] - A[0][2]*x[2] - A[0][1]*x[1])
        print("Solution Ax=b: " , x)
        print("A*x =",np.dot(B,x), "\nb = ", b,"\n\n")
        return x
"""
def createNumpyRandV(n, sigma, mu=0):
    randomV = np.random.randn(n)*sigma + mu
    return np.array(randomV)
v1 = createNumpyRandV(5, 2)
v2 = createNumpyRandV(5, 5)
v3 = createNumpyRandV(5, 0.1)
plt.figure()
plt.hist(v1.tolist())
plt.figure()
plt.hist(v2.tolist())
plt.figure()
plt.hist(v3.tolist())
plt.show()

print(createNumpyRandV(5, 2))
print(createNumpyRandV(5, 5))"""

def diskreteWärmeleitung(n):
    a, b = 0, 1
    h = b/n
    results = {}
    def startTemp(x):
        return np.sin((np.pi*x)/b)
    x, y, y_new= [], [], []
    for i in range(n+1):
        x.append(a + i*h)
        y_new.append(0)
        y.append(0)
    x.sort()
    print(x)
    for k in range(n+1):
        if k == 0 or k == n:
            y[k] = 0
        else:
            y[k] = startTemp(x[k])
    steps = 40
    for k in range(steps): # ZEIT
        for i in range(n+1): # ORT
            if i == 0 or i == n:
                y_new[i] = 0 # RB
            else:
                y_new[i] = y[i] + (1/8)*(y[i-1] -2*y[i] +y[i+1])
        results[k] = copy.deepcopy(y)
        y = copy.deepcopy(y_new)
    print(results)
    plt.plot(x, results[1], label="Schritt 1")
    plt.plot(x, results[25], label="Schritt 25")
    plt.plot(x, results[39], label="Schritt 39")
    plt.legend()
    plt.show()
#diskreteWärmeleitung(10)

def matrixMult(x):
    A = np.array([[10, 0, -6],
                  [5, 25, 7],
                  [-5, 5, 9]])
    r = []
    for row in A:
        r.append(row[0] * x[0] + row[1] * x[1] + row[2] * x[2])
    print("Result Matrx Mult: (%5.2f,%5.2f,%5.2f)" %(r[0],r[1],r[2]))

def solveIterative(steps, A, b):
    x = [1,1,1]
    for i in range(steps):
        x[0] = (1/A[0][0])*(b[0] - A[0][1]*x[1] - A[0][2])
        x[1] = (1 / A[1][1]) * (b[1] - A[1][0]*x[0] - A[1][2]*x[2])
        x[2] = (1 / A[2][2]) * (b[2] - A[2][0]*x[0] - A[2][1]*x[1])
    print("Result: (%5.2f,%5.2f,%5.2f), steps=%5.2f"%(x[0], x[1], x[2], steps))
    return x

def calcPolynomVal(x):
    res = 2+ (2/3)*(x+2) + (1/90)*(x+2)*(x+0.5)*(x-2.5)
    print("P(%5.2f)=%5.2f" %(x, res))
    return res


def calcPolynomAndPrintData(xStst, yStst, xInt):
    diffs = []
    polyCoef = []
    for i in range(len(xStst)-1):
        diffs.append((yStst[i+1] - yStst[i])/(xStst[i+1]-xStst[i]))
    polyCoef.append(diffs[0])
    calcDiffs(xStst, diffs, len(diffs), polyCoef, 1)
    print("Polynom-Koeffizienten: " + str(polyCoef))
    res = 0
    for k in range(len(polyCoef)):
        comp = polyCoef[k]
        if(k != 0):
            for h in range(k):
                comp *= (xInt-xStst[h])
        res += comp

    print()

def calcDiffs(xStst, oldDiffs, length, polyCoef, tiefe):
    diffs = []
    for i in range(length-1):
        diffs.append((oldDiffs[i+1] - oldDiffs[i])/(xStst[i+1+tiefe]-xStst[i]))
    polyCoef.append(diffs[0])
    if(len(diffs) > 1):
        tiefe += 1
        calcDiffs(xStst,diffs, len(diffs), tiefe)


