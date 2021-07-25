import numpy as np

class Kreis:

    def __init__(self, rad):
        self._radius = rad
        self.area = np.pi*self._radius**2

    def __repr__(self):
        return "Kreis mit Radius " + str(self.radius)

    def calcArea(self):
        self.area = np.pi*self._radius**2

    def __add__(self, other):
        newRadius = self.radius + other.radius
        print("Neuer Kreis mir Radius=%5.2f erzeugt"%newRadius)
        return Kreis(newRadius)

    def getRadius(self):
        print("Radius zugriff")
        return self._radius

    def setRadius(self, r):
        if(r>0):
            print("rad wird gesetzt")
            self._radius = r
            self.calcArea()
        else:
            print("radius muss größer 0 sein!")
        return

    radius = property(getRadius, setRadius)