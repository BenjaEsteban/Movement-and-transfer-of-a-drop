import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import *
import numpy as np

#datos
#masa gota de agua [kg]
m=5e-3
#gravedad
g=9.81
#dencidad del agua
den=1000
#coeficiente de arrastre
cd=0.4
#area transversal gota (circulo)
A=1e-6

#condicion inicial
t=0
vinc=0
u=vinc

#aplicamos euler
def f(u,t):
    return g-den*A*cd*u**2/(2*m)

#solucion
# se crean dos listas vacias para almacenar valores, junto con el paso de tiempo entre cada registro más el tiempo limite
tsol=[t]
usol=[u]
dt=0.4
tfin=10

#se almacenan los datos en las listas ya creadas mediante un while
while t<tfin:
    u=u+f(u,t)*dt
    t=t+dt
    usol.append(u)
    tsol.append(t)

print("velocidad de la gota",usol)
print("tiempo transcurrido",tsol)

#Impreción de la grafica
plt.scatter(tsol,usol)
plt.title('Velocidad de una gota')
plt.xlabel('Tiempo Transcurrido (seg)')
plt.ylabel('Velocidad (m/s)')
plt.show()


#interpolacion
x= np.array([0, 0.4, 0.8, 1.2000000000000002, 1.6, 2.0, 2.4, 2.8, 3.1999999999999997, 3.5999999999999996, 3.9999999999999996, 4.3999999999999995, 4.8, 5.2, 5.6000000000000005, 6.000000000000001, 6.400000000000001, 6.800000000000002, 7.200000000000002, 7.600000000000002, 8.000000000000002, 8.400000000000002, 8.800000000000002, 9.200000000000003, 9.600000000000003, 10.000000000000004])
y= np.array([0, 3.9240000000000004, 6.616177920000001, 7.038273098472358, 6.999290041777983, 7.004085154663389, 7.0035084463613035, 7.003578001700974, 7.00356961562819, 7.0035706267521585, 7.0035705048397086, 7.00357051953885, 7.003570517766556, 7.003570517980243, 7.003570517954479, 7.003570517957586, 7.003570517957211, 7.003570517957256, 7.00357051795725, 7.003570517957251, 7.003570517957252, 7.003570517957251, 7.003570517957252, 7.003570517957251, 7.003570517957252, 7.003570517957251])

plt.scatter(x, y, color='navy', s=40, marker='o', label='puntos solucion')
plt.legend(loc='upper left')

from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 

pf = PolynomialFeatures(degree = 2) 

X = pf.fit_transform(x.reshape(-1,1))  

regresion_lineal = LinearRegression() 

regresion_lineal.fit(X, y) 

print('w = ' + str(regresion_lineal.coef_) + ', b = ' + str(regresion_lineal.intercept_))

from sklearn.metrics import mean_squared_error 

prediccion_entrenamiento = regresion_lineal.predict(X)

plt.plot(x,prediccion_entrenamiento, color='gold', linewidth=2,
         label='curva de interpolacion')
plt.legend(loc='upper left')
plt.show()

mse = mean_squared_error(y_true = y, y_pred = prediccion_entrenamiento)

rmse = np.sqrt(mse)
print('Error Cuadrático Medio (MSE) = ' + str(mse))
print('Raíz del Error Cuadrático Medio (RMSE) = ' + str(rmse))
r2 = regresion_lineal.score(X, y)
print('Coeficiente de Determinación R2 = ' + str(r2))