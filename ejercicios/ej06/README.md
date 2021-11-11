## Enunciado Trabajo Práctico 6

Simulación (demostrativa) del sistema completo incluyendo sincronización.


### Objetivo

Familiarizarse con las dificultades que existen para la sincronización
de símbolo.
Conocer el método de recuperación de reloj utilizando PLL.


### Descripción

En este ejercicio se realiza una demostración de la simulación del sistema con:
- Modulador
- Canal
- Demodulador, incluyendo la sincronización de símbolo.

Se considera el siguiente sistema:

![Modulador + Canal + Demodulador](./images/ej07-sistema.png)

A diferencia de los ejercicios 03 y 05, en este caso no se asume
que el demodulador está sincronizado y por lo tanto se debe implementar
el mecanismo de sincronización.

Se discuten los distintas dificultades involucradas en el proceso de
sincronización.

Se discute la sincronización mediante la utilización de un PLL:
*square-law timing recovery*.

![Modulador + Canal + Demodulador](./images/square-law-timing-recovery.png)

Se brindan los archivos de simulación para que los alumnos puedan realizar
las simulaciones por su cuenta, realizar los gráficos y capturas que
crean necesarios.
Los alumnos son libres de analizar e investigar con mayor detalle el proceso
de sincronización.


### Entrega

La entrega se realiza directamente actualizando el archivo `README.md`
de la carpeta de la entrega.
Allí se deben incluir las distintas capturas de los gráficos y explicaciones
de lo discutido durante la demostración.

Los alumnos son libres de incluir cualquier otro tipo de información que deseen.

