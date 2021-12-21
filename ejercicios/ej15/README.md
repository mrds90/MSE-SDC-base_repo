## Enunciado Trabajo Práctico 15

Codigos convolucionales y decodificador Viterbi


### Objetivo

El objetivo de este trabajo práctico es familiarizarse con la codificación
convolucional y el decodificador viterbi.


### Descripción

TODO: Falta la descripción (ecuaciones) de las curvas teóricas.

1. Generar un script de *octave*, *python*, *matlab*, o cualquier otro lenguaje
    similar que implemente una codificación y decodificación de códigos
    convolucionales de acuerdo a una matriz de generación $G$.

2. Elegir valores para $K$, $k$ y $n$.
    Elegir los polinomios generadores.
    Construir la matriz generadora $G$.

3. Realizar la codificación.

4. Realizar la codificación mediante el algoritmo de Viterbi.

5. Repetir los pasos 1 a 4, pero introduciendo algunos bits de error en los
    bits codificados (previo a la decodificación).
    Realizar esto para distintas probabilidades de error $P_e$,
    considerando un Binary Symmetric Channel (BSC).

6. En un mismo grafico de BER vs $P_e$ (o SNR por bit) incluir las siguientes
    curvas:
    - BER sin codificación (BPSK).
    - BER con codificación (simulada).
    - Límites de BER con codificación (teórica).

7. Siéntase libre de realizar cualquier otra simulación que le parezca
    interesante.

8. Suba el script a la carpeta de entrega.

9. Complete el archivo `README.md`.


### Entrega

Se debe agregar al repositorio, en la carpeta de entrega correspondiente,
el script de simulación.

Asimismo, en la misma carpeta, se debe agregar un archivo `README.md` que
contenga las capturas de la simulación y una breve explicación de lo que se
está mostrando en cada caso.

