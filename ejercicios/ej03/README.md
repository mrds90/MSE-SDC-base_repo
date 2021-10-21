# Enunciado Trabajo Práctico 3

Se tiene el siguiente sistema:

![Modulador + Canal](./images/ej04-sistema.png)

Considere un periodo de muestreo de $T_s = \frac{1}{16}\,\mu\text{s}$ y un
tiempo de símbolo $T_\text{symb}$ de 8 o 16 veces $T_s$.

1. Generar un script de *octave*, *pthon*, *matlab*, o cualquier otro lenguaje
  similar que implemente el sistema.
    - La señal `b` es una secuencia binaria aleatoria.
      Toma los valores `0` y `1`, o alternativamente `-1` y `1`.
    - La señal `d` inserta $M-1$ ceros entre cada bit y luego le asigna un
      `1` al bit `1` y un `-1` al bit `0`.
      Puede tomar los vaores `-1`, `0` o `1`.
    - El pulso `p` puede tener varias formas:
      1. Pulso cuadrado.
      2. Pulso triangular.
      3. Pulso seno.
      4. Pulso coseno elevado.
    - La señal `x` es la señal a transmitir por el canal, se obtiene mediante la
      convolución entre `d` y `p`, o realizando el filtrado mediante el filtro
      FIR.
      En cualquier caso es importante descartar los primeros $\frac{L_p-1}{2}$
      valores, para que las señales `d` y `x` queden "sincronizadas".
    - El filtro `h` representa al canal y en este caso será una única delta en
      0.
    - La señal `n` representa a ruido blanco gausiano aditivo (AWGN) del canal.

2. Graficar las señales `d`, `x` e `y` superpuestas en un mismo gráfico.
  Realice el gráfico para cada pulso del punto anterior.
  Verificar que las deltas coinciden con los picos de los pulsos, inclusive
  para el coseno elevado.

3. Grafique la densidad espectral de las señales `x` e `y` del punto anterior
  en escala semilogaritmica.

4. Suba el script a la carpeta de entrega.

5. Complete el archivo `README.md` de la carpeta entrega con los gráficos
  del punto 2 y 3.

