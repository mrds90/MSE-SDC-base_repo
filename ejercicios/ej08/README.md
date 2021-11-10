## Enunciado Trabajo Práctico 8

Simulación del RTL del modulador y el canal:

### Objetivo

Familiarizarse con la creación de tests en el repositorio y con el bloque del
canal.

### Diagramas en bloque

![Diagrama em bloques del modulador](./images/BD-bb_modulator.png)

![Diagrama em bloques del canal](./images/BD-bb_channel.png)

### Descripción

En este ejercicio se debe simular el modulador y el canal en conjunto, para
ellos:
1. Recordar estructura del respositorio en el archivo [README](../../README.md) 
    principal del repositorio.
2. Se debe crear un nueo test en la carpeta
    `MSE-SDC-repo/modem/verification/`.
    Se debe crear la carpeta con nombre `<nombre_del_test>` y dentro de ella el
    testbench, el cual debe llevar el nombre `<nombre_del_test>.vhd`.
3. En el testbench se deben instanciar el modulador junto con el canal.
4. Se debe simular el modulador y el canal.
    Revisar el ejercicio anterior en caso de no recordar cómo simular.
4. Se deben buscar y visualizar todas las señales de interés en la simulación:
    - Tren de deltas.
    - Señal de salidas del filtro FIR.
    - Señal de salida del canal.
    - Las señales de control.

    Las señales de datos se deben mostrar en formato "analógico".


### Entrega

La entrega se realiza directamente actualizando el archivo `README.md`
de la carpeta de la entrega.
Allí se deben incluir las distintas capturas del visualizador y una breve
explicación sobre lo que se está mostrando.

Se debe guardar el archivo de configuración del visualizador `gtkw` o `wcfg`
(GTKWave o Vivado respectivamente) en la misma carpeta que el test, en este
caso en la carpeta `MSE-SDC-repo/modem/verification/tb_modulator/`, de manera
que sea simple volver a visualizar las señales de acuerdo a la configuración
utilizada por el alumno.

Los alumnos son libres de incluir cualquier otro tipo de información que deseen.

