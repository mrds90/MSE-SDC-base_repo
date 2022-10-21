# MSE-SDC: Repositorio
Maestría en Sistemas Embebidos - Sistemas Digitales para las Comunicaciones


### Descripción de las carpetas del respositorio:

Dentro del repositorio hay cuatro carpestas:
```
MSE-SDC-repo
├── clases
├── ejercicios
├── modem
└── scripts
```
La carpeta:
1. `clases`: Contiene las presentaciones utilizadas durantes el dictado de la materia
    a modo de material de consulta.
2. `ejercicios`: Contiene los enunciados de los ejercicios que se realizarán durante
    la clase y también las carpestas correspondientes para las entregas.
3. `scripts`: Contiene distintos scripts de simulación o scripts auxiliares utilizados
    en el modem.
4. `modem`: Contiene los archivos fuente de VHDL (y otros) de diseño, verificación
    e implementación del modem utilizado en la materia.

    Dentro de esta carpeta se encuetra esta estructura de trabajo:
    ```
    .
    ├── hw
    │   └── artyz7-10
    ├── src
    │   ├── channel
    │   ├── demodulator
    │   ├── lib
    │   ├── modulator
    │   └── top
    └── verification
        ├── tb_bb_modulator
        ├── ...
        └── tb_top_edu_bbt
    ```

    En la carpeta `src` se encuentran los archivos de diseño.

    Dentro de la carpeta `verificaction` hay una carpeta por cada *test*.
    Dentro de cada *test* se encuentran los *testbench* en VHDL y otros archivos para
    la simulación y verificación del sistema.

    Dentro de la carpeta `hw` se encuentran los archivos necesarios para la implementación
    del sistema.
    Estos archivos son dependientes del kit o placa utilizada para la implementación.

