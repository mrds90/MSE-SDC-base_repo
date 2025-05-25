## @file tp_final.py
# @brief Communication System Simulation with modulation and demodulation
# @details This script simulates a digital communication system including modulation, channel effects, and demodulation.
# It allows the user to analyze the system performance under different PLL parameters and SNR levels, and visualize the signals.
# @author Marcos Dominguez
# @author Lucas Monzon Languasco

# =============================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
from lib.modulator.modulator import Modulator
from lib.demodulator.demodulator import Demodulator
from lib.channel.channel import Channel
from lib.plotter.plotter import Plotter
# =============================================================================================================================

## @brief Main function that runs the simulation of the communication system.
# @details Parses command-line arguments and performs modulation, transmission through a simulated channel,
# and demodulation using different PLL configurations. It calculates and visualizes error rates for each configuration.
def main():
    ## @cond DOXYGEN_HIDE
    parser = argparse.ArgumentParser(description="Digital communication system simulation.")
    parser.add_argument('--display', '-d', action='store_true', help='Show plots and simulation details for a specific case.')
    args = parser.parse_args()
    ## @endcond

    ## @var Tsymb
    #  @brief Symbol period in seconds.
    Tsymb = 1e-3

    ## @var Ts
    #  @brief Sampling period in seconds.
    Ts = Tsymb / 16

    ## @var fs_MHz
    #  @brief Sampling frequency in MHz.
    fs_MHz = int(1/Ts/1000)

    ## @var f0_MHz
    #  @brief Carrier frequency in MHz.
    f0_MHz = int(1/Tsymb/1000)

    ## @var selected_channel_type
    #  @brief Type of channel model to be used ("delta" or "fir").
    selected_channel_type = "fir"

    ## @var delta_phase
    #  @brief Phase for the delta channel pulse.
    delta_phase = 0

    ## @var order
    #  @brief Order of the filter or number of samples per symbol.
    order = int(np.ceil(fs_MHz/f0_MHz))

    ## @var pll_kp_array
    #  @brief Array of proportional gains for the PLL.
    pll_kp_array = np.arange(0 , 5, 0.05)

    ## @var pll_ki_array
    #  @brief Array of integral gains for the PLL.
    pll_ki_array = np.arange(0 , 0.15, 0.01)

    ## @var ITERATIONS
    #  @brief Number of iterations to average results.
    ITERATIONS = 10

    ## @var snr_dB_array
    #  @brief List of SNR values in decibels to be tested.
    snr_dB_array = [30, 10, 5, 0]

    ## @var error_avg
    #  @brief Array to store the average error percentage for each combination of PLL parameters and SNR.
    error_avg = np.zeros((len(pll_ki_array), len(snr_dB_array), len(pll_kp_array)))

    # Initialize system components
    channel = Channel(channel_type=selected_channel_type, fs_MHz=fs_MHz, f0_MHz=f0_MHz, phase=delta_phase, snr_dB = snr_dB_array[0], order=order)
    modulator = Modulator(n_bytes=4, n_pre=16, n_sfd=2, pulse="rrc", Ts=Ts, Tsymb=Tsymb)
    demodulator = Demodulator(modulator.pulse, n_bytes=4, Ts=Ts, Tsymb=Tsymb, n_pre=16, n_sfd=2)

    ## @var N_TX
    #  @brief Number of transmissions (packets) to simulate.
    N_TX = 10

    ## @var N_ZEROS
    #  @brief Number of zeros to add before each packet for synchronization.
    N_ZEROS = 123

    kzeros = Ts * N_ZEROS
    khalfp = Ts * (modulator.n_fir - 1) / 2
    kmod = Tsymb * (modulator.n_pre + modulator.n_sfd + modulator.n_bytes * 8)
    k0 = kzeros + khalfp
    kend = kzeros + khalfp + kmod
    data_sent, x, d, k, bits_sent = [], np.array([]), [], np.array([]), []
    kk = np.arange(k0, kend, Tsymb)

    ## @brief Modulate and concatenate all transmission packets.
    for i in range(N_TX):
        bytes_seq = np.arange(1 + i, modulator.n_bytes + 1 + i)
        data_sent.extend(bytes_seq)
        xaux = modulator.modulate(bytes_seq)
        x = np.concatenate((x, np.zeros(N_ZEROS), xaux))
        d.extend([np.zeros(N_ZEROS), modulator.d])
        bits_sent.extend(modulator.bits)
        k = np.concatenate((k, kk + (kend + khalfp) * i))
    d = np.concatenate(d)

    ## @brief Simulation loop for each iteration, SNR, and PLL parameter combination.
    for j in range(ITERATIONS):
        print(f"Iteration {j+1}")
        for noise_index, snr_dB in enumerate(snr_dB_array):
            for ki_index, pll_ki in enumerate(pll_ki_array):
                for kp_index, pll_kp in enumerate(pll_kp_array):
                    channel.snr_dB = snr_dB
                    c = channel.transmit(x)
                    demodulator.pll_params = {'kp': pll_kp, 'ki': pll_ki, 'delay': 0}
                    hat_bytes = demodulator.demodulate(c)

                    len_diff = abs(len(data_sent) - len(hat_bytes))
                    if len(hat_bytes) > len(data_sent):
                        errores = np.count_nonzero(np.array(hat_bytes[:len(data_sent)]) != np.array(data_sent))
                    else:
                        errores = np.count_nonzero(np.array(hat_bytes) != np.array(data_sent[:len(hat_bytes)]))

                    error_pct = (errores + len_diff) / len(data_sent) * 100

                    ## @note Visual inspection case: show plots if specific PLL parameters are selected.
                    if args.display and kp_index == 10 and ki_index == 0:
                        plt_process = Plotter(num_stages=3, fs_MHz=fs_MHz)
                        plt_process.add_plot(x, d, title="Signal")
                        plt_process.add_plot(c, d, title="Signal after Channel")
                        plt_process.add_plot(demodulator.dis['y_mf'], demodulator.dis['y_mf']*demodulator.dis['en_sample']/abs(demodulator.dis['y_mf']), title="Signal after match filter")
                        bit_received = np.unpackbits(hat_bytes)
                        print(f"Errors: {errores}, Len diff: {len_diff}, Error %: {error_pct:.2f}%")
                        plt_bytes = Plotter(num_stages=1, fs_MHz=fs_MHz)
                        plt_bytes.add_plot(hat_bytes, data_sent[:len(hat_bytes)], title="Bytes", reference_label="Sent Bytes", pulse="Byte")
                        plt_bits = Plotter(num_stages=1, fs_MHz=fs_MHz)
                        plt_bits.add_plot(bits_sent, bit_received, title="Bits", reference_label="Bits sent", pulse="Bit")

                    error_avg[ki_index, noise_index, kp_index] += error_pct

    # Average the error values
    error_avg /= ITERATIONS
    error_global = np.mean(error_avg, axis=1)
    min_idx = np.unravel_index(np.argmin(error_global), error_global.shape)
    best_ki = pll_ki_array[min_idx[0]]
    best_kp = pll_kp_array[min_idx[1]]

    print(f"Best global combination: KP = {best_kp:.2f}, KI = {best_ki:.3f}")

    ## @brief Generate heatmaps for error percentages across PLL configurations.
    X_kp, Y_ki = np.meshgrid(pll_kp_array, pll_ki_array)
    num_plots = len(snr_dB_array)
    cols = math.ceil(np.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), squeeze=False)

    for idx, snr_dB in enumerate(snr_dB_array):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        Z_error = error_avg[:, idx, :]
        heatmap = ax.pcolormesh(X_kp, Y_ki, Z_error, shading='auto', cmap='viridis')
        ax.set_title(f'SNR = {snr_dB:.1f} dB')
        ax.set_xlabel('PLL KP')
        ax.set_ylabel('PLL KI')
        fig.colorbar(heatmap, ax=ax)

    for i in range(num_plots, rows * cols):
        fig.delaxes(axes[i // cols, i % cols])

    plt.suptitle('Error (%) for different SNR', fontsize=16)
    fig.text(0.5, 0.93, f"Best global combination: KP = {best_kp:.2f}, KI = {best_ki:.3f}", ha='center')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

## @brief Entry point of the script.
if __name__ == "__main__":
    main()
