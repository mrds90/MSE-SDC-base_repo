function [x_n, mis] = modulator(byte_seq, spar)
  % MODULATOR Converts a byte sequence into a modulated signal.
  %
  % [x_n, mis] = modulator(byte_seq, spar) takes a byte sequence and
  % converts it into a modulated signal by applying upsampling and pulse shaping.
  %
  % Inputs:
  %   byte_seq - Byte sequence to be modulated (values between 0 and 255)
  %   spar     - Structure of parameters with:
  %              * n_pre  : Number of bits in the preamble
  %              * n_sfd  : Number of bits in the Start Frame Delimiter (SFD)
  %              * n_pulse: Upsampling factor
  %              * pulse  : Pulse shaping waveform
  %
  % Outputs:
  %   x_n - Final modulated signal
  %   mis - Structure with internal signals used in modulation
  
    % Verify that all values in byte_seq are valid bytes (0-255)
    assert(all(byte_seq <= 255));
  
    n_bytes = length(byte_seq);
  
    % Generate preamble (alternating 0s and 1s)
    pre          = zeros(1, spar.n_pre);
    pre(2:2:end) = 1;
  
    % Generate Start Frame Delimiter (SFD)
    sfd = zeros(1, spar.n_sfd);
    if mod(spar.n_pre, 2) == 0
      sfd(1:2:end) = 1;
    else
      sfd(2:2:end) = 1;
    end
  
    % Convert bytes to bits
    data = [];
    for i = 1:n_bytes
      bin_str = dec2bin(byte_seq(i), 8); % Convert to binary (8 bits)
      bin = str2num(bin_str(:))';        % Convert string to numeric vector
      data = [data bin];
    end
  
    % Construct packet with preamble, SFD, and data
    packet = [pre sfd data];
  
    % Map bits to symbols (-1, 1)
    x = 2 * packet - 1;
  
    % Pulse shaping
    xx  = upsample(x, spar.n_pulse); % Upsampling with factor spar.n_pulse
    xxx = conv(xx, spar.pulse);      % Filtering to smooth the signal
  
    % Function outputs
    x_n = xxx;  % Modulated signal
    mis.d = packet; % Internal signal of original bits
  
  end
  