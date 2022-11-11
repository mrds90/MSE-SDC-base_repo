-- libraries
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.pkg_edu_bbt.all;

Library UNISIM;
use UNISIM.vcomponents.all;

-- entity
entity hw_top_edu_bbt is
  port(
    clk_i   : in  std_logic;                    --system clock
    arst_i  : in  std_logic;                    --ascynchronous reset
    rx_i    : in  std_logic;                    --receive pin
    tx_o    : out std_logic;
    led_o   : out std_logic_vector(3 downto 0)
  );
end entity hw_top_edu_bbt;

-- architecture
architecture rtl of hw_top_edu_bbt is
 
  COMPONENT ila_0
  PORT (
    clk : IN STD_LOGIC;
    probe0 : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    probe1 : IN STD_LOGIC_VECTOR(0 DOWNTO 0)
  );
  END COMPONENT;

  component clk_wiz_0
  port (
    -- Clock in ports
    -- Clock out ports
    clk_out1          : out    std_logic; -- 16 MHz
    -- Status and control signals
    reset             : in     std_logic;
    locked            : out    std_logic;
    clk_in1           : in     std_logic  -- 125 MHz
   );
  end component;

  signal clk_s        : std_logic;
  signal clk_locked_s : std_logic;

  signal rx_s : std_logic_vector(0 downto 0);
  signal tx_s : std_logic_vector(0 downto 0);

  signal counter_s : std_logic_vector(26 downto 0);

  -- Modulator to channel output
  signal mod_os_data_s  : std_logic_vector( 9 downto 0);
  signal mod_os_dv_s    : std_logic;
  signal mod_os_rfd_s   : std_logic;
  -- Channel output
  signal chan_os_data_s : std_logic_vector( 9 downto 0);
  signal chan_os_dv_s   : std_logic;
  signal chan_os_rfd_s  : std_logic;

  -- Modem config
  constant nm1_bytes_c  : std_logic_vector( 7 downto 0) := X"03";
  constant nm1_pre_c    : std_logic_vector( 7 downto 0) := X"07";
  constant nm1_sfd_c    : std_logic_vector( 7 downto 0) := X"03";
  constant det_th_c     : std_logic_vector(15 downto 0) := X"0040";
  constant pll_kp_c     : std_logic_vector(15 downto 0) := X"A000";
  constant pll_ki_c     : std_logic_vector(15 downto 0) := X"9000";
  -- Channel config
  constant sigma_c      : std_logic_vector(15 downto 0) := X"0040"; -- QU16.12

begin

  u_blinky : process(clk_s,arst_i)
  begin
    if arst_i = '1' then
      counter_s <= (others => '0');
    elsif rising_edge(clk_s) then
      counter_s <= std_logic_vector(unsigned(counter_s)+1);
    end if;
  end process;
  led_o <= counter_s(26 downto 23);

  u_clk_mmcm : clk_wiz_0
  port map (
    -- Clock out ports
    clk_out1 => clk_s,
    -- Status and control signals
    reset    => arst_i,
    locked   => clk_locked_s,
    -- Clock in ports
    clk_in1  => clk_i
  );
  -- clk_s <= clk_i;

  u_top : top_edu_bbt
  port map
  (
    clk_i  => clk_s,
    arst_i => arst_i,
    rx_i   => rx_i,
    tx_o   => tx_o,
    -- Config
    nm1_bytes_s => nm1_bytes_s,
    nm1_pre_s   => nm1_pre_s,
    nm1_sfd_s   => nm1_sfd_s,
    det_th_s    => det_th_s,
    pll_kp_s    => pll_kp_s,
    pll_ki_s    => pll_ki_s,
    -- Modem to channel
    mod_os_data_o => mod_os_data_s,
    mod_os_dv_o   => mod_os_dv_s,
    mod_os_rfd_i  => mod_os_rfd_s,
    -- Channel to Modem
    chan_os_data_i => chan_os_data_s,
    chan_os_dv_i   => chan_os_dv_s,
    chan_os_rfd_o  => chan_os_rfd_s
  );

  u_channel : bb_channel
  port map
  (
    -- clk, en, rst
    clk_i         => clk_s,
    en_i          => '1',
    srst_i        => arst_i,
    -- Input Stream
    is_data_i     => mod_os_data_s,
    is_dv_i       => mod_os_dv_s,
    is_rfd_o      => mod_os_rfd_s,
    -- Output Stream
    os_data_o     => chan_os_data_s,
    os_dv_o       => chan_os_dv_s,
    os_rfd_i      => chan_os_rfd_s,
    -- Control
    sigma_i       => sigma_c
  );

end architecture;

