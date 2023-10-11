# Anylitic model for the size of the simulation

class Simulation:
    def __init__(self, bytes_per_cell, padding_byte_ratio=1.37, compression_ratio=1.5):
        self.bytes_per_cell = bytes_per_cell
        self.padding_byte_ratio = padding_byte_ratio
        self.compression_ratio = compression_ratio

    def get_max_sim_size(self, gb):
        true_bytes_per_cell = self.bytes_per_cell * self.padding_byte_ratio / self.compression_ratio
        return 1024**3 * gb / true_bytes_per_cell / (1000**3)

# Model for the size of the simulation
lbm_simulation = Simulation(19*4)
em_simulation = Simulation(6*4)
fv_simulation = Simulation(5*4)
ideal_mhd_simulation = Simulation(8*4)
pic_simulation = Simulation(6*4)

# Machine sizes in GB
machine_sizes = [("Gaming PC (GB 128)", 128), ("DGX (TB 2)", 2048), ("Single GH (TB 0.65)", 650), ("GH200 SuperPod (TB 144)", 144*1024)]

# Generate latex table
print("\\begin{tabular}{|l|l|l|l|l|l|}")
print("\\hline")
print("Machine & LBM & FDTD EM & Hydro FV & Ideal MHD FV & Particle in Cell \\\\")
print("\\hline")
for machine in machine_sizes:
    print(f"{machine[0]} & {lbm_simulation.get_max_sim_size(machine[1]):.2f} & {em_simulation.get_max_sim_size(machine[1]):.2f} & {fv_simulation.get_max_sim_size(machine[1]):.2f} & {ideal_mhd_simulation.get_max_sim_size(machine[1]):.2f} & {pic_simulation.get_max_sim_size(machine[1]):.2f} \\\\")
    print("\\hline")
print("\\end{tabular}")








