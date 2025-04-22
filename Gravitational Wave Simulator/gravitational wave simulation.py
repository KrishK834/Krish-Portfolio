import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

class GravitationalWaveSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Gravitational Wave Simulator")
        self.root.geometry("1280x900")
        
        self.G = 6.67430e-11  
        self.c = 2.99792458e8 
        self.Msun = 1.989e30  
        self.Mpc = 3.0857e22  
        
        self.input_frame = ttk.Frame(root, padding="10")
        self.input_frame.pack(fill="x", expand=False)
        
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill="both", expand=True)
        
        
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.4)
        
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax1.set_title("Gravitational Waveform", fontsize=12)
        self.ax1.set_xlabel("Time (s)", fontsize=10)
        self.ax1.set_ylabel("Strain h(t)", fontsize=10)
        self.ax1.tick_params(axis='both', which='major', labelsize=8)
        
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax2.set_title("Binary Orbit", fontsize=12)
        self.ax2.set_xlabel("x (km)", fontsize=10)
        self.ax2.set_ylabel("y (km)", fontsize=10)
        self.ax2.set_aspect('equal')
        self.ax2.tick_params(axis='both', which='major', labelsize=8)
        
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax3.set_title("Frequency Evolution", fontsize=12)
        self.ax3.set_xlabel("Time (s)", fontsize=10)
        self.ax3.set_ylabel("Frequency (Hz)", fontsize=10)
        self.ax3.tick_params(axis='both', which='major', labelsize=8)
        
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax4.set_title("Power Spectrum", fontsize=12)
        self.ax4.set_xlabel("Frequency (Hz)", fontsize=10)
        self.ax4.set_ylabel("Power", fontsize=10)
        self.ax4.tick_params(axis='both', which='major', labelsize=8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.create_input_widgets()
        
    def create_input_widgets(self):

        input_grid = ttk.Frame(self.input_frame)
        input_grid.pack(fill="x", expand=True)
        
        ttk.Label(input_grid, text="System Type:", width=25).grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.system_type = tk.StringVar(value="BH-BH")
        system_combo = ttk.Combobox(input_grid, textvariable=self.system_type, width=15)
        system_combo['values'] = ("BH-BH", "NS-NS", "BH-NS")
        system_combo.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        system_combo.bind('<<ComboboxSelected>>', self.update_mass_ranges)
        
        ttk.Label(input_grid, text="Mass 1 (Solar Masses):", width=25).grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.mass1_var = tk.DoubleVar(value=30.0)
        self.mass1_entry = ttk.Entry(input_grid, textvariable=self.mass1_var, width=15)
        self.mass1_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(input_grid, text="Mass 2 (Solar Masses):", width=25).grid(row=1, column=2, padx=10, pady=5, sticky="e")
        self.mass2_var = tk.DoubleVar(value=25.0)
        self.mass2_entry = ttk.Entry(input_grid, textvariable=self.mass2_var, width=15)
        self.mass2_entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        
        ttk.Label(input_grid, text="Distance (Mpc):", width=25).grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.distance_var = tk.DoubleVar(value=100.0)
        self.distance_entry = ttk.Entry(input_grid, textvariable=self.distance_var, width=15)
        self.distance_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(input_grid, text="Lower Frequency (Hz):", width=25).grid(row=2, column=2, padx=10, pady=5, sticky="e")
        self.f_lower_var = tk.DoubleVar(value=15.0)
        self.f_lower_entry = ttk.Entry(input_grid, textvariable=self.f_lower_var, width=15)
        self.f_lower_entry.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        
        button_frame = ttk.Frame(self.input_frame)
        button_frame.pack(pady=10)
        
        self.simulate_btn = ttk.Button(button_frame, text="Simulate", command=self.simulate, width=20)
        self.simulate_btn.pack(pady=5)
    
        self.status_var = tk.StringVar(value="Ready to simulate")
        self.status_label = ttk.Label(self.input_frame, textvariable=self.status_var, font=('Arial', 10))
        self.status_label.pack(pady=5)
    
    def update_mass_ranges(self, event=None):
        system = self.system_type.get()
        
        if system == "BH-BH":
            self.mass1_var.set(30.0)
            self.mass2_var.set(25.0)
        elif system == "NS-NS":
            self.mass1_var.set(1.4)
            self.mass2_var.set(1.4)
        elif system == "BH-NS":
            self.mass1_var.set(10.0)  
            self.mass2_var.set(1.4)
    
    
    def compute_chirp_mass(self, m1, m2):
        return ((m1 * m2)**(3/5)) / ((m1 + m2)**(1/5))
    
    def time_to_merger(self, mc, f_values):

        f_values = np.asarray(f_values) 
        result = np.full_like(f_values, np.inf, dtype=np.float64) 
        positive_freq_mask = f_values > 0
        
        if np.any(positive_freq_mask):
            f_positive = f_values[positive_freq_mask]
            
            term = (5/256) * (self.c**5 / (self.G * mc)**(5/3)) * (np.pi * f_positive)**(-8/3)
            result[positive_freq_mask] = term
        
        if f_values.ndim == 0:
            return result.item()
        else:
            return result
    
    def gw_frequency_from_time(self, t_to_merger, mc):

        if t_to_merger <= 0:
            return np.inf 
        
        term = (5/256) * (self.c**5 / (self.G * mc)**(5/3)) / t_to_merger
        return (1/np.pi) * term**(3/8)
    
    def gw_strain_amplitude(self, mc, f, d):

        f = np.asarray(f) 
        
        term1 = 4 * (self.G * mc / self.c**2)**(5/3) 
        term2 = np.power(np.pi * f / self.c, 2/3, where=f>0, out=np.zeros_like(f, dtype=float))
        amplitude = term1 * term2 / (d / self.c)
        
        if f.ndim == 0:
            return amplitude.item()
        else:
            return amplitude
    
    def calculate_final_mass_spin(self, m1, m2):
        mtot = m1 + m2
        eta = (m1 * m2) / (mtot**2)  

        mf = mtot * (1 + (np.sqrt(8/9) - 1)*eta - 0.4333*eta**2 - 0.4392*eta**3)
        
        af = eta * np.sqrt(12) - 3.871*eta**2 + 4.028*eta**3
        
        return mf, min(af, 0.99)
        
    def simulate(self):
        try:
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            system_type = self.system_type.get()
            m1_solar = self.mass1_var.get()
            m2_solar = self.mass2_var.get()
            distance_mpc = self.distance_var.get()
            f_lower = self.f_lower_var.get()
            
            m1 = m1_solar * self.Msun
            m2 = m2_solar * self.Msun
            distance = distance_mpc * self.Mpc
            
            self.status_var.set(f"Simulating {system_type} merger...")
            
            mc = self.compute_chirp_mass(m1, m2)
            mtot = m1 + m2
            q = min(m1, m2) / max(m1, m2)  
            
            mf, af = self.calculate_final_mass_spin(m1, m2)
            
            r_s = 2 * self.G * mtot / (self.c ** 2)
            
            r_isco_approx = 6 * self.G * mtot / self.c**2
            f_isco_orbital_approx = (1 / (2 * np.pi)) * np.sqrt(self.G * mtot / r_isco_approx**3)
            f_upper_approx = 2 * f_isco_orbital_approx

            f_upper = max(f_lower + 1, f_upper_approx * 0.95)
            
            t_inspiral = self.time_to_merger(mc, f_lower) - self.time_to_merger(mc, f_upper)
            
            if t_inspiral <= 0 or not np.isfinite(t_inspiral):
                self.status_var.set("Error: Invalid inspiral duration calculated")
                return
            
            num_points = 10000
            frequencies = np.linspace(f_lower, f_upper, num_points)
            
            t_remaining = self.time_to_merger(mc, frequencies)
            
            time = t_remaining - t_remaining[-1]
            time = -time[::-1]  
            frequencies = frequencies[::-1]  
            
            valid_indices = np.isfinite(time) & np.isfinite(frequencies) & (frequencies > 0)
            time = time[valid_indices]
            frequencies = frequencies[valid_indices]
            
            if len(time) < 2:
                self.status_var.set("Error: Not enough valid time points generated")
                return
            
            amplitude = self.gw_strain_amplitude(mc, frequencies, distance)
            
            dt = np.diff(time, prepend=time[0] - (time[1]-time[0] if len(time)>1 else 0))
            dt = np.abs(dt)
            phase = np.cumsum(2 * np.pi * frequencies * dt)  
            phase = phase - phase[-1] 
        
            strain = amplitude * np.cos(phase)
            
            x1 = np.zeros_like(time)
            y1 = np.zeros_like(time)
            x2 = np.zeros_like(time)
            y2 = np.zeros_like(time)
            
            mu = m2 / (m1 + m2)

            r = np.zeros_like(time)
            for i, t in enumerate(time):
                t_to_merger = -t  
                if t_to_merger > 0:
                    r[i] = (256/5 * self.G**3 * mtot * mc**2 / self.c**5 * t_to_merger)**(1/4)
                else:
                    r[i] = r_s
            
            for i in range(len(time)):
                if r[i] > r_s:  
                    x1[i] = -mu * r[i] * np.cos(phase[i]/2)
                    y1[i] = -mu * r[i] * np.sin(phase[i]/2)
                    x2[i] = (1-mu) * r[i] * np.cos(phase[i]/2 + np.pi)
                    y2[i] = (1-mu) * r[i] * np.sin(phase[i]/2 + np.pi)
            
            fft_strain = np.fft.rfft(strain)
            power = np.abs(fft_strain)**2
            freqs = np.fft.rfftfreq(len(strain), np.mean(np.diff(time)))
            
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            self.ax1.set_title("Gravitational Waveform", fontsize=12)
            self.ax1.set_xlabel("Time (s)", fontsize=10)
            self.ax1.set_ylabel("Strain h(t)", fontsize=10)
            self.ax1.tick_params(axis='both', which='major', labelsize=8)
            self.ax1.grid(True, alpha=0.3)
            
            self.ax2.set_title("Binary Orbit", fontsize=12)
            self.ax2.set_xlabel("x (km)", fontsize=10)
            self.ax2.set_ylabel("y (km)", fontsize=10)
            self.ax2.tick_params(axis='both', which='major', labelsize=8)
            self.ax2.set_aspect('equal')
            self.ax2.grid(True, alpha=0.3)
            
            self.ax3.set_title("Frequency Evolution", fontsize=12)
            self.ax3.set_xlabel("Time (s)", fontsize=10)
            self.ax3.set_ylabel("Frequency (Hz)", fontsize=10)
            self.ax3.tick_params(axis='both', which='major', labelsize=8)
            self.ax3.grid(True, alpha=0.3)
            
            self.ax4.set_title("Power Spectrum", fontsize=12)
            self.ax4.set_xlabel("Frequency (Hz)", fontsize=10)
            self.ax4.set_ylabel("Power", fontsize=10)
            self.ax4.tick_params(axis='both', which='major', labelsize=8)
            self.ax4.grid(True, alpha=0.3)
            
            self.ax1.plot(time, strain, color='blue', linewidth=1.5)
            self.ax1.ticklabel_format(axis='y', style='sci', scilimits=(-23,23))
            
            step = max(1, len(time) // 1000)
            
            self.ax2.plot(x1[::step]/1000, y1[::step]/1000, 'r-', linewidth=1, alpha=0.5)
            self.ax2.plot(x2[::step]/1000, y2[::step]/1000, 'b-', linewidth=1, alpha=0.5)
            
            max_orbit_size = np.max(np.abs(np.concatenate([x1, y1, x2, y2]))) / 1000 * 1.1
            self.ax2.set_xlim(-max_orbit_size, max_orbit_size)
            self.ax2.set_ylim(-max_orbit_size, max_orbit_size)
            
            valid_idx = np.where(r > r_s)[0]
            if len(valid_idx) > 0:
                last_valid = valid_idx[-1]
                
                size1 = 8 * np.power(m1/self.Msun, 1/3)
                size2 = 8 * np.power(m2/self.Msun, 1/3)
                self.ax2.plot(x1[last_valid]/1000, y1[last_valid]/1000, 'ro', markersize=size1)
                self.ax2.plot(x2[last_valid]/1000, y2[last_valid]/1000, 'bo', markersize=size2)
            
            self.ax3.plot(time, frequencies, color='green', linewidth=1.5)
            self.ax3.set_yscale('log')  
            
            self.ax4.plot(freqs, power/np.max(power), color='purple', linewidth=1.5)
            
            if len(power) > 0:
                max_freq_idx = min(np.argmax(power) * 3, len(freqs) - 1)
                self.ax4.set_xlim(0, freqs[max_freq_idx])
            
            info_text = f"System: {system_type}\n" \
                        f"Mass 1: {m1_solar:.1f} M☉\n" \
                        f"Mass 2: {m2_solar:.1f} M☉\n" \
                        f"Chirp Mass: {mc/self.Msun:.1f} M☉\n" \
                        f"Final Mass: {mf/self.Msun:.1f} M☉\n" \
                        f"Final Spin: {af:.2f}\n" \
                        f"Distance: {distance_mpc:.1f} Mpc\n" \
                        f"Max Strain: {np.max(np.abs(strain)):.2e}\n" \
                        f"Peak Freq: {np.max(frequencies):.1f} Hz\n" \
                        f"Inspiral: {t_inspiral:.3f} s"
            
            info_box = dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8)
            self.ax2.text(0.05, 0.95, info_text, transform=self.ax2.transAxes, 
                         fontsize=9, verticalalignment='top', bbox=info_box)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.status_var.set(f"Simulation complete - {system_type} merger")
            
            warnings.filterwarnings('default', category=RuntimeWarning)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error during simulation: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GravitationalWaveSimulator(root)
    root.mainloop()