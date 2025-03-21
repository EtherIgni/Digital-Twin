import numpy             as np
import matplotlib.pyplot as plt
from scipy.optimize  import curve_fit



specific_heat_water     = 4181   #J/Kg-K
specific_heat_exchanger = 385    #J/Kg-K
C_to_K                  = 273.15 #K
Gal_min_to_Kg_s         = 0.0466 #(Kg/s)/(Gal/min)
        
        
        
def calc_wrapper(power_curve,
                 mass_flow_rate, 
                 initial_temps,
                 inlet_temps):
    node_masses        = [2470,
                          36,
                          4,
                          0.073,
                          0.008,
                          1,
                          1,
                          1,
                          1,
                          3.17,
                          1,
                          1]
    coeff_e_1 = 26
    coeff_e_2 = 25
    heater_percent     = 0.8
    max_power          = 2760
    
    
    def calc_temps(time_steps):
        power_curve_adjusted = power_curve*max_power
        time_steps           = time_steps[:int(time_steps.size/7)]
        
        # node_masses[10] = mass_1
        # node_masses[11] = mass_2

        heat_exchanger_coeff = np.array([coeff_e_1,
                                         coeff_e_2])

        specific_heats = np.array([specific_heat_water,specific_heat_water,specific_heat_water,specific_heat_water,
                                   specific_heat_exchanger,
                                   specific_heat_water,specific_heat_water,specific_heat_water,specific_heat_water,
                                   specific_heat_exchanger,
                                   specific_heat_water,specific_heat_water])

        def calc_gradient(temperatures,
                          temp_interface,
                          power_in,
                          mass_flow_rates):
            """Calculates the temperature gradient with respect to time given state parameters."""
            gradient                 = np.zeros(len(temperatures))

            loop_1_out_exchange_rate = (heat_exchanger_coeff[0])*(temperatures[2]-temperatures[4])
            loop_2_in_exchange_rate  = (heat_exchanger_coeff[0])*(temperatures[4]-temperatures[5])
            loop_2_out_exchange_rate = (heat_exchanger_coeff[1])*(temperatures[7]-temperatures[9])
            loop_3_in_exchange_rate  = (heat_exchanger_coeff[1])*(temperatures[9]-temperatures[10])

            gradient[0]  = (mass_flow_rates[0]*specific_heat_water)* (temperatures[3]-temperatures[0])    +(1-heater_percent)*power_in
            gradient[1]  = (mass_flow_rates[0]*specific_heat_water)* (temperatures[0]-temperatures[1])    +    heater_percent*power_in
            gradient[2]  = (mass_flow_rates[0]*specific_heat_water)* (temperatures[1]-temperatures[2])    -loop_1_out_exchange_rate/2
            gradient[3]  = (mass_flow_rates[0]*specific_heat_water)* (temperatures[2]-temperatures[3])    -loop_1_out_exchange_rate/2

            gradient[4]  = loop_1_out_exchange_rate-loop_2_in_exchange_rate

            gradient[5]  = (mass_flow_rates[1]*specific_heat_water)* (temperatures[8]-temperatures[5])    +loop_2_in_exchange_rate/2
            gradient[6]  = (mass_flow_rates[1]*specific_heat_water)* (temperatures[5]-temperatures[6])    +loop_2_in_exchange_rate/2
            gradient[7]  = (mass_flow_rates[1]*specific_heat_water)* (temperatures[6]-temperatures[7])    -loop_2_out_exchange_rate/2
            gradient[8]  = (mass_flow_rates[1]*specific_heat_water)* (temperatures[7]-temperatures[8])    -loop_2_out_exchange_rate/2

            gradient[9]  = loop_2_out_exchange_rate-loop_3_in_exchange_rate

            gradient[10] = (mass_flow_rates[2]*specific_heat_water)* (temp_interface-temperatures[10])     +loop_3_in_exchange_rate/2
            gradient[11] = (mass_flow_rates[2]*specific_heat_water)* (temperatures[10]-temperatures[11])   +loop_3_in_exchange_rate/2

            gradient     = gradient/(node_masses*specific_heats)

            return(gradient)

        #Allocates memory space
        num_steps       = time_steps.size
        temperatures    = np.zeros((num_steps,12))

        #Defines initial conditions
        temperatures[0] = initial_temps



        for step_num in range(1, num_steps):
            #Step forward in time
            gradient = calc_gradient(temperatures[step_num-1],
                                     inlet_temps[step_num-1],
                                     power_curve_adjusted[step_num-1],
                                     mass_flow_rates[:,step_num-1])
            
            #Updates values
            temperatures[step_num] = temperatures[step_num-1] + gradient * (time_steps[step_num]-time_steps[step_num-1])
        
        temperatures = temperatures[:,[0,1,2,3,
                                       7,8,
                                       11]]
        
        temperatures_combined = np.concat(temperatures.T)
        
        return(temperatures_combined)
    return(calc_temps)
        

input_data       = np.genfromtxt("Data/Input_Filtered.txt",      delimiter=" ").reshape((5,-1))
flow_data        = np.genfromtxt("Data/Flow_Meter_Filtered.txt", delimiter=" ").reshape((4,-1))
temperature_data = np.genfromtxt("Data/Temp_Filtered.txt",       delimiter=" ").reshape((9,-1))

power_curve  = np.interp(temperature_data[-1],input_data[-1],input_data[3])*2
flow_curve_1 = np.interp(temperature_data[-1],flow_data[-1],flow_data[0])
flow_curve_2 = np.interp(temperature_data[-1],flow_data[-1],flow_data[1])
flow_curve_3 = np.interp(temperature_data[-1],flow_data[-1],flow_data[2])

power_curve = power_curve/10

mass_flow_rates = np.array([flow_curve_1,
                            flow_curve_2,
                            flow_curve_3])
mass_flow_rates = mass_flow_rates * Gal_min_to_Kg_s

time = temperature_data[-1]/1000

temperature_data = temperature_data[:-1] + C_to_K



# for i in range(3):
#     plt.plot(time, mass_flow_rates[i], label=str(i))
# plt.legend()
# plt.show()

# plt.plot(time, power_curve)
# plt.show()

# for i in range(8):
#     plt.plot(time, temperature_data[i], label=str(i))
# plt.legend()
# plt.show()



initial_temps      = np.zeros(12)
initial_temps[0]   = temperature_data[0,0]
initial_temps[1]   = temperature_data[1,0]
initial_temps[2:4] = temperature_data[2:4,0]
initial_temps[4]   = np.mean(temperature_data[2:4,0])
initial_temps[5:7] = temperature_data[2:4,0]
initial_temps[7:9] = temperature_data[4:6,0]
initial_temps[9]   = np.mean(temperature_data[4:8,0])
initial_temps[10:] = temperature_data[6:8,0]

temperature_calculator = calc_wrapper(power_curve,
                                      mass_flow_rates,
                                      initial_temps,
                                      temperature_data[6])

time_combined      = np.tile(time,7)
temps_to_compare   = temperature_data[[0,1,2,3,4,5,7]]
temp_data_combined = np.concat(temps_to_compare)
# print(temp_data_combined)



# best_fit_param, ___ = curve_fit(temperature_calculator,time_combined,temp_data_combined,[1,1])

# print(best_fit_param)



data = temperature_calculator(time_combined)

data = data.reshape((-1,7),order="F")

colors=["saddlebrown",
        "red",
        "orange",
        "gold",
        "blue",
        "cyan",
        "green"]
node_names=["1-1",
            "1-2",
            "1-3",
            "1-4",
            "2-1",
            "2-2",
            "Out"]



for i in range(7):
    plt.plot(time, data[:,i]-C_to_K, color=colors[i], label="Calc: "+node_names[i], linestyle="dashed")
    plt.plot(time, temps_to_compare[i]-C_to_K, color=colors[i], label="True: "+node_names[i])
plt.plot(time, temperature_data[6]-C_to_K, color="black", label="True: Inlet", linestyle="dotted")
plt.legend()
plt.xlabel("Time (s)",fontsize=20)
plt.ylabel("Temperature (C)",fontsize=20)
plt.title("Model Vs True Data",fontsize=20)
plt.show()

for i in range(7):
    plt.plot(time, temps_to_compare[i]-C_to_K, color=colors[i], label="Node "+node_names[i])
plt.plot(time, temperature_data[6]-C_to_K, color="black", label="Node Inlet", linestyle="dotted")
plt.legend()
plt.xlabel("Time (s)",fontsize=20)
plt.ylabel("Temperature (C)",fontsize=20)
plt.title("Temperature Data",fontsize=20)
plt.show()

for i in range(3):
    plt.plot(time, mass_flow_rates[i], color=colors[i], label="Loop "+str(i+1))
plt.legend()
plt.xlabel("Time (s)",fontsize=20)
plt.ylabel("Flow Rate (Kg/s)",fontsize=20)
plt.title("Mass Flow Rates",fontsize=20)
plt.show()

plt.plot(time, power_curve, color="orange")
plt.xlabel("Time (s)",fontsize=20)
plt.ylabel("Power In (% of max)",fontsize=20)
plt.title("Percent Power In",fontsize=20)
plt.show()