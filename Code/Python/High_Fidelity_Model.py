import numpy as np
import matplotlib.pyplot as plt



#Constants
C_to_K                    = 273.15

exchange_constants_1      = np.array([0.001,0.001,0.001])

specific_heat_water       = 4181      # J/Kg-K



#Time Grid Initialization
start_time                = 0         # s
end_time                  = 10        # s
time_steps                = 1000     # num steps

time                      = np.linspace(start_time,end_time,time_steps+1)



#Input Values Initialization
heat_rate_in_constant     = 500       # J/s
heat_rate_in              = np.ones(time_steps+1,float)*heat_rate_in_constant

mass_flow_1_constant      = 0.3       # Kg/s
mass_flow_1               = np.ones(time_steps+1,float)*mass_flow_1_constant

mass_flow_2_constant      = 0.2       # Kg/s
mass_flow_2               = np.ones(time_steps+1,float)*mass_flow_2_constant

inlet_temp_constant       = 16+C_to_K # K
inlet_temps               = np.ones(time_steps+1,float)*inlet_temp_constant



#Loop Data Initialization
initial_temps_constant    = 18+C_to_K # K

temps_1                   = np.zeros((4,time_steps+1),float)
temps_1[:,0]              = np.ones(4)*initial_temps_constant

temps_2                   = np.zeros((1,time_steps+1),float)
temps_2[:,0]              = np.ones(1)*inlet_temp_constant



np.seterr(all='ignore')
#Calculation Loop
for i in range(1,time_steps+1):
    exchange_coeff        = exchange_constants_1[0]*np.power(mass_flow_1[i-1], -0.8) + exchange_constants_1[1]*np.power(mass_flow_2[i-1], -0.8) + exchange_constants_1[2]
    
    heat_rate_exchanger   = (1/exchange_coeff) * ((temps_1[2,i-1]-temps_2[0,i-1]) - (temps_1[3,i-1]-inlet_temps[i-1])) / (np.log(temps_1[2,i-1]-temps_2[0,i-1]) -np.log(temps_1[3,i-1]-inlet_temps[i-1]))
    if(np.isnan(heat_rate_exchanger)):
        heat_rate_exchanger = 0
    
    temp_rate_heater      = heat_rate_in[i-1]   / (mass_flow_1[i-1]*specific_heat_water)
    temp_rate_exchanger_1 = heat_rate_exchanger / (mass_flow_1[i-1]*specific_heat_water)
    temp_rate_exchanger_2 = heat_rate_exchanger / (mass_flow_2[i-1]*specific_heat_water)
    
    
    
    temps_1[0,i] = temps_1[3,i-1]
    temps_1[1,i] = temps_1[0,i-1]   + temp_rate_heater 
    temps_1[2,i] = temps_1[1,i-1]
    temps_1[3,i] = temps_1[2,i-1]   - temp_rate_exchanger_1
    
    temps_2[0,i] = inlet_temps[i-1] + temp_rate_exchanger_2
    


plt.plot(time, temps_1[0]-C_to_K,  label="1 - 0", color="saddlebrown")
plt.plot(time, temps_1[1]-C_to_K,  label="1 - 1", color="red")
plt.plot(time, temps_1[2]-C_to_K,  label="1 - 2", color="orange")
plt.plot(time, temps_1[3]-C_to_K,  label="1 - 3", color="gold")
plt.plot(time, inlet_temps-C_to_K, label="2 - 0", color="navy")
plt.plot(time, temps_2[0]-C_to_K,  label="2 - 1", color="cyan")

plt.legend()
plt.xlim(np.min(time), np.max(time))
plt.title("Simulated Temperatures")
plt.show()