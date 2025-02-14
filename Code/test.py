import numpy as np
import matplotlib.pyplot as plt

#Node Key:
#   0 - Loop 1: Heater
#   1 - Loop 1: Top Leg
#   2 - Loop 1: Bottom Leg
#   3 - Exchanger 1
#   4 - Loop 2: Bottom Left Leg
#   5 - Loop 2: Top Left Leg
#   6 - Loop 2: Top Right Leg
#   7 - Loop 2: Bottom Right Leg
#   8 - Exchanger 1
#   9 - Loop 2: Bottom Input Leg
#   10- Loop 2: Top Output Leg


#Physical Parameters
num_nodes=11
node_masses=np.array([1,1,1,
                      1,
                      1,1,1,1,
                      1,
                      1,1])

heat_exchanger_coeff=np.array([1,1])
heat_exchanger_areas=np.array([1,1])

specific_heat_water=1
specific_heat_exchanger=1

specific_heats=np.array([specific_heat_water,specific_heat_water,specific_heat_water,
                         specific_heat_exchanger,
                         specific_heat_water,specific_heat_water,specific_heat_water,specific_heat_water,
                         specific_heat_exchanger,
                         specific_heat_water,specific_heat_water])

def calc_gradient(temperatures,
                  temp_interface,
                  power_in,
                  mass_flow_rates):
    """Calculates the temperature gradient with respect to time given state parameters."""
    gradient=np.zeros(len(temperatures))

    loop_1_out_exchange_rate = (heat_exchanger_coeff[0]*heat_exchanger_areas[0])*(temperatures[1]-temperatures[3])
    loop_2_in_exchange_rate  = (heat_exchanger_coeff[0]*heat_exchanger_areas[0])*(temperatures[3]-temperatures[4])
    loop_2_out_exchange_rate = (heat_exchanger_coeff[1]*heat_exchanger_areas[1])*(temperatures[6]-temperatures[8])
    loop_3_in_exchange_rate  = (heat_exchanger_coeff[1]*heat_exchanger_areas[1])*(temperatures[8]-temperatures[9])

    gradient[0]  = (mass_flow_rates[0]*specific_heat_water)* (temperatures[2]-temperatures[0])    +power_in
    gradient[1]  = (mass_flow_rates[0]*specific_heat_water)* (temperatures[0]-temperatures[1])    -loop_1_out_exchange_rate/2
    gradient[2]  = (mass_flow_rates[0]*specific_heat_water)* (temperatures[1]-temperatures[2])    -loop_1_out_exchange_rate/2

    gradient[3]  = loop_1_out_exchange_rate-loop_2_in_exchange_rate

    gradient[4]  = (mass_flow_rates[1]*specific_heat_water)* (temperatures[7]-temperatures[4])    +loop_2_in_exchange_rate/2
    gradient[5]  = (mass_flow_rates[1]*specific_heat_water)* (temperatures[4]-temperatures[5])    +loop_2_in_exchange_rate/2
    gradient[6]  = (mass_flow_rates[1]*specific_heat_water)* (temperatures[5]-temperatures[6])    -loop_2_out_exchange_rate/2
    gradient[7]  = (mass_flow_rates[1]*specific_heat_water)* (temperatures[6]-temperatures[7])    -loop_2_out_exchange_rate/2

    gradient[8]  = loop_2_out_exchange_rate-loop_3_in_exchange_rate

    gradient[9]  = (mass_flow_rates[2]*specific_heat_water)* (temp_interface-temperatures[9])     +loop_3_in_exchange_rate/2
    gradient[10] = (mass_flow_rates[2]*specific_heat_water)* (temperatures[9]-temperatures[10])   +loop_3_in_exchange_rate/2

    gradient=gradient/(node_masses*specific_heats)

    return(gradient)

#Simulation parameters
num_steps=5000
step_size=0.01

#Allocates memory space
times=np.linspace(0,num_steps*step_size,num_steps)
temperatures=np.zeros((num_steps,num_nodes))
temp_interfaces=np.zeros(num_steps)
power_ins=np.zeros(num_steps)
mass_flow_rates=np.zeros((num_steps,3))

#Parameters for generating fake input data
#Power input as ramp
max_power_in=0   #units
ramp_slope=10     #units/s
ramp_start_time=10 #s
#Constant flow rates
loop_1_flowrate=15
loop_2_flowrate=15
loop_3_flowrate=5
#Constant inlet temperature
room_temperature=26

#Defines initial conditions
temperatures[0]=np.ones(num_nodes)*room_temperature
power_ins[0]=0
temp_interfaces[0]=room_temperature-5
mass_flow_rates[0]=np.array([loop_1_flowrate,loop_2_flowrate,loop_3_flowrate])



for step_num in range(1,num_steps):
    #Step forward in time
    gradient=calc_gradient(temperatures[step_num-1],
                           temp_interfaces[step_num-1],
                           power_ins[step_num-1],
                           mass_flow_rates[step_num-1])
    
    #Updates values
    temperatures[step_num]=temperatures[step_num-1]+gradient*step_size

    #Read data
    power_ins[step_num]=np.max([0,np.min([max_power_in,max_power_in*(times[step_num]-0.5)])])
    temp_interfaces[step_num]=room_temperature-5
    mass_flow_rates[step_num]=np.array([loop_1_flowrate,loop_2_flowrate,loop_3_flowrate])


plt.plot(times,power_ins)
plt.title("Power Input")
plt.xlim(0,np.max(times))
plt.ylim(bottom=0)
plt.savefig("Code/Images/Power Input.png",dpi=1000)
plt.clf()

plt.plot(times,temp_interfaces)
plt.title("Input Coolant Temperature")
plt.xlim(0,np.max(times))
plt.ylim(bottom=0)
plt.savefig("Code/Images/Input Coolant Temperature.png",dpi=1000)
plt.clf()

plt.plot(times,mass_flow_rates[:,0],label="Loop 1 Flow Rate")
plt.plot(times,mass_flow_rates[:,1],label="Loop 2 Flow Rate")
plt.plot(times,mass_flow_rates[:,2],label="Loop 3 Flow Rate")
plt.title("Mass Flow Rates")
plt.xlim(0,np.max(times))
plt.ylim(bottom=0)
plt.legend()
plt.savefig("Code/Images/Mass Flow Rates.png",dpi=1000)
plt.clf()

plt.plot(times,temperatures[:,0],label="Node 1")
plt.plot(times,temperatures[:,1],label="node 2")
plt.plot(times,temperatures[:,2],label="Node 3")
plt.title("Loop 1 Temperatures")
plt.xlim(0,np.max(times))
plt.ylim(bottom=0)
plt.legend()
plt.savefig("Code/Images/Loop 1 Temperatures.png",dpi=1000)
plt.clf()

plt.plot(times,temperatures[:,4],label="Node 1")
plt.plot(times,temperatures[:,5],label="node 2")
plt.plot(times,temperatures[:,6],label="Node 3")
plt.plot(times,temperatures[:,7],label="Node 4")
plt.title("Loop 2 Temperatures")
plt.xlim(0,np.max(times))
plt.ylim(bottom=0)
plt.legend()
plt.savefig("Code/Images/Loop 2 Temperatures.png",dpi=1000)
plt.clf()

plt.plot(times,temperatures[:,9],label="Node 1")
plt.plot(times,temperatures[:,10],label="node 2")
plt.title("Loop 3 Temperatures")
plt.xlim(0,np.max(times))
plt.ylim(bottom=0)
plt.legend()
plt.savefig("Code/Images/Loop 3 Temperatures.png",dpi=1000)
plt.clf()

plt.plot(times,temperatures[:,3],label="Exchanger 1")
plt.plot(times,temperatures[:,8],label="Exchanger 2")
plt.title("Heat Exchanger Internal Temperatures")
plt.xlim(0,np.max(times))
plt.ylim(bottom=0)
plt.legend()
plt.savefig("Code/Images/Heat Exchanger Internal Temperatures.png",dpi=1000)
plt.clf()