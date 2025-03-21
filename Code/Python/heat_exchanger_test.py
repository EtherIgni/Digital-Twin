import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

heat_exchange_coeff = 1
specific_heat       = 1
density             = 1

pipe_area_1         = 1
pipe_area_2         = 1

exchanger_size      = 5
exchanger_area_1    = 1
exchanger_area_2    = 1

num_seconds         = 5


num_time_steps      = 1000
num_spacial_steps   = 100



time_node_space                     = np.linspace(0,num_seconds,num_seconds*num_time_steps+1)
space_node_space                    = np.linspace(0,exchanger_size,num_spacial_steps+1)

# inlet_temps_1                       = np.ones(num_time_steps*num_seconds+1)*25
# inlet_temps_1[:num_time_steps]      = np.linspace(20,25,num_time_steps,False) 
# inlet_temps_2                       = np.ones(num_time_steps*num_seconds+1)*30
# inlet_temps_2[:num_time_steps]      = np.linspace(25,30,num_time_steps,False)

inlet_temps_1                       = np.linspace(25,30,num_time_steps*num_seconds+1)
inlet_temps_2                       = np.linspace(20,25,num_time_steps*num_seconds+1)

mass_flow_rate_1                    = np.ones(num_time_steps*num_seconds+1)*7
mass_flow_rate_1[num_time_steps*3:] = np.ones(num_time_steps*(num_seconds-3)+1)*3
mass_flow_rate_2                    = np.ones(num_time_steps*num_seconds+1)*10
mass_flow_rate_2[num_time_steps*3:] = np.ones(num_time_steps*(num_seconds-3)+1)*5

# mass_flow_rate_1                    = np.ones(num_time_steps*num_seconds+1)*7
# mass_flow_rate_2                    = np.ones(num_time_steps*num_seconds+1)*10

nodes_1                             = np.ones((num_time_steps*num_seconds+1,num_spacial_steps+1))*inlet_temps_1[0]
nodes_2                             = np.ones((num_time_steps*num_seconds+1,num_spacial_steps+1))*inlet_temps_2[0]


def exp(x,a,b,c,d):
    return(np.exp((x-d)*a)*b+c)

for i in range(num_time_steps*num_seconds):
    #Get Velocities
    velocity_1       = mass_flow_rate_1[i]/(density*pipe_area_1)
    velocity_2       = mass_flow_rate_2[i]/(density*pipe_area_2)
    
    nodes_1[i+1]      = np.interp(space_node_space, space_node_space+velocity_1/num_time_steps, nodes_1[i])
    nodes_1[i+1,0]    = inlet_temps_1[i+1]
    nodes_2[i+1]      = np.interp(space_node_space, space_node_space-velocity_2/num_time_steps, nodes_2[i])
    nodes_2[i+1,-1]   = inlet_temps_2[i+1]
    
    heat_flux         = heat_exchange_coeff*(nodes_1[i+1]-nodes_2[i+1])
    heat_transferred  = ((heat_flux[:-1]+heat_flux[1:])/2)*(exchanger_size/num_spacial_steps)

    nodes_1[i+1,1:]   = nodes_1[i+1,1:] - heat_transferred*specific_heat*(mass_flow_rate_1[i]/num_time_steps)
    nodes_2[i+1,1:]   = nodes_2[i+1,1:] + heat_transferred*specific_heat*(mass_flow_rate_2[i]/num_time_steps)



# fig, axes = plt.subplots(2)

# axes[0].plot(time_node_space, nodes_1[:,0],  label="Inlet")
# axes[0].plot(time_node_space, nodes_1[:,-1], label="Outlet")
# axes[0].legend()
# axes[0].set_title("Size 2")
# axes[0].set_xlim(0,num_seconds)

# axes[1].plot(time_node_space, nodes_2[:,0],  label="Inlet")
# axes[1].plot(time_node_space, nodes_2[:,-1], label="Outlet")
# axes[1].legend()
# axes[1].set_title("Side 2")
# axes[1].set_xlim(0,num_seconds)

# plt.show()



fig, ax = plt.subplots(3)
fig.set_figheight(20)
fig.set_figwidth(7)
  
def gif_Function(frame):
    step=frame*25
    if(step>num_time_steps*num_seconds):
        step=num_time_steps*num_seconds
    
    fig.suptitle("Time: "+str(step/num_time_steps))
    
    ax[0].cla()
    ax[0].plot(space_node_space, nodes_1[step], color="gold",   label="Side 1")
    ax[0].plot(space_node_space, nodes_2[step], color="purple", label="Side 2")
    ax[0].legend()
    ax[0].set_title("Temperature Distribution Across Exchanger")
    ax[0].set_xlim(0,exchanger_size)
    
    ax[1].cla()
    ax[1].plot(time_node_space[:step], nodes_1[:step,0],  color="red",    label="Inlet Side 1")
    ax[1].plot(time_node_space[:step], nodes_1[:step,-1], color="orange", label="Outlet Side 2")
    ax[1].plot(time_node_space[:step], nodes_2[:step,0],  color="blue",   label="Inlet Side 2")
    ax[1].plot(time_node_space[:step], nodes_2[:step,-1], color="green",  label="Outlet Side 2")
    ax[1].legend()
    ax[1].set_title("Inlet and Outlet Temperatures")
    ax[1].set_xlim(0,num_seconds)
    
    ax[2].cla()
    ax[2].plot(time_node_space[:step], mass_flow_rate_1[:step], color="grey",  label="Side 1")
    ax[2].plot(time_node_space[:step], mass_flow_rate_2[:step], color="brown", label="Side 2")
    ax[2].legend()
    ax[2].set_title("Mass Flow Rates")
    ax[2].set_xlim(0,num_seconds)

gif    = FuncAnimation(fig, gif_Function, frames=int((num_time_steps*(num_seconds+1))/25), interval=25)
writer = PillowWriter(fps=25,
                      metadata=dict(artist='Me'),
                      bitrate=1800)
gif.save('Code/Images/New_Exchanger_Model.gif', writer=writer)
# plt.show()