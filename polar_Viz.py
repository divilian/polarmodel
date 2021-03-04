
import numpy as np
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer

from polar import CitizenAgent, SocialWorld

colors = ["Red","Green","Blue","Orange"]

if __name__ == "__main__":

    N = UserSettableParameter("slider","Number of agents (N)",20,1,50,5)
    T = UserSettableParameter("slider","Number of iterations (T)",20,1,500,10)
    I = UserSettableParameter("slider","Number of issues (I)",10,1,20,1)
    Cthresh = UserSettableParameter("slider","Comparison threshold",.5,0,1,.05)
    p = UserSettableParameter("slider","ER edge probability (p)",.2,0.05,1,.05)

    issueplots = [ ChartModule(
        [{"Label":"agent"+str(a)+"_iss"+str(i),"Color":colors[a]}
            for a in range(4) for i in range(3)],
        data_collector_name="datacollector") ]

    server = ModularServer(SocialWorld, issueplots,
        "Polarizers",
        { "N":N, "T":T, "I":I, "Cthresh": Cthresh, "ER_p":p })
        
    server.port = 8081
    server.launch()
