import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from iapws import IAPWS97
from iapws import IAPWS97 as WSP
import numpy as np
import math as M
from sympy import *
from IPython.display import HTML, display

#Объем диска для регулирующей ступени
dk 
dval
sd_reg

Vd_reg = 3.14/4*(dk^2-dval^2)*sd_reg 

#Объем раб.лопаток для регулирующей ступени
F_atl
b2_reg
b2_atl
l2_reg
z2

Vlop_reg = F_atl*l2_reg*z2*(b2_reg/b2_atl)^2

#Объем бандажа для регулирующей ступени
B2
B_band = B2+0.01 #м
delta_band = 0.005 #м
dp

Vband_reg = B_band*delta_band*3.14*dp

#Масса диска рег.ступени
rho = 7800 #кг/м3
md_reg = (Vd_reg+Vlop_reg+Vband_reg)*rho





#Объем диска для нерегулирующей ступени
dk_n 
dval
sd_nereg
z_nereg

Vd_nereg = 3.14/4*(dk_n^2-dval_n^2)*sd_nereg*z_nereg 

#Объем раб.лопаток для нерегулирующей ступени
F_atl
b2_nereg
b2_atl
l_21
l_2z
l2_nereg = (l_21+l_2z)/2
z_2z
z_21=(3.14*d_21)/(0.65*b2) #who is b2???
Z2_sr = (z_21+z_2z)/2

Vlop_nereg = F_atl*l2_nereg*Z2_sr*z_nereg*(b2_nereg/b2_atl)^2

#Объем бандажа для нерегулирующей ступени
B2
B_band = B2+0.01 #м
delta_band = 0.005 #м
dp1
dpz
dp_sr = (dp1+dpz)/2

Vband_nereg = B_band*delta_band*3.14*dp_sr*z_nereg

#Масса диска нерег.ступени
md_nereg = (Vd_nereg+Vlop_nereg+Vband_nereg)*rho

#Масса вала
dsh_right
dsh_left #меньше правого
d_otv = 0.1 #м
L2
L3
L4

m_val = 3.14/4*((dval^2-d_otv^2)*L4-(dsh_left^2-d_otv^2)*L2-(dsh_right^2-d_otv^2)*L3)*rho




#Масса ротора
m_rot = m_val+md_nereg+md_reg
