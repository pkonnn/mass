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
dk = 1.1
dval= 0.6
sd_reg

Vd_reg = 3.14/4*(dk^2-dval^2)*sd_reg 

#Объем раб.лопаток для регулирующей ступени
F_atl
b2_nereg
b2_atl
l2_reg
z2

Vlop_reg = F_atl*l2_reg*z2*(b2_nereg/b2_atl)^2

#Объем бандажа для регулирующей ступени

B2
B_band = B2+0.01
delta_band = 0.005
dp

Vband_reg = B_band*delta_band*3.14*dp


#Объем лопаток нерег случайно начала
l_21
l_2z
l2_sr = (l_21+l_2z)/2
z_2z
z_21=(3.14*d_21)/(0.65*b2) #who is b2???
z_nereg



