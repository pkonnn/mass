import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

st.subheader('Определение массы ротора ЦВД')

st.write('_Конструкция ротора_')

z = st.number_input(Количество ступеней z', value = 9)
drs = 1100
rrs = 1100/2

d_val = 540
r_val = 540 / 2

d_st = 820
r_st = d_st/ 2

r_r = 410
r_r2 = r_val - 160

fig = plt.figure()

plt.plot([0,0],[0,rrs],c="g")
plt.plot([0,120],[rrs,rrs],c="g")
plt.plot([120,120],[rrs,0],c="g")
plt.plot([120,240],[r_val,r_val],c="g")
plt.plot([0,-540],[r_val,r_val],c="g")
n1 = 0
n2 = 0
k = 1

for i in range(z-1):
  if i % 2 != 0:
    plt.plot([240+n1,360+n1],[r_val,r_val],c="g")
    plt.plot([360+n1,360+n1],[0,r_r],c="g")
    plt.plot([360+n1,420+n1],[r_r,r_r],c="g")
    plt.plot([420+n1,420+n1],[r_r,0],c="g")
    n1 = n1 + 180
  else:
    plt.plot([-540-n2,-660-n2],[r_val,r_val],c="g")
    plt.plot([-660-n2,-660-n2],[0,r_r],c="g")
    plt.plot([-660-n2,-720-n2],[r_r,r_r],c="g")
    plt.plot([-720-n2,-720-n2],[r_r,0],c="g")
    n2 = n2 + 180
# Слева
plt.plot([-720-n2+180,-720-n2+180-1440],[r_val,r_val],c="g")
plt.plot([-720-n2+180-1440,-720-n2+180-1440],[r_val,0],c="g")
plt.plot([-720-n2+180-1440,-720-n2+180-1440-480],[r_r2,r_r2],c="g")
plt.plot([-720-n2+180-1440-480,-720-n2+180-1440-480],[r_r2,0],c="g")
# Справа
plt.plot([420+n1-180,420+n1-180+960],[r_val,r_val],c="g")
plt.plot([420+n1-180+960,420+n1-180+960],[r_val,0],c="g")
plt.plot([420+n1-180+960,420+n1-180+960+480],[r_r2,r_r2],c="g")
plt.plot([420+n1-180+960+480,420+n1-180+960+480],[r_r2,0],c="g")

plt.plot([-720-n2+180-1440-480-300,420+n1-180+960+480+300],[0,0],linewidth = '1',c="blue")
plt.grid()

st.pyplot(fig)



st.write('_Масса ротора_')

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
