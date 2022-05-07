import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

st.text("Конструкция ротора ЦВД")
z = 9
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
