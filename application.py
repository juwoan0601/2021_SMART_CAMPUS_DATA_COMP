from tkinter import *
import time
import random
import os
import pickle
import numpy as np
import config

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

window = Tk()
window.title('POSTECH HEAD-COUNT')
win_WIDTH = 1200
win_HEIGHT = 500

SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS
DEFAULT_VALUES = [2022,1,3,17,30,4,  0,0,0,0,0,0,0,0,0,0,0,0,    25,0,0,0,0,0,0,0,0,0,0,0,   0,0,0,0]

input_params = []
for i in range(len(SELECTED_COLUMNS)):
    var = StringVar()
    var.set(DEFAULT_VALUES[i])
    Label(window, text = SELECTED_COLUMNS[i]).grid(row = i+1,column = 0)
    Entry(window, textvariable=var).grid(row = i+1,column = 1)
    input_params.append(var)

result_text = Label(window, text = "Result is Empty")
result_text.grid(row = len(SELECTED_COLUMNS)+2,column = 0,columnspan=5)

def predict_m1():
    MODEL_PATH = './saved_model/tpot_34_column_26.19.pkl'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
    a = [2022,1,3,17,30,4,  0,0,0,0,0,0,0,0,0,0,0,0,    25,0,0,0,0,0,0,0,0,0,0,0,   0,0,0,0]
    input = np.reshape(np.array(a),(1,-1))
    result = loaded_model.predict(input)
    print("예상 식수인원은 ", int(result), "명 입니다.")
    result_text.configure(text = "예상 식수인원은 "+str(int(result))+"명 입니다.")

def predict_m2():
    MODEL_PATH = './saved_model/XGBoost_29.79.pkl'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
    params = [float(input_params[i].get()) for i in range(len(SELECTED_COLUMNS))]
    params = np.reshape(np.array(params),(1,-1))
    result = loaded_model.predict(params)
    result_week = [[],[],[]]
    result_month = [[],[],[]]
    time = [7,11,17]
    for j in range(3):
        for i in range(7):
            temp = params
            temp[0,3] = time[j]
            temp[0,5] = (temp[0,5]+1)%7+1
            result_week[j].append(loaded_model.predict(temp))
        for i in range(28):
            temp = params
            temp[0,3] = time[j]
            temp[0,5] = (temp[0,5]+1)%7+1
            temp[0,2] = (temp[0,2]+1)%31+1
            result_month[j].append(loaded_model.predict(temp))
    result_text.configure(text = "예상 식수인원은 "+str(int(result))+"명 입니다.")
    plotWeekLine(result_week)
    plotMonthLine(result_month)

def plotWeekLine(y):
    plt.grid(True)
    fig = plt.figure()
    ax1 = plt.subplot(111, xlim=(0, 6), ylim=(0, 300))
    ax1.plot(np.arange(7),y[0], lw=1, c='green',ms=1)
    ax2 = plt.subplot(111, xlim=(0, 6), ylim=(0, 300))
    ax2.plot(np.arange(7),y[1], lw=1, c='blue',ms=1)
    ax3 = plt.subplot(111, xlim=(0, 6), ylim=(0, 300))
    ax3.plot(np.arange(7),y[2], lw=1, c='red',ms=1)
    ax1.set_title("Head Count for a Week")
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(column=3,row=1,rowspan=int(len(SELECTED_COLUMNS)/2))

def plotMonthLine(y):
    plt.grid(True)
    fig = plt.figure()
    ax1 = plt.subplot(111, xlim=(0, 27), ylim=(0, 300))
    ax1.plot(np.arange(28),y[0], lw=1, c='green',ms=1)
    ax2 = plt.subplot(111, xlim=(0, 27), ylim=(0, 300))
    ax2.plot(np.arange(28),y[1], lw=1, c='blue',ms=1)
    ax3 = plt.subplot(111, xlim=(0, 27), ylim=(0, 300))
    ax3.plot(np.arange(28),y[2], lw=1, c='red',ms=1)
    ax1.set_title("Head Count for a Month")
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(column=3,row=1+int(len(SELECTED_COLUMNS)/2),rowspan=int(len(SELECTED_COLUMNS)/2))

b_s1 = Button(window,text='Search!',command=predict_m1).grid(row = len(SELECTED_COLUMNS)+1,column = 0)
b_s2 = Button(window,text='Search!',command=predict_m2).grid(row = len(SELECTED_COLUMNS)+1,column = 1)

window.mainloop()