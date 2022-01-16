from tkinter import *
from tkinter import filedialog # import filedialog module
import time
import random
import os
import pickle
import numpy as np
from datetime import datetime
import copy

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DATE_COLUMNS = [
    'Year', 
    'Month', 
    'Day', 
    'Hour',
    'Week N'
]

MENU_COLUMNS = [
    'pork', 
    'beef', 
    'chicken', 
    'duck', 
    'processed meat', 
    'molluscs', 
    'fish', 
    'egg', 
    'dairy', 
    'tuna', 
    'shrimp', 
    'special', 
]

WEATHER_COLUMNS = [
    'temperature(C)', 
    'precipitation(mm)', 
    'wind dir(deg)', 
    'wind speed(m/s)', 
    'spot atmospheric pressure(hPa)', 
    'sea-level pressure(hPa)', 
    'humidity(%)', 
    'sun radiation(MJ/m2)', 
    'bright sunshine(Sec)', 
    'fine dust concentration(ug/m3)', 
    'Cold wave warning', 
    'Heat wave warning',
    'Typoon'
]

SCHEDULE_COLUMNS = [
    'exam', 
    'class registration', 
    'make-up-class', 
    'vacation',
]

MENU_COLUMNS_KR = [
    '돼지고기', 
    '소고기', 
    '닭고기', 
    '오리고기', 
    '다진 고기', 
    '유제품', 
    '생선', 
    '달걀', 
    '채소', 
    '연어', 
    '새우', 
    '특식', 
]

SCHEDULE_COLUMNS_KR = [
    '시험 기간', 
    '수강 신청', 
    '보충 수업', 
    '방학',
]

WEATHER_COLUMNS_KR = [
    '기온(C)', 
    '강수량(mm)', 
    '풍향(deg)', 
    '풍속(m/s)', 
    '기압(hPa)', 
    '해수면기준 기압(hPa)', 
    '상대습도(%)', 
    '일조량(MJ/m2)', 
    '일조 시간(Sec)', 
    '미세먼지 농도(ug/m3)', 
    '한파', 
    '폭염',
    '태풍'
]

class EntryWithPlaceholder(Entry):
    #https://stackoverflow.com/questions/27820178/how-to-add-placeholder-to-an-entry-in-tkinter
    def __init__(self, master=None, placeholder="PLACEHOLDER", color='grey', textvariable=None):
        super().__init__(master)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']
        self.textvariable = textvariable

        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)

        self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

    def foc_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self['fg'] = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()

window = Tk()
window.title('POSTECH HEAD-COUNT')
win_WIDTH = 1200
win_HEIGHT = 500

SELECTED_COLUMNS = DATE_COLUMNS + MENU_COLUMNS + WEATHER_COLUMNS + SCHEDULE_COLUMNS
DEFAULT_VALUES = [2022,1,3,17,4,  0,0,0,0,0,0,0,0,0,0,0,0,    25,0,0,0,0,0,0,0,0,0,0,0,0,0,   0,0,0,0]
DEFAULT_VALUES_WEATHER = [12.5,0,231,1.3,1020,1021,50,5,13797,30,0,0,0]

NCOL_SEC1 = 2
NROW_SEC1 = 0
NCOL_SEC2 = 10
NROW_SEC2 = 0
section1 = Label(window, text="정보 입력",font=50).grid(row=NROW_SEC1,column=NCOL_SEC1,columnspan=6)
section2 = Label(window, text="정보 출력",font=50).grid(row=NROW_SEC1,column=NCOL_SEC2)

section1_1 = Label(window, text="날짜 정보").grid(row=NCOL_SEC1+1,column=NCOL_SEC1,columnspan=2)
input1 = IntVar()
input1.set(2021)
ilabel1 = Label(window, text="년").grid(row=NCOL_SEC1+2,column=NCOL_SEC1)
Entry(window, textvariable=input1).grid(row=NCOL_SEC1+2,column=NCOL_SEC1+1)
input2 = IntVar()
input2.set(6)
ilabel2 = Label(window, text="월").grid(row=NCOL_SEC1+3,column=NCOL_SEC1)
Entry(window, textvariable=input2).grid(row=NCOL_SEC1+3,column=NCOL_SEC1+1)
input3 = IntVar()
input3.set(1)
ilabel3 = Label(window, text="일").grid(row=NCOL_SEC1+4,column=NCOL_SEC1)
Entry(window, textvariable=input3).grid(row=NCOL_SEC1+4,column=NCOL_SEC1+1)
input4=IntVar()
input4.set(7)
radio1=Radiobutton(window, text="조식", value=7, variable=input4).grid(row=NCOL_SEC1+5,column=NCOL_SEC1+1)
radio2=Radiobutton(window, text="중식", value=11, variable=input4).grid(row=NCOL_SEC1+6,column=NCOL_SEC1+1)
radio3=Radiobutton(window, text="석식", value=17, variable=input4).grid(row=NCOL_SEC1+7,column=NCOL_SEC1+1)

section1_2 = Label(window, text="학사 일정").grid(row=NCOL_SEC1+1,column=NCOL_SEC1+2)
input_sch = []
for i in range(len(SCHEDULE_COLUMNS)):
    temp = IntVar()
    Checkbutton(window, text=SCHEDULE_COLUMNS_KR[i], variable=temp).grid(row=NCOL_SEC1+2+i,column=NCOL_SEC1+2)
    input_sch.append(temp)

section1_2 = Label(window, text="메뉴 정보").grid(row=NCOL_SEC1+1,column=NCOL_SEC1+3)
input_menu=[]
for i in range(len(MENU_COLUMNS)):
    temp = IntVar()
    Checkbutton(window, text=MENU_COLUMNS_KR[i], variable=temp).grid(row=NCOL_SEC1+2+i,column=NCOL_SEC1+3)
    input_menu.append(temp)

section1_3 = Label(window, text="날씨 정보").grid(row=NCOL_SEC1+1,column=NCOL_SEC1+4,columnspan=2)
input_weather = []
for i in range(len(WEATHER_COLUMNS)-3):
    temp = IntVar()
    temp.set(DEFAULT_VALUES_WEATHER[i])
    ilabel1 = Label(window, text=WEATHER_COLUMNS_KR[i]).grid(row=NCOL_SEC1+2+i,column=NCOL_SEC1+4)
    Entry(window, textvariable=temp).grid(row=NCOL_SEC1+2+i,column=NCOL_SEC1+5)
    input_weather.append(temp)
for i in range(3):
    temp = IntVar()
    temp.set(DEFAULT_VALUES_WEATHER[10+i])
    Checkbutton(window, text=WEATHER_COLUMNS_KR[10+i], variable=temp).grid(row=NCOL_SEC1+12+i,column=NCOL_SEC1+3,columnspan=2)
    input_weather.append(temp)

number_text = Label(window, text = "0", font=300)
number_text.grid(row = 2,column = NCOL_SEC2)
result_text = Label(window, text = "[예측하기] 버튼을 눌러 식수인원을 확인하세요!")
result_text.grid(row = 3,column = NCOL_SEC2)
result_text1 = Label(window, text = "")
result_text1.grid(row = 4,column = NCOL_SEC2)
result_text2 = Label(window, text = "")
result_text2.grid(row = 5,column = NCOL_SEC2)
result_text3 = Label(window, text = "")
result_text3.grid(row = 6,column = NCOL_SEC2)
result_text4 = Label(window, text = "")
result_text4.grid(row = 7,column = NCOL_SEC2)

processed_value = Label(window, text="...").grid(row=100,column=1,columnspan=100)
inputs = {}
inputs['A'] = np.zeros(5)
inputs['B'] = np.zeros(17)
inputs['C'] = np.zeros(18)
inputs['D'] = np.zeros(9)
inputs['E'] = np.zeros(34)
def process_model_input():
    input_value = np.zeros(34)
    date = "{0}-{1}-{2}".format(input1.get(),input2.get(),input3.get())
    dt = datetime.strptime(date, '%Y-%m-%d')
    input_value[0] = input1.get()
    input_value[1] = input2.get()
    input_value[2] = input3.get()
    input_value[3] = input4.get()
    input_value[4] = dt.weekday()+1
    for i in range(len(DATE_COLUMNS)): #5
        inputs['A'][i] = input_value[i]
        inputs['B'][i] = input_value[i]
        inputs['C'][i] = input_value[i]
        inputs['D'][i] = input_value[i]
    for i in range(len(MENU_COLUMNS)): #12
        input_value[5+i] = input_menu[i].get()
        inputs['B'][5+i] = input_menu[i].get()
    for i in range(len(WEATHER_COLUMNS)): #13
        input_value[17+i] = input_weather[i].get()
        inputs['C'][5+i] = input_weather[i].get()
    for i in range(len(SCHEDULE_COLUMNS)): #4
        input_value[30+i] = input_sch[i].get()*2
        inputs['D'][5+i] = input_sch[i].get()*2
    inputs['E'] = input_value
    #Label(window, text=','.join(list(map(str,input_value)))).grid(row=100,column=NCOL_SEC1+1)
#Button(window,text='Check!',command=process_model_input).grid(row = 100,column = NCOL_SEC1)

def predict():
    process_model_input()
    MODEL_PATH_A = './saved_model/tpot_36_column_27.12_A.pkl'
    MODEL_PATH_B = './saved_model/tpot_36_column_26.95_B.pkl'
    MODEL_PATH_C = './saved_model/tpot_36_column_27.88_C.pkl'
    MODEL_PATH_D = './saved_model/tpot_36_column_27.16_D.pkl'
    MODEL_PATH_E = './saved_model/tpot_36_column_25.41_E.pkl'
    loaded_model_A = pickle.load(open(MODEL_PATH_A, 'rb'))
    loaded_model_B = pickle.load(open(MODEL_PATH_B, 'rb'))
    loaded_model_C = pickle.load(open(MODEL_PATH_C, 'rb'))
    loaded_model_D = pickle.load(open(MODEL_PATH_D, 'rb'))
    loaded_model_E = pickle.load(open(MODEL_PATH_E, 'rb'))
    input_A = inputs['A']
    input_B = inputs['B']
    input_C = inputs['C']
    input_D = inputs['D']
    input_E = inputs['E']
    input_A = np.reshape(np.array(input_A),(1,-1))
    input_B = np.reshape(np.array(input_B),(1,-1))
    input_C = np.reshape(np.array(input_C),(1,-1))
    input_D = np.reshape(np.array(input_D),(1,-1))
    input_E = np.reshape(np.array(input_E),(1,-1))
    result_A = loaded_model_A.predict(input_A)
    result_B = loaded_model_B.predict(input_B)
    result_C = loaded_model_C.predict(input_C)
    result_D = loaded_model_D.predict(input_D)
    result_E = loaded_model_E.predict(input_E)
    print("예상 식수인원은 ", int(result_E), "명 입니다.")
    print("날짜로 {0}명, 메뉴로 {1}명, 날씨로 {2}명, 학사일정으로 {3}명".format(int(result_A), int(result_B-result_A),int(result_C-result_A),int(result_D-result_A)))
    print("A: {0}".format(int(result_A)))
    print("B: {0}".format(int(result_B)))
    print("C: {0}".format(int(result_C)))
    print("D: {0}".format(int(result_D)))
    print("E: {0}".format(int(result_E)))

    number_text.configure(text = str(int(result_E)))
    result_text.configure(text = "예상 식수인원은 "+str(int(result_E))+"명 입니다.")
    result_text1.configure(text = "날짜 정보로 "+str(int(result_A))+"명의 식수인원이 예측되었습니다.")
    result_text2.configure(text = "메뉴 정보는 "+str(int(result_B-result_A))+"명 정도 영향을 줍니다.")
    result_text3.configure(text = "날씨 정보는 "+str(int(result_C-result_A))+"명 정도 영향을 줍니다.")
    result_text4.configure(text = "학사일정 정보는 "+str(int(result_D-result_A))+"명 정도 영향을 줍니다.")

    week_value = [[],[],[]]
    tl = [7,11,17]
    for i in range(3):
        week_input = copy.deepcopy(input_E)
        week_input[0,3] = tl[i]
        for j in range(7):
            week_input[0,2] = week_input[0,2] + 1
            week_input[0,4] = (week_input[0,4] + 1)%7 + 1
            week_value[i].append(loaded_model_E.predict(week_input))
    plotWeekLine(week_value)

def plotWeekLine(y):
    plt.grid(True)
    fig = plt.figure()
    plt.plot(np.arange(7),y[0], lw=1, c='green',ms=1,label='Breakfast')
    plt.plot(np.arange(7),y[1], lw=1, c='blue',ms=1,label='Lunch')
    plt.plot(np.arange(7),y[2], lw=1, c='red',ms=1,label='Dinner')
    plt.title("Head Count for a Week")
    plt.legend(loc=1)
    plt.xlabel("weekand")
    plt.ylabel("Head Count")
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(column=NCOL_SEC2,row=8,rowspan=9)

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
    canvas.get_tk_widget().grid(column=NCOL_SEC2,row=1+int(len(SELECTED_COLUMNS)/2),rowspan=int(len(SELECTED_COLUMNS)/2))

b_pred = Button(window,text='예측하기',command=predict).grid(row = 50,column = 0, columnspan=9)

# def browseFiles():
#     filename = filedialog.askopenfilename(initialdir = "/",
#                                           title = "Select a File",
#                                           filetypes = (("Text files",
#                                                         "*.txt*"),
#                                                        ("all files",
#                                                         "*.*")))
#     # Change label contents
#     label_file_explorer.configure(text="File Opened: "+filename)

# label_file_explorer = Label(window,
#                             text = "File Explorer using Tkinter",
#                             width = 100, height = 4,
#                             fg = "blue").grid(row=0, column=0)
# button_explore = Button(window,
#                         text = "Browse Files",
#                         command = browseFiles).grid(row=0, column=1)

window.mainloop()