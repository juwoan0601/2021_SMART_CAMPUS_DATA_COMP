# 2021_SMART_CAMPUS_DATA_COMP
2021 POSTECH 제 1회 스마트캠퍼스데이터경진대회 중 학식 식수인원 예측 과제

# How to Run Application
Required python 3.8 or higher
~~~
~/$ git clone https://github.com/juwoan0601/2021_SMART_CAMPUS_DATA_COMP.git
~/$ cd 2021_SMART_CAMPUS_DATA_COMP
~/2021_SMART_CAMPUS_DATA_COMP$ pip install -r requirements.txt
~/2021_SMART_CAMPUS_DATA_COMP$ python application.py
~~~

# Freeze Python Package
~~~
pip list --format=freeze > requirements.txt
~~~

# Install python Package
~~~
pip install -r requirements.txt
~~~

# How To Use TPOT in script
~~~
python -m train.automl_tpot.py
~~~

# How to Build Application
~~~
cd application
~/application$ pyinstaller --onefile --noconsole application.py 
~~~