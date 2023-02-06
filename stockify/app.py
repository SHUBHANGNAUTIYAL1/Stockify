
import pandas as pd
from flask import Flask, render_template, request,make_response,redirect,session
import numpy as np
from datetime import date
import pandas_datareader as web
import mysql.connector
import os
import datetime as dt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
from statsmodels.tsa.arima.model import ARIMA
from nsepy import get_history
warnings.filterwarnings("ignore")

app=Flask(__name__)
app.secret_key = os.urandom(24)

conn = mysql.connector.connect(
    host="sql12.freesqldatabase.com", user="sql12594119", password="NFM7iTi6Ui", database="sql12594119")
cursor = conn.cursor()

now=dt.datetime.now()
start_date = '2022-01-01'
end_date = now.strftime("20%y-%m-%d")
start = datetime.strptime(start_date, "20%y-%m-%d")
end = datetime.strptime(end_date, "20%y-%m-%d")
#List=['WIPRO.NS','RELIANCE.NS','TATAMOTORS.NS','HCLTECH.NS','M&M.NS','KOTAKBANK.NS','BAJAJ-AUTO.NS','NTPC.NS','ITC.NS','INFY.NS','HINDUNILVR.NS',
      # 'TCS.NS','HDFC.NS','NESTLEIND.NS','ASIANPAINT.NS','MARUTI.NS','LT.NS','TITAN.NS','BRITANNIA.NS','DRREDDY.NS','PIDILITIND.NS','3MINDIA.NS','AARTIDRUGS.NS',
      # ]
List=['WIPRO','RELIANCE','TATAMOTORS','HCLTECH','M&M','KOTAKBANK','BAJAJ-AUTO','NTPC','ITC','INFY','HINDUNILVR',
       'TCS','HDFC','NESTLEIND','MARUTI','LT','TITAN','BRITANNIA','DRREDDY','PIDILITIND','3MINDIA'
       ]
List1=[]

length=len(List)
for i in range(length):
      List2=[]
      start_date = '2019-01-01'
      end_date = now.strftime("20%y-%m-%d")
      start = datetime.strptime(start_date, "20%y-%m-%d")
      end = datetime.strptime(end_date, "20%y-%m-%d")
    
      df = get_history(symbol=List[i], start=date(start.year, start.month, start.day), end=date(end.year, end.month, end.day))
      #df=web.DataReader(List[i],data_source='yahoo',start='2021-01-01',end=now.strftime("20%y-%m-%d"))
      print(i)
      df['MA50']=df['Prev Close'].rolling(50).mean()
      df['MA100']=df['Prev Close'].rolling(100).mean()
      df['MA200']=df['Prev Close'].rolling(200).mean()

      rows=df.shape[0]

      diff=0
      diff=df.values[rows-1][3]-df.values[rows-2][3]
      a=diff/df.values[rows-2][3]
      diffcheck=df.values[rows-1][7]-df.values[rows-1][3]
      
      NEW={
         'OPEN':[round(df.values[rows-1][3],2)],
         'CLOSES':[round(df.values[rows-1][7],2)],
         'DIFFERENCE':[round(diff,2)],     
         'MA_50':[round(df.values[rows-1][14],2)],
         'MA_100 ':[round(df.values[rows-1][15],2)],
         'MA_200 ':[round(df.values[rows-1][16],2)],
         'CHANGE':[round(a*100,2)],
         'DIFFCHECK':[round(diffcheck,2)]
         }
      
      df1=pd.DataFrame(NEW)
      #print(df1)
      
      for j in range(8):
         List2.append(df1.values[0][j])
      List2.append(List[i])
      #print(List2)
      List1.append(List2)   
      #print(List1)

   #print(df1.shape)
#print(List1)
List5=List1
List1=sorted(List1,key=lambda x:int(x[2]))
#print(List1)

#List5=sorted(List5,key=lambda x:int (x[7]))
#print(List5)
List1.reverse()
print(List1)

na1=List1[0][8]
na2=List1[1][8]
na3=List1[2][8]
na4=List1[3][8]
na5=List1[4][8]
na6=List1[5][8]
na7=List1[6][8]
na8=List1[7][8]
na9=List1[8][8]
na10=List1[9][8]
na11=List1[10][8]
na12=List1[11][8]

h2=List1[1][2]
h1=List1[0][2]
h3=List1[2][2]
h4=List1[3][2]
h5=List1[4][2]
h6=List1[5][2]
h7=List1[6][2]
h8=List1[7][2]
h9=List1[8][2]
h10=List1[9][2]
h11=List1[10][2]
h12=List1[11][2]


a1=List1[0][3]
a2=List1[1][3]
a3=List1[2][3]
a4=List1[3][3]
a5=List1[4][3]
a6=List1[5][3]
a7=List1[6][3]
a8=List1[7][3]
a9=List1[8][3]
a10=List1[9][3]
a11=List1[10][3]
a12=List1[11][3]

b1=List1[0][4]
b2=List1[1][4]
b3=List1[2][4]
b4=List1[3][4]
b5=List1[4][4]
b6=List1[5][4]
b7=List1[6][4]
b8=List1[7][4]
b9=List1[8][4]
b10=List1[9][4]
b11=List1[10][4]
b12=List1[11][4]

c1=List1[0][5]
c2=List1[1][5]
c3=List1[2][5]
c4=List1[3][5]
c5=List1[4][5]
c6=List1[5][5]
c7=List1[6][5]
c8=List1[7][5]
c9=List1[8][5]
c10=List1[9][5]
c11=List1[10][5]
c12=List1[11][5]


f1=List1[0][0]
f2=List1[1][0]
f3=List1[2][0]
f4=List1[3][0]
f5=List1[4][0]
f6=List1[5][0]
f7=List1[6][0]
f8=List1[7][0]
f9=List1[8][0]
f10=List1[9][0]
f11=List1[10][0]
f12=List1[11][0]

g1=List1[0][1]
g2=List1[1][1]
g3=List1[2][1]
g4=List1[3][1]
g5=List1[4][1]
g6=List1[5][1]
g7=List1[6][1]
g8=List1[7][1]
g9=List1[8][1]
g10=List1[9][1]
g11=List1[10][1]
g12=List1[11][1]

l1=List1[0][6]
l2=List1[1][6]
l3=List1[2][6]
l4=List1[3][6]
l5=List1[4][6]
l6=List1[5][6]
l7=List1[6][6]
l8=List1[7][6]
l9=List1[8][6]
l10=List1[9][6]
l11=List1[10][6]
l12=List1[11][6]
ca=0
#for bucket company


#for colours

if    h1>0:
   color1='limegreen'
else:
   color1='red'
   
if h2>0:
   color2='limegreen'
else:
   color2='red'
   
if h3>0:
   color3='limegreen'
else:
   color3='red'
   
if  h4>0:
   color4='limegreen'
else:
   color4='red'
   
if h5>0:
   color5='limegreen'
else:
   color5='red'
   
if  h6>0:
   color6='limegreen'
else:
   color6='red'
   
   
if h7>0:
   color7='limegreen'
else:
   color7='red'
   
   
if h8>0:
   color8='limegreen'
else:
   color8='red'
   
   
if h9>0:
   color9='limegreen'
else:
   color9='red'
   
   
if h10>0:
   color10='limegreen'
else:
   color10='red'
   
if h11 > 0:
   color11 = 'limegreen'
else:
   color11= 'red'
   
if h12 > 0:
   color12 = 'limegreen'
else:
   color12 = 'red'
   
@app.route('/')
def web1():
   if 'user_id' in session:
      return redirect('/index')
   else:
      return render_template("login.html")

   
@app.route('/add', methods=['POST'])
def add_user():
   name = request.form.get('uname')
   email = request.form.get('uemail')
   password = request.form.get('upassword')
   cursor.execute("""INSERT INTO `users` (`Name`,`Email`,`Password`) VALUES ('{}' ,'{}','{}')""".format(
       name, email, password))
   conn.commit()
   cursor.execute("""SELECT * FROM `users` WHERE `Email` Like '{}'  """.format(email))
   myuser=cursor.fetchall()
   session['user_id']=myuser[0][0]
   return redirect('/index')


@app.route('/loginvalidation', methods=['POST'])
def login_validation():
   email = request.form.get('email')
   password = request.form.get('password')
   cursor.execute(
       """SELECT * FROM `users` WHERE `Email` LIKE '{}' AND `Password` LIKE '{}'  """.format(email, password))
   users = cursor.fetchall()
   print(users)
   if len(users) > 0:
      session['user_id'] = users[0][0]
      return redirect('/index')
   else:
      return redirect('/')


@app.route('/logout')
def logout():
   session.pop('user_id')
   return redirect('/')


@app.route('/index')
def web2():
   
   
 

   if 'user_id' in session:
     cursor.execute("""SELECT * FROM `users` WHERE `Id` Like '{}'  """.format(session['user_id']))
     
     myuser=cursor.fetchall()
     print(myuser)
     name=myuser[0][1]
     emai=myuser[0][2]
     return render_template("index.html",uname=name,uemail=emai,name1=na1,a1=f1,b1=g1,d1=h1,c1=a1,e1=50,k1=l1,colo1=color1,name2=na2,a2=f2,b2=g2,d2=h2,c2=a2,e2=50,k2=l2,colo2=color2,name3=na3,a3=f3,b3=g3,d3=h3,c3=a3,e3=50,k3=l3,colo3=color3,
                          name4=na4,a4=f4,b4=g4,d4=h4,c4=a4,k4=l4,colo4=color4,name5=na5,a5=f5,b5=g5,d5=h5,c5=a5,k5=l5,colo5=color5,name6=na6,a6=f6,b6=g6,d6=h6,c6=a6,k6=l6,colo6=color6,name7=na7,a7=f7,b7=g7,d7=h7,c7=a7,k7=l7,colo7=color7,
                         name8=na8, a8=f8,b8=g8,d8=h8,c8=a8,k8=l8,colo8=color8,name9=na9,a9=f9,b9=g9,d9=h9,c9=a9,k9=l9,colo9=color9,name10=na10,a10=f10,b10=g10,d10=h10,c10=a10,k10=l10,colo10=color10,name11=na11,a11=f11,b11=g11,d11=h11,c11=a11,k11=l11,colo11=color11,name12=na12,a12=f12,b12=g12,d12=h12,c12=a12,k12=l12,colo12=color12)
   else:
      return redirect('/')
        
      
      
      
   
    
@app.route('/invest')  
def invest1():  
    return render_template("invest.html")  
 
@app.route('/AboutUs')  
def about():  
    return render_template("AboutUs.html")      
  
@app.route('/stocks',methods=['POST'])
def index3():
   mt=request.form['name']
   mt1=mt
   
      
   now=dt.datetime.now()
   start_date = '2019-01-01'
   end_date = now.strftime("20%y-%m-%d")
   start = datetime.strptime(start_date, "20%y-%m-%d")
   end = datetime.strptime(end_date, "20%y-%m-%d")
   dk = get_history(symbol=mt, start=date(start.year, start.month, start.day), end=date(end.year, end.month, end.day))
   #w=request.form['MC']
   wa='MA50'
   #w=int(w)
   dk[wa]=dk['Prev Close'].rolling(50).mean()
   rows=dk.shape[0]
   print(dk)
   diff1=0
  # if dk.values[rows-2][3]>dk.values[rows-1][3] :
    #diff1=dk.values[rows-2][3]-dk.values[rows-1][3]
   #if  dk.values[rows-2][3]<dk.values[rows-1][3] : 
   print(dk.values[rows-1][3])
   print(dk.values[rows-2][3])
   diff1=round(dk.values[rows-1][3],2)-round(dk.values[rows-2][3],2)
   print(diff1)
   alpha=diff1/(round(dk.values[rows-2][3],2))
   alpha=alpha*100
   NEW1={'OPEN':[round(dk.values[rows-1][2],2)],
         'CLOSES':[round(dk.values[rows-1][3],2)],    
               wa:[round(dk.values[rows-1][6],2)],
         'change':[round(alpha,2)]
       }
   dk1=pd.DataFrame(NEW1)
   #print(dk1)
   diff2=round(diff1,2)
   #print(diff2)
   #print(dk1.values[0][3])
   
   
   
   
   
   
      

 
   return render_template('index4.html',namea=mt1,np1=dk1.values[0][0],np2=dk1.values[0][1],np3=dk1.values[0][2],eap=50,np4=diff2,np5=dk1.values[0][3]) 
       
@app.route('/risk',methods=['POST'])  
def web4():
   now=dt.datetime.now()
   start_date = '2019-01-01'
   end_date = now.strftime("20%y-%m-%d")
   start = datetime.strptime(start_date, "20%y-%m-%d")
   end = datetime.strptime(end_date, "20%y-%m-%d")

   end_date = '2022-01-01'
   List=['WIPRO','TATAMOTORS','HCLTECH','M&M','KOTAKBANK','BAJAJ-AUTO','NTPC','ITC','INFY','HINDUNILVR',
      'TCS', 'HDFC', 'NESTLEIND', 'MARUTI', 'LT', 'TITAN', 'BRITANNIA', 'DRREDDY', 'PIDILITIND', '3MINDIA', 'AARTIDRUGS',
      'AMBUJACEM','HINDALCO','PGHH','FEDERALBNK','TATASTEEL','SKFINDIA','ASHOKLEY','CUMMINSIND','TATACOMM','ABBOTINDIA',
       'ASTRAZEN','SANOFI','PFIZER','GSFC','HINDUNILVR','AKZOINDIA','NOCIL','TATACHEM','SCHAEFFLER','GRINDWELL',
       'BLISSGVS','ALEMBICLTD','BAYERCROP','COROMANDEL','DEEPAKNTR','SUDARSCHEM','ALKYLAMINE','EIHOTEL','INDHOTEL',
       'CASTROLIND','APOLLOTYRE','CEATLTD','FINPIPE','TATAINVEST','CHOLAHLDNG','EICHERMOT','GMMPFAUDLR','NESCO',
       'WESTLIFE','ZEEL','IFBIND','BERGEPAINT','GRAPHITE','GARFIBRES','HEG','HUHTAMAKI','SUPREMEIND','VSTIND',
       'CANFINHOME','DHANUKA','GILLETTE','VIPIND','APOLLOHOSP','NAVNETEDUL']


         
   length=len(List)
   print(length)
   
   length=len(List)
   print(length)
   
   mt4 = request.form['shu']
   mt5 = request.form['MAC']
   mt6 = request.form['nam']
   if mt5 == "Low":
      List10=[]
      for i in range(length):
         print(i)
         start_date = '2019-01-01'
         end_date = now.strftime("20%y-%m-%d")
         start = datetime.strptime(start_date, "20%y-%m-%d")
         end = datetime.strptime(end_date, "20%y-%m-%d")
         
         df = get_history(symbol=List[i], start=date(start.year, start.month, start.day), end=date(end.year, end.month, end.day))
         #df=df.dropna()
         print(df.tail())
         rows=df.shape[0]
         #print(df.shape)


         #stepwise_fit=auto_arima(df['Open'],trace=True,suppress_warnings=True)
         #stepwise_fit.summary()
         train=df.iloc[:-30]
         test=df.iloc[-30:]
         #print(train.shape,test.shape)
         model=ARIMA(train['High'],order=(1,0,5))
         model=model.fit()
         #print(model.summary())
         start=len(train)
         end=len(train)+len(test)-1
         pred=model.predict(start=start,end=end,type='levels')
         pred.index=df.index[start:end+1]
         #print(pred)
         model2=ARIMA(df['High'],order=(1,0,5))
         model2=model2.fit()
         #print(df.tail())
         #index_future_dates=pd.date_range(start="2022-12-18",end=now.strftime("20%y-%m-%d"))
         index_future_dates = pd.date_range(start="2023-01-29", end="2023-02-10")
         pred=model2.predict(start=len(df),end=len(df)+12,type='levels').rename('ARIMA PREDICTION')
         pred.index=index_future_dates
         #print(pred)
         if(pred[12]>pred[0]):
           List10.append(List[i])
           print(List[i])

      lot1 = List10[0]
      lot2 = List10[1]
      lot3 = List10[2]
      lot4 = List10[3]
      lot5 = List10[4]
      lot6 = List10[5]
      lot7 = List10[6]
      lot8 = List10[7]
      lot9 = List10[8]
      lot10 = List10[9]
   
      
      
     
   if mt5=="Medium":
      List10=[]
      for i in range(length):
            print(i)
            start_date = '2019-01-01'
            end_date = now.strftime("20%y-%m-%d")
            start = datetime.strptime(start_date, "20%y-%m-%d")
            end = datetime.strptime(end_date, "20%y-%m-%d")
            df = get_history(symbol=List[i], start=date(start.year, start.month, start.day), end=date(end.year, end.month, end.day))
            rows=df.shape[0]
            df=df.dropna()
            print(df.tail()) 
            #df=df.set_index(pd.DatetimeIndex(df['Date']))    
            df3=df.copy()
            df3['Numbers']=list(range(0,len(df3)))
            X=np.array(df3[['Numbers']])
            Y=df3['Close'].values
            lin_model=LinearRegression().fit(X,Y)
            print('Intercept:',lin_model.intercept_)
            print('Slope:',lin_model.coef_)
            y_pred=lin_model.coef_ *X + lin_model.intercept_   #y=mx+b
            df3['Pred']=y_pred
            print(r2_score(df3['Close'],df3['Pred']))
            result=lin_model.coef_ *len(df3)+30 +lin_model.intercept_
            print(result)
            if(result>df.values[rows-1][7]):
               List10.append(List[i])
      lot1 = List10[0]
      lot2 = List10[1]
      lot3 = List10[2]
      lot4 = List10[3]
      lot5 = List10[4]
      lot6 = List10[5]
      lot7 = List10[6]
      lot8 = List10[7]
      lot9 = List10[8]
      lot10 = List10[9]
      
    
   if mt5=="High":
      List10=[]
      for i in range(length):
               print(i)
               start_date = '2019-01-01'
               end_date = now.strftime("20%y-%m-%d")
               start = datetime.strptime(start_date, "20%y-%m-%d")
               end = datetime.strptime(end_date, "20%y-%m-%d")
               df = get_history(symbol=List[i], start=date(start.year, start.month, start.day), end=date(end.year, end.month, end.day))
               rows=df.shape[0]
               df3=df.copy()
               df=df.dropna()
               print(df.tail())
               df= df[['Close']]
               future_days=30
               df['Prediction']=df[['Close']].shift(-future_days)
               X=np.array(df.drop(['Prediction'],1))[:-future_days]
               Y=np.array(df['Prediction'])[:-future_days]
               x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)
               tree=DecisionTreeRegressor().fit(x_train,y_train)
               x_future=df.drop(['Prediction'],1)[:-future_days]
               x_future=x_future.tail(future_days)
               x_future=np.array(x_future)
               tree_prediction=tree.predict(x_future)
               #print(tree_prediction[29])
               if(tree_prediction[29]>df3.values[rows-1][7]):
                     List10.append(List[i])

      lot1 = List10[0]
      lot2 = List10[1]
      lot3 = List10[2]
      lot4 = List10[3]
      lot5 = List10[4]
      lot6 = List10[5]
      lot7 = List10[6]
      lot8 = List10[7]
      lot9 = List10[8]
      lot10 = List10[9]
   
   lot11=mt6  
          
   return render_template('index3.html',nam1=lot11,lc1=lot1,lc2=lot2,lc3=lot3,lc4=lot4,lc5=lot5,lc6=lot6,lc7=lot7,lc8=lot8,lc9=lot9,lc10=lot10)
  
if __name__ == '__main__':

   app.run(debug=True,port=3000)


