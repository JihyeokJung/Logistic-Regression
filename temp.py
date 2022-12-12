import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
#사이킷런의 로지스틱 회귀 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

path = "C://Users//정지혁//Desktop//Pytohn-workspace//CSV Files"
data = pd.read_csv(path + "\ThoraricSurgery.csv")
data.head()

print(data) #[470 rows x 18 columns]

#타겟 데아터 따로 저장
target = data['Survival']

data.drop(labels=['id', 'Survival', 'Type of tumor'], axis=1, inplace=True)

print(data)

#훈련 데이터와 테스트 데이터 분리
#train_test_split은 별도 설정 없을 시 훈련 데이터80%, 테스트 데이터 20%
train_input, test_input, train_target, test_target = train_test_split(
    data, target, random_state=40)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
#로지스틱 회귀 인스턴스 생성
lr = LogisticRegression()
#훈련 데이터로 모델 훈련
# -> fit()에 훈련 데이터셋을 전달하고 모델을 훈련시킴.
lr.fit(train_input, train_target)

#예측 결과 출력
print(lr.predict(test_input))

#변수 종류 출력
print(data.head(0))
#각 특징(변수, feature)들의 가중치
print(lr.coef_)
# -> 어떤 변수가 결과에 큰 영향을 미치는가

#내가 폐암에 걸렸을 때를 가정한 조건
pred = lr.predict([[2.2, 1, 1, 0, 1, 1, 0, 0, 52]])
pred2 = lr.predict([[2.8, 0, 0, 1, 0, 0, 1, 1, 60]])

if(pred[0] == 0):
    print('AI : case1 사망하실 것으로 예측됩니다.\n')
else:
    print('AI : case1 생존하실 것으로 예측됩니다.\n')

if(pred2[0] == 0):
    print('AI : case2 사망하실 것으로 예측됩니다.\n')
else:
    print('AI : case2 생존하실 것으로 예측됩니다.\n')
    
#음성 클래스 / 양성 클래스의 확률
print('케이스1 ) 양성 클래스 / 음성 클래스 : {}'.format(lr.predict_proba([[2.2, 1, 0, 0, 1, 1, 1, 0, 52]])))
print('케이스2 ) 양성 클래스 / 음성 클래스 : {}'.format(lr.predict_proba([[2.8, 0, 0, 1, 0, 0, 1, 1, 60]])))

x = np.arange(-9, 9, 0.1)
phi = 1 / (1 + np.exp(-x))

plt.plot(x, phi)
plt.xlabel('x')
plt.ylabel('phi')

plt.show()