스윙 분석 및 분류
===
스윙 분석 및 분류에는 스마트워치에서 50Hz로 수집된 IMU 센서의 6축 데이터(가속도, 각속도)를 사용하며, 스윙 분석에서 영상을 추가로 사용한다. IMU 데이터는 지자기, 드리프트 보정이 완료되었으며, 상수 값들은 실험적∙정성적으로 결정하였다.

스윙 분석
---
### 스윙 분석 전제사항
- 입력으로 하이클리어 스윙 시 영상과 IMU 데이터가 사용된다.
-	전문가들의 스윙 모범 자세는 서로 일관적이며, 실력에 영향을 주는 부분일수록 그렇다.
-	사용자들의 배드민턴 실력에 대해 정성적으로 명확히 순위를 매길 수 있다.   
#
***
IMU 데이터 스윙 분석
---
### IMU 데이터 스윙 분석 순서   
1.  스윙 데이터를 전처리한다. 데이터 자르기, 특징 추가, 정규화(normalization), windowing을 포함한다. z방향 가속도 고점을 기준으로 앞 10시점, 뒤 5시점을 분석 대상으로 삼았으며, window size는 8로 정했다. 각 window의 key는 (시작 시점, 특징) 튜플이며, 8개 값으로 구성된 list를 value로 갖는다.
2.	전문가 데이터간 같은 window의 평균 및 전문가 데이터와 전문가 평균 간 유클리디안 거리 표준편차를 구하고, 표준편차를 통해 각 window의 Z score를 산출한다.
3.	사용자 데이터와 전문가 평균 및 Z score를 활용하여 사용자의 실력을 측정하여 점수화하고, 실력 측정에 대한 특징들의 기여도를 제시한다.
4.	각 스윙 단계(백스윙, 임팩트, 팔로스로우)에서의 사용자-전문가 데이터 간 차이를 비교하고, 이를 해석하여 사용자에게 제시한다.
#
### IMU 데이터 스윙 분석 세부사항 및 성능 평가   
스마트 기기를 활용한 배드민턴 스윙 분석, 한국소프트웨어종합학술대회 논문집, 2023-12, 1436p
#
***
영상 데이터 스윙 분석
---
영상 데이터 스윙 분석은 모범 배드민턴 자세에 대해 연구한 여러 스포츠 과학 논문에 근거하여  임팩트 시점의 어깨 각도, 팔꿈치 각도, 스윙 전반의 몸통 회전각을 실력 측정 요소로 활용하였다.   
#
### 영상 데이터 스윙 분석 순서는 다음과 같다.
1.  MoveNet 사용을 위해 스윙 동작 영상을 이미지 배열(shape: 480, 360, 3)로 전처리 한다.
2.	MoveNet을 활용하여 각 관절의 좌표를 추출한다. 만약 각 관절 좌표에 대한 예측 스코어나 영상 길이가 기준치 이하일 경우, 영상 분석은 반려된다.
3.	각도는 내적, 몸통 회전각은 두 골반 끝 좌표의 유클리디안 거리를 활용하여 실력 측정 요소를 반환한다. 각도의 경우, 사용 손목의 y좌표가 가장 높을 때인 임팩트 순간을 기준으로 하였으며, 몸통 회전각은 거리의 최대값과 최소값의 차를 기준으로 하였다.
4.	전문가 실력 요소 평균과 사용자 수치를 비교하여, 임의 값을 경계로 실력을 점수화 한다.   
#
스윙 분류
---
스윙 분류는 경기 전반의 IMU 데이터가 사용되며, 스윙 시점 감지 및 전처리, 스윙 분류, 각 스윙 점수 산출 세 단계로 구성된다. 각 단계에 대한 세부사항은 다음과 같다.
#
### 1. 스윙 시점 감지 및 전처리   
 스윙 시점 감지 및 전처리 과정은 다음과 같이 세 단계로 나뉜다. 피크 검출은 SciPy라이브러리의 signal.find_peaks 함수를 사용하였으며, 과정 결과물은 각 스윙의 시점, 종점, 특징값들이다. 6개 측정값 각각에 대하여 역치를 넘는 구간에서의 피크를 구하고, 피크를 중심으로 30개 시점을 한 구간으로 정했다. 이때 역치: 가속도 2m/s^2, 각속도 200도/s로 정했다.
 이 구간이 3개 이상 겹치는 구역에 대해서, z방향 가속도가 1m/s^2이상일 때 피크를 검출하여 임팩트 시점으로 보았다.   
 찾아낸 임팩트 시점을 중심으로 30개의 시점을 스윙 구간으로 삼고, 각 측정값 혹은 값들 간의 range, minimum, maximum, {, absolute} average, kurtosis {f, p}, skewness {statistic, p value}, standard deviation, vector angle, inter quartile range, relative difference를 해당 스윙의 특징으로 계산했다.
 라벨링 된 423개 스윙 데이터에 대해 스윙 시점 감지 알고리즘의 정확도는 다음과 같아, 위양성을 유념한 알고리즘의 설계 의도를 만족한다.
 # 
 정확도: 0.877, 정밀도: 0.956, 재현율: 0.915, f1 점수: 0.935
 #
### 2. 스윙 분류   
 스윙 분류는 앞선 단계의 결과물과 머신 러닝 기법을 사용한다. 스윙 분류는 분석과 달리 단일한 모범 자세를 골라내기 위함이 아님에 주의하였다. 고전적 ML 기법 Logistic Regression, KNN, Linear SVC, Decision Tree, Random Forest와 LSTM을 테스트한 결과, 우수한 성능을 보였고, 의도에 적합한 KNN을 스윙 분류에 활용하기로 하였다. 12종류의 스윙에 대해 라벨링 된, 4:1로 나뉘어진 387개 스윙 데이터의 KNN 분류 정확도는 다음과 같다.
 #
 정확도: 0.9, 정밀도: 0.924, 재현율: 0.91, f1 점수: 0.913
 #
 다만 훈련 및 검증에 쓰인 데이터의 양이 부족하고 편중되었으며, 스윙 종류를 분류한 다른 시도에서도 12개와 같이 많은 종류를 구분한 사례가 없는 것으로 보아, 향후 분류 제반을 발전시켜 나가야 할 것이다.
 #
### 3. 각 스윙 점수 산출   
분류된 각 스윙에 대한 스윙 점수 산출은 IMU 데이터 스윙 분석과 같은 방법으로 진행하였다. 이를 위해 기 존재하는 하이클리어 뿐만 아니라 12개 모든 스윙의 전문가 데이터를 수집∙활용함이 마땅하나, 수집 환경 제한으로 아마추어 스윙 데이터를 대신 사용하였다. 또한 기존 방법의 분석 결과 해석은 하이클리어에 특화되어 있기에, 이 부분은 생략되었다.

---   

깃헙 링크   
---
코어 로직   
https://github.com/rkdthdah/cokcok_data_core   
   
데이터 수집용 안드로이드 및 스마트 워치 앱   
https://github.com/rkdthdah/smart-phone-app-for-testing   
https://github.com/rkdthdah/smart-watch-app-for-testing   
