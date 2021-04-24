# humid_obs를 변수로 xgboost를 이용한 prediction test
## precedure
- timesteps: 720, 2160, 4320, 8640(즉, 1달, 3달, 6달, 12달 간격으로 입력값의 timesteps을 지정)
- model: xgboost를 이용
- parameter tunning: booster:dart, objective:mse로 접근, n_estimator:1000, learning_rate:0.1로 지정
- data: ulsan_data, dangjin_data에 대해 humid_obs 변수만을 가지고 진행

## Result
- ulsan_data와 dangjin_data를 비교할 경우, ulsan_data에 대해 단일 변수에 대한 예측이 가시적으로 우수하다고 판단됨
- 단, 3일 이후의 예측부터는 실제값과 예측값의 추세에서 큰 차이가 발생하는 것을 알 수 있음
- 예측값은 7일 이후로 추세를 예측하지 못하고 특정 offset에서 oscilating하는 
