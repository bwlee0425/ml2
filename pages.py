from sklearn.model_selection import train_test_split
import streamlit as st
import FinanceDataReader as fdr
from lightgbm import LGBMClassifier  # 수정 1: LightGBM 분류 모델을 사용하기 위한 import 추가
import pandas as pd

stocks = fdr.StockListing('KOSPI') # KOSPI: 940 종목
stocks_df = stocks[['Code', 'Name']]
company_dict_comp = { company.Name : company.Code for idx, company in stocks_df.iterrows()}
# {한화우:005885}

# 주식 데이터를 가져오는 함수 추가
def get_company_stock_data(company_code):  # 수정 2: 회사 주식 데이터를 가져오는 함수 정의
    return fdr.DataReader(company_code)

predict_data={}
def page2():
    st.header("최고의 주식분석 서비스!!")
    options = st.multiselect(
    "관심 기업을 선택하세요",
    list(company_dict_comp.keys()),
    list(company_dict_comp.keys())[:5],
    )
    #"What are your favorite colors",
    # ["Green", "Yellow", "Red", "Blue"], # 숨어있는 애들
    # ["Yellow", "Red"], # 이미 화면에 나와있는애들
    
    #print(type(options))
    st.write("You selected:", options)

    # 버튼을 누르면
    # 관심기업 리스트를 순차적으로 돌면서
    # 다음날의 예측값을 리턴한다
    # 이 내용들을 종합해서 dataframe(표 형식)으로 보여준다
    """if st.button("분석 시작"):
        for company_code in options:
            company_data = get_company_stock_data(company_dict_comp[company_code])
            company_data['target'] = (company_data['Close'].shift(-1) > company_data['Close']).astype(int)
            company_df = company_data.dropna()
            # 특성과 타겟 분리
            X = company_df.drop('Target', axis=1)
            # train test 데이터 분할
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # X_train = X.iloc[:-1]
            # y_train = y[:-1]
            # X_test = X.iloc[[-1]]
            # y_test = y[-1]
            # classifier=LGBMClassifier()
            # classifier.fit(X_train, y_train)
            # y_pred = classifier.predict(X_test)
            # st.header(y_pred)
            # break"""
    if st.button("분석 시작"):
        for company_code in options:
            # 회사 코드로 주가 데이터 가져오기
            company_data = get_company_stock_data(company_dict_comp[company_code])  # 수정 3: get_company_stock_data 함수 사용
            company_data['target'] = (company_data['Close'].shift(-1) > company_data['Close']).astype(int)
            company_df = company_data.dropna()
            
            # 특성과 타겟 분리
            X = company_df.drop('target', axis=1)  # 수정 4: target 열 이름을 소문자로 통일
            y = company_df['target']  # 수정 5: 타겟 변수를 정의

            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            X_test = X.iloc[[-1]]
            y_test = y.iloc[-1]
            
            classifier = LGBMClassifier()
            classifier.fit(X_train, y_train)
            
            y_pred = classifier.predict(X_test)
            st.header(y_pred)
            break