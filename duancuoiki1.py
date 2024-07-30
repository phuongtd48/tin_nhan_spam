import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn tới tệp CSV
file_path = 'spam.csv'

st.title('Phân loại Email Spam')

# 1. Tải dữ liệu
st.header('1. Tải Dữ Liệu')

if not os.path.isfile(file_path):
    st.error(f"Tệp '{file_path}' không tìm thấy. Vui lòng kiểm tra đường dẫn và tên tệp.")
else:
    # Đọc dữ liệu
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    st.write("Dữ liệu đã được tải thành công!")
    st.dataframe(df)

    # 2. Trực quan hóa dữ liệu
    st.header('2. Trực Quan Hóa Dữ Liệu')

    # Đếm số lượng tin nhắn spam và ham
    spam_count = df[df['label'] == 'spam'].shape[0]
    ham_count = df[df['label'] == 'ham'].shape[0]

    # Hiển thị số lượng tin nhắn spam và ham
    st.write(f"Số lượng tin nhắn spam: {spam_count}")
    st.write(f"Số lượng tin nhắn không phải spam: {ham_count}")

    # Vẽ biểu đồ thanh
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=df, ax=ax)
    ax.set_title('Số lượng tin nhắn Spam và Không phải Spam')
    st.pyplot(fig)

    # Chuyển đổi nhãn 'spam' và 'ham' thành nhãn số
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    # 3. Xây dựng và huấn luyện mô hình
    st.header('3. Xây Dựng và Huấn Luyện Mô Hình')

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuyển đổi văn bản thành các đặc trưng số
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Huấn luyện và đánh giá các mô hình
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }

    accuracies = {}

    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred) * 100
        accuracies[model_name] = accuracy

    # Hiển thị kết quả
    st.write("### Độ chính xác của các mô hình")
    accuracy_df = pd.DataFrame(list(accuracies.items()), columns=['Mô hình', 'Độ chính xác'])
    st.dataframe(accuracy_df)

    # 4. Sử Dụng Mô Hình Để Dự Đoán
    st.header('4. Sử Dụng Mô Hình Để Dự Đoán')

    st.write("## Phát Hiện Tin Nhắn Spam")
    user_input = st.text_area("Nhập tin nhắn của bạn:", "")

    def classify_message(message, model):
        user_input_vec = vectorizer.transform([message])
        prediction = model.predict(user_input_vec)
        prediction_label = "Spam" if prediction[0] == 1 else "Không phải Spam"
        return prediction_label

    selected_model_name = st.selectbox("Chọn mô hình", list(models.keys()))
    selected_model = models[selected_model_name]

    if st.button("Dự Đoán"):
        if user_input:
            prediction_label = classify_message(user_input, selected_model)
            st.write(f"Kết quả dự đoán: {prediction_label}")

            # Thêm thông tin nhận biết tin nhắn spam
            if prediction_label == "Spam":
                st.write("### Dấu Hiệu Nhận Biết Tin Nhắn Spam:")
                st.write("""
                    - Nội dung chứa các từ khóa như 'Win (thắng)', 'Free (miễn phí)', 'Promotion (khuyến mãi)'.
                    - Địa chỉ email của người gửi không rõ ràng hoặc không đáng tin cậy.
                    - Yêu cầu thông tin cá nhân hoặc thông tin tài khoản ngân hàng.
                    - Chứa các liên kết đến các trang web không xác minh.
                """)
        else:
            st.write("Vui lòng nhập một tin nhắn để dự đoán.")
