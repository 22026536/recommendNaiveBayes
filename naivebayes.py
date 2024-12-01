from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_anime = pd.DataFrame(list(db["Anime"].find()))
df_favorites = pd.DataFrame(list(db["UserFavorites"].find()))

# Chuyển đổi _id và Anime_id thành chuỗi
df_anime['_id'] = df_anime['_id'].astype(str)
df_anime['Anime_id'] = df_anime['Anime_id'].astype(str)
df_favorites['_id'] = df_favorites['_id'].astype(str)

# Tiền xử lý dữ liệu:

# 1. Phân loại Members vào các mốc: 0-200000, 200000-500000, và 500000+
def categorize_members(members):
    if members <= 200000:
        return '0-200000'
    elif members <= 500000:
        return '200000-500000'
    else:
        return '500000+'

df_anime['Members_Category'] = df_anime['Member'].apply(categorize_members)

# 2. Mã hóa cột JapaneseLevel
le_japanese_level = LabelEncoder()
df_anime['JapaneseLevel'] = le_japanese_level.fit_transform(df_anime['JanpaneseLevel'])


# Tạo một đối tượng OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Áp dụng OneHotEncoder cho cột 'Genres'
genres_encoded = encoder.fit_transform(df_anime['Genres'].apply(lambda x: x.split(',')).values)

# Chuyển đổi kết quả thành DataFrame với các tên cột mới
genres_encoded_df = pd.DataFrame(genres_encoded, columns=encoder.get_feature_names_out(['Genres']))

# Kết hợp dữ liệu One-Hot đã mã hóa với DataFrame gốc
df_anime = pd.concat([df_anime, genres_encoded_df], axis=1)
df_anime.drop('Genres', axis=1, inplace=True)

# 4. Mã hóa Type thành giá trị số
le = LabelEncoder()
df_anime['Type'] = le.fit_transform(df_anime['Type'])

# Lấy ra các thông tin Anime mà người dùng đã yêu thích từ bảng UserFavorites
favorite_animes = df_favorites[['User_id', 'favorites']]

# Tạo DataFrame kết hợp các bộ phim yêu thích của người dùng
favorite_data = df_anime[df_anime['Anime_id'].isin(favorite_animes['favorites'])]

# Đặc trưng đầu vào (features) cho mô hình: Loại bỏ 'Status' và 'Producers', thêm 'Members_Category' và 'JapaneseLevel'
features = favorite_data[['Score', 'Type', 'Members_Category', 'JapaneseLevel'] + [col for col in df_anime.columns if col not in ['_id', 'Anime_id', 'Name', 'English_Name', 'Favorites', 'Scored_By', 'Member', 'Image_URL', 'JanpaneseLevel', 'LastestEpisodeAired']]]
target = favorite_data['Anime_id']  # Mục tiêu là gợi ý Anime_id cho người dùng

# Tiền xử lý cho Naive Bayes: Chuyển đổi các cột phân loại thành số
# 1. Mã hóa cột 'Members_Category'
le_members = LabelEncoder()
features['Members_Category'] = le_members.fit_transform(features['Members_Category'])

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = nb.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Hàm gợi ý phim cho người dùng
@app.post("/")
async def recommend(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10
    # Lấy các bộ phim yêu thích của người dùng từ bảng UserFavorites
    user_favorites = df_favorites[df_favorites['User_id'] == user_id]['favorites'].tolist()
    
    # Lọc ra các bộ phim chưa được yêu thích (để tránh gợi ý lại phim đã yêu thích)
    potential_animes = df_anime[~df_anime['Anime_id'].isin(user_favorites)]
    
    # Dự đoán các phim mà người dùng có thể thích
    features = potential_animes[['Score', 'Type', 'Members_Category', 'JapaneseLevel'] + [col for col in df_anime.columns if col not in ['_id', 'Anime_id', 'Name', 'English_Name', 'Favorites', 'Scored_By', 'Member', 'Image_URL', 'JanpaneseLevel', 'LastestEpisodeAired']]]
    features['Members_Category'] = le_members.transform(features['Members_Category'])  # Mã hóa lại Members_Category
    
    predicted = nb.predict(features)
    
    # Lấy các Anime_id dự đoán từ mô hình
    recommended_animes = potential_animes[potential_animes['Anime_id'].isin(predicted)].head(n)  # Lấy top 5 gợi ý
    
    recommendations = recommended_animes[['Anime_id', 'Name', 'English_Name', 'Score', 'Genres']].to_dict(orient='records')
    return recommendations
