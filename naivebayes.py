def recommend_by_naive_bayes(user_id, n):
    """
    Gợi ý phim sử dụng mô hình Naive Bayes, trả về toàn bộ thông tin anime.
    """
    # Lọc dữ liệu đánh giá của người dùng
    user_ratings = df_ratings[df_ratings['User_id'] == user_id]
    if user_ratings.empty:
        return {"error": "Người dùng không có đánh giá."}
    
    # Dự đoán
    predictions = naive_bayes_model.predict(user_ratings[['User_id', 'Rating']])
    
    # Loại bỏ kết quả trùng lặp
    recommended_ids = list(dict.fromkeys(predictions))  # Anime_id đã được mã hóa (Name_encoded)
    unique_anime_ids = label_encoder.inverse_transform(recommended_ids)  # Giải mã ra Anime_id gốc
    
    # Lấy toàn bộ thông tin của anime
    recommended_anime = df_anime[df_anime['Anime_id'].isin(unique_anime_ids)]
    
    # Giới hạn số lượng kết quả trả về
    recommended_anime = recommended_anime.head(n)
    
    return recommended_anime.to_dict(orient="records")  # Trả về dưới dạng danh sách dict


# Endpoint cho /recommend3
@app.post("/")
async def recommend3(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý mặc định là 10

    if user_id is None:
        return {"error": "Vui lòng cung cấp user_id"}

    # Gọi hàm recommend_by_naive_bayes
    result = recommend_by_naive_bayes(user_id, n)
    return {"recommendations": result}

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sẽ cung cấp cổng trong biến PORT
    uvicorn.run("naivebayes:app", host="0.0.0.0", port=port)