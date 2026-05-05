# Reflection
## Hoàng Tuấn Anh - 2A202600075
### Chế độ nào tối ưu cho loại truy vấn nào và tại sao?
- **Keyword Search (BM25):** Tốt nhất cho truy vấn chính xác, mã ID, hoặc từ vựng hiếm vì nó dựa trên tần suất từ khóa (TF-IDF).
- **Semantic Search (Vector):** Thắng khi tìm kiếm diễn giải (paraphrase) hoặc theo ngữ cảnh. Nó nắm bắt ý nghĩa qua dense embeddings, tìm ra kết quả liên quan dù từ khóa không khớp hoàn toàn.
- **Hybrid Search (RRF):** Vượt trội với truy vấn "hỗn hợp" (mixed) chứa cả từ khóa cụ thể lẫn khái niệm rộng. Nó kết hợp thế mạnh của cả hai phương pháp trên để mang lại kết quả chuẩn xác nhất.

### Khi nào không nên dùng Hybrid search?
Nên tránh dùng hybrid search khi **độ trễ (latency) thấp và tiết kiệm tài nguyên** là ưu tiên hàng đầu. Hybrid yêu cầu chạy song song hai truy vấn rồi hợp nhất lại, làm tăng chi phí tính toán và thời gian phản hồi. Ngoài ra, với các tác vụ chỉ cần khớp chính xác tuyệt đối (như tra cứu log, ID hệ thống), thành phần ngữ nghĩa sẽ thừa thãi, gây tốn tài nguyên và có thể làm nhiễu kết quả.
