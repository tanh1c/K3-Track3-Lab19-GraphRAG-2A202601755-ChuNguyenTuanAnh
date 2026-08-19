# Báo cáo Lab 19 — GraphRAG so với Flat RAG

**Sinh viên:** Chu Nguyễn Tuấn Anh  
**MSSV:** 2A202601755

---

# Phần 1 — Thuyết minh kỹ thuật

## 1. Đồng tham chiếu có thể sai trong tình huống nào?

Lỗi đồng tham chiếu có thể xảy ra khi một đại từ hoặc một cụm từ tham chiếu chung có nhiều tiền đề hợp lệ trong cùng một đoạn văn bản. Quy trình chỉ gọi LLM khi phát hiện dấu hiệu cần xử lý đồng tham chiếu và giữ nguyên văn bản nếu ngữ cảnh còn mơ hồ. Những tham chiếu chưa giải quyết được vẫn được lưu tại điểm kiểm tra để phục vụ kiểm tra và truy vết.

Thiết kế này ưu tiên **độ chính xác** vì một lỗi đồng tham chiếu có thể tạo ra cạnh sai trong đồ thị tri thức và khiến lỗi lan truyền sang các bước hợp nhất thực thể, truy hồi và suy luận.

## 2. Ngưỡng hợp nhất thực thể là bao nhiêu và vì sao?

Ngưỡng ứng viên theo độ tương đồng vector là **0.90**. Sau bước tìm ứng viên gần nhất bằng FAISS ANN, quy trình vẫn bắt buộc kiểm tra thêm điều kiện từ vựng và kiểu thực thể trước khi gộp.

Ngưỡng cao được chọn nhằm ưu tiên độ chính xác. Trong đồ thị tri thức dùng cho suy luận, một trường hợp **gộp sai** thường nguy hiểm hơn một trường hợp **tách sai**, vì gộp sai có thể làm nhiều thực thể khác nhau bị nhập thành một nút và ảnh hưởng tới nhiều truy vấn phía sau.

## 3. Ví dụ độ tương đồng cao nhưng không nên gộp

Một ví dụ thực tế từ bảng kiểm tra:

**Ivy Tech Community College - Columbus ↔ Ivy Tech Community College (độ tương đồng = 0.899)**

Dù độ tương đồng cao, quy trình vẫn không gộp nếu cặp ứng viên không vượt qua các điều kiện an toàn. Cơ chế này đặc biệt giúp ngăn các trường hợp tên công ty và tên sản phẩm chỉ giống phần đầu, hoặc hai tên người chỉ trùng họ.

## 4. Ba nút có bậc cao nhất

1. **Microsoft** — loại `Company`, bậc = **9**
2. **artificial intelligence** — loại `Technology`, bậc = **7**
3. **ServiceNow** — loại `Company`, bậc = **5**

Trong lần chạy đầy đủ hiện tại chưa xuất hiện siêu nút thực sự theo ngưỡng `degree > 100`. Tuy nhiên quy trình vẫn có cơ chế bảo vệ: nếu bậc vượt 100 thì chỉ lấy tối đa 50 cạnh gần nhất theo thời gian, đồng thời áp dụng giới hạn số cạnh toàn cục để tránh bùng nổ ngữ cảnh.

## 5. Vì sao ưu tiên cạnh mới nhất có thể vừa đúng vừa sai?

Ưu tiên cạnh mới nhất hợp lý với các câu hỏi về trạng thái hiện tại vì thông tin gần đây thường phản ánh trạng thái mới nhất của thực thể. Cách này cũng giúp hạn chế số lượng cạnh khi gặp nút có bậc lớn.

Tuy nhiên với câu hỏi lịch sử, cạnh cũ có thể là bằng chứng quan trọng nhất. Vì vậy chính sách chỉ giới hạn mạnh khi gặp siêu nút, đồng thời toàn bộ cạnh vẫn giữ thông tin nguồn gốc và ngày xuất bản để có thể truy vết khi phân tích lỗi.

## 6. Flat RAG thắng ở trường hợp nào?

Kết quả theo từng nhóm được lưu tại `outputs/graphrag_vs_flatrag_summary.csv`.

Trường hợp GraphRAG kém hơn Flat RAG nhiều nhất là:

- **Mã câu hỏi:** `G5000-44`
- **Nhóm:** `multi-hop`
- **Chênh lệch chất lượng Graph - Flat:** **-3.00**

Điều này cho thấy GraphRAG không phải lúc nào cũng tốt hơn. Nếu đồ thị thiếu cạnh quan trọng hoặc thực thể khởi đầu chưa chính xác thì truy hồi vector thuần có thể lấy được đoạn bằng chứng trực tiếp tốt hơn.

## 7. GraphRAG thắng ở trường hợp nào?

Trường hợp GraphRAG cải thiện nhiều nhất là:

- **Mã câu hỏi:** `G5000-08`
- **Nhóm:** `multi-hop`
- **Chênh lệch chất lượng Graph - Flat:** **+3.33**

Duyệt đồ thị đặc biệt hữu ích khi câu trả lời cần nối thông tin qua nhiều thực thể hoặc nhiều tài liệu khác nhau thay vì chỉ tìm một đoạn văn bản có độ tương đồng cao.

## 8. Đánh đổi giữa độ trễ và số token

Kết quả trung bình trên Golden Dataset:

- **Độ trễ Flat RAG:** 2.69 giây
- **Độ trễ GraphRAG:** 5.40 giây
- **Số token Flat RAG:** 949.8
- **Số token GraphRAG:** 1770.0

Như vậy GraphRAG tốn nhiều thời gian và token hơn do phải thực hiện ghép thực thể, duyệt đồ thị và kết hợp thêm ngữ cảnh đồ thị. Đổi lại, nó cải thiện khả năng tổng hợp thông tin đa bước và liên tài liệu.

## 9. Tác nhân lập trình AI từng đề xuất gì nhưng không được sử dụng, và vì sao?

Quy trình không sử dụng độ tương đồng cosine theo từng cặp với độ phức tạp **O(N²)** cho khử trùng lặp gần hoặc hợp nhất thực thể vì cách này không phù hợp khi dữ liệu tăng lớn.

Thay vào đó:

- Khử trùng lặp gần sử dụng **SimHash + LSH**.
- Hợp nhất thực thể sử dụng **FAISS ANN** để tạo ứng viên, sau đó mới áp dụng điều kiện từ vựng và kiểu thực thể để quyết định có gộp hay không.
- Golden Dataset không được sử dụng để lựa chọn các đoạn đem đi trích xuất vì điều đó sẽ gây **rò rỉ dữ liệu đánh giá**.

Tác nhân lập trình AI được sử dụng để tăng tốc phần mã lặp lại, viết kiểm thử và hỗ trợ triển khai, nhưng các quyết định kiến trúc và tiêu chí chấp nhận vẫn được kiểm soát bằng kiểm thử, CI và các điều kiện bất biến của quy trình.

## 10. Nếu mở rộng lên toàn bộ tập dữ liệu thì điểm nghẽn đầu tiên là gì?

Điểm nghẽn đầu tiên là **trích xuất bằng LLM và giới hạn tốc độ API**. Sau đó mới tới nhúng vector, lập chỉ mục, hợp nhất thực thể và độ phân nhánh của đồ thị.

Hướng mở rộng phù hợp gồm:

- hàng đợi bền vững và cơ chế lưu tiến độ/tiếp tục;
- các tiến trình xử lý bất đồng bộ theo hạn mức API;
- ANN và phân khối ứng viên cho hợp nhất thực thể;
- ghi Neo4j theo lô bằng `UNWIND`;
- phân vùng đồ thị hoặc tóm tắt cộng đồng;
- bộ nhớ đệm cho truy hồi và cho các kết quả LLM có thể tái sử dụng.

## Kết quả thực nghiệm chính

- **Tính đầy đủ:** Flat = **3.64**, Graph = **3.88**
- **Tính trung thực theo bằng chứng:** Flat = **3.68**, Graph = **3.88**
- **Khả năng suy luận đa bước:** Flat = **3.44**, Graph = **3.74**
- **Số nút trong đồ thị:** 418
- **Số cạnh trong đồ thị:** 268
- **Số cạnh thiếu thông tin nguồn gốc:** 0

---

# Phần 2 — Phân tích các trường hợp thành công và thất bại

## Trường hợp 1 — GraphRAG có lợi thế

**Câu hỏi `G5000-08`:** Những tổ chức bên ngoài nào được kết nối với các nỗ lực AI tạo sinh của ServiceNow trong dữ liệu đã chọn, và mỗi tổ chức đóng vai trò gì?

### Câu trả lời của Flat RAG

Flat RAG xác định hai tổ chức chính:

1. **NVIDIA:** hợp tác với ServiceNow để phát triển năng lực AI tạo sinh cấp doanh nghiệp nhằm hỗ trợ tự động hóa quy trình làm việc nhanh hơn và thông minh hơn.
2. **Deloitte:** mở rộng liên minh với ServiceNow để tích hợp Now Assist vào các dịch vụ quản lý thế hệ mới, hỗ trợ khách hàng xử lý nhu cầu vận hành và công nghệ.

Flat RAG không tìm ra đầy đủ các tổ chức bên ngoài còn lại trong bằng chứng được cung cấp.

### Câu trả lời của GraphRAG

GraphRAG xác định được ba tổ chức và vai trò riêng biệt:

1. **NVIDIA:** hợp tác với ServiceNow để phát triển năng lực AI tạo sinh cấp doanh nghiệp.
2. **Accenture:** hợp tác cùng ServiceNow và NVIDIA để thúc đẩy việc ứng dụng AI tạo sinh trong doanh nghiệp, đóng vai trò hỗ trợ triển khai và mở rộng mức độ áp dụng.
3. **Deloitte:** mở rộng liên minh với ServiceNow nhằm tích hợp Now Assist vào các dịch vụ quản lý phục vụ nhu cầu vận hành và công nghệ.

### Đánh giá

- **Chênh lệch chất lượng:** +3.33 cho GraphRAG.
- **Nguyên nhân:** Flat RAG xếp hạng từng đoạn văn bản độc lập, trong khi GraphRAG có thể nối các thực thể đã chuẩn hóa và các cạnh có thông tin nguồn gốc trước khi kết hợp thêm bằng chứng vector. Điều này giúp GraphRAG tổng hợp được nhiều mối quan hệ liên quan hơn trong một câu hỏi đa bước.

## Trường hợp 2 — GraphRAG thất bại tương đối

**Câu hỏi `G5000-44`:** Hai hệ sinh thái đối tác khác nhau nào kết nối L&T Technology Services với hạ tầng tiên tiến trong năm 2023: một hệ cho 5G đường sắt đô thị và một hệ cho bảo mật OT?

### Câu trả lời của Flat RAG

Flat RAG lấy được hai quan hệ phù hợp:

1. Với **5G cho đường sắt đô thị**, L&T Technology Services hợp tác với **Qualcomm** và **Thales** để hỗ trợ mạng 5G riêng cho hệ thống đường sắt đô thị.
2. Với **bảo mật OT**, L&T Technology Services hợp tác với **Palo Alto Networks** với vai trò MSSP để cung cấp nền tảng bảo mật OT cho các ngành công nghiệp.

### Câu trả lời của GraphRAG

GraphRAG xác định đúng quan hệ với **Palo Alto Networks** cho bảo mật OT, nhưng phần 5G đường sắt bị suy diễn từ một quan hệ mua lại với Smart World & Communication thay vì lấy đúng quan hệ hợp tác với Qualcomm và Thales.

### Đánh giá

- **Chênh lệch chất lượng:** -3.00 cho GraphRAG.
- **Các nguyên nhân cần kiểm tra:** thiếu cạnh ở bước trích xuất, thực thể khởi đầu chưa tối ưu hoặc truy hồi không đi đúng nhánh quan hệ cần thiết.
- Các trường `graph_route`, các thực thể khởi đầu được ghép và số lượng cạnh đều được xuất theo từng câu hỏi để có thể truy vết thay vì đoán nguyên nhân.

---

# Phần 3 — Tự đánh giá và kế hoạch cải tiến

**Sinh viên:** Chu Nguyễn Tuấn Anh  
**MSSV:** 2A202601755

## Liên hệ nội dung bài học với phần triển khai

| Khái niệm | Phần triển khai | Quan sát |
|---|---|---|
| Đồng tham chiếu thận trọng | `resolve_coreferences()` | Chỉ gọi LLM với đoạn có dấu hiệu đại từ/tham chiếu chung; trường hợp mơ hồ vẫn có thể kiểm tra lại. |
| Danh sách schema cho phép nghiêm ngặt | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Quan hệ ngoài schema bị loại trước khi đưa vào cơ sở dữ liệu. |
| Hợp nhất thực thể | `build_resolution_map()` | FAISS ANN chỉ dùng để tìm ứng viên; điều kiện từ vựng và kiểu thực thể quyết định việc gộp có an toàn hay không. |
| Ghi dữ liệu theo lô | `Neo4jStore.bulk_insert_*()` | Dùng `UNWIND` theo lô, không ghi từng dòng qua mạng. |
| Xử lý siêu nút | `GraphRetriever.retrieve()` | Nút có bậc > 100 bị giới hạn tối đa 50 cạnh gần nhất và còn chịu giới hạn cạnh toàn cục. |
| Đánh giá | `evaluate()` + `Judge` | Cùng một Golden Dataset và ba tiêu chí chấm giúp đo rõ đánh đổi giữa chất lượng và chi phí. |

## Bài học từ quá trình sửa lỗi

Một kiểm thử khử trùng lặp gần có tính xác định cho thấy việc băm `title + body` có thể bỏ sót các bản sao được phát hành lại khi tiêu đề thay đổi nhưng phần nội dung chính gần như giống nhau. Quy trình được sửa để tạo dấu vân tay từ phần nội dung bài viết và lưu bảng kiểm tra cho các quyết định khử trùng lặp gần.

CI cũng từng phát hiện quy trình GitHub Actions bị kích hoạt trùng nhiều lần. Sau đó hệ thống được tách thành:

- CI nhẹ cho các lần đẩy mã thông thường;
- quy trình tích hợp chỉ chạy khi tệp kích hoạt được thay đổi có chủ đích.

Cách này giảm việc gọi API không cần thiết và giúp quá trình thử nghiệm dễ kiểm soát hơn.

## Kế hoạch cải tiến nếu đưa lên môi trường vận hành thực tế

Nếu phát triển thành một trợ lý tri thức phục vụ thực tế, các nguyên tắc cần giữ gồm:

- lưu thông tin nguồn gốc trên mọi quan hệ;
- phân vùng dữ liệu đồ thị theo tập dữ liệu và phiên bản;
- chỉ gộp thực thể khi có cả bằng chứng ngữ nghĩa và bằng chứng từ vựng/kiểu thực thể phù hợp;
- dùng truy hồi tự hiệu chỉnh trước khi mở rộng bán kính đồ thị quá lớn;
- dùng báo cáo cộng đồng cho câu hỏi mang tính toàn cục;
- dùng BFS cục bộ cho câu hỏi tập trung vào thực thể cụ thể;
- bổ sung bộ nhớ đệm, giám sát và cơ chế kiểm soát chi phí LLM.

---

# Phần 4 — Kiểm chứng điểm thưởng

## Điểm thưởng A — Truy hồi cấp thấp và cấp cao

Quy trình có cả hai tầng truy hồi:

- **Cấp thấp:** truy hồi theo thực thể và vùng lân cận đồ thị cho câu hỏi cụ thể.
- **Cấp cao:** truy hồi theo báo cáo cộng đồng cho câu hỏi tổng quan trên toàn bộ tập tài liệu.

Bộ định tuyến truy vấn được kiểm chứng bằng dữ liệu minh họa:

- Số truy vấn được chuyển sang **cục bộ:** 2
- Số truy vấn được chuyển sang **toàn cục:** 2

Bằng chứng được lưu tại `outputs/retrieval_router_demo.csv`.

## Điểm thưởng B — Tìm kiếm toàn cục qua báo cáo cộng đồng

Quy trình sử dụng phương án dự phòng NetworkX với thuật toán `networkx.greedy_modularity_communities` để phát hiện cộng đồng trên đồ thị, sau đó ghi `community_id` trở lại Neo4j bằng `UNWIND` theo lô.

Kết quả lần chạy đầy đủ:

- **Số cộng đồng:** 164
- **Số cộng đồng được LLM tóm tắt:** 164/164
- **Tỷ lệ tóm tắt bằng LLM:** 100%

Các báo cáo cộng đồng được dùng làm ngữ cảnh cấp cao cho tìm kiếm toàn cục.

## Điểm thưởng C — Truy hồi đồ thị tự hiệu chỉnh

Cơ chế tự hiệu chỉnh triển khai đúng chuỗi:

1. Truy hồi **hop 2**.
2. LLM kiểm tra ngữ cảnh đã đủ hay chưa.
3. Nếu chưa đủ, mở rộng sang **hop 3**.
4. LLM kiểm tra lại lần thứ hai.
5. Nếu vẫn chưa đủ, thực hiện **truy hồi vector dự phòng**.
6. Mọi nhánh đều có **điều kiện dừng** rõ ràng.

Kết quả kiểm chứng trên 12 câu điểm thưởng:

- `hop2`: 3 câu
- `hop3`: 1 câu
- `hop3+vector`: 8 câu
- Tỷ lệ ngữ cảnh được LLM đánh giá đủ ở hop 2 ban đầu: **25%**
- Tỷ lệ ngữ cảnh đủ sau tự hiệu chỉnh: **100%**
- Mức cải thiện: **+75 điểm phần trăm**
- Độ dài ngữ cảnh trung bình: từ **2333.667** ký tự lên **5517.500** ký tự
- Cả 12/12 trường hợp đều có điều kiện dừng cuối cùng

Bằng chứng được lưu tại `outputs/self_correction_bonus_eval.csv` và `outputs/bonus_before_after.csv`.

## Điểm thưởng khử trùng lặp gần

Khử trùng lặp gần sử dụng **SimHash + LSH** thay vì so sánh theo từng cặp với độ phức tạp O(N²).

- **Số bản ghi gần trùng lặp được loại:** 5

Bằng chứng được lưu tại `outputs/near_dedup_audit.csv`.

---

# Phần 5 — Tổng kết kết quả bài Lab

Lần chạy đầy đủ cuối cùng sử dụng chính sách **chỉ lấy 5.000 dòng đầu tiên** của nguồn dữ liệu theo thứ tự xác định, không lấy mẫu ngẫu nhiên.

Các chỉ số chính:

- Dòng dữ liệu nguồn: **5.000**
- Bài viết sau khử trùng lặp: **2.114**
- Đoạn được lập chỉ mục: **2.114**
- Đoạn được LLM trích xuất: **400**
- Bộ ba quan hệ hợp lệ: **268**
- Nút trong Neo4j: **418**
- Golden Dataset đã được đánh giá: **50/50**
- Số dòng kiểm tra hợp nhất thực thể: **16**
- Bản ghi gần trùng lặp bị loại: **5**
- Báo cáo cộng đồng: **164**
- Cạnh thiếu thông tin nguồn gốc: **0**
- Lỗi trích xuất còn lại: **0**

Toàn bộ kết quả chi tiết được lưu trong thư mục `outputs/`, báo cáo trong `reports/`, và notebook đã được chạy đầy đủ để phục vụ việc kiểm tra và chấm bài.
