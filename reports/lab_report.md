# Báo cáo Lab 19 — GraphRAG so với Flat RAG

**Sinh viên:** Chu Nguyễn Tuấn Anh  
**MSSV:** 2A202601755

---

# Phần 1 — Thuyết minh kỹ thuật

## 1. Coreference có thể sai trong tình huống nào?

Lỗi đồng tham chiếu có thể xảy ra khi một đại từ hoặc một cụm từ tham chiếu chung có nhiều tiền đề hợp lệ trong cùng một đoạn văn bản. Pipeline sử dụng cơ chế kích hoạt thận trọng: chỉ gọi LLM khi phát hiện dấu hiệu cần xử lý coreference, đồng thời giữ nguyên văn bản nếu ngữ cảnh còn mơ hồ. Các mention chưa giải quyết được vẫn được checkpoint để phục vụ kiểm tra và truy vết.

Thiết kế này ưu tiên **độ chính xác (precision)** vì một coreference sai có thể tạo ra cạnh sai trong knowledge graph và khiến lỗi lan truyền sang các bước entity resolution, retrieval và reasoning.

## 2. Ngưỡng Entity Resolution là bao nhiêu và vì sao?

Ngưỡng ứng viên theo vector similarity là **0.90**. Sau bước tìm ứng viên gần nhất bằng ANN, pipeline vẫn bắt buộc kiểm tra thêm lexical guard và type guard trước khi merge.

Ngưỡng cao được chọn nhằm ưu tiên precision. Trong knowledge graph dùng cho reasoning, một **false merge** thường nguy hiểm hơn một **false split**, vì merge sai có thể làm nhiều thực thể khác nhau bị gộp thành một node và ảnh hưởng tới nhiều truy vấn phía sau.

## 3. Ví dụ similarity cao nhưng không nên merge

Một ví dụ thực tế từ audit:

**Ivy Tech Community College - Columbus ↔ Ivy Tech Community College (similarity = 0.899)**

Dù similarity cao, pipeline vẫn không merge nếu không vượt qua các guard cần thiết. Cơ chế guard đặc biệt giúp ngăn các trường hợp như company và product chỉ giống prefix, hoặc hai tên người chỉ trùng họ.

## 4. Ba node có degree cao nhất

1. **Microsoft** — loại `Company`, degree = **9**
2. **artificial intelligence** — loại `Technology`, degree = **7**
3. **ServiceNow** — loại `Company`, degree = **5**

Trong lần chạy full hiện tại chưa xuất hiện super-node thực sự theo ngưỡng `degree > 100`. Tuy nhiên pipeline vẫn có cơ chế bảo vệ: nếu degree vượt 100 thì chỉ lấy tối đa 50 cạnh gần nhất theo thời gian, đồng thời áp dụng global edge cap để tránh context explosion.

## 5. Vì sao ưu tiên cạnh mới nhất có thể vừa đúng vừa sai?

Ưu tiên cạnh mới nhất hợp lý với các câu hỏi về trạng thái hiện tại vì thông tin gần đây thường phản ánh trạng thái mới nhất của thực thể. Cách này cũng giúp hạn chế số lượng cạnh khi gặp node có degree lớn.

Tuy nhiên với câu hỏi lịch sử, cạnh cũ có thể là bằng chứng quan trọng nhất. Vì vậy policy chỉ kích hoạt mạnh khi gặp super-node, đồng thời toàn bộ cạnh vẫn giữ provenance và ngày xuất bản để có thể truy vết khi phân tích lỗi.

## 6. Flat RAG thắng ở trường hợp nào?

Kết quả theo từng nhóm được lưu tại `outputs/graphrag_vs_flatrag_summary.csv`.

Trường hợp GraphRAG kém hơn Flat RAG nhiều nhất là:

- **Mã câu hỏi:** `G5000-44`
- **Nhóm:** `multi-hop`
- **Chênh lệch chất lượng Graph - Flat:** **-3.00**

Điều này cho thấy GraphRAG không phải lúc nào cũng tốt hơn; nếu graph thiếu cạnh quan trọng hoặc entity seed chưa chính xác thì vector retrieval thuần có thể lấy được đoạn bằng chứng trực tiếp tốt hơn.

## 7. GraphRAG thắng ở trường hợp nào?

Trường hợp GraphRAG cải thiện nhiều nhất là:

- **Mã câu hỏi:** `G5000-08`
- **Nhóm:** `multi-hop`
- **Chênh lệch chất lượng Graph - Flat:** **+3.33**

Graph traversal đặc biệt hữu ích khi câu trả lời cần nối thông tin qua nhiều entity hoặc nhiều tài liệu khác nhau thay vì chỉ tìm một đoạn văn bản có độ tương đồng cao.

## 8. Đánh đổi giữa latency và token

Kết quả trung bình trên Golden Dataset:

- **Flat RAG latency:** 2.69 giây
- **GraphRAG latency:** 5.40 giây
- **Flat RAG token:** 949.8 token
- **GraphRAG token:** 1770.0 token

Như vậy GraphRAG tốn nhiều thời gian và token hơn do phải thực hiện entity matching, graph traversal và kết hợp thêm graph context. Đổi lại, nó cải thiện khả năng tổng hợp thông tin đa bước và liên tài liệu.

## 9. AI Coding Agent từng đề xuất gì nhưng không được sử dụng, và vì sao?

Pipeline không sử dụng pairwise cosine similarity theo kiểu **O(N²)** cho near-dedup hoặc entity resolution vì cách này không phù hợp khi dữ liệu tăng lớn.

Thay vào đó:

- Near-dedup sử dụng **SimHash + LSH**.
- Entity resolution sử dụng **FAISS ANN** để tạo candidate, sau đó mới dùng lexical/type guard để quyết định merge.
- Golden Dataset không được sử dụng để lựa chọn extraction chunks vì điều đó sẽ gây **benchmark leakage**.

AI Coding Agent được sử dụng để tăng tốc boilerplate, viết test và hỗ trợ triển khai, nhưng các quyết định kiến trúc và acceptance criteria vẫn được kiểm soát bằng test, CI và các invariant của pipeline.

## 10. Nếu scale lên toàn bộ dataset thì bottleneck đầu tiên là gì?

Bottleneck đầu tiên là **LLM extraction và giới hạn API/rate limit**. Sau đó mới tới embedding/indexing, entity resolution và graph fan-out.

Hướng mở rộng phù hợp gồm:

- hàng đợi bền vững và checkpoint/resume;
- worker bất đồng bộ theo quota API;
- ANN/blocking cho entity resolution;
- batch `UNWIND` khi ghi Neo4j;
- partition hoặc community summary;
- cache retrieval và cache các kết quả LLM có thể tái sử dụng.

## Kết quả thực nghiệm chính

- **Tính đầy đủ (Comprehensiveness):** Flat = **3.64**, Graph = **3.88**
- **Tính trung thực theo bằng chứng (Faithfulness):** Flat = **3.68**, Graph = **3.88**
- **Khả năng suy luận đa bước (Multi-hop reasoning):** Flat = **3.44**, Graph = **3.74**
- **Số node trong graph:** 418
- **Số edge trong graph:** 268
- **Số edge thiếu provenance:** 0

---

# Phần 2 — Phân tích các trường hợp thành công và thất bại

## Trường hợp 1 — GraphRAG có lợi thế

**Câu hỏi `G5000-08`:** Những tổ chức bên ngoài nào được kết nối với các nỗ lực generative AI của ServiceNow trong dữ liệu đã chọn, và mỗi tổ chức đóng vai trò gì?

### Câu trả lời của Flat RAG

Flat RAG xác định hai tổ chức chính:

1. **NVIDIA:** hợp tác với ServiceNow để phát triển năng lực generative AI cấp doanh nghiệp nhằm hỗ trợ tự động hóa quy trình làm việc nhanh hơn và thông minh hơn.
2. **Deloitte:** mở rộng liên minh với ServiceNow để tích hợp Now Assist vào các dịch vụ managed service thế hệ mới, hỗ trợ khách hàng xử lý nhu cầu vận hành và công nghệ.

Flat RAG không tìm ra đầy đủ các tổ chức bên ngoài còn lại trong bằng chứng được cung cấp.

### Câu trả lời của GraphRAG

GraphRAG xác định được ba tổ chức và vai trò riêng biệt:

1. **NVIDIA:** hợp tác với ServiceNow để phát triển năng lực generative AI cấp doanh nghiệp.
2. **Accenture:** hợp tác cùng ServiceNow và NVIDIA để thúc đẩy việc ứng dụng generative AI trong doanh nghiệp, đóng vai trò hỗ trợ triển khai và mở rộng adoption.
3. **Deloitte:** mở rộng liên minh với ServiceNow nhằm tích hợp Now Assist vào managed services phục vụ nhu cầu vận hành và công nghệ.

### Đánh giá

- **Chênh lệch chất lượng:** +3.33 cho GraphRAG.
- **Nguyên nhân:** Flat RAG xếp hạng từng chunk độc lập, trong khi GraphRAG có thể nối các canonical entity và các cạnh có provenance trước khi kết hợp thêm bằng chứng vector. Điều này giúp GraphRAG tổng hợp được nhiều mối quan hệ liên quan hơn trong một câu hỏi multi-hop.

## Trường hợp 2 — GraphRAG thất bại tương đối

**Câu hỏi `G5000-44`:** Hai hệ sinh thái đối tác khác nhau nào kết nối L&T Technology Services với hạ tầng tiên tiến trong năm 2023: một hệ cho 5G đường sắt đô thị và một hệ cho bảo mật OT?

### Câu trả lời của Flat RAG

Flat RAG lấy được hai quan hệ phù hợp:

1. Với **5G cho đường sắt đô thị**, L&T Technology Services hợp tác với **Qualcomm** và **Thales** để hỗ trợ mạng 5G private cho hệ thống đường sắt đô thị.
2. Với **bảo mật OT**, L&T Technology Services hợp tác với **Palo Alto Networks** với vai trò MSSP để cung cấp nền tảng bảo mật OT cho các ngành công nghiệp.

### Câu trả lời của GraphRAG

GraphRAG xác định đúng quan hệ với **Palo Alto Networks** cho bảo mật OT, nhưng phần 5G đường sắt bị suy diễn từ một quan hệ acquisition với Smart World & Communication thay vì lấy đúng partnership với Qualcomm và Thales.

### Đánh giá

- **Chênh lệch chất lượng:** -3.00 cho GraphRAG.
- **Các nguyên nhân cần kiểm tra:** thiếu cạnh extraction, entity seed chưa tối ưu hoặc retrieval không đi đúng nhánh quan hệ cần thiết.
- Các trường `graph_route`, matched seeds và số lượng edge đều được export theo từng câu hỏi để có thể truy vết thay vì đoán nguyên nhân.

---

# Phần 3 — Reflection và kế hoạch cải tiến

**Sinh viên:** Chu Nguyễn Tuấn Anh  
**MSSV:** 2A202601755

## Liên hệ nội dung bài học với phần triển khai

| Khái niệm | Phần triển khai | Quan sát |
|---|---|---|
| Coreference thận trọng | `resolve_coreferences()` | Chỉ gọi LLM với chunk có trigger đại từ/tham chiếu chung; trường hợp mơ hồ vẫn có thể audit. |
| Allowlist schema nghiêm ngặt | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Relation ngoài schema bị loại trước khi ingest. |
| Entity Resolution | `build_resolution_map()` | FAISS ANN chỉ dùng để tìm candidate; lexical/type guard quyết định merge có an toàn hay không. |
| Bulk ingestion | `Neo4jStore.bulk_insert_*()` | Dùng batch `UNWIND`, không ghi từng row qua network. |
| Xử lý Super-node | `GraphRetriever.retrieve()` | Node có degree > 100 bị giới hạn tối đa 50 cạnh gần nhất và còn chịu global edge cap. |
| Evaluation | `evaluate()` + `Judge` | Cùng một Golden Dataset và ba tiêu chí judge giúp đo rõ đánh đổi chất lượng/chi phí. |

## Bài học từ quá trình debug

Một test near-dedup deterministic cho thấy việc hash `title + body` có thể bỏ sót các bản sao được syndicate khi tiêu đề thay đổi nhưng phần nội dung chính gần như giống nhau. Pipeline được sửa để fingerprint phần nội dung bài viết và lưu audit table cho các quyết định near-dedup.

CI cũng từng phát hiện workflow bị trigger trùng nhiều lần. Sau đó pipeline được tách thành:

- lightweight CI cho push thông thường;
- integration workflow chỉ chạy khi sentinel được thay đổi có chủ đích.

Cách này giảm việc chạy API không cần thiết và giúp quá trình thử nghiệm dễ kiểm soát hơn.

## Kế hoạch cải tiến nếu đưa lên production

Nếu phát triển thành một knowledge assistant production, các nguyên tắc cần giữ gồm:

- lưu provenance trên mọi relation;
- namespace dữ liệu graph theo dataset/version;
- chỉ merge entity khi có cả semantic evidence và lexical/type evidence phù hợp;
- dùng self-correcting retrieval trước khi mở rộng bán kính graph quá lớn;
- dùng community reports cho câu hỏi mang tính toàn cục;
- dùng local BFS cho câu hỏi tập trung vào entity cụ thể;
- bổ sung cache, monitoring và cơ chế kiểm soát chi phí LLM.

---

# Phần 4 — Kiểm chứng Bonus

## Bonus A — Low-level / High-level Retrieval

Pipeline có cả hai tầng truy hồi:

- **Low-level:** truy hồi theo entity và graph neighborhood cho câu hỏi cụ thể.
- **High-level:** truy hồi theo community reports cho câu hỏi tổng quan trên toàn bộ corpus.

Query router được kiểm chứng bằng demo:

- Số query được route sang **local**: 2
- Số query được route sang **global**: 2

Evidence được lưu tại `outputs/retrieval_router_demo.csv`.

## Bonus B — Global Search qua Community Reports

Pipeline sử dụng fallback NetworkX với thuật toán `networkx.greedy_modularity_communities` để phát hiện cộng đồng trên graph, sau đó ghi `community_id` trở lại Neo4j bằng batch `UNWIND`.

Kết quả full run:

- **Số community:** 164
- **Số community được LLM tóm tắt:** 164/164
- **Tỷ lệ LLM summary:** 100%

Các community report được dùng làm high-level context cho global search.

## Bonus C — Self-Correction Graph Retrieval

Self-correction triển khai đúng chuỗi:

1. Truy hồi **hop 2**.
2. LLM kiểm tra context đã đủ hay chưa.
3. Nếu chưa đủ, mở rộng sang **hop 3**.
4. LLM kiểm tra lại lần thứ hai.
5. Nếu vẫn chưa đủ, thực hiện **vector fallback**.
6. Mọi nhánh đều có **stop condition** rõ ràng.

Kết quả kiểm chứng trên 12 câu bonus:

- `hop2`: 3 câu
- `hop3`: 1 câu
- `hop3+vector`: 8 câu
- Tỷ lệ context được LLM đánh giá đủ ở hop 2 ban đầu: **25%**
- Tỷ lệ context đủ sau self-correction: **100%**
- Mức cải thiện: **+75 điểm phần trăm**
- Độ dài context trung bình: từ **2333.667** ký tự lên **5517.500** ký tự
- Cả 12/12 trường hợp đều có terminal stop condition

Evidence được lưu tại `outputs/self_correction_bonus_eval.csv` và `outputs/bonus_before_after.csv`.

## Bonus Near-Dedup

Near-dedup sử dụng **SimHash + LSH** thay vì so sánh pairwise O(N²).

- **Số near-duplicate được loại:** 5

Evidence được lưu tại `outputs/near_dedup_audit.csv`.

---

# Phần 5 — Tổng kết kết quả bài Lab

Lần chạy full cuối cùng sử dụng chính sách **chỉ lấy 5.000 dòng đầu tiên** của nguồn dữ liệu theo thứ tự xác định, không sampling.

Các chỉ số chính:

- Dòng dữ liệu nguồn: **5.000**
- Bài viết sau dedup: **2.114**
- Chunk được index: **2.114**
- Chunk được LLM extraction: **400**
- Triple hợp lệ: **268**
- Node trong Neo4j: **418**
- Golden Dataset đã evaluate: **50/50**
- Entity Resolution audit rows: **16**
- Near-duplicate bị loại: **5**
- Community reports: **164**
- Edge thiếu provenance: **0**
- Extraction error còn lại: **0**

Toàn bộ kết quả chi tiết được lưu trong thư mục `outputs/`, báo cáo trong `reports/`, và notebook đã được chạy đầy đủ để phục vụ việc kiểm tra và chấm bài.
