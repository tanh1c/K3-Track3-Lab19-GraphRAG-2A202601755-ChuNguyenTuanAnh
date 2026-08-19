# Technical Defense — Lab 19 GraphRAG vs Flat RAG

## 1. Coreference sai ở tình huống nào?
False coreference xảy ra khi đại từ/generic mention có nhiều antecedent hợp lệ trong cùng chunk. Pipeline dùng conservative trigger và giữ nguyên văn bản khi mơ hồ; unresolved mentions được checkpoint để audit. Điều này ưu tiên precision vì một false coreference có thể tạo false edge lan truyền trong graph.

## 2. Entity-resolution threshold là bao nhiêu, vì sao?
Vector candidate threshold là **0.90**. Sau ANN candidate search, lexical/type guard vẫn bắt buộc. Mức này ưu tiên precision; false merge nguy hiểm hơn false split trong knowledge graph dùng cho reasoning.

## 3. Candidate similarity cao nhưng không nên merge?
Ví dụ audit thực nghiệm: **Ivy Tech Community College - Columbus ↔ Ivy Tech Community College (sim=0.899)**. Guard đặc biệt chặn company-vs-product prefix và person chỉ trùng họ.

## 4. Top 3 super-node và degree?
- 1. Microsoft (Company), degree=9
- 2. artificial intelligence (Technology), degree=7
- 3. NVIDIA (Company), degree=5

## 5. Vì sao ưu tiên edge mới nhất có thể đúng/sai?
Đúng cho câu hỏi trạng thái hiện tại và kiểm soát context explosion; sai với câu hỏi lịch sử vì cạnh cũ có thể là evidence quyết định. Vì vậy policy chỉ kích hoạt khi degree >100, cap 50, đồng thời giữ provenance/date để failure analysis.

## 6. Flat RAG thắng nhóm nào?
Kết quả theo group nằm trong `outputs/graphrag_vs_flatrag_summary.csv`. Ca có Graph-minus-Flat thấp nhất là **G5000-44 (multi-hop)**, delta quality=-3.00.

## 7. GraphRAG thắng nhóm nào?
Ca có Graph-minus-Flat cao nhất là **G5000-08 (multi-hop)**, delta quality=3.33. Graph traversal hữu ích khi evidence phải nối qua nhiều entity/document.

## 8. Latency/token trade-off?
Flat latency trung bình=2.69s; GraphRAG=5.40s. Flat token trung bình=949.8; GraphRAG=1770.0. GraphRAG trả giá bằng traversal + graph context để tăng khả năng multi-hop.

## 9. AI Coding Agent đề xuất gì mà không dùng, vì sao?
Không dùng pairwise cosine O(N²) cho near-dedup/entity resolution. Near-dedup dùng SimHash+LSH; entity resolution dùng FAISS ANN + lexical guard. Không dùng golden metadata để chọn extraction chunks vì đó là benchmark leakage.

## 10. Scale lên toàn bộ dataset: bottleneck đầu tiên là gì?
LLM extraction/rate limit là bottleneck đầu tiên, sau đó mới tới embedding/indexing và graph fan-out. Hướng scale: durable queue + checkpoint, async workers theo quota, ANN/blocking cho resolution, batch UNWIND, partition/community summaries và cache retrieval.

## Empirical metrics
- **Comprehensiveness:** Flat=3.64, Graph=3.88
- **Faithfulness:** Flat=3.68, Graph=3.88
- **Multi-hop reasoning:** Flat=3.44, Graph=3.74
- Graph nodes=415, edges=266, invalid provenance=0.
