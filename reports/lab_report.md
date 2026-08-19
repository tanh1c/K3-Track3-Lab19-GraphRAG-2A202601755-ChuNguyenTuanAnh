# Lab 19 Report — GraphRAG vs Flat RAG

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


---

# Failure Analysis — Lab 19

## Case 1 — GraphRAG advantage
- **Question:** G5000-08 — Which external organizations are connected to ServiceNow's generative-AI efforts in the selected data, and what distinct role does each play?
- **Flat answer:** ServiceNow's generative-AI efforts involve partnerships with two key external organizations, each playing a distinct role:

1. **NVIDIA**: Partnered with ServiceNow to develop powerful enterprise-grade generative AI capabilities aimed at transforming business processes through faster, more intelligent workflow automation [chunk_id=https://www.tmcnet.com/usubmit/2023/05/17/9816504.htm::c0000].

2. **Deloitte**: Expanded its alliance with ServiceNow to integrate Now Assist generative AI capabilities into next-generation managed services. Deloitte focuses on helping customers manage ongoing operational and technology needs strategically, enhancing employee and enterprise experiences [chunk_id=https://www.businesswire.com/news/home/20231025715145/en/Deloitte-and-ServiceNow-Expand-Alliance-to-Integrate-Now-Assist-Generative-AI-Capabilities-With-Next-Generation-Managed-Services::c0000].

No other external organizations connected to ServiceNow's generative AI efforts are mentioned in the provided data.
- **Graph answer:** ServiceNow's generative-AI efforts are connected to the following external organizations, each playing a distinct role:

1. **NVIDIA**: Partnered with ServiceNow to develop powerful enterprise-grade generative AI capabilities aimed at transforming business processes through faster, more intelligent workflow automation [source_row=746].

2. **Accenture**: Partnered alongside ServiceNow and NVIDIA to accelerate generative AI adoption for enterprises, indicating a collaborative role in implementation and scaling of generative AI solutions [source_row=648].

3. **Deloitte**: Expanded its alliance with ServiceNow to integrate Now Assist generative AI capabilities with next-generation managed services, focusing on helping customers manage ongoing operational and technology needs using generative AI [source_row=411].

These partnerships highlight NVIDIA's role in technology development, Accenture's role in enterprise adoption acceleration, and Deloitte's role in integrating generative AI into managed services for operational support.
- **Quality delta:** 3.33
- **Root cause interpretation:** Flat retrieval ranks chunks independently; GraphRAG can connect canonical entities and provenance-bearing edges before combining vector evidence.

## Case 2 — GraphRAG failure / weakest relative case
- **Question:** G5000-44 — What two distinct partner ecosystems connect L&T Technology Services to advanced infrastructure in 2023: one for urban-rail 5G and one for OT security?
- **Flat answer:** In 2023, L&T Technology Services connected to advanced infrastructure through two distinct partner ecosystems: 

1. For urban-rail 5G, L&T Technology Services partnered with Qualcomm and Thales to enable 5G private networks in urban railways [chunk_id=http://www.itnewsonline.com/BusinessWire/LT-Technology-Services-and-Qualcomm-Selected-by-Thales-for-Enabling-5G-Private-Networks-in-Urban-Railways/30401].

2. For OT (Operational Technology) security, L&T Technology Services joined forces with Palo Alto Networks as an MSSP partner to offer comprehensive security platforms for OT and address industrial segments [chunk_id=https://finance.yahoo.com/news/l-t-technology-services-joins-083000328.html::c0000].
- **Graph answer:** In 2023, L&T Technology Services connects to advanced infrastructure through two distinct partner ecosystems: 

1. For urban-rail 5G, the partnership is implied through their acquisition of the Smart World & Communication (SWC) business of Larsen & Toubro Ltd (L&T), which likely supports advanced communication infrastructure [source_row=1007].

2. For OT (Operational Technology) security, L&T Technology Services partnered with Palo Alto Networks as an MSSP (Managed Security Service Provider) partner to offer comprehensive OT security platforms addressing industrial segments [source_row=471].
- **Quality delta:** -3.00
- **Likely root causes to audit:** missing extraction edge, imperfect seed/entity resolution, or super-node recency cap. `graph_route`, matched seeds and edge count are exported per question for tracing rather than guessing.


---

# Reflection & Action Plan — ChuNguyenTuanAnh

## Lecture → code mapping
| Concept | Implementation | Observation |
|---|---|---|
| Conservative coreference | `resolve_coreferences()` | LLM calls only for chunks with pronoun/generic-reference triggers; ambiguity remains auditable. |
| Strict schema allowlist | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Unsupported relation labels are rejected before ingestion. |
| Entity resolution | `build_resolution_map()` | FAISS ANN is only candidate generation; lexical/type guard decides merge safety. |
| Bulk ingestion | `Neo4jStore.bulk_insert_*()` | Uses `UNWIND` batches; no row-by-row network writes. |
| Super-node mitigation | `GraphRetriever.retrieve()` | degree >100 is capped to at most 50 recent edges plus a global edge cap. |
| Evaluation | `evaluate()` + `Judge` | Same generator family, golden reference and 3 judge dimensions expose quality/cost trade-offs. |

## Debugging lesson
A deterministic near-dedup test exposed that hashing `title + body` missed syndicated copies whose title changed while body stayed the same. The fix fingerprints article body and keeps an audit table. CI also exposed duplicate GitHub Action triggers, which were separated into lightweight push CI versus sentinel-only integration runs.

## Action plan
For a production knowledge assistant, keep provenance on every relation, namespace graph ingestion by dataset/version, gate entity merges with both semantic and lexical evidence, and add a self-correcting retrieval route before expanding graph radius. Community reports should serve global questions while local BFS handles entity-centric questions.

## Bonus evidence
- Near-duplicates removed: 5
- Community reports built: 163
