# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Dương Trịnh Hoài An  
**Nhóm:** An, Quyền, Dũng 
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
Hai đoạn văn có high cosine similarity khi embedding của chúng trỏ gần cùng một hướng trong không gian vector, tức là chúng mang ý nghĩa gần nhau. Điều này thường xảy ra khi hai câu nói về cùng chủ đề, cùng hành động, hoặc cùng thông tin cốt lõi dù cách diễn đạt có thể khác.

**Ví dụ HIGH similarity:**
- Sentence A: `Python is widely used for machine learning projects.`
- Sentence B: `Many machine learning systems are built with Python.`
- Tại sao tương đồng: Cả hai đều nói về vai trò của Python trong machine learning.

**Ví dụ LOW similarity:**
- Sentence A: `Metadata filters improve retrieval precision.`
- Sentence B: `Bananas grow best in warm tropical climates.`
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau, gần như không chia sẻ ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
=> Cosine similarity tập trung vào hướng của vector thay vì độ lớn tuyệt đối, nên phù hợp hơn với embedding văn bản nơi ý nghĩa thường nằm ở hướng biểu diễn. Euclidean distance dễ bị ảnh hưởng bởi scale và thường kém ổn định hơn khi so sánh các vector biểu diễn ngôn ngữ.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
Áp dụng công thức:

`num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`

`= ceil((10000 - 50) / (500 - 50))`  
`= ceil(9950 / 450)`  
`= ceil(22.11)`  
`= 23`

**Đáp án:** `23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Khi overlap tăng lên 100:

`num_chunks = ceil((10000 - 100) / (500 - 100))`  
`= ceil(9900 / 400)`  
`= ceil(24.75)`  
`= 25`

=> Nếu overlap tăng lên 100 thì số chunk tăng từ `23` lên `25` vì phần chồng lặp giữa hai chunk lớn hơn, nên bước nhảy nhỏ hơn. Overlap nhiều hơn thường cần trong các bài toán phải giữ ngữ cảnh liên tục, như RAG trên tài liệu dài, hướng dẫn nhiều bước, hoặc tài liệu pháp lý/chính sách, vì thông tin quan trọng có thể nằm ngay ranh giới giữa hai chunk.

---

## 2. Document Selection - Nhóm (10 điểm)

**Chosen domain:** `Python programming basics`

**Why did the group choose this domain?**
> The group chose Python tutorial documents because all files share the same context (basic Python), making it easy to create verifiable benchmark queries. The documents are also structured by topic and section, which is ideal for comparing different chunking strategies and testing metadata filtering.

### Shared document set

| # | File | Source | Characters | Metadata |
|---|------|--------|------------|----------|
| 1 | `python_intro_and_variables.txt` | Python Tutorial - Introduction | 16,468 | `topic=intro_variables`, `source=python_docs`, `level=basic` |
| 2 | `python_conditionals_loops_functions.txt` | Python Tutorial - Control Flow Tools | 35,177 | `topic=control_flow`, `source=python_docs`, `level=basic` |
| 3 | `python_dictionaries_sets_list_tuples.txt` | Python Tutorial - Data Structures | 22,041 | `topic=collections`, `source=python_docs`, `level=basic` |
| 4 | `python_input_output.txt` | Python Tutorial - Input and Output | 18,141 | `topic=input_output`, `source=python_docs`, `level=basic` |
| 5 | `python_error_exception.txt` | Python Tutorial - Errors and Exceptions | 21,243 | `topic=exceptions`, `source=python_docs`, `level=basic` |
| 6 | `python_module.txt` | Python Tutorial - Modules | 23,266 | `topic=modules`, `source=python_docs`, `level=basic` |

### Metadata schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|---|---|---|---|
| `topic` | string | `control_flow`, `modules` | Pre-filter queries by subtopic, increasing precision when the question domain is already known. |
| `source` | string | `python_docs` | Traces a chunk back to its source file; useful for debugging and reporting. |
| `section` | string | `introduction`, `exceptions` | Identifies which section a chunk came from; supports chunk coherence analysis. |

---

## 3. Chunking Strategy - Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu mẫu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|---|---|---:|---:|---|
| `python_intro_and_variables.txt` | FixedSizeChunker (`fixed_size`) | 37 | 493.7 | Khá |
| `python_intro_and_variables.txt` | SentenceChunker (`by_sentences`) | 48 | 341.6 | Tốt |
| `python_intro_and_variables.txt` | RecursiveChunker (`recursive`) | 48 | 341.1 | Trung bình |
| `python_conditionals_loops_functions.txt` | FixedSizeChunker (`fixed_size`) | 79 | 494.6 | Khá |
| `python_conditionals_loops_functions.txt` | SentenceChunker (`by_sentences`) | 70 | 500.6 | Tốt |
| `python_conditionals_loops_functions.txt` | RecursiveChunker (`recursive`) | 98 | 354.8 | Trung bình |
| `python_module.txt` | FixedSizeChunker (`fixed_size`) | 52 | 496.5 | Khá |
| `python_module.txt` | SentenceChunker (`by_sentences`) | 47 | 493.0 | Tốt |
| `python_module.txt` | RecursiveChunker (`recursive`) | 68 | 335.6 | Trung bình |

### Strategy Của Tôi

**Loại:** FixedSizeChunker

**Mô tả cách hoạt động:**
> `FixedSizeChunker` cắt văn bản thành các chunk có độ dài cố định tối đa `chunk_size` ký tự, với phần chồng lặp `overlap` ký tự giữa các chunk liên tiếp. Bước nhảy giữa hai chunk là `chunk_size - overlap`. Strategy hoàn toàn dựa trên vị trí ký tự, không phụ thuộc vào ngôn ngữ hay cấu trúc tài liệu.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> `FixedSizeChunker` đảm bảo chunk count và chunk length nhất quán và dễ dự đoán, phù hợp với pipeline đơn giản. Với benchmark Python basic, mỗi file có độ dài đủ lớn để tận dụng overlap nhằm giảm nguy cơ cắt giữa câu; kết quả cho thấy top-3 retrieval đều trả về chunk đúng file ở mọi query nhóm.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|---|---|---:|---:|---|
| `python_module.txt` | best baseline = RecursiveChunker | 68 | 335.6 | Tốt, giữ cấu trúc section |
| `python_module.txt` | **của tôi = FixedSizeChunker** | 52 | 496.5 | Ổn định, dễ dự đoán |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Chunks | Relevant@top1 | Avg cosine score | Điểm mạnh | Điểm yếu |
|---|---|---:|---:|---:|---|---|
| Tôi (An) | FixedSizeChunker | 306 | 5/5 | 0.7052 | Đơn giản, chunk count ổn định | Dễ cắt giữa câu khi không có overlap lớn |
| Quyền | SentenceChunker | 277 | 5/5 | 0.7237 | Giữ câu trọn ý, chunk dễ đọc | Chunk size không đều, dense section có thể vượt limit |
| Dũng | RecursiveChunker | 389 | 5/5 | 0.7726 | Tôn trọng paragraph/section boundary | Chunk count cao nhất, nhiều chunk nhỏ hơn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với bộ Python tutorial notes có nhiều section và heading rõ ràng, `RecursiveChunker` cho kết quả retrieval tốt nhất vì tôn trọng cấu trúc tài liệu. `SentenceChunker` cũng cạnh tranh tốt nhờ chunk trọn câu. `FixedSizeChunker` là baseline đơn giản và đủ dùng cho pipeline cơ bản nhưng đứng sau hai strategy kia về chất lượng context.

---

---

## 4. My Approach - Cá nhân (10 điểm)

Giải thích cách tiếp cận khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** - approach:  
Tôi dùng regex tách câu theo các dấu kết thúc câu phổ biến (`.`, `!`, `?`) đi kèm khoảng trắng hoặc hết chuỗi. Sau đó, các câu được gom theo `max_sentences_per_chunk`, strip khoảng trắng dư và ghép lại thành chunk dễ đọc.

**`RecursiveChunker.chunk` / `_split`** - approach:  
Tôi viết base case cho text rỗng và text đã ngắn hơn `chunk_size`, sau đó triển khai đệ quy theo thứ tự separator ưu tiên. Thuật toán luôn cố ghép các mảnh nhỏ lại nếu còn nằm trong `chunk_size`, và chỉ fallback sang split cứng khi không còn separator nào phù hợp.

### EmbeddingStore

**`add_documents` + `search`** - approach:  
Tôi chuẩn hóa mỗi tài liệu thành một record gồm `id`, `content`, `metadata`, `embedding`, rồi lưu trong một danh sách in-memory làm source of truth cho test. Khi search, store embed query, tính dot product với embedding đã lưu, sort giảm dần theo score và trả về top-k.

**`search_with_filter` + `delete_document`** - approach:  
Tôi filter theo metadata trước rồi mới chạy similarity search để giảm noise từ đầu. Với `delete_document`, tôi gắn `doc_id` vào metadata của mọi record, từ đó có thể xóa toàn bộ chunk cùng document một cách ổn định.

### KnowledgeBaseAgent

**`answer`** - approach:  
Agent lấy top-k chunk liên quan từ `EmbeddingStore`, ghép chúng thành phần context có đánh số, rồi dựng prompt theo RAG pattern đơn giản: dùng context để trả lời. Sau đó agent gọi `llm_fn(prompt)` để giữ phần generation injectable và dễ test.

### Test Results

```text
pytest tests -q
..........................................
42 passed, 1 warning in 0.14s
```

**Số tests pass:** `42 / 42`

---

## 5. Similarity Predictions - Cá nhân (5 điểm)

Embedder: `all-MiniLM-L6-v2` (local). Score range: −1 → 1. Threshold: high ≥ 0.5, low < 0.2.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|---|---|---|---|---:|---|
| 1 | Python is a popular programming language. | Python is widely used in software development. | high | 0.8567 | Có |
| 2 | I enjoy studying machine learning. | Today it is raining heavily outside. | low | -0.0369 | Có |
| 3 | A vector store is used to save embeddings. | A vector database helps find similar items. | high | 0.4842 | Có |
| 4 | The cat is sleeping on the chair. | Object-oriented programming uses classes and objects. | low | 0.0587 | Có |
| 5 | Chunking helps split a document into small parts. | Dividing text into chunks makes retrieval more effective. | high | 0.7375 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về embeddings biểu diễn nghĩa?**
> Cặp 3 (vector store vs. vector database) cho score 0.4842 — thấp hơn kỳ vọng dù cùng domain. Điều này cho thấy embedding mã hóa cả hành động ("save" vs. "find similar"), không chỉ chủ đề; hai câu nói về hai việc khác nhau nên vector không hoàn toàn cùng hướng. Ngược lại cặp 1 đạt 0.8567 vì cả hai diễn đạt gần như cùng một mệnh đề theo cách song song.
---

## 6. Results - Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân trong package `src`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer | Chunk dự kiến chứa thông tin |
|---|---|---|---|
| 1 | What does range(5) generate in Python? | range(5) generates 0, 1, 2, 3, 4, and the end point is not included. | `python_conditionals_loops_functions.txt` |
| 2 | How can a list be used as a stack in Python? | Use append() to push items and pop() without an index to remove the last item in LIFO order. | `python_dictionaries_sets_list_tuples.txt` / `python_intro_and_variables.txt` |
| 3 | What is the difference between str() and repr()? | str() returns a human-readable representation, while repr() returns an interpreter-readable or debugging representation. | `python_intro_and_variables.txt` |
| 4 | What happens if an exception type matches the except clause? | The except clause executes, and execution then continues after the try/except block. | `python_error_exception.txt` |
| 5 | What does import fibo do and how can you access fib() after importing? | import fibo binds the module name fibo in the current namespace, and the function is called as fibo.fib(...). | `python_module.txt` |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tom tat) | Score | Relevant? | Agent Answer (tom tat) |
|---|---|---|---|---|---|
| 1 | What does range(5) generate in Python? | Chunk tu `python_conditionals_loops_functions.txt` co vi du range() in ra 0 1 2 3 4, end point never part of sequence | 0.5710 | Yes | range(5) generates 0, 1, 2, 3, 4 - endpoint 5 is not included |
| 2 | How can a list be used as a stack in Python? | Chunk tu `python_dictionaries_sets_list_tuples.txt` giai thich list as stack: last-in first-out, append() va pop() | 0.6559 | Yes | Use append() to push and pop() without index to remove last item (LIFO) |
| 3 | What is the difference between str() and repr()? | Chunk tu `python_input_output.txt` mo ta str() human-readable vs repr() cho interpreter/debugging | 0.7742 | Yes | str() returns human-readable output; repr() returns interpreter/debug representation |
| 4 | What happens if an exception type matches the except clause? | Chunk tu `python_error_exception.txt` giai thich type match thi except clause chay, sau do tiep tuc sau block | 0.6935 | Yes | The matching except clause executes, then execution continues after the try/except block |
| 5 | What does import fibo do and how can you access fib() after importing? | Chunk tu `python_module.txt` mo ta import fibo va namespace, ham goi qua fibo.fib(...) | 0.6770 | Yes | import fibo binds module name fibo in namespace; function called as fibo.fib(...) |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm - Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi thấy điểm quan trọng nhất là gán metadata đúng chủ đề. Chỉ cần topic rõ như `control_flow` và `modules` là benchmark có thể filter để truy vấn chính xác hơn rất nhiều.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Demo cho thấy cùng một bộ tài liệu nhưng chunking khác nhau có thể cho cảm nhận retrieval khác nhau. Strategy đơn giản chưa chắc tối ưu, nhưng với benchmark Python cơ bản thì cân bằng giữa context và chunk count quan trọng hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm `heading`/`section` metadata sâu hơn, chia từng file theo chủ đề trước khi chunk, và chạy benchmark ít nhất trên 2 strategy tồn tại song song để so sánh tốt hơn. Nếu có thể, tôi sẽ thử thêm custom strategy theo header trong tài liệu Python docs.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|---|---|---|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **65 / 100 + phần nhóm** |
