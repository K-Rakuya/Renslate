# RENVI ARCHITECTURAL QUALITY STANDARDS & CODE CONVENTIONS
# TIÊU CHUẨN CHẤT LƯỢNG & QUY TẮC PHÁP LỆNH KỸ THUẬT RENVI (2026+)

Toàn bộ các kỹ sư và AI agents khi tham gia phát triển, bảo trì hoặc mở rộng codebase Renvi (`renslate`) BẮT BUỘC phải tuân thủ nghiêm ngặt 8 Trụ cột Tiêu chuẩn Chất lượng & Pháp lệnh Kỹ thuật dưới đây:

---

## 1. ĐẶT TỔNG QUÁT REN'PY LÊN HÀNG ĐẦU (GENERAL-PURPOSE REN'PY PARADIGM)
- Mọi module, giải thuật và bộ phân tích cú pháp phải vận hành chuẩn xác trên **mọi game Visual Novel Ren'Py bất kỳ** (hỗ trợ cả kịch bản text `.rpy` và mã nhị phân bytecode `.rpyc`, cấu trúc nhiều file, nhiều chương, tách biệt kịch bản thoại khỏi mã giao diện UI/Screens).
- **Tuyệt đối không viết giải pháp chắp vá (ad-hoc patches/band-aids) hay regex đối phó** cho riêng một câu thoại, một góc biên hay một game mẫu cụ thể. Mọi hành vi ngôn ngữ phải xuất phát từ các quy luật cấu trúc diễn ngôn tổng quát.

## 2. TUYỆT ĐỐI KHÔNG HARDCODE & KHÔNG SUY ĐOÁN (ZERO HARDCODING & ZERO SPECULATION)
- **Nghiêm cấm tuyệt đối** việc đưa bất kỳ tên nhân vật, ID, tag phát ngôn, đại từ xưng hô, danh xưng gia tộc, hay quy tắc quan hệ của game mẫu vào mã nguồn dưới bất kỳ hình thức nào (kể cả inline tuple, inline list hay constant set).
- **100% dữ liệu phải được trích xuất động và suy biến từ chứng cứ kịch bản thực tế:** Cây cú pháp AST Ren'Py, đồ thị mạng lưới đồng xuất hiện ($N \times N$ co-occurrence matrix), sự hiện diện của sprite (`show`/`hide`), và chưng cất cửa sổ tương tác then chốt bằng LLM theo schema có cấu trúc.
- Không suy đoán mù mờ: Nếu không có chứng cứ cấu trúc hoặc ngữ cảnh, phải trả về trạng thái trung tính/mặc định chứ không được gán ép giả định.

## 3. KHÔNG COMMENT RÁC – MÃ NGUỒN TỰ DIỄN GIẢI (CLEAN CODE & NO JUNK COMMENTS)
- Mã nguồn phải tự diễn giải (Self-documenting), phân tách ranh giới mạch lạc theo chuẩn Hexagonal Architecture (Ports and Adapters) và Domain-Driven Design (DDD).
- **Nghiêm cấm viết comment rác:** Không viết comment giải thích vụn vặt, không viết lời xin lỗi, không để lại ghi chú workaround hay ghi chú tạm bợ.
- Chỉ duy trì standard docstrings súc tích, chuẩn mực cho các Public API và giao diện Protocol thực sự cần thiết.

## 4. KHÔNG GIẢN LƯỢC HÓA & ĐỔI MỚI TRIỆT ĐỂ (DEEP MODULES & BOLD INNOVATION OVER LEGACY)
- Xây dựng **Deep Modules** (Giao diện API bên ngoài tối giản, nhưng ẩn giấu bên trong giải thuật tính toán sâu rộng và mạnh mẽ). Không viết hàm giả lập (stub), không đi đường tắt nửa vời làm mất mát dữ liệu (lossy shortcuts).
- **Không bị ràng buộc bởi logic cũ:** Nếu một giải pháp thiết kế mới từ đầu (ground-up clean architecture) mang lại hiệu quả thực sự vượt trội, rõ ràng và thanh thoát, **hãy mạnh dạn đổi mới và thay thế toàn bộ logic cũ**. Tuyệt đối không giữ lại hay vá víu những heuristic chắp vá đã bộc lộ sự yếu kém.

## 5. CHỈ DÙNG DỰ ÁN THỰC TẾ ĐỂ CHẤM ĐIỂM CHẤT LƯỢNG (REAL-WORLD BENCHMARK VERIFICATION)
- Mọi đánh giá chất lượng phân tích, độ chính xác của trích xuất quan hệ và hiệu quả dịch thuật **bắt buộc phải được chạy và thẩm định trực tiếp trên cây thư mục của các tựa game Ren'Py thực tế** (bao gồm cả các bản phân phối kịch bản nguồn .rpy và các bản đóng gói bytecode .rpyc thương mại).
- Các bộ test nhân tạo (synthetic unit tests) được dùng để bảo toàn hồi quy (regression) và tính chất toán học, nhưng thước đo năng lực tối thượng của hệ thống là khả năng vận hành mượt mà, chính xác trên các game thương mại thật.

## 6. KHÔNG GIỮ RÁC CŨ & TRIỆT TIÊU ĐƯỜNG DẪN FALLBACK LÙI (ZERO LEGACY DEAD CODE & ZERO STEALTH FALLBACK)
- Khai tử triệt để các hệ thống lỗi thời đã chứng minh là cản trở (Xóa sổ 100% Glossary, xóa bỏ bảng map string phẳng `pronoun_map={"y": "tôi - cô"}`, xóa bỏ chuỗi regex đoán mò người nghe trên text trần).
- **Kiểm toán đường dẫn thực thi (Callsite Audit):** Toàn bộ các luồng runtime trong `cli.py`, `runner.py`, `translate.py`, `builder.py`, `resolver.py` bắt buộc phải **thực sự và duy nhất** sử dụng các module kiến trúc mới 2026:
  * AST Grounding Engine (`src/renslate/renpy/grounding.py`)
  * Vector Quan hệ 4 chiều & Toán tử Chiếu đại số $\Pi(\mathbf{R})$ (`src/renslate/domain/relational.py`)
  * 2-Pass Hierarchical Bootstrap (`src/renslate/engine/analyze/topology.py` & `distill.py`)
  * TransGraph In-Context Micro-Framing (`build_transgraph_frame()`)
- Tuyệt đối không cho phép tồn tại bất kỳ đường dẫn ngầm (stealth fallback) nào âm thầm lùi về heuristic cũ.

## 7. QUY CHUẨN KỸ SƯ HÀNG ĐẦU THẾ GIỚI & PURE STANDARD LIBRARY (WORLD-CLASS ENGINEERING & PURE STDLIB)
- **100% Pure Python 3.10+ Standard Library:** Tuyệt đối không bổ sung bất kỳ thư viện ngoài nào vào runtime (`dependencies = []` trong `pyproject.toml`).
- Vận hành tin cậy trên môi trường đơn luồng (`workers=1`), lưu trữ giao dịch SQLite ACID bền vững với composite indexes.
- **Bảo đảm Tính chất Toán học:** Mọi cặp xưng hô đối ứng tiếng Việt bắt buộc phải thỏa mãn tính khép kín đại số (Algebraic Reciprocal Invariant):
  $$\Pi(\mathbf{R}_{(B, A)}, g_B, g_A) \equiv \sigma(\Pi(\mathbf{R}_{(A, B)}, g_A, g_B))$$
- Bảo đảm 100% test suite chạy xanh (0 failure), linter `ruff` sạch sẽ không cảnh báo (All checks passed!).

## 8. TRIỆT ĐỂ BẢO VỆ DỮ LIỆU & VÔ DANH HÓA TOÀN BỘ BENCHMARK (ZERO BENCHMARK LEAKAGE & CLEAN TRACKING)
- **Nghiêm cấm tuyệt đối** việc đưa bất kỳ thông tin, tên dự án, đường dẫn tuyệt đối/cục bộ, hay định danh cụ thể của bất kỳ tựa game/dữ liệu nào đã dùng để huấn luyện và kiểm nghiệm vào tracking hoặc lịch sử của toàn bộ repo (kể cả trong tài liệu kỹ thuật, mã nguồn, tests, scripts, commit messages hay git metadata).
- **100% kiểm thử thực tế phải trừu tượng hóa và nạp động:** Toàn bộ đường dẫn game thực tế phục vụ benchmark bắt buộc phải được nạp động thông qua biến môi trường độc lập (ví dụ: `RENVI_TEST_GAME_RPY`, `RENVI_TEST_GAME_RPYC`), không commit hay hardcode bất kỳ thư mục game cá nhân nào vào repo.
- **Vô danh hóa toàn diện tài liệu & script:** Mọi tài liệu kỹ thuật (docs/ADRs), mã kiểm thử (tests), script khởi chạy (launchers) bắt buộc sử dụng danh xưng trừu tượng trung tính (`CommercialVN-Alpha`, `CommercialVN-Beta`, `/path/to/...`), không bao giờ để lộ cây thư mục lưu trữ hay tên dự án cá nhân.
