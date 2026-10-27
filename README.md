# vitrans

Bộ công cụ tự động hoá dịch game Ren'Py sang tiếng Việt bằng model local (chạy qua `llama.cpp` `llama-server`, LM Studio hoặc server tương thích OpenAI API). Không sử dụng bản cắt giảm, tích hợp trọn vẹn toàn bộ quy trình: trích xuất, phân tích nhân vật và ngữ cảnh, dịch theo lô kèm bộ nhớ dịch và thuật ngữ, kiểm định chất lượng (QA), xuất/nhập TSV review thủ công và ghi ngược lại vào game.

Thiết kế chi tiết xem [docs/DESIGN.md](docs/DESIGN.md).

## Yêu cầu

- Python 3.10 trở lên, **zero external dependency** ở runtime (chỉ dùng thư viện chuẩn Python).
- Server LLM local: `llama.cpp` (`llama-server`) hoặc LM Studio / Ollama chạy cổng `http://127.0.0.1:8080/v1`.
- Ren'Py SDK (để sinh khung dịch ban đầu bằng `renpy translate`).

## Cài đặt

```bash
pip install -e ".[dev]"
```

Hoặc chạy trực tiếp không cần cài đặt: `PYTHONPATH=src python -m vitrans --help`.

## Quy trình sử dụng hoàn chỉnh

```bash
# 1. Khởi tạo dự án dịch
vitrans init -p /đường/dẫn/game --sdk /đường/dẫn/renpy-sdk

# 2. Sinh khung dịch bằng Ren'Py SDK (game/tl/vietnamese/*.rpy)
vitrans generate -p /đường/dẫn/game

# 3. Trích xuất đơn vị dịch vào kho SQLite (.vitrans/project.db)
vitrans extract -p /đường/dẫn/game

# 4. Phân tích nhân vật, phân cảnh và khai thác ứng viên thuật ngữ
vitrans analyze -p /đường/dẫn/game

# 5. Dịch tự động bằng model local qua llama.cpp
vitrans translate -p /đường/dẫn/game

# 6. Kiểm định chất lượng bản dịch (QA)
vitrans qa -p /đường/dẫn/game

# 7. (Tuỳ chọn) Xuất TSV để biên tập thủ công trên Excel / Google Sheets
vitrans export -p /đường/dẫn/game -o review.tsv
# Sau khi sửa xong trên TSV, nhập lại (ưu tiên manual, tự động đồng bộ vào TM):
vitrans import -p /đường/dẫn/game -i review.tsv

# 8. Xem thống kê tiến độ
vitrans status -p /đường/dẫn/game

# 9. Ghi bản dịch an toàn vào game (có sao lưu tự động và kiểm tra tính toàn vẹn)
vitrans apply -p /đường/dẫn/game --setup-font
```

## Bảng lệnh CLI

| Lệnh | Tác dụng |
|------|----------|
| `init` | Tạo `.vitrans/config.json` và cơ sở dữ liệu SQLite trong dự án. |
| `generate` | Gọi Ren'Py SDK (`renpy translate <ngôn ngữ> --no-todo`) để sinh/cập nhật khung dịch. |
| `extract` | Quét tệp dịch, trích xuất và đồng bộ đơn vị dịch vào DB; idempotent, an toàn khi chạy lại. |
| `analyze` | Quét kịch bản phát hiện nhân vật, xưng hô, phân đoạn cảnh và khai thác glossary. |
| `translate` | Dịch theo lô qua `llama.cpp` server; tự động dùng seed UI, tái sử dụng TM exact match, quản lý ngân sách ngữ cảnh (context budgeting). |
| `qa` | Kiểm tra chất lượng: toàn vẹn placeholder, cú pháp Ren'Py, rò rỉ ngôn ngữ, glossary compliance, tỷ lệ độ dài. |
| `export` | Xuất bản dịch ra tệp TSV (UTF-8 BOM) để review thủ công. |
| `import` | Nhập bản dịch đã sửa tay từ TSV, kiểm tra markup và cập nhật vào Translation Memory. |
| `status` | Thống kê số lượng đơn vị theo loại (`dialogue`, `string`, `ui`) và trạng thái. |
| `apply` | Ghi bản dịch vào tệp `.rpy` của game; `--dry-run` để xem trước, `--setup-font` để tạo cấu hình phông chữ tiếng Việt. |

## Cấu trúc mã nguồn

| Mô-đun | Vai trò |
|--------|---------|
| `llm` | Client HTTP kết nối `llama-server` (`/v1/chat/completions`), exponential backoff, trích xuất JSON tin cậy. |
| `context` | Xây dựng prompt tối ưu, phân bổ ngân sách token, ghép glossary, hồ sơ nhân vật, tóm tắt cảnh và lookback. |
| `translate` | Điều phối dịch: gom lô theo cảnh, tái sử dụng seed UI và TM, gọi LLM, xử lý retry và phân rã lô khi lỗi. |
| `qa` | Bộ kiểm tra chất lượng dịch: tính toàn vẹn placeholder, rò rỉ ngoại ngữ, vi phạm thuật ngữ, làm sạch hallucination. |
| `tm` | Translation Memory: so khớp chính xác tức thì và mờ (fuzzy matching) với chỉ mục n-gram tăng tốc. |
| `glossary` | Quản lý bảng thuật ngữ: khớp từ theo ranh giới và độ dài giảm dần, khai thác ứng viên tự động. |
| `analyze` | Phân tích kịch bản: phát hiện `Character()`, lập hồ sơ nhân vật, phân đoạn cảnh theo label. |
| `tsvio` | Xuất/nhập TSV tương thích bảng tính, kiểm định placeholder và khóa nguồn gốc `manual`. |
| `builtin_ui` | Từ điển hạt giống dịch sẵn cho hàng trăm chuỗi giao diện Ren'Py phổ biến. |
| `mask` | Che và khôi phục markup (`[var]`, `{i}`, `[[`, `%%`) bằng placeholder, có kiểm tra. |
| `parser` | Phân tích tệp dịch thành đơn vị dịch và danh sách vấn đề. |
| `literals` | Mã hoá/giải mã chuỗi lời thoại Ren'Py và chuỗi `old`/`new`. |
| `db`, `models` | Kho SQLite (lược đồ hỗ trợ migration) và các kiểu dữ liệu domain. |
| `config`, `project` | Cấu hình có kiểm tra chặt và bố cục thư mục dự án. |
| `writer` | Ghi bản dịch nguyên tử vào tệp dịch, drift detection và sinh `vitrans_setup.rpy`. |
| `sdk`, `cli` | Gọi Ren'Py SDK và giao diện dòng lệnh. |

## Tài liệu kỹ thuật & Hướng dẫn vận hành

- [Tài liệu Thiết kế Kiến trúc (DESIGN.md)](docs/DESIGN.md)
- [Hồ sơ Quyết định Kiến trúc (ADR - Architecture Decision Records)](docs/ARCHITECTURE_DECISIONS.md)
- [Hướng dẫn Vận hành & Chạy ngầm (RUNBOOK.md)](docs/RUNBOOK.md)

## Dịch tự động qua đêm (Unattended Overnight Runner)

Để tiến hành dịch toàn bộ visual novel lớn (hàng chục nghìn câu thoại) hoàn toàn tự động, tự phục hồi khi server LLM gặp sự cố và tránh lặp vô tận:

```bash
python3 scripts/unattended_runner.py -p /đường/dẫn/game --auto-apply
```

## Phát triển & Kiểm thử

```bash
pytest -v
ruff check .
mypy src tests
```

