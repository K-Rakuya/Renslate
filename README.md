# vitrans

Bộ công cụ tự động hoá dịch game Ren'Py sang tiếng Việt bằng model local (Gemma 4 26B MoE, chạy qua `llama.cpp`). Thiết kế đầy đủ xem [docs/DESIGN.md](docs/DESIGN.md).

Trạng thái: **giai đoạn M1** — đọc, lưu và ghi lại các tệp dịch của Ren'Py một cách an toàn. Các bước dịch bằng model (M2 trở đi) chưa có.

## Yêu cầu

- Python 3.10 trở lên, không có phụ thuộc ngoài thư viện chuẩn.
- Ren'Py SDK (để sinh khung dịch bằng `renpy translate`).

## Cài đặt

```bash
pip install -e ".[dev]"
```

Hoặc chạy trực tiếp không cần cài: `PYTHONPATH=src python -m vitrans --help`.

## Quy trình

```bash
vitrans init     -p /đường/dẫn/game --sdk /đường/dẫn/renpy-sdk
vitrans generate -p /đường/dẫn/game      # sinh game/tl/vietnamese/*.rpy
vitrans extract  -p /đường/dẫn/game      # nạp các chuỗi vào .vitrans/project.db
vitrans status   -p /đường/dẫn/game
vitrans apply    -p /đường/dẫn/game      # ghi bản dịch vào tệp dịch (có sao lưu)
```

| Lệnh | Tác dụng |
|------|----------|
| `init` | Tạo `.vitrans/config.json` và cơ sở dữ liệu trong thư mục dự án. |
| `generate` | Gọi `renpy translate <ngôn ngữ> --no-todo` để sinh hoặc cập nhật khung dịch. |
| `extract` | Đọc các tệp dịch, tạo hoặc cập nhật đơn vị dịch; chạy lại bao nhiêu lần cũng an toàn. Trả mã thoát 1 nếu có lỗi cú pháp. |
| `apply` | Ghi bản dịch đã lưu vào đúng dòng của tệp dịch. `--dry-run` chỉ báo cáo. |
| `status` | Thống kê đơn vị theo loại và trạng thái. |

`apply` không bao giờ đụng vào `.rpy` gốc của game, chỉ sửa các tệp trong `game/tl/<ngôn ngữ>/`. Bản gốc của từng tệp được sao lưu vào `.vitrans/backup/` trước khi ghi, và việc ghi bị huỷ toàn bộ nếu tệp đã thay đổi kể từ lần `extract` gần nhất hoặc có bản dịch chứa markup lỗi.

## Phát triển

```bash
pytest
ruff check .
mypy
```

Cấu trúc mã nguồn:

| Mô-đun | Vai trò |
|--------|---------|
| `literals` | Mã hoá/giải mã chuỗi lời thoại Ren'Py và chuỗi `old`/`new`. |
| `mask` | Che và khôi phục markup (`[var]`, `{i}`, `[[`, `%%`) bằng placeholder, có kiểm tra. |
| `parser` | Phân tích tệp dịch thành đơn vị dịch và danh sách vấn đề. |
| `config`, `project` | Cấu hình có kiểm tra chặt và bố cục thư mục dự án. |
| `db`, `models` | Kho SQLite (lược đồ có migration) và các kiểu dữ liệu. |
| `extract`, `writer` | Đồng bộ kho với tệp dịch và ghi bản dịch ngược lại. |
| `sdk`, `cli` | Gọi Ren'Py SDK và giao diện dòng lệnh. |
