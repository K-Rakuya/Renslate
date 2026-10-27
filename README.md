# Renslate

> Bộ công cụ tự động hoá dịch game Ren'Py sang tiếng Việt bằng Local LLM — Giữ đúng ngữ cảnh, chuẩn xưng hô và bảo toàn tuyệt đối mã game.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-0%20(Pure%20Stdlib)-success.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/Tests-1272%20Passed-brightgreen.svg)](tests/)
[![Typing](https://img.shields.io/badge/Typing-Strict%20Mypy-blueviolet.svg)](src/renslate/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#giấy-phép-license)

---

## Giới thiệu (Overview)

**Renslate** là bộ công cụ mã nguồn mở giúp tự động hoá quy trình bản địa hoá (localization) visual novel nền tảng Ren'Py sang tiếng Việt chất lượng cao bằng các mô hình ngôn ngữ lớn (LLM) chạy cục bộ (như **Qwen 2.5**, **Qwen 3.5**, **MoE** thông qua `llama.cpp`, Ollama hoặc LM Studio).

Renslate được thiết kế để giải quyết triệt để các vấn đề cốt lõi:

- 🎭 **Chuẩn xưng hô & ngữ cảnh**: Tự động nhận diện mối quan hệ giữa các nhân vật để chọn đúng đại từ xưng hô tự nhiên.
- 🛡️ **Bảo vệ mã game tuyệt đối**: Tự động nhận diện và che chắn toàn bộ biến số (`[player]`), thẻ định dạng (`{b}`, `{color}`) và ký tự đặc biệt của Ren'Py.
- 📦 **Hỗ trợ game đóng gói (`.rpyc`)**: Tự động đọc và trích xuất câu thoại trực tiếp từ file bytecode `.rpyc` mà không bắt buộc phải có mã nguồn gốc hay cài đặt Ren'Py SDK.
- ⚡ **Siêu nhẹ & Tiện lợi**: 100% viết bằng thư viện chuẩn Python 3.10+ (**zero external runtime dependencies**), khởi động tức thì, không lo xung đột môi trường.
- 🖥️ **Giao diện trực quan**: Tích hợp Studio Wizard tương tác từng bước (`renslate wizard`) và bảng điều khiển terminal thời gian thực (`renslate run`).

---

## Cài đặt (Installation)

### Yêu cầu
- **Python**: Phiên bản 3.10 trở lên.
- **Server LLM**: `llama-server` (`llama.cpp`), LM Studio, Ollama hoặc bất kỳ server nào tương thích OpenAI API (mặc định tại `http://127.0.0.1:8080/v1`).

### Cài đặt nhanh

```bash
git clone https://github.com/<username>/renslate.git
cd renslate
pip install -e .
```

*Mẹo: Bạn cũng có thể chạy trực tiếp từ thư mục mã nguồn mà không cần cài đặt:*
```bash
python3 -m renslate --help
```

---

## Bắt đầu nhanh (Quick Start)

### Cách 1: Sử dụng Studio Wizard (Khuyên dùng)

Giao diện tương tác từng bước trên terminal — không cần ghi nhớ câu lệnh:

```bash
renslate wizard
# hoặc chỉ cần gõ:
renslate
```

Trình hướng dẫn sẽ tự động:
1. Phát hiện phiên bản game và tệp kịch bản (`.rpy` hoặc `.rpyc`).
2. Khởi tạo dự án và trích xuất câu thoại vào cơ sở dữ liệu.
3. Phân tích nhân vật, quan hệ và dịch bảng thuật ngữ.
4. Kiểm tra kết nối với server LLM.
5. Tiến hành dịch và hiển thị tiến độ trực quan.
6. Cài đặt font tiếng Việt và áp dụng bản dịch vào game.

---

### Cách 2: Chạy dịch tự động qua đêm (Autonomous Mode)

Chỉ với một câu lệnh duy nhất, hệ thống sẽ tự động dịch, tự phục hồi khi gặp sự cố mạng và áp dụng ngay vào game khi hoàn tất:

```bash
renslate run -p /đường/dẫn/đến/game --auto-apply
```

---

### Cách 3: Quy trình dòng lệnh từng bước (Step-by-step CLI)

Nếu bạn muốn kiểm soát chi tiết từng công đoạn:

```bash
# 1. Khởi tạo dự án dịch
renslate init -p /đường/dẫn/game

# 2. Trích xuất câu thoại (hỗ trợ cả .rpy lẫn .rpyc)
renslate extract -p /đường/dẫn/game

# 3. Phân tích nhân vật & thuật ngữ bằng LLM
renslate analyze -p /đường/dẫn/game --bootstrap

# 4. Tiến hành dịch thoại theo lô
renslate translate -p /đường/dẫn/game

# 5. Kiểm tra chất lượng và phát hiện lỗi bản dịch (QA)
renslate qa -p /đường/dẫn/game

# 6. Cấu hình phông tiếng Việt và áp dụng vào game
renslate apply -p /đường/dẫn/game --setup-font
```

---

## Bảng tra cứu lệnh CLI (Command Reference)

| Lệnh | Cú pháp tiêu biểu | Chức năng |
|:---|:---|:---|
| `wizard` | `renslate wizard` | Giao diện hướng dẫn tương tác trực quan 6 pha (Studio Mode). |
| `run` / `auto` | `renslate run [--auto-apply]` | Dịch tự động liên tục với TUI Dashboard theo dõi tiến độ. |
| `init` | `renslate init -p <game>` | Khởi tạo cấu hình và database cho dự án dịch. |
| `extract` | `renslate extract -p <game>` | Trích xuất câu thoại và chuỗi UI từ game vào database. |
| `decompile` | `renslate decompile <target.rpyc>` | Dịch ngược tệp bytecode `.rpyc` thành mã nguồn `.rpy`. |
| `analyze` | `renslate analyze -p <game> [--bootstrap]` | Phân tích nhân vật, đồ thị quan hệ và khai thác thuật ngữ. |
| `translate` | `renslate translate -p <game>` | Dịch các câu thoại theo lô qua LLM server. |
| `qa` | `renslate qa -p <game>` | Kiểm tra tính toàn vẹn cú pháp, biến số và đại từ xưng hô. |
| `dashboard` | `renslate dashboard` | Bảng điều khiển terminal thời gian thực xem tốc độ và tiến độ. |
| `preview` | `renslate preview -p <game> [-n 10]` | Xem trước các câu đã dịch song ngữ song song trên terminal. |
| `export` | `renslate export -p <game> -o review.tsv` | Xuất dữ liệu ra file TSV để biên tập thủ công trên Excel/Sheets. |
| `import` | `renslate import -p <game> -i review.tsv` | Nạp bản dịch đã sửa tay từ TSV và đồng bộ vào Translation Memory. |
| `apply` | `renslate apply -p <game> [--setup-font]` | Ghi bản dịch vào tệp dịch của game kèm cấu hình phông tiếng Việt. |

---

## Biên tập thủ công (Human Review)

Renslate hỗ trợ xuất/nhập tệp bảng tính TSV (tương thích hoàn toàn với Microsoft Excel và Google Sheets):

```bash
# Xuất các câu cần xem xét ra file TSV
renslate export -p /đường/dẫn/game -o review.tsv

# Sau khi chỉnh sửa xong, nạp lại vào dự án:
renslate import -p /đường/dẫn/game -i review.tsv
```

*Các câu chỉnh sửa thủ công sẽ được gắn nhãn `manual` và tự động lưu vào Bộ nhớ dịch (Translation Memory) để ưu tiên tái sử dụng, không bị ghi đè bởi máy dịch.*

---

## Cấu hình LLM Server (Local Inference)

Mặc định, Renslate kết nối tới `http://127.0.0.1:8080/v1` (tương thích OpenAI API). Bạn có thể tùy chỉnh endpoint hoặc model trong tệp `.renslate/config.json`.

Dự án có sẵn script khởi chạy `llama.cpp` (`llama-server`) được tối ưu hoá cho cấu hình máy tính cá nhân (tự động cân đối VRAM, hạn chế hao mòn SSD và tiết kiệm tài nguyên):

```bash
./scripts/run_qwen_server.sh
```

---

## Cấu trúc không gian làm việc (`.renslate/`)

Khi bạn làm việc trên một tựa game, Renslate lưu trữ dữ liệu tại thư mục `.renslate/` bên trong thư mục trò chơi:

- `config.json`: Cấu hình dự án (ngôn ngữ đích, endpoint LLM, kích thước batch).
- `project.db`: Cơ sở dữ liệu SQLite lưu trữ toàn bộ đơn vị dịch và Translation Memory.
- `characters.json`: Danh sách hồ sơ nhân vật, hình mẫu tính cách và quy tắc xưng hô.
- `relationships.json`: Mạng lưới đồ thị quan hệ giữa các nhân vật (cho phép xem và chỉnh sửa).
- `glossary.json`: Bảng thuật ngữ chuyên ngành đã dịch.
- `backups/`: Các bản sao lưu tự động trước mỗi lần ghi tệp vào game.

---

## Kiểm thử & Phát triển (Contributing)

Dự án duy trì tiêu chuẩn kiểm thử tự động nghiêm ngặt:

```bash
# Chạy bộ test suite (1,272 test cases):
pytest -q

# Kiểm tra linting và tĩnh kiểu:
ruff check .
mypy src tests
```

---

## Giấy phép (License)

Dự án được phân phối dưới giấy phép **MIT License**.
