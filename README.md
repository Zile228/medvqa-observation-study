# Mô tả ảnh trung gian trong hỏi đáp ảnh y khoa

Mã nguồn cho các thí nghiệm đánh giá ảnh hưởng của mô tả ảnh trung gian đến độ chính xác trả lời câu hỏi y khoa ở chế độ zero-shot. Nghiên cứu so sánh trả lời trực tiếp từ ảnh và câu hỏi với trả lời có thêm mô tả do mô hình sinh. Zero-shot ở đây nghĩa là không tinh chỉnh mô hình trên dữ liệu thí nghiệm; kết quả không xác định cơ chế bên trong mô hình hoặc giá trị sử dụng lâm sàng.

Hai mô hình trả lời là **LLaVA-1.5-7B** (`L`) và **Qwen2.5-VL-7B-Instruct** (`Q`). Mô tả do chính mô hình trả lời (`SELF`) hoặc **GPT-4o-mini-2024-07-18** (`G`) tạo, theo dạng tổng quát (`GEN`), danh mục giải phẫu (`ANA`) hoặc khung quan sát lâm sàng (`GRD`). `NONE` là điều kiện không có mô tả; ví dụ `Q-SELF-GRD` là Qwen trả lời với mô tả GRD do Qwen tạo.

SLAKE và VQA-RAD tạo nhóm radiology (`RAD`); 300 ảnh chọn từ OmniMedVQA tạo nhóm native (`NAT`). Tập câu hỏi đóng gồm **688 câu RAD và 300 câu NAT, thuộc 542 ảnh**. RAD đã được khảo sát trong giai đoạn phát triển. NAT là tập held-out được chọn riêng cho đánh giá.

## Cấu trúc mã nguồn

| Thư mục | Nội dung |
| --- | --- |
| `medvqa/` | Chuẩn bị dữ liệu, dựng prompt, chạy mô hình và phân tích thống kê |
| `notebooks/` | Các bước chạy nghiên cứu, dùng trực tiếp package `medvqa` |
| `resources/`, `configs/` | Prompt, đầu vào cố định, định tuyến và phiên bản mô hình/dataset |
| `prep/`, `artifacts/` | Tập ảnh tham chiếu, bản đồ token nhãn và kế hoạch chấm/can thiệp |
| `evidence/` | Các bảng kết quả và thông tin thực nghiệm để đối chiếu với báo cáo |
| `scripts/`, `tests/` | Chuẩn bị đầu vào, xác minh bản phân phối và kiểm thử CPU |

Trong `medvqa`, bắt đầu từ `data_preparation.py` để đọc bước chuẩn bị, `models.py` để đọc phần chạy mô hình, `prompts.py` để đọc cách dựng prompt và `analysis/` để đọc cách chấm/thống kê.

## Cài và kiểm tra

Dùng **Python 3.12**. Từ thư mục chứa README, tạo rồi kích hoạt `.venv` theo hệ điều hành trước khi cài:

```bash
python -m venv .venv
python -m pip install -e ".[analysis,dev]"
python -B -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
python scripts/verify_manifest.py
python scripts/verify_evidence.py
```

Phân tích dùng NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1 và statsmodels 0.15.0. Kiểm thử CPU kiểm cách chấm, tính chỉ số, bootstrap và xử lý đầu vào; các kiểm tra cần GPU/tokenizer được bỏ qua khi thiếu phụ thuộc. Quy ước định dạng mã nằm trong `pyproject.toml`.

Chỉ cần bộ nguồn để chạy kiểm thử và kiểm các bảng minh chứng. Chạy GPU/API hoặc tính lại điểm từ từng câu cần thêm dữ liệu và kết quả thô. Các nhóm phụ thuộc `[models]`, `[api]`, `[data]` phục vụ từng bước riêng. Môi trường phải khớp `run_params.json`, kể cả bản CUDA của PyTorch: Qwen được đo trên Colab L4, LLaVA trên Kaggle T4×2. Trường `env` trong `configs/models.json` là nhãn mặc định được ghi vào bản ghi chạy; bộ phiên bản thực sự áp dụng do `medvqa.execution.env_for` chọn (Qwen trên Colab dùng `ENV-COLAB-L4`), xem `evidence/runtime_versions.csv`. Nếu snapshot không còn cung cấp, ghi rõ phần không tái lập được thay vì tự thay mô hình.

## Chạy nghiên cứu

Notebook mặc định đọc mã từ thư mục hiện tại. Nếu đặt mã ở nơi khác, đặt `MEDVQA_SOURCE_ROOT` đến thư mục chứa `medvqa`; mount Drive hoặc gắn dataset trước khi cấu hình đường dẫn. Mỗi notebook có ô cấu hình đầu vào, đầu ra và mô hình.

| Công việc | Notebook |
| --- | --- |
| Chuẩn bị ảnh và câu hỏi | `01_prepare_data` |
| Sinh mô tả bằng mô hình hoặc GPT | `02_generate_observations`, `03_generate_external_observations` |
| Trả lời, chấm mô tả, phân tích chính | `04_answer_questions`, `05_evaluate_observations`, `06_analyze_results` |
| Ba đối chứng danh sách trên Qwen | `07_neutral_list_controls` |
| Hiệu chỉnh một ảnh, rồi sinh mô tả GPT chi tiết cao | `08_check_image_detail`, `09_generate_high_detail_observations` |
| Trả lời với mô tả chi tiết cao trên hai môi trường riêng | `10_answer_high_detail_Q`, `11_answer_high_detail_L` |

**Dữ liệu.** Phiên bản SLAKE (`BoKelvin/SLAKE`) và OmniMedVQA (`foreverbeliever/OmniMedVQA`) nằm trong `configs/datasets.json`. VQA-RAD cần `VQA_RAD_Dataset_Public_original.json` và thư mục `original_public_vqarad_images`. Chuẩn bị bằng notebook 01; nếu đã có `prep_out`, tạo bộ đầu vào bằng:

```bash
python scripts/prepare_inputs.py --output data/inputs --vqa-rad /path/to/vqa_rad --prepared-data /path/to/prep_out
```

`prep_out` chứa đầu vào suy luận; `prep_sealed` chứa nhãn chấm riêng. Sinh mô tả và trả lời không đọc nhãn chấm; định tuyến không dựa vào câu hỏi/đáp án. Bộ mã không chứa toàn bộ ảnh gốc, trọng số, nhãn chấm hoặc kết quả thô. Có câu hỏi công khai, mô tả lưu sẵn và một ảnh thử PNG mã hóa base64 trong hai file hiệu chỉnh.

**Thực thi.** Notebook chính dùng `MODE="PILOT"` để kiểm định dạng và lỗi kỹ thuật trước `MODE="TEST"`; không dùng độ chính xác pilot để chọn cấu hình. Mô tả giới hạn 900 token; câu trả lời có/không và trắc nghiệm 8 token, ngắn/mở 24 token. Khi tiếp tục lượt bị gián đoạn, giữ cùng cấu hình và thư mục. Mỗi phiên lưu `records.jsonl`, `run_config.json`, `integrity.json`; đầu ra đã hoàn tất không bị ghi đè.

Notebook 08–11 mặc định `EXECUTE=False` để kiểm đầu vào trước. Sau lần thử GPT, đối chiếu usage và số dư trên dashboard trước khi chạy 599 ảnh. API key đọc từ môi trường hoặc ô nhập ẩn. Yêu cầu thiếu phản hồi, lệch token trên 15% hoặc vượt trần chi phí đều dừng để đối chiếu; không xóa dấu vết để gọi lại yêu cầu chưa rõ trạng thái. Có thể dùng lại mô tả lưu sẵn; sinh lại qua API không bảo đảm trùng từng byte dù cùng snapshot và seed.

Các phần bổ sung có lệnh Python tương ứng; dùng `--help` để xem tham số:

```bash
python -m medvqa.detail_batch --help
python -m medvqa.high_detail_answers --help
```

**Phân tích kết quả đã lưu.** Thư mục chính cần chứa các run và `prep/prep_out`, `prep/prep_sealed`, cùng cấu hình tập đánh giá. Các lệnh sau không gọi mô hình:

```bash
medvqa-analyze --runs-dir /path/to/runs --out-dir results/main
python -m medvqa.analysis.supplementary --runs-dir /path/to/runs --out-dir results/supplementary
python -m medvqa.analysis.neutral_controls --base-runs /path/to/runs --control-run /path/to/control_run --output results/neutral_controls
python -m medvqa.analysis.high_detail --base-runs /path/to/runs --q-run /path/to/high_Q --l-run /path/to/high_L --inputs resources/detail_probe/high_detail_answer_inputs.json --observations /path/to/high_observations/shard_00/records.jsonl --output results/high_detail
```

## Đọc số liệu

`evidence/manifest.json` ghi tên bảng nguồn, số dòng, tên cột và mã SHA-256. Các CSV chứa số liệu thực nghiệm, tách biệt với dữ liệu giả trong bộ kiểm thử. Các giá trị số được giữ theo nguồn; bảng định tuyến chỉ lấy các cột ước lượng và thông tin mẫu, không kèm cột diễn giải. Nhóm ảnh dự phòng được báo riêng cho câu đóng và cho toàn bộ câu hỏi:

| Nhóm bảng | Nội dung |
| --- | --- |
| `main_*.csv` | Chỉ số theo điều kiện, kiểm định xác nhận, so sánh mô tả và chất lượng đầu ra |
| `supplementary_comparisons.csv` | 76 dòng phân tích bổ sung trên kết quả đã có |
| `neutral_list_*.csv` | 14 tương phản và các phép đo chiều trả lời, đổi đáp án, phân bố xác suất |
| `high_detail_*.csv` | 12 tương phản, 16 dòng chỉ số chính và kiểm tra mô tả chi tiết cao |
| `main_routing_sensitivity.csv`, `routing_answer_overlap.csv`, `main_sensitivity.csv` | Tập câu hỏi định tuyến, mức trùng đáp án và độ nhạy; đối chiếu Bảng 5.6–5.8 |
| `correction_by_type.csv` | Kết quả can thiệp theo loại sửa mô tả |
| `judge_*.csv`, `observation_support.csv`, `observation_contradiction.csv` | Hiệu chuẩn, bao phủ, chất lượng đầu ra và kết quả chấm; đối chiếu Bảng 5.10–5.13 |
| `neutral_list_token_*.csv` | Độ dài token của văn bản và từng tên mục đối chứng |
| `answer_execution_counts.csv`, `runtime_versions.csv`, `experiment_counts.csv` | Đơn vị đếm, môi trường, số đếm chạy thử và chuẩn bị dữ liệu |
| `fallback_text_comparison.csv`, `fallback_answer_changes.csv` | So sánh mô tả và dự đoán trên nhóm ảnh dự phòng |

Chỉ số chính là độ chính xác trên câu hỏi đóng. `n_questions` đếm câu hỏi; các thứ tự lựa chọn của cùng câu được gộp trước khi báo kết quả. Bootstrap ghép cặp theo **cụm ảnh, 10.000 mẫu, seed 20261016**, RAD phân tầng theo dataset và NAT theo nhóm ảnh. `ci_low`/`ci_high` là khoảng tin cậy 95%. Hiệu độ chính xác tính bằng điểm phần trăm; cần đọc cột `metric` vì không phải mọi ước lượng đều là phần trăm, chẳng hạn Brier.

`main_confirmatory.csv` giữ nguyên các họ kiểm định xác nhận và Holm. Các phân tích bổ sung là thăm dò. Đối chứng danh sách dùng biên tương đương ±3 điểm phần trăm, báo Holm trên bảy tương phản RAD như phân tích độ nhạy; phần chi tiết cao báo p chưa hiệu chỉnh. Không kết luận tương đương chỉ vì khoảng tin cậy chứa 0, hoặc chia mức giảm thành tỷ trọng nhân quả của từng thành phần.

LLaVA-Med có mã hỗ trợ dự kiến nhưng chưa có thực nghiệm bổ sung hoàn tất. Phần mô hình hỗn hợp có script tại `resources/statistics/mixed_models.R`; phải kiểm hội tụ và phương pháp thực tế trước khi dùng hệ số. Các bảng OR chưa được xác minh đầy đủ không nằm trong `evidence`.

Mã kiểm tra và phiên bản trong cấu hình giúp phát hiện thay đổi đầu vào và đối chiếu đúng dữ liệu thực nghiệm. Kiểm checksum xác nhận tính nhất quán của tệp; kiểm thử CPU không thay cho việc tái chạy mô hình.

Mã tương phản và tên nhóm trong CSV giữ theo quy ước ban đầu, tương ứng với cách gọi trong báo cáo như sau:

| Trong báo cáo | Trong CSV và mã nguồn |
| --- | --- |
| Nhóm OMNI | `NAT` |
| C1–C3 (RQ1), họ F1 | `C1`–`C3`, họ `F1-RAD`, `F1-NAT` |
| C4, C5, C6 (RQ2), họ F2 | `C7`, `C8`, `C9`, họ `F3-RAD`, `F3-NAT` |
| C7–C9 (đổi đáp án), C10–C12 (TV), C13 (BLANK), họ F3 | `C10`–`C12`, `C13`–`C15`, `C16`, họ `F-INT` |
| EMPTY, SHUF, CORR, PARA, BLANK | `Q-INT-EMPTY`, `Q-INT-SHUF`, `Q-INT-CORR`, `Q-INT-PARA`, `Q-INT-BLANK` |

Dùng mô tả phép so sánh để nối đúng các hàng. `fallback_answer_changes.csv` ghi số câu đổi dự đoán qua ít nhất một thứ tự lựa chọn; đây không phải chênh lệch độ chính xác.
