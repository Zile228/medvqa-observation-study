# Prompt cho bộ sinh can thiệp (text-only, GPT-4o-mini)

Editor chính EDIT-G: `gpt-4o-mini-2024-07-18`, temperature 0, `seed=20261016`, `max_tokens=900`, chỉ văn bản (không gửi ảnh, câu hỏi, options, đáp án). 
Editor thay thế EDIT-ALT: `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083`, greedy, `max_new_tokens=900`, cùng prompt (dạng một lượt user), cùng kiểm bằng code. Chỉ dùng theo R-GPT-LATE: thay **toàn bộ** một loại can thiệp, không trộn hai editor trong một loại.
Đầu vào duy nhất: observation ARM-GRD đã mang ID trung tính:
- PAIR-MAIN (CELL-Q-SELF-GRD): INT-PARA, INT-CORR.
- PAIR-EXT (CELL-L-G-GRD), chỉ trong thiết kế mở rộng: INT2-PARA (cùng prompt PARA). INT2-SHUF và INT2-EMPTY sinh bằng code như INT-SHUF, INT-EMPTY.
Payload API: chỉ `model`, `messages` (text), `temperature`, `seed`, `max_tokens`; không `user`/`metadata`.

## INT-PARA — paraphrase (sàn nhiễu, L2)

```
Rewrite the text below in different words while keeping every fact exactly the same.
Rules:
- Keep the same number of lines, the same order, the same item number and item name at the start of each line.
- Keep the middle field of each line (visible / partly visible / not visible / not assessable) unchanged.
- Keep every side, location, number, size word and negation unchanged in meaning.
- Change only the wording of the last field of each line. Do not add or remove information.
- Keep the length within 15% of the original.
Return only the rewritten text.

Text:
{OBSERVATION}
```

Kiểm bằng code (fail → gọi lại tối đa 2 lần với cùng prompt nhưng `seed` 20261017, 20261018 (EDIT-ALT: không có seed, greedy, nên không thử lại); vẫn fail → ảnh được đánh dấu `int_missing`, mọi câu của ảnh đó bị loại khỏi contrast INT-PARA và báo riêng số lượng):
1. Số dòng, số thứ tự và tên item khớp 100%.
2. Trường trạng thái (giữa hai dấu `|`) khớp 100%.
3. Đếm các từ khóa sau ở mỗi dòng không đổi: `left, right, image-left, image-right, upper, lower, anterior, posterior, medial, lateral, no, not, without, absent`.
4. Độ dài (số từ) trong [0,85; 1,15] lần bản gốc.

## INT-CORR — sai có chủ đích

Loại edit khai báo trước:
- `PRES` (có/không): đổi một phát hiện đang được mô tả là có thành không có, hoặc ngược lại, trong MỘT dòng có trạng thái `visible`/`partly visible`.
- `LAT` (vị trí/bên): đổi bên (left↔right, image-left↔image-right) hoặc vị trí (upper↔lower, anterior↔posterior, medial↔lateral) trong MỘT dòng.
- `SEV` (mức độ): đổi một từ mức độ/kích thước (small↔large, mild↔marked, slight↔severe, narrow↔wide, thin↔thick, few↔many) trong MỘT dòng.

Gán loại edit cho từng ảnh (tất định, không nhìn câu hỏi):
1. Code quét observation để biết loại nào khả thi: `LAT` nếu có ≥1 từ bên/vị trí trong danh sách; `SEV` nếu có ≥1 từ mức độ trong danh sách; `PRES` luôn khả thi nếu có ≥1 dòng `visible`/`partly visible` (nếu không có dòng nào như vậy: `PRES` đổi một dòng `not visible` thành mô tả có một vùng bất thường — xem prompt).
2. Thứ tự ảnh: sắp theo `sha256(neutral_image_id + "CORR-v1")`. Đi lần lượt, mỗi ảnh nhận loại khả thi có số lượng đã gán ít nhất (hòa: thứ tự PRES, LAT, SEV). Dòng bị sửa: dòng khả thi đầu tiên theo thứ tự `sha256(neutral_image_id + item_no)`.
3. Bảng gán `corruption_plan.json` được sinh và băm (SHA ghi ở experiment manifest) TRƯỚC khi chạy answerer trên INT-CORR.

```
Edit the text below by changing exactly one fact, as specified. Change nothing else.
Edit type: {EDIT_TYPE_DESCRIPTION}
Line to edit: line {LINE_NO}
Rules:
- Change only the last field of line {LINE_NO}. Keep every other line character-for-character identical.
- The edited sentence must stay fluent and keep about the same length.
- Do not add words such as "incorrectly", "changed" or any hint that the text was edited.
Return a JSON object only: {"edited_text": "<the full text after the edit>", "original_span": "<words replaced>", "new_span": "<new words>"}

Text:
{OBSERVATION}
```

`EDIT_TYPE_DESCRIPTION`:
- PRES: `Reverse the presence of one described finding (present becomes absent, or absent becomes present).`
- LAT: `Swap one side or position word to its opposite (left/right, upper/lower, anterior/posterior, medial/lateral).`
- SEV: `Swap one degree or size word to its opposite (small/large, mild/marked, narrow/wide, thin/thick, few/many).`

Kiểm bằng code: JSON hợp lệ; mọi dòng khác dòng `LINE_NO` giống hệt; dòng `LINE_NO` khác bản gốc; `original_span` có trong dòng gốc, `new_span` có trong dòng mới; với LAT/SEV, cặp từ đổi thuộc đúng danh sách; độ dài dòng mới trong ±10 từ. Fail → thử lại tối đa 2 lần (seed 20261017, 20261018); vẫn fail → `int_missing`, báo riêng theo loại.

## INT-EMPTY, INT-SHUF, INT-BLANK
Không cần LLM; sinh bằng code (xem `build_prompts.py` cho INT-EMPTY và `medvqa/interventions.py` cho SHUF/BLANK).
