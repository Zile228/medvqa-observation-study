# Prompt cho LLM judge (RQ2 và câu mở)

Judge: J1 `internlm/internlm3-8b-instruct` @ `28c99415adaf61767bd1c619f4f99f308fdfd223`; J2 `ibm-granite/granite-3.3-8b-instruct` @ `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`. Text-only, greedy, `max_new_tokens` ghi ở từng prompt, chat template mặc định của model, không system prompt riêng.
Precision của judge được chọn theo cấu hình chạy. Dự phòng nếu một judge không load được ở pilot: `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083`.
Judge KHÔNG thấy: tên observer, arm, ID gốc, đáp án gold (trừ J-OPEN, nơi gold là tham chiếu bắt buộc). Thứ tự các observation được trộn theo seed trước khi chấm.
Output không phải JSON hợp lệ theo schema → `judge_invalid` (không sửa, không gọi lại), báo tỷ lệ theo judge.

## J-AX1 — đúng với annotation (chỉ SLAKE test 96 ảnh), `max_new_tokens=400`

```
You compare statements about a medical image with the reference annotation of that image.
The annotation lists labelled regions with their location in the image. It may be incomplete: a structure that is not listed may still be present.
For each statement, choose one label:
- "supported": the annotation confirms the statement.
- "contradicted": the annotation clearly conflicts with the statement (for example wrong side, or a structure placed where the annotation shows something else).
- "not_verifiable": the annotation does not cover the statement.
Judge only against the annotation. Do not use outside knowledge about what is likely.

Annotation:
{ANNOTATION_TEXT}

Statements:
{NUMBERED_STATEMENTS}

Return JSON only: {"labels": [{"id": <number>, "label": "supported|contradicted|not_verifiable"}, ...]} with one entry per statement, in order.
```
`ANNOTATION_TEXT`: dựng tất định từ `detection.json` và `mask.png` (source96): mỗi nhãn một dòng `<label>: region <image-left|centre|image-right>, <upper|middle|lower>; bounding box x=<..>, y=<..>, w=<..>, h=<..> (pixels, image size W×H)`. Vùng 3×3 tính theo tâm bbox.

## J-AX2 — tuân thủ framework (panel 216 ảnh), `max_new_tokens=200`

```
Below is a list of requested observation items for one type of medical image, and a written observation.
Answer three questions.
1. For how many of the requested items does the observation give a usable statement about that item (a description or an explicit "not assessable")? Give a number.
2. Does the observation name a disease or give a diagnosis? yes or no.
3. Does the observation make any claim of these unsupported kinds: {UNSUPPORTED_LIST}? yes or no.

Requested items:
{GRD_ITEM_LIST_FOR_ROUTE}

Observation:
{OBSERVATION}

Return JSON only: {"items_addressed": <integer>, "diagnosis": "yes|no", "unsupported_claim": "yes|no"}
```
`GRD_ITEM_LIST_FOR_ROUTE` luôn là danh sách item ARM-GRD của route ảnh đó, ở MỌI arm (kể cả GEN, ANA). `UNSUPPORTED_LIST` lấy từ `unsupported_inferences` của các field trong route (hợp nhất, bỏ trùng), cộng 3 mục chung: `whole-examination normality`, `size in physical units without a scale`, `change over time`.
Vì GRD được yêu cầu đúng các item này, câu 1 là **manipulation check**, không phải kết quả về chất lượng. Câu 2–3 là chỉ số vi phạm, trung lập giữa các arm.

## J-AX3 — liên quan câu hỏi (ARM-GRD, 4 observer, panel), `max_new_tokens=20`

```
Here is a question about a medical image and a written observation of the same image. You do not need to answer the question.
Rate how much the observation contains information that is relevant for answering the question:
0 = nothing relevant; 1 = some relevant information but not enough to answer; 2 = information that directly bears on the answer.
Do not judge whether the observation is correct.

Question: {QUESTION}
Observation:
{OBSERVATION}

Return JSON only: {"relevance": 0|1|2}
```
Không đưa đáp án gold và không đưa options vào J-AX3.

## J-OPEN — đúng/sai của câu trả lời mở (E-S8), `max_new_tokens=20`

```
A question about a medical image has a reference answer. Decide whether the candidate answer means the same as the reference answer. Accept synonyms and equivalent wording. Extra details are acceptable only if they do not contradict the reference.

Question: {QUESTION}
Reference answer: {GOLD}
Candidate answer: {PREDICTION}

Return JSON only: {"verdict": "correct|incorrect"}
```

## J-PROBE — probe biết trước đáp án
- 250 statement dựng tất định (seed 20261016) từ annotation SLAKE test:
  - Mẫu A, 100 đúng: `<label> is visible in the <vùng 3×3 theo tâm bbox>` (đáp án `supported`).
  - Mẫu A, 100 đảo bên: cùng nhãn, vùng image-left ↔ image-right, chỉ dùng bbox có tâm ngoài cột giữa (đáp án `contradicted`).
  - Mẫu B (diễn đạt kiểu observation), 25 đúng + 25 đảo bên: `<Vùng viết hoa chữ đầu>: the <label> appears in this part of the image, with no other structure described there.`.
- Statement probe gần nguyên văn annotation nên dễ hơn statement thật: độ chính xác probe là **cận trên** của độ đúng judge. Báo theo lớp và theo mẫu A/B.
- Trộn vào lượt J-AX1 theo khối 10 statement/lượt, cùng prompt J-AX1, annotation của đúng ảnh đó. Judge không biết đâu là probe.
- Báo độ chính xác từng judge theo từng lớp. Ngưỡng khóa: ≥ 0,70 ở **mỗi** lớp cho **cả hai** judge thì mới viết kết luận về E-R2a/E-R2b.

## Đồng thuận
- J-AX1 (3 lớp), J-AX2 câu 2–3: % đồng ý, **Gwet AC1** (chỉ số chính), Cohen's kappa (báo kèm; dễ bị nghịch lý kappa khi `not_verifiable` áp đảo).
- J-AX3: kappa trọng số bậc hai + % đồng ý. E-R2c: ICC(2,1). KTC 95% bootstrap theo ảnh.

## Kiểm bias
- **Vị trí/thứ tự:** 10% lượt J-AX1 (chọn theo seed 20261016, phân tầng theo arm) chạy lại với thứ tự statement đảo ngược; báo tỷ lệ nhãn giữ nguyên theo statement, và kappa giữa hai lượt.
- **Độ dài:** báo hệ số Spearman giữa số token observation và (a) tỷ lệ supported, (b) items_addressed, theo từng judge; so ANA với GRD (độ dài prompt đã khớp).
- **Họ model:** J1, J2 khác họ với mọi observer và answerer (không GPT, Qwen, Llama/Vicuna, Mistral).
