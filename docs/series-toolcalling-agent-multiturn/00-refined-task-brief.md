# Refined Prompt (Task Brief)

## Mục tiêu
Xây dựng một series tài liệu thực chiến, có thể dùng ngay, hướng dẫn cách dùng LLaMA-Factory để huấn luyện Agent cho bài toán tool-calling multi-turn, làm rõ từ khâu thiết kế dữ liệu đến huấn luyện và đánh giá mô hình.

## Deliverables bắt buộc
1. Tài liệu tổng quan series và lộ trình học.
2. Tài liệu thiết kế dữ liệu tool-calling multi-turn:
   - schema chuẩn,
   - tiêu chí chất lượng,
   - cách tạo chosen/rejected,
   - cách tạo dữ liệu cho RM/DPO/KTO/PPO.
3. Tài liệu pipeline build dữ liệu cho LLaMA-Factory:
   - mapping schema sang định dạng dataset,
   - template,
   - processor/collator liên quan.
4. Tài liệu training config và recipes:
   - RM,
   - DPO/ORPO/SimPO,
   - KTO,
   - PPO,
   - lưu ý đặc thù tool-calling multi-turn.
5. Tài liệu evaluation, debugging, anti-patterns.
6. Một runbook end-to-end với checklist vận hành.

## Yêu cầu chất lượng
- Bám sát code path thực tế của LLaMA-Factory hiện tại.
- Viết theo hướng thực dụng: có cấu hình mẫu, KPI, failure modes.
- Tách rõ phần đã có trong source và phần mở rộng tham chiếu.
- Ưu tiên khả năng tái lập (reproducibility): seed, artifact, logging.

## Đối tượng đọc
- ML Engineer mới vào team RLHF/tool-calling.
- Research Engineer cần đưa thí nghiệm vào pipeline vận hành.
- Tech lead cần checklist setup và đánh giá chất lượng.
