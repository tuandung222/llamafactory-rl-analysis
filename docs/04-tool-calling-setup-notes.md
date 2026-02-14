# Lưu ý khi setup training config RL cho bài toán tool-calling multi-turn

## 1) Mục tiêu đúng cho tool-calling
Không chỉ tối ưu text quality; cần tối ưu hành vi:
- Chọn đúng tool.
- Truyền đúng arguments.
- Biết dừng tool khi đủ thông tin.
- Biết recovery sau lỗi tool.

## 2) Dataset design
Nên lưu theo message-based format:
- `system`
- `user`
- `assistant` (có thể chứa tool plan)
- `function_call`
- `observation`
- `assistant` final

Với DPO/KTO/RM:
- Mỗi sample phải có cặp preferred/non-preferred rõ ràng.

Với PPO:
- Prompt pool đa dạng task và ràng buộc.
- Reward model hoặc reward API chấm trajectory.

## 3) Các config quan trọng nhất
- `template`: phải phù hợp model family và tool-call format.
- `cutoff_len`: đủ chứa multi-turn context.
- `pref_beta`: bắt đầu nhỏ (0.05-0.2) khi data noisy.
- `pref_ftx`: bật nhẹ nếu muốn giữ khả năng hội thoại nền.
- PPO: tune `ppo_target` + `ppo_epochs` để tránh over-optimization.

## 4) Reward rubric khuyến nghị cho tool-calling
Ví dụ composite reward:
- `R_tool_select`: đúng tool hay không.
- `R_args`: tham số đúng schema và semantic.
- `R_efficiency`: số lượt gọi tool hợp lý.
- `R_recovery`: xử lý lỗi tool đúng quy trình.
- `R_final`: câu trả lời cuối đáp ứng yêu cầu.

Total: `R = w1*R_tool_select + w2*R_args + w3*R_efficiency + w4*R_recovery + w5*R_final`

## 5) Failure mode thường gặp
1. Over-calling tools: gọi tool dù đã đủ thông tin.
2. Premature final answer: kết luận khi chưa verify.
3. Schema drift: output JSON/tool args sai format.
4. Context truncation: mất lượt tool cũ do `cutoff_len` thấp.

## 6) Checklist trước khi train
1. In 20 sample đã render và đọc bằng mắt.
2. Chạy sanity inference với checkpoint base.
3. Kiểm tra tỉ lệ invalid tool args.
4. Đảm bảo eval set có hard cases multi-turn.
