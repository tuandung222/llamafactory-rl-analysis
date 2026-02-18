# Data Design cho Tool-Calling Multi-Turn

## 1) Định nghĩa "đúng" cho Agent tool-calling
Một trajectory tốt cần đồng thời:
- Chọn đúng tool.
- Truyền đúng arguments.
- Gọi tool đủ số bước (không thiếu, không thừa).
- Kết luận cuối đúng với evidence từ tool outputs.

## 2) Schema mẫu (message-based)
Mỗi sample nên gồm:
- `system`: policy và ràng buộc.
- `messages`: chuỗi multi-turn gồm:
  - `user`
  - `assistant` (plan hoặc call)
  - `function_call`
  - `observation` (tool response)
  - `assistant` final

## 3) Dataset theo mục tiêu huấn luyện

### 3.1 RM/DPO (pairwise)
Một record có:
- cùng prompt/context,
- `chosen` trajectory,
- `rejected` trajectory.

Chosen nên tốt hơn rejected theo rubric cụ thể:
- đúng tool,
- args hợp lệ,
- hiệu quả số bước,
- final answer chính xác.

### 3.2 KTO (desirable/undesirable)
Mỗi sample có tag:
- desirable hoặc undesirable.
Phù hợp khi dữ liệu feedback không đầy đủ pairwise.

### 3.3 PPO
Dữ liệu prompt pool đa dạng task;
reward sẽ chấm online theo trajectory model sinh ra.

## 4) Rubric chấm chất lượng dữ liệu
Điểm mỗi trajectory:
- `S_tool_select` (0-1)
- `S_args_valid` (0-1)
- `S_step_efficiency` (0-1)
- `S_recovery` (0-1)
- `S_final_correct` (0-1)

Tổng: `S = w1*S_tool_select + w2*S_args_valid + w3*S_step_efficiency + w4*S_recovery + w5*S_final_correct`

## 5) Anti-pattern dữ liệu
1. Chosen/rejected đều tệ.
2. Rejected lỗi quá dễ (model học shortcut).
3. Mất turn tool output do truncate.
4. Không thống nhất format tool arguments (JSON schema drift).

## 6) Data split khuyến nghị
- Train: 80%
- Dev: 10%
- Test: 10%

Test phải có hard cases:
- ambiguous tool selection
- missing information recovery
- long horizon (4-8 tool turns)
