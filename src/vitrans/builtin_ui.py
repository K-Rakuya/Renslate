"""Built-in Ren'Py UI string translation seeds.

Ren'Py has hundreds of standard interface strings across navigation, preferences,
file slots, history, confirmation dialogs, accessibility, and error screens.
Seeding these with idiomatic Vietnamese translations ensures game UI is instantly
translated without wasting LLM tokens.
"""

from __future__ import annotations

# Maps original English UI string to idiomatic Vietnamese translation.
BUILTIN_UI_SEEDS: dict[str, str] = {
    # ── Main / Game Menu ──────────────────────────────────────────────
    "Start": "Bắt đầu",
    "History": "Lịch sử",
    "Save": "Lưu game",
    "Load": "Tải game",
    "Preferences": "Cài đặt",
    "End Replay": "Kết thúc xem lại",
    "Main Menu": "Menu chính",
    "About": "Giới thiệu",
    "Help": "Trợ giúp",
    "Quit": "Thoát",
    "Return": "Quay lại",
    "Back": "Trở về",
    "Skip": "Tua nhanh",
    "Auto": "Tự động",
    "Menu": "Menu",
    "Options": "Tùy chọn",
    "Continue": "Tiếp tục",
    "Restart": "Khởi động lại",
    # ── Common actions / Buttons ──────────────────────────────────────
    "Yes": "Có",
    "No": "Không",
    "Cancel": "Hủy",
    "OK": "OK",
    "Confirm": "Xác nhận",
    "Close": "Đóng",
    "Delete": "Xóa",
    "Rename": "Đổi tên",
    "Select": "Chọn",
    "Copy": "Sao chép",
    "Paste": "Dán",
    "Reset": "Đặt lại",
    "Default": "Mặc định",
    "Page": "Trang",
    "Page {}": "Trang {}",
    "Automatic saves": "Lưu tự động",
    "Quick saves": "Lưu nhanh",
    "{#auto_page}A": "{#auto_page}TĐ",
    "{#quick_page}Q": "{#quick_page}N",
    # ── Save / Load Slots ─────────────────────────────────────────────
    "empty slot": "ô trống",
    "<Empty Slot>": "<Ô trống>",
    "{#file_time}%A, %B %d %Y, %H:%M": "{#file_time}%H:%M, %d/%m/%Y",
    "{#file_time}%b %d %Y, %H:%M": "{#file_time}%H:%M, %d/%m/%Y",
    "Slot {}": "Ô {}",
    # ── Confirmation Dialogs ──────────────────────────────────────────
    "Are you sure you want to quit?": "Bạn có chắc muốn thoát không?",
    "Are you sure you want to delete this save?": "Bạn có chắc muốn xóa bản lưu này không?",
    "Are you sure you want to overwrite your save?": (
        "Bạn có chắc muốn ghi đè lên bản lưu này không?"
    ),
    "Loading will lose unsaved progress. Are you sure you want to do this?": (
        "Tải game sẽ làm mất tiến trình chưa lưu. Bạn có chắc muốn tiếp tục không?"
    ),
    "Are you sure you want to quit and return to the main menu?": (
        "Bạn có chắc muốn thoát về menu chính không?"
    ),
    "Are you sure you want to end the replay?": "Bạn có chắc muốn kết thúc xem lại không?",
    "Slow Skip": "Tua chậm",
    "Fast Skip": "Tua nhanh",
    # ── Preferences Screen ────────────────────────────────────────────
    "Display": "Hiển thị",
    "Window": "Cửa sổ",
    "Fullscreen": "Toàn màn hình",
    "Rollback Side": "Vùng cuộn lại",
    "Disable": "Tắt",
    "Left": "Trái",
    "Right": "Phải",
    "Unseen Text": "Lời thoại chưa đọc",
    "After Choices": "Sau lựa chọn",
    "Transitions": "Chuyển cảnh",
    "Text Speed": "Tốc độ chữ",
    "Auto-Forward Time": "Thời gian tự chuyển",
    "Music Volume": "Âm lượng nhạc nền",
    "Sound Volume": "Âm lượng hiệu ứng",
    "Voice Volume": "Âm lượng giọng đọc",
    "Mute All": "Tắt toàn bộ âm thanh",
    "All": "Tất cả",
    "None": "Không",
    "Keep": "Giữ",
    "Language": "Ngôn ngữ",
    # ── History Screen ────────────────────────────────────────────────
    "The dialogue history is empty.": "Lịch sử đối thoại hiện đang trống.",
    # ── Help Screen: Keyboard & Mouse ─────────────────────────────────
    "Keyboard": "Bàn phím",
    "Mouse": "Chuột",
    "Gamepad": "Tay cầm",
    "Enter": "Enter",
    "Advances dialogue and activates the interface.": "Chuyển lời thoại và kích hoạt giao diện.",
    "Space": "Phím cách",
    "Advances dialogue without selecting choices.": "Chuyển lời thoại mà không chọn lựa chọn.",
    "Arrow Keys": "Phím mũi tên",
    "Navigate the interface.": "Điều hướng giao diện.",
    "Escape": "Escape",
    "Accesses the game menu.": "Mở menu trò chơi.",
    "Ctrl": "Ctrl",
    "Skips dialogue while held down.": "Tua nhanh lời thoại khi giữ phím.",
    "Tab": "Tab",
    "Toggles dialogue skipping.": "Bật/tắt chế độ tua nhanh lời thoại.",
    "Page Up": "Page Up",
    "Rolls back to earlier dialogue.": "Cuộn lại lời thoại trước đó.",
    "Page Down": "Page Down",
    "Rolls forward to later dialogue.": "Cuộn tới lời thoại tiếp theo.",
    "Hides the user interface.": "Ẩn giao diện người dùng.",
    "Takes a screenshot.": "Chụp ảnh màn hình.",
    "Toggles assistive {a=https://www.renpy.org/l/voicing}self-voicing{/a}.": (
        "Bật/tắt tính năng {a=https://www.renpy.org/l/voicing}đọc màn hình hỗ trợ{/a}."
    ),
    "Opens the accessibility menu.": "Mở menu trợ năng.",
    "Left Click": "Chuột trái",
    "Middle Click": "Chuột giữa",
    "Right Click": "Chuột phải",
    "Mouse Wheel Up\nClick Rollback Side": "Cuộn chuột lên\nNhấp vào vùng cuộn",
    "Mouse Wheel Down": "Cuộn chuột xuống",
    "Right Trigger\nA/Bottom Button": "Cò phải\nNút A/Dưới",
    "Left Trigger\nLeft Shoulder": "Cò trái\nNút vai trái",
    "Right Shoulder": "Nút vai phải",
    "D-Pad, Sticks": "D-Pad, Cần điều khiển",
    "Start, Guide, B/Right Button": "Start, Guide, Nút B/Phải",
    # ── About Screen ──────────────────────────────────────────────────
    "Made with {a=https://www.renpy.org/}Ren'Py{/a} [renpy.version_only].\n\n[renpy.license!t]": (
        "Được phát triển với {a=https://www.renpy.org/}Ren'Py{/a} "
        "[renpy.version_only].\n\n[renpy.license!t]"
    ),
    "Version [config.version!t]\n": "Phiên bản [config.version!t]\n",
    # ── Accessibility ─────────────────────────────────────────────────
    "Accessibility": "Trợ năng",
    "Self-Voicing": "Tự động đọc",
    "Clipboard Voicing": "Đọc bảng nhớ tạm",
    "Debug Voicing": "Gỡ lỗi đọc",
    "Text-to-speech has been enabled.": "Tính năng chuyển văn bản thành giọng nói đã được bật.",
    "Text-to-speech has been disabled.": "Tính năng chuyển văn bản thành giọng nói đã tắt.",
    "High Contrast Text": "Độ tương phản cao",
    "Font": "Phông chữ",
    "System Cursor": "Con trỏ hệ thống",
    # ── System / Ren'Py Runtime Errors ────────────────────────────────
    "An error has occurred.": "Đã xảy ra lỗi.",
    "The error message was:": "Thông báo lỗi:",
    (
        "You may also use try to use rollback to get back to the state the game was in "
        "before the error occurred."
    ): ("Bạn cũng có thể thử cuộn lại để trở về trạng thái trước khi xảy ra lỗi."),
    "Rollback": "Cuộn lại",
    "Ignore": "Bỏ qua",
    "Reload": "Tải lại",
    "Open Traceback": "Mở nhật ký lỗi",
    "Copy to Clipboard": "Sao chép vào bộ nhớ tạm",
    "Quits the game.": "Thoát trò chơi.",
    "Parsing the script failed.": "Phân tích kịch bản thất bại.",
}


def get_builtin_ui_translation(source: str) -> str | None:
    """Return the Vietnamese translation seed for a Ren'Py UI string if known."""
    return BUILTIN_UI_SEEDS.get(source.strip())
