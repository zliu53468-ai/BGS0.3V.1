# wsgi.py
import os
import sys
from pathlib import Path

# 將專案根目錄放進 sys.path，避免相對匯入失敗
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 依序嘗試多種常見路徑匯入 Flask app
app = None
_import_errors = []

for candidate in (
    "server:app",        # repo 根目錄有 server.py
    "app.server:app",    # app/server.py
    "src.server:app",    # src/server.py
):
    module_name, attr = candidate.split(":")
    try:
        module = __import__(module_name, fromlist=[attr])
        app = getattr(module, attr)
        break
    except Exception as e:
        _import_errors.append(f"{candidate} -> {e}")

if app is None:
    msg = (
        "❌ 無法在任何候選路徑找到 Flask app。\n"
        "請確認 server.py 的實際位置與名稱（大小寫），\n"
        "並確保對應資料夾含有 __init__.py（若在子資料夾）。\n\n"
        "嘗試過的路徑與錯誤：\n- " + "\n- ".join(_import_errors)
    )
    raise RuntimeError(msg)

# 本地執行（非 gunicorn）時使用
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
