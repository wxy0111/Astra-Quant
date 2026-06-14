"""Start the local OKX New dashboard.

This entrypoint mirrors the older `C:\\okx` habit: one command starts the local
dashboard, while `start.bat` can also launch Cloudflare Tunnel for phone access.

本入口保持旧项目的使用习惯：启动本地看板；`start.bat` 可同时启动隧道供手机访问。
"""

from __future__ import annotations

from src.dashboard import run_dashboard


def main() -> None:
    """Run the dashboard server."""
    run_dashboard()


if __name__ == "__main__":
    main()
