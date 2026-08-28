# Manual search and shutdown acceptance test

This Windows-only check validates the integration with Windows file associations and
the lifecycle of the packaged `pythonw.exe` processes. It complements the automated
HTTP, security, retrieval-field, and tray lifecycle tests.

1. Launch AXPIndexerNG and open **Search...** from the tray menu.
2. Search for `mixing time bleach`.
3. Confirm that relevant documents appear and that every result retains its source,
   path, and snippet.
4. Confirm that `mixing`, `time`, and `bleach` occurrences are highlighted in yellow
   and that every result has a **Relevance _n_%** badge.
5. Select **Open file** and confirm Windows launches the original document using its
   configured application. Repeat with a local path and, when available, a mapped
   drive or UNC source.
6. Close the opened document, then select **Exit AXPIndexerNG** from the tray menu.
7. Wait briefly and confirm that no daemon or web-client AXPIndexerNG `pythonw.exe`
   process remains. Do not use a blanket Python-process check as a cleanup action.
8. Launch AXPIndexerNG again and confirm that the daemon starts normally when
   `auto_start_daemon` is enabled.

## Heartbeat and watchdog resilience (Windows)

### Normal AC power

1. Launch AXPIndexerNG and confirm the daemon reaches idle or scanning state.
2. Confirm the heartbeat updates normally, index a source, and confirm there are no watchdog warnings.

### Battery/throttled operation

1. Disconnect AC power and enable Windows battery saver or low-power mode.
2. Start a significant indexing operation while other normal applications remain active.
3. Confirm scheduling delays below 90 seconds do not cause a restart and that exactly one daemon remains active.

### Synthetic stale heartbeat

1. In a development environment, hold the catalog daemon instance lock and make its published heartbeat stale.
2. Confirm the tray log contains `Daemon heartbeat stale but daemon instance is still active; duplicate restart
   suppressed`.
3. Confirm that no new `pythonw` daemon process starts. Do not terminate processes by executable name.
