# Secure Windows background indexing

## Architecture

In **interactive mode**, the tray starts, monitors, and stops the indexing daemon. In **background mode**, Windows Task
Scheduler starts `AXPIndexerDaemon.pyw`; the tray only monitors and controls that independently owned daemon. Closing the
tray therefore does not stop background indexing, and the task can continue after session sign-out or disconnection.

The task runs at logon of the current user, whether that user is logged on or not, with least privilege. It does not add a
boot trigger, wake the computer, start the search client, or expose a new network listener.

## Credentials and security

Windows' own visible `schtasks.exe` prompt collects the account password during registration. AXP never reads, receives,
logs, or stores that password. Windows Task Scheduler retains what it needs so normal Kerberos/NTLM authentication can
reach resources already authorized for the user. AXP does not store network credentials and does not create network drive
mappings.

AXPIndexerNG never runs the indexer as SYSTEM for this feature. It also never uses LocalService or NetworkService. The
daemon intentionally has the current user's effective filesystem permissions rather than machine-wide privileges. It does
not elevate, change firewall or security policy, grant batch-logon rights, or bypass Task Scheduler, domain, share, NTFS,
VPN, or network policy. A refusal is reported and interactive mode remains available.

## Mapped drives and catalog identity

Interactive drive letters are resolved through the Windows `WNetGetConnectionW` API before activation. Only a non-secret
translation such as `K:` to `\\server\share` is saved. The background scanner reads the UNC **access path**, while the
catalog retains the original `K:\...` **logical path**. Consequently enabling background mode requires no schema change,
reindex, vector/chunk rebuild, or search/ranking change and creates no duplicate UNC document identity.

If a source or VPN is unavailable, its scan is incomplete/offline and existing documents are preserved for a later retry.
Search opens only a document selected by database ID; if its logical drive is unavailable, the trusted stored translation
may be used as a fallback.

## Lifecycle and managed workstations

Enable **Keep indexing after sign-out**, review preflight, confirm the explanation, and answer Windows' credential prompt.
After registration the task starts immediately. On a later sign-in the tray reconnects to its heartbeat. **Disable
background indexing** stops the daemon, validates ownership before deleting the task, and returns to interactive mode.
Moving the portable installation preserves its installation ID, so the old action is reported as needing repair rather than
creating an unrelated task. Repair or credential refresh re-registers the owned task and lets Windows prompt again.

Some corporate environments prohibit non-interactive tasks or batch logon. AXP reports that restriction and does not
bypass or weaken it; contact the organization's IT administrator if background indexing is required.
