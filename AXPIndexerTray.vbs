Option Explicit
Dim shell, fso, root, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
command = Chr(34) & root & "\python\pythonw.exe" & Chr(34) & " " & Chr(34) & root & "\AXPIndexerTray.pyw" & Chr(34)
shell.Run command, 0, False
