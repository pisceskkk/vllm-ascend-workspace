param(
    [string]$Target = "Codex:Jiguang:AccessToken",
    [string]$SecretFile
)
$ErrorActionPreference = "Stop"

if (-not ("Jiguang.NativeCredentialWrite" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace Jiguang {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct CredentialWrite {
    public UInt32 Flags;
    public UInt32 Type;
    public string TargetName;
    public string Comment;
    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public UInt32 CredentialBlobSize;
    public IntPtr CredentialBlob;
    public UInt32 Persist;
    public UInt32 AttributeCount;
    public IntPtr Attributes;
    public string TargetAlias;
    public string UserName;
  }
  public static class NativeCredentialWrite {
    [DllImport("advapi32.dll", EntryPoint="CredWriteW", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern bool CredWrite(ref CredentialWrite credential, UInt32 flags);
  }
}
'@
}

$secure = if ($SecretFile) {
    $resolved = (Resolve-Path -LiteralPath $SecretFile).Path
    ConvertTo-SecureString ("@file:" + $resolved) -AsPlainText -Force
}
else {
    Read-Host "Paste the secret for '$Target'" -AsSecureString
}
$blob = [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($secure)
try {
    $length = $secure.Length * 2
    $credential = [Jiguang.CredentialWrite]::new()
    $credential.Type = 1
    $credential.TargetName = $Target
    $credential.CredentialBlobSize = $length
    $credential.CredentialBlob = $blob
    $credential.Persist = 2
    $credential.UserName = "Bearer"
    if (-not [Jiguang.NativeCredentialWrite]::CredWrite([ref]$credential, 0)) {
        throw "CredWrite failed with Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
    }
    Write-Output "Stored Jiguang token in Windows Credential Manager target '$Target'."
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($blob)
}
