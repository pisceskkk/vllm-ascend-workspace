param(
    [Parameter(Mandatory = $true)][string]$Machine,
    [Parameter(Mandatory = $true)][string]$Target,
    [string]$Distribution = "Ubuntu",
    [string]$RepoRoot = "/home/q00946761/vllm-ascend-workspace"
)
$ErrorActionPreference = "Stop"

if ($Machine -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$') {
    throw "invalid machine alias"
}
if ($Target -notmatch '^[A-Za-z0-9][A-Za-z0-9:_.-]{2,127}$') {
    throw "invalid Windows credential target"
}
if ($Distribution -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$') {
    throw "invalid WSL distribution"
}
if ($RepoRoot -notmatch '^/[A-Za-z0-9/_.-]+$') {
    throw "invalid WSL repository path"
}

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

$alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%+-_"
$bytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $rng.GetBytes($bytes)
}
finally {
    $rng.Dispose()
}
$chars = for ($i = 0; $i -lt $bytes.Length; $i++) {
    $alphabet[$bytes[$i] % $alphabet.Length]
}
$password = -join $chars

$output = $password | & wsl.exe -d $Distribution --cd $RepoRoot python3 .agents/skills/jiguang-runtime-management/scripts/jiguang_device_access.py --machine $Machine --password-stdin --confirm
if ($LASTEXITCODE -ne 0) {
    throw "Jiguang device access configuration failed: $output"
}

$secure = ConvertTo-SecureString $password -AsPlainText -Force
$blob = [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($secure)
try {
    $credential = [Jiguang.CredentialWrite]::new()
    $credential.Type = 1
    $credential.TargetName = $Target
    $credential.CredentialBlobSize = $secure.Length * 2
    $credential.CredentialBlob = $blob
    $credential.Persist = 2
    $credential.UserName = "root"
    if (-not [Jiguang.NativeCredentialWrite]::CredWrite([ref]$credential, 0)) {
        throw "CredWrite failed with Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
    }
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($blob)
    $password = $null
}

Write-Output $output
Write-Output "Stored a generated Jiguang device password in Windows Credential Manager target '$Target'."
