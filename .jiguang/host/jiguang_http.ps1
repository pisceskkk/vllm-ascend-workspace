$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if (-not ("Jiguang.NativeCredential" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace Jiguang {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct Credential {
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
  public static class NativeCredential {
    [DllImport("advapi32.dll", EntryPoint="CredReadW", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern bool CredRead(string target, UInt32 type, UInt32 flags, out IntPtr credential);
    [DllImport("advapi32.dll", SetLastError=true)]
    public static extern void CredFree(IntPtr credential);
  }
}
'@
}

function Write-Result([hashtable]$Payload, [int]$ExitCode = 0) {
    [Console]::Out.Write(($Payload | ConvertTo-Json -Depth 30 -Compress))
    exit $ExitCode
}

function Get-JiguangToken([string]$Target) {
    $pointer = [IntPtr]::Zero
    if (-not [Jiguang.NativeCredential]::CredRead($Target, 1, 0, [ref]$pointer)) {
        throw "Windows credential '$Target' was not found"
    }
    try {
        $credential = [Runtime.InteropServices.Marshal]::PtrToStructure(
            $pointer,
            [type][Jiguang.Credential]
        )
        if ($credential.CredentialBlobSize -le 0) {
            throw "Windows credential '$Target' is empty"
        }
        return [Runtime.InteropServices.Marshal]::PtrToStringUni(
            $credential.CredentialBlob,
            [int]($credential.CredentialBlobSize / 2)
        )
    }
    finally {
        [Jiguang.NativeCredential]::CredFree($pointer)
    }
}

function Resolve-JiguangSecret([string]$Target) {
    $stored = Get-JiguangToken $Target
    if ($stored.StartsWith("@file:", [StringComparison]::Ordinal)) {
        $path = $stored.Substring(6)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "file-backed Windows credential '$Target' cannot be resolved"
        }
        return Get-Content -LiteralPath $path -Raw
    }
    return $stored
}

try {
    $raw = [Console]::In.ReadToEnd()
    if ([string]::IsNullOrWhiteSpace($raw)) {
        throw "host bridge requires one JSON request on stdin"
    }
    $request = $raw | ConvertFrom-Json
    $method = ([string]$request.method).ToUpperInvariant()
    if ($method -notin @("GET", "POST", "PUT", "PATCH", "DELETE")) {
        throw "unsupported HTTP method"
    }
    $path = [string]$request.path
    if (-not $path.StartsWith("/api/") -or $path.Contains("..") -or $path.Contains("://")) {
        throw "invalid API path"
    }
    $baseUrl = if ($env:JIGUANG_BASE_URL) { $env:JIGUANG_BASE_URL } else { "https://jiguang.ascend.huawei.com" }
    $baseUri = [Uri]$baseUrl
    if ($baseUri.Scheme -ne "https" -or $baseUri.Host -ne "jiguang.ascend.huawei.com") {
        throw "JIGUANG_BASE_URL must be https://jiguang.ascend.huawei.com"
    }
    $builder = [UriBuilder]::new([Uri]::new($baseUri, $path))
    $pairs = [System.Collections.Generic.List[string]]::new()
    if ($request.query) {
        foreach ($property in $request.query.PSObject.Properties) {
            if ($null -eq $property.Value) { continue }
            $values = if ($property.Value -is [System.Array]) { $property.Value } else { @($property.Value) }
            foreach ($value in $values) {
                $pairs.Add(
                    [Uri]::EscapeDataString($property.Name) + "=" + [Uri]::EscapeDataString([string]$value)
                )
            }
        }
    }
    $builder.Query = [string]::Join("&", $pairs)
    $token = Resolve-JiguangSecret ([string]$request.credential_target)
    if ($token.StartsWith("Bearer ", [StringComparison]::OrdinalIgnoreCase)) {
        $token = $token.Substring(7)
    }
    $headers = @{ Authorization = "Bearer $token"; Accept = "application/json" }
    $timeout = if ($request.timeout_seconds) { [int]$request.timeout_seconds } else { 30 }
    $invoke = @{
        Uri = $builder.Uri.AbsoluteUri
        Method = $method
        Headers = $headers
        TimeoutSec = $timeout
    }
    $invokeWebRequest = Get-Command Invoke-WebRequest
    if ($invokeWebRequest.Parameters.ContainsKey("NoProxy")) {
        $invoke.NoProxy = $true
    }
    else {
        [System.Net.WebRequest]::DefaultWebProxy = [System.Net.WebProxy]::new()
    }
    if ($null -ne $request.body -and $method -ne "GET") {
        $secretTarget = $request.body.__secret_credential_target
        $secretField = $request.body.__secret_field
        if ($secretTarget -or $secretField) {
            if (-not $secretTarget -or $secretField -notin @("secret", "relay_secret")) {
                throw "invalid secret injection request"
            }
            $storedSecret = Resolve-JiguangSecret ([string]$secretTarget)
            $request.body.PSObject.Properties.Remove("__secret_credential_target")
            $request.body.PSObject.Properties.Remove("__secret_field")
            $request.body | Add-Member -NotePropertyName ([string]$secretField) -NotePropertyValue $storedSecret -Force
        }
        $invoke.ContentType = "application/json; charset=utf-8"
        $invoke.Body = $request.body | ConvertTo-Json -Depth 30 -Compress
    }
    $response = Invoke-WebRequest @invoke
    $data = $null
    if (-not [string]::IsNullOrWhiteSpace($response.Content)) {
        try { $data = $response.Content | ConvertFrom-Json } catch { $data = $response.Content }
    }
    Write-Result @{ ok = $true; status = [int]$response.StatusCode; data = $data }
}
catch {
    $status = $null
    $detail = $_.Exception.Message
    if ($_.Exception.Response) {
        try { $status = [int]$_.Exception.Response.StatusCode } catch {}
    }
    Write-Result @{ ok = $false; status = $status; error = $detail } 2
}
