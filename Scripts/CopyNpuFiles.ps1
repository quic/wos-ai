<#
.SYNOPSIS
    Copies the required files from a QAIRT SDK to a specified destination directory.

.DESCRIPTION
    Usage:
        CopyNpuFiles.ps1 <Source SDK Path> <Destination Directory> <Architecture>

    Parameters:
        - <Source SDK Path> : Root directory of the QAIRT SDK installation  
                              (e.g., C:\Qualcomm\AIStack\QAIRT\2.42.0.251225)

        - <Destination Directory> : Directory where the required runtime files will be copied.  
                                   If the directory does not exist, it will be created automatically.  
                                   (e.g., C:\Users\HCKTest\Desktop\sanjeev\workspace\QAIRT_dlls)
.EXAMPLE
    .\CopyNpuFiles.ps1 "C:\Qualcomm\AIStack\QAIRT\2.42.0.251225" "C:\Users\HCKTest\Desktop\sanjeev\workspace\QAIRT_dlls"
#>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $source_SDK,

    [Parameter(Mandatory = $true, Position = 1)]
    [string] $dest_dir
)

# Resolve source path (fail if not found)
try {
    $source_SDK = (Resolve-Path -Path $source_SDK -ErrorAction Stop).Path
} catch {
    Write-Error "Source SDK path not found: $source_SDK"
    exit 1
}

# Resolve destination if it exists; otherwise keep as provided
$destResolved = $null
try {
    $destResolved = Resolve-Path -Path $dest_dir -ErrorAction Stop
} catch {
    # ignore; will create directory
}
if ($destResolved) {
    $dest_dir = $destResolved.Path
}

Write-Host "Source SDK : $source_SDK"
Write-Host "Destination: $dest_dir"
Write-Host ""

# Ensure destination exists (create if missing; if present, just continue)
if (-not (Test-Path -Path $dest_dir)) {
    Write-Host "Destination does not exist. Creating: $dest_dir"
    New-Item -ItemType Directory -Path $dest_dir -Force | Out-Null
} else {
    Write-Host "Destination exists. Skipping creation."
}

# Define relative file paths to copy from source_SDK
$filesToCopy = @(
    # arm64x-windows-msvc DLLs
    "lib\arm64x-windows-msvc\QnnHtp.dll",
    "lib\arm64x-windows-msvc\QnnHtpPrepare.dll",
    "lib\arm64x-windows-msvc\QnnSystem.dll",
    "lib\arm64x-windows-msvc\QnnHtpNetRunExtensions.dll",
    "lib\arm64x-windows-msvc\QnnHtpV73Stub.dll",
    "lib\arm64x-windows-msvc\QnnHtpV81Stub.dll",

    # Hexagon v73 unsigned
    "lib\hexagon-v73\unsigned\libQnnHtpV73Skel.so",
    "lib\hexagon-v73\unsigned\libqnnhtpv73.cat",

    # Hexagon v81 unsigned
    "lib\hexagon-v81\unsigned\libQnnHtpV81Skel.so",
    "lib\hexagon-v81\unsigned\libqnnhtpv81.cat"
)

# Perform copy
$copied = 0
$missing = @()

foreach ($relPath in $filesToCopy) {
    $src = Join-Path -Path $source_SDK -ChildPath $relPath
    if (Test-Path -Path $src) {
        try {
            Copy-Item -Path $src -Destination $dest_dir -Force
            Write-Host "Copied: $src -> $dest_dir"
            $copied++
        } catch {
            Write-Warning "Failed to copy: $src. Error: $($_.Exception.Message)"
        }
    } else {
        Write-Warning "Missing source file (skipped): $src"
        $missing += $src
    }
}

Write-Host ""
Write-Host "Summary:"
Write-Host "  Files copied : $copied"
Write-Host "  Files missing: $($missing.Count)"
if ($missing.Count -gt 0) {
    Write-Host "  Missing list:"
    $missing | ForEach-Object { Write-Host "   - $_" }
}
