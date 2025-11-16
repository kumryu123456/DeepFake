# Convert MP4 to 16kHz Mono WAV
# FFmpeg-based conversion for audio processing

param(
    [string]$InputDir = "data\input_videos",
    [string]$OutputDir = "data\converted_wav"
)

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " MP4 to WAV Conversion (16kHz Mono)" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Check if FFmpeg is available
try {
    $null = & ffmpeg -version 2>&1
} catch {
    Write-Host "ERROR: FFmpeg not found. Please install FFmpeg and add it to PATH" -ForegroundColor Red
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Get all MP4 files
$mp4Files = Get-ChildItem -Path $InputDir -Filter "*.mp4"

if ($mp4Files.Count -eq 0) {
    Write-Host "ERROR: No MP4 files found in $InputDir" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($mp4Files.Count) MP4 files" -ForegroundColor Yellow
Write-Host "Output directory: $OutputDir`n" -ForegroundColor Yellow

$successCount = 0
$failCount = 0

foreach ($file in $mp4Files) {
    $inputPath = $file.FullName
    $outputName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name) + ".wav"
    $outputPath = Join-Path $OutputDir $outputName

    Write-Host "[$($successCount + $failCount + 1)/$($mp4Files.Count)] Converting: $($file.Name)" -ForegroundColor White

    # FFmpeg command: Convert to 16kHz mono WAV
    $ffmpegArgs = @(
        "-i", $inputPath,
        "-vn",                  # No video
        "-acodec", "pcm_s16le", # PCM 16-bit
        "-ar", "16000",         # 16kHz sample rate
        "-ac", "1",             # Mono
        "-y",                   # Overwrite
        $outputPath
    )

    try {
        $process = Start-Process -FilePath "ffmpeg" -ArgumentList $ffmpegArgs -NoNewWindow -Wait -PassThru -RedirectStandardError "$env:TEMP\ffmpeg_error.txt"

        if ($process.ExitCode -eq 0 -and (Test-Path $outputPath)) {
            $fileSize = (Get-Item $outputPath).Length / 1KB
            Write-Host "  ✓ Success - Output: $outputName ($([math]::Round($fileSize, 1)) KB)" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "  ✗ Failed - Check FFmpeg error log" -ForegroundColor Red
            $failCount++
        }
    } catch {
        Write-Host "  ✗ Error: $_" -ForegroundColor Red
        $failCount++
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " Conversion Complete" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Total files: $($mp4Files.Count)" -ForegroundColor White
Write-Host "✓ Success: $successCount" -ForegroundColor Green
Write-Host "✗ Failed: $failCount" -ForegroundColor Red
Write-Host "============================================================`n" -ForegroundColor Cyan
