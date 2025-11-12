# =============================================================================
#
# Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

<#  
    The MLC_Setup.ps1 PowerShell script automatesthe setup process for MLC LLM by downloading and installing necessary components, including Miniconda and various dependencies.
    It creates and activates a virtual environment, upgrades pip, and installs required Python packages. 
    The function also runs scripts to check and ensure all dependencies are correctly set up, providing a complete and successful installation for MLC LLM tutorials. 
    By default, $rootDirPath is set to C:\Qualcomm_AI, where all files will be downloaded.
	
    Note: Users can modify values such as rootDirPath, QNN SDK version, etc, if desired
#>


############################ Define the URL for download ###############################################

$condaUrl             = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$mlcLlmUtilsUrl       = "https://codelinaro.jfrog.io/artifactory/clo-472-adreno-opensource-ai/mlc-llm/2025.06.r1/mlc_llm-utils-win-x86-2025.06.r1.zip"
$mingwUrl             = "https://github.com/niXman/mingw-builds-binaries/releases/download/14.2.0-rt_v12-rev0/x86_64-14.2.0-release-win32-seh-msvcrt-rt_v12-rev0.7z"
$gitUrl               = "https://github.com/git-for-windows/git/releases/download/v2.47.0.windows.2/Git-2.47.0.2-64-bit.exe"
$sevenZipUrl          = "https://7-zip.org/a/7z2408-arm64.exe"
$mlcWheelFileUrl      = "https://codelinaro.jfrog.io/artifactory/clo-472-adreno-opensource-ai/mlc-llm/2025.06.r1/mlc_llm_adreno_cpu_clml_2025_06_r1-0.1.dev0-cp312-cp312-win_amd64.whl"
$mlcFileName          = "mlc_llm_adreno_cpu_clml_2025_06_r1-0.1.dev0-cp312-cp312-win_amd64.whl" 
$tvmWheelFileUrl      = "https://codelinaro.jfrog.io/artifactory/clo-472-adreno-opensource-ai/mlc-llm/2025.06.r1/tvm_adreno_cpu_clml_2025_06_r1-0.20.dev0-cp312-cp312-win_amd64.whl"
$tvmFileName          = "tvm_adreno_cpu_clml_2025_06_r1-0.20.dev0-cp312-cp312-win_amd64.whl" 

# Visual Studio dependency for compiling and converting ONNX model to C++ & binary, used for generating model.dll file
$vsStudioUrl = "https://download.visualstudio.microsoft.com/download/pr/7593f7f0-1b5b-43e1-b0a4-cceb004343ca/09b5b10b7305ae76337646f7570aaba52efd149b2fed382fdd9be2914f88a9d0/vs_Enterprise.exe"

$vsInstallerPath = "C:\VS\Common7\Tools\Launch-VsDevShell.ps1"
$SUGGESTED_VS_BUILDTOOLS_VERSION = "14.34"
$SUGGESTED_WINSDK_VERSION = "10.0.22621"
$SUGGESTED_VC_VERSION = "19.34"

$global:CHECK_RESULT = 1
$global:tools = @{}
$global:tools.add( 'vswhere', "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" )

############################ Define the Installation paths ###############################################

$condaInstallPath     = "C:\ProgramData\miniconda3"
$mingwInstallPath     = "C:\MinGW"
$gitInstallPath       = "C:\Program Files\Git"
$sevenZipInstallPath  = "C:\Program Files\7-Zip"
$sevenZipPath         = "C:\Program Files\7-Zip\7z.exe"



########################################      Functions       ##########################################

Function Set_Variables {
    param (
        [string]$rootDirPath = "C:\WoS_AI"
    )
    # Create the Root folder if it doesn't exist
    if (-Not (Test-Path $rootDirPath)) {
        New-Item -ItemType Directory -Path $rootDirPath
    }
    Set-Location -Path $rootDirPath
    # Define download directory inside the working directory for downloading all dependency files and SDK.
    $global:DIR_PATH = "$rootDirPath"
    $global:downloadDirPath = "$rootDirPath\Downloads"
    # Create the Root folder if it doesn't exist
    if (-Not (Test-Path $downloadDirPath)) {
        New-Item -ItemType Directory -Path $downloadDirPath
    }
    # Define the path where the installer will be downloaded.
    $global:condaDownloaderPath       = "$downloadDirPath\miniconda.exe"
    $global:sevenZipDownloadPath      = "$downloadDirPath\7zInstaller.exe"
    $global:mingwDownloaderPath       = "$downloadDirPath\x86_64-14.2.0-release-win32-seh-msvcrt-rt_v12-rev0.7z"
    $global:gitDownloadPath           = "$downloadDirPath\Git-2.47.0.2-64-bit.exe"
    $global:vsStudioDownloadPath      = "$downloadDirPath\vs_Enterprise.exe"
    $global:debugFolder               = "$rootDirPath\Debug_Logs"
    # Create the Root folder if it doesn't exist
    if (-Not (Test-Path $debugFolder)) {
        New-Item -ItemType Directory -Path $debugFolder
    }
}

Function Show-Progress {
    param (
        [int]$percentComplete,
        [int]$totalPercent
    )
    $progressBar = ""
    $progressWidth = 100
    $progress = [math]::Round((($percentComplete/$totalPercent)*100) / 100 * $progressWidth)
    for ($i = 0; $i -lt $progressWidth; $i++) {
        if ($i -lt $progress) {
            $progressBar += "#"
        } else {
            $progressBar += "-"
        }
    }
    # Write-Progress -Activity "Progress" -Status "$percentComplete% Complete" -PercentComplete $percentComplete
    Write-Host "[$progressBar] ($percentComplete/$totalPercent) Setup Complete"
}

Function install_VS_Studio {
    param()
    process {
        # Install the VS Studio
        Start-Process -FilePath $vsStudioDownloadPath -ArgumentList "--installPath C:\VS --passive --wait --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --add Microsoft.VisualStudio.Component.VC.14.34.17.4.x86.x64 --add Microsoft.VisualStudio.Component.VC.14.34.17.4.ARM64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset" -Wait -PassThru
        # Check if the VS Studio
        check_MSVC_components_version
    }
}


Function show_recommended_version_message {
    param (
        [String] $SuggestVersion,
        [String] $FoundVersion,
        [String] $SoftwareName
    )
    process {
        Write-Warning "The version of $SoftwareName $FoundVersion found has not been validated. Recommended to use known stable $SoftwareName version $SuggestVersion"
    }
}

Function show_required_version_message {
    param (
        [String] $RequiredVersion,
        [String] $FoundVersion,
        [String] $SoftwareName
    )
    process {
        Write-Host "ERROR: Require $SoftwareName version $RequiredVersion. Found $SoftwareName version $FoundVersion" -ForegroundColor Red
    }
}


Function compare_version {
    param (
        [String] $TargetVersion,
        [String] $FoundVersion,
        [String] $SoftwareName
    )
    process {
        if ( (([version]$FoundVersion).Major -eq ([version]$TargetVersion).Major) -and (([version]$FoundVersion).Minor -eq ([version]$TargetVersion).Minor) ) { }
        elseif ( (([version]$FoundVersion).Major -ge ([version]$TargetVersion).Major) -and (([version]$FoundVersion).Minor -ge ([version]$TargetVersion).Minor) ) {
            show_recommended_version_message $TargetVersion $FoundVersion $SoftwareName
        }
        else {
            show_required_version_message $TargetVersion $FoundVersion $SoftwareName
            $global:CHECK_RESULT = 0
        }
    }
}

Function locate_prerequisite_tools_path {
    param ()
    process {
        # Get and Locate VSWhere
        if (!(Test-Path $global:tools['vswhere'])) {
            Write-Host "No Visual Studio Instance(s) Detected, Please Refer To The Product Documentation For Details" -ForegroundColor Red
        }
    }
}

Function detect_VS_instance {
    param ()
    process {
        locate_prerequisite_tools_path

        $INSTALLED_VS_VERSION = & $global:tools['vswhere'] -latest -property installationVersion
        $INSTALLED_PATH = & $global:tools['vswhere'] -latest -property installationPath
        $productId = & $global:tools['vswhere'] -latest -property productId

        return $productId, $INSTALLED_PATH, $INSTALLED_VS_VERSION
    }
}

Function check_VS_BuildTools_version {
    param (
        [String] $SuggestVersion = $SUGGESTED_VS_BUILDTOOLS_VERSION
    )
    process {
        $INSTALLED_PATH = & $global:tools['vswhere'] -latest -property installationPath
        $version_file_path = Join-Path $INSTALLED_PATH "VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"
        if (Test-Path $version_file_path) {
            $INSTALLED_VS_BUILDTOOLS_VERSION = Get-Content $version_file_path
            compare_version $SuggestVersion $INSTALLED_VS_BUILDTOOLS_VERSION "VS BuildTools"
            return $INSTALLED_VS_BUILDTOOLS_VERSION
        }
        else {
            Write-Error "VS BuildTools not installed"
            $global:CHECK_RESULT = 0
        }
        return "Not Installed"
    }
}

Function check_WinSDK_version {
    param (
        [String] $SuggestVersion = $SUGGESTED_WINSDK_VERSION
    )
    process {
        $INSTALLED_WINSDK_VERSION = Get-ItemPropertyValue -Path 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\Windows\v10.0' -Name ProductVersion
        if($?) {
            compare_version $SuggestVersion $INSTALLED_WINSDK_VERSION "Windows SDK"
            return $INSTALLED_WINSDK_VERSION
        }
        else {
            Write-Error "Windows SDK not installed"
            $global:CHECK_RESULT = 0
        }
        return "Not Installed"
    }
}

Function check_VC_version {
    param (
        [String] $VsInstallLocation,
        [String] $BuildToolVersion,
        [String] $Arch,
        [String] $SuggestVersion = $SUGGESTED_VC_VERSION
    )
    process {
        $VcExecutable = Join-Path $VsInstallLocation "VC\Tools\MSVC\" | Join-Path -ChildPath $BuildToolVersion | Join-Path -ChildPath "bin\Hostx64" | Join-Path -ChildPath $Arch | Join-Path -ChildPath "cl.exe"

        if(Test-Path $VcExecutable) {
            #execute $VcExecutable and retrieve stderr since version is in it.
            $process_alloutput = & "$VcExecutable" 2>&1
            $process_stderror = $process_alloutput | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }
            $CMD = $process_stderror | Out-String | select-string "Version\s+(\d+\.\d+\.\d+)" # The software version is output in STDERR
            $INSTALLED_VC_VERSION = $CMD.matches.groups[1].value
            if($INSTALLED_VC_VERSION) {
                compare_version $SuggestVersion $INSTALLED_VC_VERSION ("Visual C++(" + $Arch + ")")
                return $INSTALLED_VC_VERSION
            }
            else {
                Write-Error "Visual C++ not installed"
                $global:CHECK_RESULT = 0
            }
        }
        return "Not Installed"
    }
}

Function check_MSVC_components_version {
    param ()
    process {
        $check_result = @()
        $productId, $vs_install_path, $vs_installed_version = detect_VS_instance
        if ($productId) {
            $check_result += [pscustomobject]@{Name = "Visual Studio"; Version = $vs_installed_version}
        }
        else {
            $check_result += [pscustomobject]@{Name = "Visual Studio"; Version = "Not Installed"}
            $global:CHECK_RESULT = 0
        }
        $buildtools_version = check_VS_BuildTools_version
        $check_result += [pscustomobject]@{Name = "VS Build Tools"; Version = $buildtools_version}
        $check_result += [pscustomobject]@{Name = "Visual C++(x86)"; Version = check_VC_version $vs_install_path $buildtools_version "x64"}
        $check_result += [pscustomobject]@{Name = "Visual C++(arm64)"; Version = check_VC_version $vs_install_path $buildtools_version "arm64"}
        $check_result += [pscustomobject]@{Name = "Windows SDK"; Version = check_WinSDK_version}
        Write-Host ($check_result | Format-Table| Out-String).Trim()
    }
}

Function download_install_VS_Studio {
    param()
    process {
        # Checking if VStudio already installed
        # If yes
        if (Test-Path $vsInstallerPath) {
            Write-Output "VS-Studio already installed."
        }
        # Else downloading and installing VStudio
        else {
            Write-Output "Downloading the VS Studio..." 
            $result = download_file -url $vsStudioUrl -downloadfile $vsStudioDownloadPath
            # Checking for successful download
            if ($result) {
                Write-Output "VS Studio is downloaded at : $vsStudioDownloadPath" 
                Write-Output "installing VS-Studio..."
                if (install_VS_Studio) {
                    Write-Output "VS-Studio installed successfully." 
                }
                else{
                    Write-Output "VS-Studio installation failed..  from : $vsStudioDownloadPath"  
                }
            } 
            else{
                Write-Output "VS Studio download failed... Downloaded the VS Studio from :  $vsStudioUrl and install." 
            }
        }
    }
}

Function download_file {
    param (
        [string]$url,
        [string]$downloadfile
    )
    process {
        try {
	    # Download the file
	    Invoke-WebRequest -Uri $url -OutFile $downloadfile
	    return $true
        }
        catch {
            return $false
        }
    }
}



Function download_and_extract_mlc_utils {
    param (
        [string]$artifactsUrl,
        [string]$rootDirPath
    )
    process {
        $zipFilePath = "$rootDirPath\Downloads\mlc_utils.zip"
	if (Test-Path $zipFilePath) {
            	Write-Output "MLC already exists at : $condaDownloaderPath"
        } 
	else {
		# Download the ZIP file
		Invoke-WebRequest -Uri $artifactsUrl -OutFile $zipFilePath
	}

        # Extract the ZIP file
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zipFilePath, "$rootDirPath\Mlc_llm")
	return $true
    }  
}

Function install_conda {
    param()
    process {
	Start-Process -FilePath $condaDownloaderPath -ArgumentList "/S" -Wait
        $env:Path += ";$condaInstallPath;$condaInstallPath\Scripts;$condaInstallPath\Library\bin"
        [System.Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine) + ";" + [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
        & "$condaInstallPath\Scripts\conda.exe" init
		return $true
    }   
}

Function Install-Mingw {
    param()
    process {
        # Create the Root folder if it doesn't exist
        if (-Not (Test-Path $mingwInstallPath)) {
            New-Item -ItemType Directory -Path $mingwInstallPath
        }
        & $sevenZipPath x $mingwDownloaderPath "-o$mingwInstallPath"
        # Check if MinGW was installed successfully
        if (Test-Path "$mingwInstallPath\mingw64\bin\gcc.exe") {
            Write-Output "MinGW installed successfully."
            # Get the current PATH environment variable
            $envPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
            # Add the new paths if they are not already in the PATH
            if ($envPath -notlike "*$mingwInstallPath\mingw64\bin*") {
                $envPath = "$mingwInstallPath\mingw64\bin;$envPath"
                [System.Environment]::SetEnvironmentVariable("Path", $envPath, [System.EnvironmentVariableTarget]::User)
            }
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine) + ";" + [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
            return $true
        }
        return $false
    }
}

Function install_git {
    param()
    process {
        # Install Git
        Start-Process -FilePath $gitDownloadPath -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
        # Check if Git was installed successfully
        if (Test-Path "$gitInstallPath\bin\git.exe") {
            Write-Output "Git installed successfully."
            # Get the current PATH environment variable
            $envPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
            # Add the new paths if they are not already in the PATH
            if ($envPath -notlike "*$gitInstallPath\bin*") {
                $envPath = "$gitInstallPath\bin;$envPath"
                [System.Environment]::SetEnvironmentVariable("Path", $envPath, [System.EnvironmentVariableTarget]::User)
            }
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine) + ";" + [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
            return $true
        }
        return $false
    }
}

Function install_7z {
    param()
    process {
        # Install 7-Zip
        Start-Process -FilePath $sevenZipDownloadPath -ArgumentList "/S" -Wait
        # Check if 7-Zip was installed successfully
        if (Test-Path "$sevenZipInstallPath\7z.exe") {
            Write-Output "7-Zip installed successfully."
            # Get the current PATH environment variable
            $envPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
            # Add the new paths if they are not already in the PATH
            if ($envPath -notlike "*$sevenZipInstallPath*") {
                $envPath = "$sevenZipInstallPath;$envPath"
                [System.Environment]::SetEnvironmentVariable("Path", $envPath, [System.EnvironmentVariableTarget]::User)
            }
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine) + ";" + [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
            # # Verify 7-Zip installation
            # & "$sevenZipInstallPath\7z.exe" --help
            return $true
        }
        return $false
    }
}

Function download_install_conda {
    param()
    process {
        # Define the path to check if Miniconda is installed
        $minicondaPath = "$condaInstallPath\_conda.exe"
        # Check if Miniconda is already installed
        if (Test-Path $minicondaPath) {
            Write-Output "Miniconda is already installed at: $minicondaPath"
        }
        else {
            Write-Output "Downloading the conda file ..."
            $result = download_file -url $condaUrl -downloadfile $condaDownloaderPath
            # Checking for successful download
            if ($result) {
                Write-Output "conda File is downloaded at : $condaDownloaderPath"
                Write-Output "Installing conda..."
                if (install_conda) {
                    Write-Output "conda installed successfully."
                }
                else {
                    Write-Output "conda installation failed..  from : $condaDownloaderPath"  
                }
            } 
            else {
                Write-Output "conda download failed. Download the conda file from : $condaUrl and install."
            }
        }
    }
}

function download_install_mingw {
    param(
    )
    process {
        # Check if MinGW is already installed
        if (Test-Path "$mingwInstallPath\mingw64\bin\gcc.exe") {
            Write-Output "MinGW already installed."
        }
        # Else downloading and installing MinGW
        else {
            Write-Output "Downloading the MinGW file ..."
            $result = download_file -url $mingwUrl -downloadfile $mingwDownloaderPath
            # Checking for successful download
            if ($result) {
                Write-Output "MinGW file is downloaded at: $mingwDownloaderPath"
                Write-Output "Installing MinGW..."
                if (Install-Mingw) {
                    Write-Output "MinGW installed successfully."
                }
                else {
                    Write-Output "MinGW installation failed. Please install MinGW from: $mingwDownloaderPath"
                }
            }
            else {
                Write-Output "MinGW download failed. Download the MinGW file from: $mingwUrl and install."
            }
        }
    }
}


Function download_install_7z {
    param()
    process {
        # Checking if 7-Zip is already installed
        if (Test-Path "$sevenZipInstallPath\7z.exe") {
            Write-Output "7-Zip already installed."
        }
        # Else downloading and installing 7-Zip
        else {
            Write-Output "Downloading the 7-Zip file ..."
            $result = download_file -url $sevenZipUrl -downloadfile $sevenZipDownloadPath
            # Checking for successful download
            if ($result) {
                Write-Output "7-Zip file is downloaded at : $sevenZipDownloadPath"
                Write-Output "Installing 7-Zip..."
                if (install_7z) {
                    Write-Output "7-Zip installed successfully."
                }
                else {
                    Write-Output "7-Zip installation failed. Please install 7-Zip from : $sevenZipDownloadPath"
                }
            }
            else {
                Write-Output "7-Zip download failed. Download the 7-Zip file from : $sevenZipUrl and install."
            }
        }
    }
}

Function download_install_git {
    param()
    process {
        # Checking if Git is already installed
        if (Test-Path "$gitInstallPath\bin\git.exe") {
            Write-Output "Git already installed."
        }
        # Else downloading and installing Git
        else {
            Write-Output "Downloading the Git file ..."
            $result = download_file -url $gitUrl -downloadfile $gitDownloadPath
            # Checking for successful download
            if ($result) {
                Write-Output "Git file is downloaded at : $gitDownloadPath"
                Write-Output "Installing Git..."
                if (install_git) {
                    Write-Output "Git 2.47.0.2 installed successfully."
                }
                else {
                    Write-Output "Git installation failed. Please install Git 2.47.0.2 from : $gitDownloadPath"
                }
            }
            else {
                Write-Output "Git download failed. Download the Git file from : $gitUrl and install."
            }
        }
    }
}

Function Check_Setup {
    param(
        [string]$logFilePath
    )
    process {
        $results = @()
        # Check if MiniCondao is installed
        if (Test-Path "$condaInstallPath\_conda.exe") {
            $results += [PSCustomObject]@{
                Component = "MiniConda"
                Status    = "Successful"
                Comments  = "$(conda --version)"
            }
        } else {
            $results += [PSCustomObject]@{
                Component = "MiniConda"
                Status    = "Failed"
                Comments  = "Download from $vsStudioUrl"
            }
        }

        # Check if GCC is installed
        if (Test-Path "$mingwInstallPath\mingw64\bin\gcc.exe") {
            $results += [PSCustomObject]@{
                Component = "Mingw64 GCC"
                Status    = "Successful"
                Comments  = "$(gcc --version)"
            }
        } else {
            $results += [PSCustomObject]@{
                Component = "Mingw64 GCC"
                Status    = "Failed"
                Comments  = "Download from $mingwUrl"
            }
        }

 	# Check if Visual Studio is installed
        if (Test-Path $vsInstallerPath) {
            $results += [PSCustomObject]@{
                Component = "Microsoft Visual Studio"
                Status    = "Successful"
                Comments  = "Microsoft Visual Studio version 17.10.4"
            }
        } else {
            $results += [PSCustomObject]@{
                Component = "Microsoft Visual Studio"
                Status    = "Failed"
                Comments  = "Download from $vsStudioUrl"
            }
        }

        # Check if Git is installed
        if (Test-Path "$gitInstallPath") {
            $results += [PSCustomObject]@{
                Component = "Git"
                Status    = "Successful"
                Comments  = "$(git --version)"
            }
        } else {
            $results += [PSCustomObject]@{
                Component = "Git"
                Status    = "Failed"
                Comments  = "Download from $gitUrl"
            }
        }

        # Output the results as a table
        $results | Format-Table -AutoSize

        # Store the results in a debug.log file
        $results | Out-File -FilePath $logFilePath
    }
}

################################# Main Code #####################################

Function MLC_LLM_Setup {
    param(
        [string]$rootDirPath = "C:\WoS_AI"
        )
    process {
        Set_Variables -rootDirPath $rootDirPath
        download_install_conda
        Show-Progress -percentComplete 1 5
        download_install_7z
        download_install_mingw
        Show-Progress -percentComplete 2 5
	download_install_VS_Studio
 	Show-Progress -percentComplete 3 5
        download_install_git
        Show-Progress -percentComplete 4 5
	Write-Output "Creating the conda env ... "
	conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
	conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
	conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
        conda create -n MLC_VENV -c conda-forge "llvmdev=15" "cmake>=3.24" git rust numpy==1.26.4 decorator psutil typing_extensions scipy attrs git-lfs python=3.12 onnx clang_win-64 -y
        #conda activate MLC_VENV
        download_and_extract_mlc_utils -artifactsUrl $mlcLlmUtilsUrl -rootDirPath $rootDirPath
        cd $rootDirPath
	if (Test-Path "$downloadDirPath\$mlcFileName") {
 		Write-Output "MLC wheel already exists at : $condaDownloaderPath"
        } 
	else {
		Write-Output "Downloading MLC wheel file..."
		# Invoke-WebRequest -o "$downloadDirPath\mlc_llm_adreno_cpu-0.1.dev0-cp312-cp312-win_amd64.whl" https://codelinaro.jfrog.io/artifactory/clo-472-adreno-opensource-ai/mlc-llm/mlc_llm_adreno_cpu-0.1.dev0-cp312-cp312-win_amd64.whl
        Invoke-WebRequest -o "$downloadDirPath\$mlcFileName" "$mlcWheelFileUrl"
		Write-Output "MLC wheel downloaded at : $downloadDirPath\$mlcFileName"
	}
	if (Test-Path "$downloadDirPath\$tvmFileName") {
            	Write-Output "TVM wheel already exists at : $condaDownloaderPath"
        } 
	else {
		Write-Output "Downloading TVM wheel file..."
		# Invoke-WebRequest -o "$downloadDirPath\tvm_adreno_cpu-0.19.dev0-cp312-cp312-win_amd64.whl" https://codelinaro.jfrog.io/artifactory/clo-472-adreno-opensource-ai/mlc-llm/tvm_adreno_cpu-0.19.dev0-cp312-cp312-win_amd64.whl
        Invoke-WebRequest -o "$downloadDirPath\$tvmFileName" "$tvmWheelFileUrl"
		Write-Output "TVM wheel downloaded at : $downloadDirPath\$tvmFileName"
	}
	$mlcEnvPath = (conda info --base) + "\envs\MLC_VENV"
	# Install the package into the specified Conda environment
	& "$mlcEnvPath\Scripts\pip.exe" install "$downloadDirPath\$mlcFileName" --prefix $mlcEnvPath
	& "$mlcEnvPath\Scripts\pip.exe" install "$downloadDirPath\$tvmFileName" --prefix $mlcEnvPath
	Show-Progress -percentComplete 5 5
        Write-Output "***** Installation of MLC LLM *****"
        Check_Setup -logFilePath "$debugFolder\MLC_LLM_Setup_Debug.log"
    }
}
