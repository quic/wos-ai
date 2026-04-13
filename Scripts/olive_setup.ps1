# =============================================================================
#
# Copyright (c) 2024-26, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

<#  
    The ort_setup.ps1 PowerShell script automates the setup of various ONNX Runtime (ORT) Execution Providers (EP) by downloading and installing necessary components.
    Such as Python, ONNX models, required artifacts, and redistributable packages. Separate functions are defined for each ORT EP. 
    Each function checks for the existence of a virtual environment at a rootDirPath and creates one if it doesn’t exist. 
    They then activate the virtual environment, upgrade pip, and install the required packages: onnxruntime for CPU EP, onnxruntime-directml for GPU EP, onnxruntime-qnn for QNN EP, and optimum[onnxruntime] for Huggingface tutorials. 
    It is not necessary to install files for all ORT EP, users are free to try any one EP or all EPs based on their needs, and the script will handle the installation accordingly. After installation, a success message will be shown.
    The ORT_QNN_setup function also copies specific DLL files to the rootDirPath, which are needed to run the model on NPU. 
    By default, $rootDirPath is set to C:\WoS_AI, where all files will be downloaded and the Python environment will be created. 
#>

############################ Define the URL for download ##################################

# URL for downloading the python 3.12.6 
<#  For Python 3.12.6 dependency:
    - Any version of Python can be used for AMD architecture.
    - For ARM architecture, install Python 3.11.x only. ORT QNN EP supports only Python ARM or AMD installations.
    - Other ORT EPs require the AMD version of Python.
    - To use ORT QNN EP on ARM, it is advised to create two Python environments: one for pre- and post-processing, and a second ARM environment for execution.
    Note: Python ARM has limitations with other dependencies such as torch, onnx, etc.
    Therefore, we recommend using the AMD version to avoid these issues.
#>
$pythonUrl = "https://www.python.org/ftp/python/3.12.6/python-3.12.6-amd64.exe"

# ONNX model file for image prediction used in tutorials.
# $modelUrl =  "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/mobilenet_v2.onnx"

# URL for downloading the Visual Studio Redistributable for ARM64. Visual studio is used during model exection on HTP(NPU) backend.
$vsRedistributableUrl = "https://aka.ms/vs/17/release/vc_redist.arm64.exe"

<# Required files 
    - License             : License document
#>
$licenseUrl        = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/LICENSE"

<#  Artifacts for tutorials, including:
    - io_utils.py         : Utility file for preprocessing images and postprocessing to get top 5 predictions.
#>
# $io_utilsUrl       = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Artifacts/io_utils.py"


############################ python installation path ##################################
# Retrieves the value of the Username
$username =  (Get-ChildItem Env:\Username).value

$pythonInstallPath = "C:\Users\$username\AppData\Local\Programs\Python\Python312"
$pythonScriptsPath = $pythonInstallPath+"\Scripts"

#### GIT INSTALL ####
# Git download URL (64-bit Windows installer)
$gitUrl             = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe"
$gitVersion         = "2.47.1"
#### GIT INSTALL ####


<#
    Each tutorial section will have its own individual Python environment:

    - ORT CPU EP           : Uses SDX_ORT_CPU_ENV, which has specific Python package dependencies.
    - ORT GPU EP           : Uses SDX_ORT_CPU_ENV, which has specific Python package dependencies.
    - ORT QNN EP           : Uses SDX_ORT_QNN_ENV, which has specific Python package dependencies.
    - Hugging Face Optimum : Uses SDX_HF_ENV, which has specific Python package dependencies.

    Note: Each section has dependencies that cannot be used in conjunction with other Python packages.
    For example, ORT QNN EP and ORT CPU EP cannot install packages in the same Python environment.
    Users are advised to create separate Python environments for each case.

    Define the paths for each environment to be created in the root directory 
	
    Note: Users can change this path to another location if desired.
#>

$OLIVE_ENV_Path = "Python_Venv\SDX_OLIVE_ENV"
$Olive_Folder_path = "Models\Olive"
$Olive_json_url = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Scripts/olive_mobilenet_qnn_ep.json"
$Olive_data_url = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Scripts/olive_download_files.py"
$Olive_user_script_url = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Scripts/olive_user_script.py"

####################################################################################
############################      Function        ##################################


Function Set_Variables {
    param (
        [string]$rootDirPath = "C:\WoS_AI"
    )
    # Create the Root folder if it doesn't exist
    if (-Not (Test-Path $rootDirPath)) {
        New-Item -ItemType Directory -Path $rootDirPath
    }
    Set-Location -Path $rootDirPath
    # Define download directory inside the working directory for downloading all dependency files.
    $global:downloadDirPath = "$rootDirPath\Downloads"
    # Create the Root folder if it doesn't exist
    if (-Not (Test-Path $downloadDirPath)) {
        New-Item -ItemType Directory -Path $downloadDirPath
    }
    # Define the path where the installer will be downloaded.
    $global:pythonDownloaderPath = "$downloadDirPath\python-3.12.6-amd64.exe" 
    $global:vsRedistDownloadPath = "$downloadDirPath\vc_redist.arm64.exe"

     #### GIT INSTALL ####
    # Define the path where the Git installer will be downloaded.
    $global:gitDownloaderPath = "$downloadDirPath\Git-$gitVersion-64-bit.exe"
    #### GIT INSTALL ####

    # Define the license download path.
    $global:lincensePath      = "$rootDirPath\License"

    $global:debugFolder    = "$rootDirPath\Debug_Logs"
    # Create the Root folder if it doesn't exist.
    if (-Not (Test-Path $debugFolder)) {
        New-Item -ItemType Directory -Path $debugFolder
    }
    # Define folder path for Olive artifacts.
    $global:OliveFolder = "$rootDirPath\$Olive_Folder_path"
    # Create the Root folder if it doesn't exist
    if (-Not (Test-Path $OliveFolder)) {
        New-Item -ItemType Directory -Path $OliveFolder
    }
    
}

Function download_file {
    param (
        [string]$url,
        [string]$downloadfile
    )
    # Download the file
    process {
        try {
            Invoke-WebRequest -Uri $url -OutFile $downloadfile
            return $true
        }
        catch {
            return $false
        }
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

Function install_vsRedistributable
{
    param()
    process{
        Start-Process -FilePath $vsRedistDownloadPath -ArgumentList "/install", "/quiet", "/norestart" -Wait 
        if(Test-Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\arm64"){
            return $true
        }
        return $false
    }
}

Function install_python {
    param()
    process {
        # Install Python
        Start-Process -FilePath $pythonDownloaderPath -ArgumentList "/quiet InstallAllUsers=1 TargetDir=$pythonInstallPath" -Wait
        # Check if Python was installed successfully
        if (Test-Path "$pythonInstallPath\python.exe") {
            Write-Output "Python installed successfully."
            # Get the current PATH environment variable
            $envPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)

            # Add the new paths if they are not already in the PATH
            if ($envPath -notlike "*$pythonScriptsPath*") {
                $envPath = "$pythonScriptsPath;$pythonInstallPath;$envPath"
                [System.Environment]::SetEnvironmentVariable("Path", $envPath, [System.EnvironmentVariableTarget]::User)
            }

            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine) + ";" + [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)

            # Verify Python installation
            return $true
        } 
        else {
            return $false
        }
        
    }
}

#### GIT INSTALL ####
Function install_git {
    param()
    process {
        # Run Git installer silently
        Start-Process -FilePath $gitDownloaderPath -ArgumentList "/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
        # Refresh PATH so git.exe is visible in the current session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine) + ";" + [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
        # Verify Git installation
        try {
            $installedVer = (git --version 2>&1)
            if ($installedVer -match "\d+\.\d+\.\d+") {
                return $true
            }
        } catch {}
        return $false
    }
}

Function get_installed_git_version {
    param()
    process {
        try {
            $verString = git --version 2>&1
            if ($verString -match "(\d+\.\d+\.\d+)") {
                return $matches[1]
            }
        } catch {}
        return ""
    }
}

Function download_install_git {
    param()
    process {
        $installedVer = get_installed_git_version
        # If Git is already installed with the expected version, skip
        if ($installedVer -eq $gitVersion) {
            Write-Output "Git $gitVersion is already installed. Skipping."
            return
        }
        # If a different version is installed, notify the user
        if ($installedVer -ne "") {
            Write-Output "Git $installedVer is currently installed. Installing Git $gitVersion..."
        } else {
            Write-Output "Git not found. Downloading Git $gitVersion..."
        }
        $result = download_file -url $gitUrl -downloadfile $gitDownloaderPath
        if ($result) {
            Write-Output "Git installer downloaded at: $gitDownloaderPath"
            Write-Output "Installing Git $gitVersion..."
            if (install_git) {
                Write-Output "Git $gitVersion installed successfully."
            } else {
                Write-Output "Git installation failed. Please install manually from: $gitUrl"
            }
        } else {
            Write-Output "Git download failed. Download manually from: $gitUrl"
        }
    }
}
#### GIT INSTALL ####

Function download_script_license{
    param()
    process{
        # License 
        # Checking if License already present 
        # If yes
        if(Test-Path $lincensePath){
            Write-Output "License is already downloaded at : $lincensePath"
        }
        # Else dowloading
        else{
            $result = download_file -url $licenseUrl -downloadfile $lincensePath
            if($result){
                Write-Output "License is downloaded at : $lincensePath"
            }
            else{
                Write-Output "License download failed. Download from $licenseUrl"
            }
        }
    }
}

Function download_install_python {
    param()
    process {
        # Check if python already installed
        # If Yes
        if (Test-Path "$pythonInstallPath\python.exe") {
            Write-Output "Python already installed."
        }
        # Else downloading and installing python
        else{
            Write-Output "Downloading the python file ..." 
            $result = download_file -url $pythonUrl -downloadfile $pythonDownloaderPath
            # Checking for successful download
            if ($result) {
                Write-Output "Python File is downloaded at : $pythonDownloaderPath"
                Write-Output "Installing python..."
                if (install_python) {
                    Write-Output "Python installed successfully." 
                }
                else {
                    Write-Output "Python installation failed.. Please installed python from : $pythonDownloaderPath"  
                }
            } 
            else{
                Write-Output "Python download failed. Download the python file from : $pythonUrl and install." 
            }
        }
    }
}


Function download_install_redistributable {
    param()
    process {
        # Download redistributable file 
        # Checking if redistributable already present 
        # If yes
        # if (Test-Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\arm64") {
        #     Write-Output "VS-Redistributable is already installed."
        # }
        # # Else downloading and installing redistributable
        # else {
        Write-Output "Downloading VS-Redistributable..." 
        $result = download_file -url $vsRedistributableUrl -downloadfile $vsRedistDownloadPath
        if ($result) {
            Write-Output "VS-Redistributable File is downloaded at : $vsRedistDownloadPath" 
            Write-Output "installing VS-Redistributable..."
            if (install_vsRedistributable) {
                Write-Output "VS-Redistributable is installed successfully." 
            }
            else {
                Write-Output "VS-Redistributable installation failed... from : $vsRedistDownloadPath" 
            }
        } 
        else{
            Write-Output "VS-Redistributable download failed.... Download the VS-Redistributable file from :  $vsRedistributableUrl and install" 
        }
        # }
    }
}

############################## Main code ##################################]

Function Check_Setup {
    param(
        [string]$logFilePath
    )
    process {
        $results = @()

        # Check if Python is installed
        if (Test-Path "$pythonInstallPath\python.exe") {
            $results += [PSCustomObject]@{
                Component = "Python"
                Status    = "Successful"
                Comments  = "$(python --version)"
            }
        } else {
            $results += [PSCustomObject]@{
                Component = "Python"
                Status    = "Failed"
                Comments  = "Download from $pythonUrl"
            }
        }

        # Check if Visual Studio Redistributable is installed
        if (Test-Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\arm64") {
            $results += [PSCustomObject]@{
                Component = "VS-Redistributable"
                Status    = "Successful"
                Comments  = "Visual Studio C++ redistributable 14.42.3"
            }
        } else {
            $results += [PSCustomObject]@{
                Component = "VS-Redistributable"
                Status    = "Failed"
                Comments  = "Download from $vsRedistributableUrl"
            }
        }

        #### GIT INSTALL ####
        # Check if Git is installed
        $installedGitVer = get_installed_git_version
        if ($installedGitVer -ne "") {
            $results += [PSCustomObject]@{
                Component = "Git"
                Status    = "Successful"
                Comments  = "git version $installedGitVer"
            }
        } else {
            $results += [PSCustomObject]@{
                Component = "Git"
                Status    = "Failed"
                Comments  = "Download from $gitUrl"
            }
        }
        #### GIT INSTALL ####

        # Output the results as a table
        $results | Format-Table -AutoSize

        # Capture System Info 
        $systemInfo = Get-ComputerInfo | Out-String

        # Store the results in a debug.log file with additional lines
        $logContent = @(
            "Status of the installation:"
            $results | Format-Table -AutoSize | Out-String
            "------ System Info ------"
            $systemInfo
        )
        # Store the results in a debug.log file
        $logContent | Out-File -FilePath $logFilePath
    }
}


Function OLIVE_Setup {
    param(
        [string]$rootDirPath = "C:\WoS_AI"
        )
    process {
    	# Set the permission on PowerShell to execute the command. If prompted, accept and enter the desired input to provide execution permission.
     	Set-ExecutionPolicy RemoteSigned 
        Set_Variables -rootDirPath $rootDirPath
        download_install_python
        Show-Progress -percentComplete 1 5
        download_install_redistributable
        Show-Progress -percentComplete 2 5
        download_script_license
        Show-Progress -percentComplete 3 5
        download_install_git
        Show-Progress -percentComplete 4 5
		download_file -url $Olive_data_url -downloadfile $Olive_Folder_path\olive_download_files.py
		download_file -url $Olive_user_script_url -downloadfile $Olive_Folder_path\olive_user_script.py
        $SDX_OLIVE_ENV_Path = "$rootDirPath\$OLIVE_ENV_Path"
        # Check if virtual environment was created
        if (-Not (Test-Path -Path  $SDX_OLIVE_ENV_Path))
        {
           py -3.12 -m venv $SDX_OLIVE_ENV_Path
        }
        # Check if the virtual environment was created successfully
        if (Test-Path "$SDX_OLIVE_ENV_Path\Scripts\Activate.ps1") {
            # Activate the virtual environment
            & "$SDX_OLIVE_ENV_Path\Scripts\Activate.ps1"
            python -m pip install --upgrade pip
            pip install onnxruntime-qnn==1.24.4
			pip install olive-ai==0.9.1
			pip install transformers==4.56.0
            pip install pillow
			pip install requests
			pip install torchvision
        }
        Show-Progress -percentComplete 5 5
        Write-Output "***** Installation for ONNX-QNN *****"
        Check_Setup -logFilePath "$debugFolder\ORT_QNN_Setup_Debug.log"
        Invoke-Command { & "powershell.exe" } -NoNewScope
    }
}

Function Activate_OLIVE_VENV {
    param ( 
        [string]$rootDirPath = "C:\WoS_AI" 
    )
    process {
        $SDX_OLIVE_ENV_Path = "$rootDirPath\$OLIVE_ENV_Path"
        $global:DIR_PATH      = $rootDirPath
        cd "$DIR_PATH\$Olive_Folder_path"
        & "$SDX_OLIVE_ENV_Path\Scripts\Activate.ps1"
    }  
}


