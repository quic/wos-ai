# =============================================================================
#
# Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
$modelUrl =  "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/mobilenet_v2.onnx"

# URL for downloading the Visual Studio Redistributable for ARM64. Visual studio is used during model exection on HTP(NPU) backend.
$vsRedistributableUrl = "https://aka.ms/vs/17/release/vc_redist.arm64.exe"

<# Required files 
    - License             : License document
#>
$licenseUrl        = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/LICENSE"

<#  Artifacts for tutorials, including:
    - io_utils.py         : Utility file for preprocessing images and postprocessing to get top 5 predictions.
#>
$io_utilsUrl       = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Artifacts/io_utils.py"


############################ python installation path ##################################
# Retrieves the value of the Username
$username =  (Get-ChildItem Env:\Username).value

$pythonInstallPath = "C:\Users\$username\AppData\Local\Programs\Python\Python312"
$pythonScriptsPath = $pythonInstallPath+"\Scripts"


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

$ORT_CPU_ENV_Path = "Python_Venv\SDX_ORT_CPU_ENV"
$ORT_QNN_ENV_Path = "Python_Venv\SDX_ORT_QNN_ENV"
$ORT_HF_ENV_Path  = "Python_Venv\SDX_ORT_HF_ENV"

$Mobilenet_Folder_path = "Models\Mobilenet_V2"

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

    # Define the license download path.
    $global:lincensePath      = "$rootDirPath\License"

    $global:debugFolder    = "$rootDirPath\Debug_Logs"
    # Create the Root folder if it doesn't exist.
    if (-Not (Test-Path $debugFolder)) {
        New-Item -ItemType Directory -Path $debugFolder
    }
    # Define folder path for mobilenet artifacts.
    $global:mobilenetFolder = "$rootDirPath\$Mobilenet_Folder_path"
    # Create the Root folder if it doesn't exist
    if (-Not (Test-Path $mobilenetFolder)) {
        New-Item -ItemType Directory -Path $mobilenetFolder
    }
    # Define the artifacts download path.
    $global:io_utilsPath       = "$mobilenetFolder\io_utils.py"
    # Define the mobilenet model download path.
    $global:modelFilePath      = "$mobilenetFolder\mobilenet_v2.onnx"
    
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

Function download_mobilenet_artifacts{
    param ()
    process{
        # io_utils for pre and post processing for the mobilenet 
        # Checking if io_utils.py already present
        # If yes
        if(Test-Path $io_utilsPath){
            Write-Output "io_utils.py is already downloaded at : $io_utilsPath"
        }
        # Else dowloading
        else{
            $result = download_file -url $io_utilsUrl -downloadfile $io_utilsPath
            if($result){
                Write-Output "io_utils.py is downloaded at : $io_utilsPath"
            }
            else{
                Write-Output "io_utils.py download failed. Download from $io_utilsUrl"
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

Function download_onnxmodel {
    param()
    process {
        # Download Model file 
        # Checking if mobilenet.onnx already present 
        # If yes
        if (Test-Path $modelFilePath) {
            Write-Output "ONNX File already present at : $modelFilePath" # -ForegroundColor Green
        }
        # Else downloading
        else {
            Write-Output "Downloading the onnx model ..." 
            $result = download_file -url $modelUrl -downloadfile $modelFilePath
            # Checking for successful download
            if ($result) {
                Write-Output "Onnx File is downloaded at : $modelFilePath" 
            } 
            else{
                Write-Output "Onnx download failed. Download the onnx file from :  $modelUrl" 
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

Function mobilenet_artifacts{
    param ()
    process {
        download_onnxmodel
        download_mobilenet_artifacts
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


Function ORT_CPU_Setup {
    param(
        [string]$rootDirPath = "C:\WoS_AI"
        )
    process {
    	# Set the permission on PowerShell to execute the command. If prompted, accept and enter the desired input to provide execution permission.
	Set-ExecutionPolicy RemoteSigned 
        Set_Variables -rootDirPath $rootDirPath
        download_install_python
        Show-Progress -percentComplete 1 4
        download_install_redistributable
        Show-Progress -percentComplete 2 4
        download_script_license
        mobilenet_artifacts
        Show-Progress -percentComplete 3 4
        $SDX_ORT_CPU_ENV_Path = "$rootDirPath\$ORT_CPU_ENV_Path"
        # Check if virtual environment was created
        if (-Not (Test-Path -Path  $SDX_ORT_CPU_ENV_Path))
        {
           py -3.12 -m venv $SDX_ORT_CPU_ENV_Path
        }
        # Check if the virtual environment was created successfully
        if (Test-Path "$SDX_ORT_CPU_ENV_Path\Scripts\Activate.ps1") {
            # Activate the virtual environment
            & "$SDX_ORT_CPU_ENV_Path\Scripts\Activate.ps1"
            python -m pip install --upgrade pip
            pip install onnxruntime
            pip install pillow
	    pip install requests
        }
        Show-Progress -percentComplete 4 4
        Write-Output "***** Installation for ORT-CPU *****"
        Check_Setup -logFilePath "$debugFolder\ORT_CPU_Setup_Debug.log"
        Invoke-Command { & "powershell.exe" } -NoNewScope
    }
}

Function Activate_ORT_CPU_VENV {
    param ( 
        [string]$rootDirPath = "C:\WoS_AI" 
    )
    process {
        $SDX_ORT_CPU_ENV_Path = "$rootDirPath\$ORT_CPU_ENV_Path"
        $global:DIR_PATH      = $rootDirPath
        cd "$DIR_PATH\$Mobilenet_Folder_path"
        & "$SDX_ORT_CPU_ENV_Path\Scripts\Activate.ps1"
    }  
}

Function ORT_HF_Setup {
    param(
        [string]$rootDirPath = "C:\WoS_AI"
        )
    process {
    	# Set the permission on PowerShell to execute the command. If prompted, accept and enter the desired input to provide execution permission.
	Set-ExecutionPolicy RemoteSigned 
        Set_Variables -rootDirPath $rootDirPath
        download_install_python
        Show-Progress -percentComplete 1 4
        download_install_redistributable
        Show-Progress -percentComplete 2 4
        download_script_license
        Show-Progress -percentComplete 3 4
        $SDX_ORT_HF_ENV_Path = "$rootDirPath\$ORT_HF_ENV_Path"
        # Check if virtual environment was created
        if (-Not (Test-Path -Path  $SDX_ORT_HF_ENV_Path))
        {
           py -3.12 -m venv $SDX_ORT_HF_ENV_Path
        }
        # Check if the virtual environment was created successfully
        if (Test-Path "$SDX_ORT_HF_ENV_Path\Scripts\Activate.ps1") {
            # Activate the virtual environment
            & "$SDX_ORT_HF_ENV_Path\Scripts\Activate.ps1"
            python -m pip install --upgrade pip
            pip install optimum[onnxruntime]
            pip install onnxruntime-directml
            pip install pillow
	    pip install requests
        }
        Show-Progress -percentComplete 4 4
        Write-Output "***** Installation for Hugging Face Optimum + ONNX-RT *****"
        Check_Setup -logFilePath "$debugFolder\ORT_HF_Setup_Debug.log"
        Invoke-Command { & "powershell.exe" } -NoNewScope
    }
}

Function Activate_ORT_HF_VENV {
    param ( 
        [string]$rootDirPath = "C:\WoS_AI" 
    )
    process {
        $SDX_ORT_HF_ENV_Path = "$rootDirPath\$ORT_HF_ENV_Path"
        $global:DIR_PATH     = $rootDirPath
        cd "$DIR_PATH\$Mobilenet_Folder_path"
        & "$SDX_ORT_HF_ENV_Path\Scripts\Activate.ps1"
    }  
}

Function ORT_QNN_Setup {
    param(
        [string]$rootDirPath = "C:\WoS_AI"
        )
    process {
    	# Set the permission on PowerShell to execute the command. If prompted, accept and enter the desired input to provide execution permission.
     	Set-ExecutionPolicy RemoteSigned 
        Set_Variables -rootDirPath $rootDirPath
        download_install_python
        Show-Progress -percentComplete 1 4
        download_install_redistributable
        Show-Progress -percentComplete 2 4
        download_script_license
        mobilenet_artifacts
        Show-Progress -percentComplete 3 4
        $SDX_ORT_QNN_ENV_Path = "$rootDirPath\$ORT_QNN_ENV_Path"
        # Check if virtual environment was created
        if (-Not (Test-Path -Path  $SDX_ORT_QNN_ENV_Path))
        {
           py -3.12 -m venv $SDX_ORT_QNN_ENV_Path
        }
        # Check if the virtual environment was created successfully
        if (Test-Path "$SDX_ORT_QNN_ENV_Path\Scripts\Activate.ps1") {
            # Activate the virtual environment
            & "$SDX_ORT_QNN_ENV_Path\Scripts\Activate.ps1"
            python -m pip install --upgrade pip
            pip install onnxruntime-qnn
            pip install pillow
	    pip install requests
        }
        Show-Progress -percentComplete 4 4
        Write-Output "***** Installation for ONNX-QNN *****"
        Check_Setup -logFilePath "$debugFolder\ORT_QNN_Setup_Debug.log"
        Invoke-Command { & "powershell.exe" } -NoNewScope
    }
}

Function Activate_ORT_QNN_VENV {
    param ( 
        [string]$rootDirPath = "C:\WoS_AI" 
    )
    process {
        $SDX_ORT_QNN_ENV_Path = "$rootDirPath\$ORT_QNN_ENV_Path"
        $global:DIR_PATH      = $rootDirPath
        cd "$DIR_PATH\$Mobilenet_Folder_path"
        & "$SDX_ORT_QNN_ENV_Path\Scripts\Activate.ps1"
    }  
}


