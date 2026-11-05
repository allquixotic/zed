param (
    [Parameter(Mandatory = $true)]
    [string]$filePath
)

$ErrorActionPreference = 'Stop'
$params = @{}

$endpoint = $ENV:ENDPOINT
if ([string]::IsNullOrWhiteSpace($endpoint)) {
    throw "The 'ENDPOINT' env is required."
}
$params["Endpoint"] = $endpoint

$trustedSigningAccountName = $ENV:ACCOUNT_NAME
if ([string]::IsNullOrWhiteSpace($trustedSigningAccountName)) {
    throw "The 'ACCOUNT_NAME' env is required."
}
$params["CodeSigningAccountName"] = $trustedSigningAccountName

$certificateProfileName = $ENV:CERT_PROFILE_NAME
if ([string]::IsNullOrWhiteSpace($certificateProfileName)) {
    throw "The 'CERT_PROFILE_NAME' env is required."
}
$params["CertificateProfileName"] = $certificateProfileName

$fileDigest = $ENV:FILE_DIGEST
if ([string]::IsNullOrWhiteSpace($fileDigest)) {
    throw "The 'FILE_DIGEST' env is required."
}
$params["FileDigest"] = $fileDigest

$timeStampDigest = $ENV:TIMESTAMP_DIGEST
if ([string]::IsNullOrWhiteSpace($timeStampDigest)) {
    throw "The 'TIMESTAMP_DIGEST' env is required."
}
$params["TimestampDigest"] = $timeStampDigest

$timeStampServer = $ENV:TIMESTAMP_SERVER
if ([string]::IsNullOrWhiteSpace($timeStampServer)) {
    throw "The 'TIMESTAMP_SERVER' env is required."
}
$params["TimestampRfc3161"] = $timeStampServer

$params["Files"] = $filePath

if ($env:ZED_SIGNING_AUTH -eq 'azure-cli') {
    Import-Module TrustedSigning -RequiredVersion 0.5.8 -ErrorAction Stop
    foreach ($credential in @('Environment', 'WorkloadIdentity', 'ManagedIdentity', 'SharedTokenCache', 'VisualStudio', 'VisualStudioCode', 'AzurePowerShell', 'AzureDeveloperCli', 'InteractiveBrowser')) {
        $params["Exclude${credential}Credential"] = $true
    }
    $params['ExcludeAzureCliCredential'] = $false
}

$trace = $ENV:TRACE
if (-Not [string]::IsNullOrWhiteSpace($trace)) {
    if ([System.Convert]::ToBoolean($trace)) {
        Set-PSDebug -Trace 2
    }
}

Invoke-TrustedSigning @params

if ($env:ZED_WINDOWS_PUBLISHER) {
    foreach ($file in ($filePath -split ',')) {
        $signature = Get-AuthenticodeSignature -LiteralPath $file
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -ne $env:ZED_WINDOWS_PUBLISHER -or $null -eq $signature.TimeStamperCertificate) {
            throw "Missing, invalid, untimestamped or unexpected signature: $file ($($signature.Status))"
        }
        $record = [ordered]@{
            file = [IO.Path]::GetFileName($file)
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant()
            publisher = $signature.SignerCertificate.Subject
            certificate_sha1 = $signature.SignerCertificate.Thumbprint
            timestamp_publisher = $signature.TimeStamperCertificate.Subject
        }
        if ($env:ZED_SIGNING_EVIDENCE) {
            $record | ConvertTo-Json -Compress | Add-Content -Encoding utf8 $env:ZED_SIGNING_EVIDENCE
        }
        Write-Host "Verified timestamped signature: $file"
    }
}
