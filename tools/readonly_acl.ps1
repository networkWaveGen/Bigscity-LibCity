param(
    [Parameter(Mandatory=$true)][ValidateSet('protect','restore')][string]$Action,
    [Parameter(Mandatory=$true)][string]$TargetPath
)
$ErrorActionPreference = 'Stop'
$target = Get-Item -LiteralPath $TargetPath
$marker = Join-Path $target.Parent.FullName '.runtime-readonly-sandbox'
if (-not $target.PSIsContainer -or $target.Name -notin @('source','data') -or -not (Test-Path -LiteralPath $marker)) {
    throw 'ACL changes are restricted to generated source/data snapshots in a marked sandbox.'
}
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$rights = [System.Security.AccessControl.FileSystemRights]::Write -bor
          [System.Security.AccessControl.FileSystemRights]::Delete -bor
          [System.Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles
$inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
               [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
$rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
    $identity, $rights, $inheritance,
    [System.Security.AccessControl.PropagationFlags]::None,
    [System.Security.AccessControl.AccessControlType]::Deny
)
$acl = Get-Acl -LiteralPath $target.FullName
if ($Action -eq 'protect') { $acl.AddAccessRule($rule) }
else { $acl.RemoveAccessRuleSpecific($rule) }
Set-Acl -LiteralPath $target.FullName -AclObject $acl
Write-Output "$Action completed: $($target.FullName)"
