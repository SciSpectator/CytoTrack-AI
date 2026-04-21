; Inno Setup installer script for CytoTrack AI
; Build the portable bundle first:  build_windows_exe.bat
; Then compile this script with Inno Setup 6:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss

#define AppName      "CytoTrack AI"
#define AppVersion   "1.0"
#define AppPublisher "CytoTrack"
#define AppExe       "CytoTrackAI.exe"
#define SourceDir    "..\dist\CytoTrackAI"

[Setup]
AppId={{3F8B3A2E-6D53-4C21-9A8D-9BE8D1C0E2A7}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://cytotrack.local
DefaultDirName={autopf}\CytoTrack AI
DefaultGroupName=CytoTrack AI
DisableProgramGroupPage=yes
OutputBaseFilename=CytoTrackAI-{#AppVersion}-setup
OutputDir=..\installer_output
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: checkedonce
Name: "startmenuicon"; Description: "Create a &Start Menu shortcut"; GroupDescription: "Additional icons:"; Flags: checkedonce

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\CytoTrack AI"; Filename: "{app}\{#AppExe}"; Tasks: startmenuicon
Name: "{group}\Uninstall CytoTrack AI"; Filename: "{uninstallexe}"
Name: "{commondesktop}\CytoTrack AI"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon; IconFilename: "{app}\{#AppExe}"

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch CytoTrack AI"; Flags: nowait postinstall skipifsilent
