SET CONDAPATH=C:\conda
SET IPYNBPATH=C:\Users\Public\Documents\jupyter

cd /d %IPYNBPATH%
SET PATH=%PATH%;%CONDAPATH%;%CONDAPATH%\Scripts;%CONDAPATH%\Library\bin;
call %CONDAPATH%\Scripts\activate.bat %CONDAPATH%
jupyter lab --notebook-dir=%IPYNBPATH%

