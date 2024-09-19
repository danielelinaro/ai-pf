@echo off

FOR /r %%f IN (AC_configs\*.json) DO (
	python run_PF.py AC -v 2 %%f
)
