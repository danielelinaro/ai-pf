
@echo off

rem python run_PF.py AC -v 2 -o V2020_Rete_Sardegna_2021_06_03cr_AC_EqX_MIMC_I2201TR1_-2.5_1.0_20_dP=0.1.npz .\config\Sardinia_AC_config.json > AC.log
python run_PF.py AC -v 2 -o V2020_Rete_Sardegna_2021_06_03cr_AC_EqX_MIMC_I2201TR1_SULCTI0202GGR2_OFF_-2.5_1.0_20_dP=0.1.npz .\config\Sardinia_AC_config_SULCTI0202GGR2_OFF.json > AC_SULCTI0202GGR2_OFF.log
python run_PF.py AC -v 2 -o V2020_Rete_Sardegna_2021_06_03cr_AC_EqX_MIMC_I2201TR1_FSACTI0201GGR3_OFF_-2.5_1.0_20_dP=0.1.npz .\config\Sardinia_AC_config_FSACTI0201GGR3_OFF.json > AC_FSACTI0201GGR3_OFF.log
