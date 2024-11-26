#!/bin/bash

workdir=../..
cd $workdir

scratchdir=/Odyssey/private/m19beauc/DMI/results/lightning_logs
#scratchdir=$SCRATCH/lightning_logs

ckpt_path=/homes/m19beauc/4dvarnet-starter/ckpt/DMI/SST/sst_dmi_baltic_wcoarse_wgeo.ckpt
ckpt_path=/homes/m19beauc/4dvarnet-starter/ckpt/DMI/SST/sst_dmi_baltic_dm3_wcoarse_wgeo.ckpt

dm=baltic_ext
dm=baltic_dm3_eval

xp_name=dt7_linweights_wcoarse

list_outputs=()
# run 4DVarNet sequentially for the 12 months
for month in {1..12} ; do
  echo "Run 4DVarNet on month"${month}
  firstdayofmonth=2021-${month}-01
  start=$(date +%Y-%m-%d -d "$firstdayofmonth - 8 day")
  stop=$(date +%Y-%m-%d -d "$firstdayofmonth + 39 day")
  cp -rf config/xp/DMI/SST/sed_dmi_sst_test_wcoarse_wgeo.yaml config/xp/DMI/SST/dmi_sst_test.yaml
  sed -i -e 's|__START__|'${start}'|g' config/xp/DMI/SST/dmi_sst_test.yaml
  sed -i -e 's|__STOP__|'${stop}'|g' config/xp/DMI/SST/dmi_sst_test.yaml
  sed -i -e 's|__CKPT_PATH__|'${ckpt_path}'|g' config/xp/DMI/SST/dmi_sst_test.yaml
  sed -i -e 's|__DOMAIN__|'${dm}'|g' config/xp/DMI/SST/dmi_sst_test.yaml
  #CUDA_VISIBLE_DEVICES=5 HYDRA_FULL_ERROR=1 python main.py xp=DMI/SST/dmi_sst_test
  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python main.py xp=DMI/SST/dmi_sst_test
  N=`ls -Art ${scratchdir} | tail -n 1 | cut -f2 -d'_'`
  mm1=$((month-1))
  mp1=$((month+1))
  if [ ${mm1} == 0 ]; then
    mm1=12
  fi
  if [ ${mp1} == 13 ]; then
    mp1=1
  fi
  # extract only relevant days from NetCDF files
  cdo delete,month=$mm1 ${scratchdir}/version_${N}/test_data.nc ${scratchdir}/version_${N}/test_data_${month}_tmp.nc
  cdo delete,month=$mp1 ${scratchdir}/version_${N}/test_data_${month}_tmp.nc ${scratchdir}/version_${N}/test_data_${month}.nc
  rm -rf ${scratchdir}/version_${N}/test_data_${month}_tmp.nc
  # append to the list of NetCDF files to merge
  list_outputs+=("${scratchdir}/version_${N}/test_data_${month}.nc")
  rm -rf config/xp/DMI/SST/dmi_sst_test.yaml
done

echo ${list_outputs[*]}
# merge the NetCDF files
ncrcat ${list_outputs[*]} -O ${scratchdir}/DMI-L4_GHRSST-SSTfnd-DMI_4DVarNet-NSEABALTIC_2021_${xp_name}_${dm}.nc
mv -f ${scratchdir}/DMI-L4_GHRSST-SSTfnd-DMI_4DVarNet-NSEABALTIC_2021_${xp_name}_${dm}.nc /Odyssey/private/m19beauc/DMI/4DVarNet_outputs

