sh updateSrc.sh
echo "finishing updateSrc!"
echo "transferring code... "; echo; echo;

scp -r model/ mainprog* $asus:/media/anhvt89/seagateRepo/spparks-18May17/examples/potts_pfm/testLarge2D_AsusTower_ParamsFitFPE_17Mar18/fp1d
echo "done"

