$DATASET='cifar100'  #cub flower imagenet_sub
$GPU=0
$NET='resnet32' # resnet32
$LR=1e-6 
$EPOCH=51 
$SAVE=50
$LOSS='triplet_no_hard_mining' #triplet
$TASK=11 #6
$BASE=50 #17
$SEED=1

$EM='_EM' # _EM

$Method='Finetuning'
$Tradeoff=0

$NAME="${Method}_${Tradeoff}_${DATASET}_${LOSS}_${NET}_${LR}_${EPOCH}epochs_task${TASK}_base${BASE}_seed${SEED}${EM}"
Write-Verbose $NAME

#python train.py -base ${BASE} -seed ${SEED} -task ${TASK} -data ${DATASET} -tradeoff ${Tradeoff} -exp ${Tradeoff} -net ${NET} -method ${Method} -lr ${LR} -dim 512  -num_instances 8 -BatchSize 32 -loss ${LOSS} -epochs ${EPOCH} -log_dir ${DATASET}_seed${SEED}/${NAME} -save_step ${SAVE} -gpu ${GPU} --em

python test.py -seed ${SEED} -base ${BASE} -task ${TASK} -epochs ${EPOCH} -data ${DATASET} -gpu ${GPU} -method ${Method} -r checkpoints\${DATASET}_seed${SEED}\${NAME} | Out-File -FilePath .\results\${DATASET}\${NAME}_old_mean.txt 

python test.py -seed ${SEED} -base ${BASE} -task ${TASK} -epochs ${EPOCH} -data ${DATASET} -gpu ${GPU} -method ${Method} -r checkpoints\${DATASET}_seed${SEED}\${NAME} -mapping_test -sigma_test 0.20 | Out-File -FilePath .\results\${DATASET}\${NAME}_SDC_sigma_test${SIGMA_TEST}.txt 