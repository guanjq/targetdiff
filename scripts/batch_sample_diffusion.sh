TOTAL_TASKS=100
BATCH_SIZE=50

if [ $# != 5 ]; then
    echo "Error: 5 arguments required."
    exit 1
fi

CONFIG_FILE=$1
RESULT_PATH=$2
NODE_ALL=$3
NODE_THIS=$4
START_IDX=$5

for ((i=$START_IDX;i<$TOTAL_TASKS;i++)); do
    NODE_TARGET=$(($i % $NODE_ALL))
    if [ $NODE_TARGET == $NODE_THIS ]; then
        echo "Task ${i} assigned to this worker (${NODE_THIS})"
        python -m scripts.sample_diffusion ${CONFIG_FILE} -i ${i} --batch_size ${BATCH_SIZE} --result_path ${RESULT_PATH}
    fi
done
