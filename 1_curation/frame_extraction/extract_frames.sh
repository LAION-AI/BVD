#!/bin/bash
# set -x

# ------- User editable paths ------------------------
PROJECT_ROOT=""
SOURCE=""
TARGET=""
# -------

NUM_CPUS=16
NUM_WORKERS=32
START_ID=$1
END_ID=$2

# FILES="$SOURCE/*.transferred"
FILES=$(ls $SOURCE/*.transferred | sed 's/\.transferred$//' | sort)

FILES_ARRAY=($FILES)
for ((i=START_ID-1; i<END_ID; i++)); do
    FILE=${FILES_ARRAY[i]}

    # Exit if FILE is empty or unset
    if [[ -z "$FILE" ]]; then
        echo "No more files at index $i — exiting loop."
        break
    fi

    # Get the base name of the file (without the directory path)
    PARQUET=$(basename "$FILE")
    sbatch --dependency=singleton << EOT
#!/bin/bash
#SBATCH --job-name=extract-$PARQUET   # Job name
#SBATCH --output=$TARGET-logs/$PARQUET.log           # Standard output file
#SBATCH --error=$TARGET-logs/$PARQUET.error          # Standard error file
#SBATCH --partition=<partition>    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --mem-per-cpu=<mem>
#SBATCH --cpus-per-task=$NUM_CPUS     # Number of CPU cores per task
#SBATCH --time=24:00:00               # Maximum runtime (D-HH:MM:SS)
#SBATCH --account=<account>       # Slurm account


source ~/.bashrc
cd $PROJECT_ROOT/1_curation/frame_extraction
source .venv/bin/activate

python extract_frames.py \
    --source $SOURCE/ \
    --parquet_name $PARQUET \
    --target $TARGET/ \
    --workers $NUM_WORKERS \
    --blackdetect 'd=0.1:pix_th=0.1' \
    --scene_threshold 0.1
EOT
    done
