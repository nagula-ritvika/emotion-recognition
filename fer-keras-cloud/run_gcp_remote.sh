export BUCKET_NAME=fer-cs6140-data
export JOB_NAME="fer_basic_cnn_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --runtime-version 1.10 \
  --module-name trainer.task \
  --package-path ./trainer \
  --region $REGION \
  --python-version 3.5 \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --train-file gs://$BUCKET_NAME/fer2013.csv