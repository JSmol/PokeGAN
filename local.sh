IMAGE_URI=torch-gpu:pokemon
docker build -t ${IMAGE_URI} .

docker run \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/keys/gcp.json:ro \
    -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gcp.json \
    ${IMAGE_URI} \
    --epochs=1 \
    --batch-size=801 \
    --learning-rate=0.002 \
    --beta1=0.5 \
    --beta2=0.9 \
    --weight-decay=0 \
