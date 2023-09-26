docker-build:
	docker build -t hs-detector:latest .

docker-build-gcp:
	docker build --platform linux/amd64 -t ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} .

docker-run:
	docker run -d -p 8501:8501 --name hs-detector hs-detector

docker-run-gcp:
	docker run -d -e PORT=8501 -p 8501:8501 --name hs-detector ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker-stop:
	docker stop hs-detector && docker rm hs-detector

docker-push-gcp:
	docker push ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

gcp-auth-docker:
	gcloud auth configure-docker

gcp-deploy:
	gcloud run deploy --port 8501 --image ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region ${GCR_REGION} --memory 768M

