.PHONY: build space

NS ?= betheredge

IMAGE_NAME ?= dev
CONTAINER_NAME ?= dev


build: Dockerfile
	docker build -t $(NS)/$(IMAGE_NAME) -f Dockerfile .
	#docker run --rm -v $(pwd):/app -v $(pwd)/logs:/logs $(IMAGE_NAME) python test_agent.py TestAgent.test_SpaceInvaders_v2
	#docker run --rm -v $(pwd):/app -v $(pwd)/logs:/logs $(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_SpaceInvaders_v2

space:
	docker-compose run --rm agent python -m unittest test_agent.TestAgent.test_SpaceInvaders_v0
	#docker run --rm -v $(pwd):/app -v $(pwd)/logs:/logs $(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_SpaceInvaders_v0
