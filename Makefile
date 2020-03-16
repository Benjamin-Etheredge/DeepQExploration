.PHONY: build space

NS ?= betheredge

IMAGE_NAME ?= dev
CONTAINER_NAME ?= dev
DIR := ${CURDIR}


build: Dockerfile
	docker build -t $(NS)/$(IMAGE_NAME) -f Dockerfile .

bash: build
	docker run --rm -v $(DIR):/app -v $(DIR)/logs:/logs $(IMAGE_NAME) bash

brick: build
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_Breakout

space:
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_SpaceInvaders_v4