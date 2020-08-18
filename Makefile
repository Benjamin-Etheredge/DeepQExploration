.PHONY: build space

NS ?= betheredge

IMAGE_NAME ?= gym_dev
CONTAINER_NAME ?= gym_dev
DIR := ${CURDIR}


build: Dockerfile
	docker build -t $(NS)/$(IMAGE_NAME) -f Dockerfile .

main: build
	docker run --rm --gpus all -v $(DIR):/app -v $(DIR)/logs:/logs $(NS)/$(IMAGE_NAME) python src/main.py

pong: build
	docker run --rm --gpus all -v $(DIR):/app -v $(DIR)/logs:/logs $(NS)/$(IMAGE_NAME) python src/main.py --environment "Pong-v4"

main_cpu: build
	docker run --rm -e CUDA_VISIBLE_DEVICES='-1' -v $(DIR):/app -v $(DIR)/logs:/logs $(NS)/$(IMAGE_NAME) python src/main.py

bash: build
	docker run --rm -v $(DIR):/app -v $(DIR)/logs:/logs $(IMAGE_NAME) bash

brick: build
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_Breakout

profile: build
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m cProfile -o breakout.prof test_agent.py

space:
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_SpaceInvaders_v4

vanilla:
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_vanilla

vanilla:
	docker run --name vanilla --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test.test_agent.TestAgent.test_vanilla
double:
	docker run --name double --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_double
duel:
	docker run --name duel --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_duel

double_duel:
	docker run --name double_duel --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_double_duel

clipped_double_duel:
	docker run --name clipped_double_duel --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_clipped_double_duel

networks:
	make -j 3 vanilla double duel

test_env:
	docker run --rm -v $(DIR):/app $(NS)/$(IMAGE_NAME) python test_env.py

requirements:
	docker run --rm -v $(DIR):/app $(NS)/$(IMAGE_NAME) pip freeze > requirements.txt