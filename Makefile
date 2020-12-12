.PHONY: build space

NS ?= betheredge

IMAGE_NAME ?= gym_dev
CONTAINER_NAME ?= gym_dev
FULL_NAME := $(NS)/$(IMAGE_NAME)
DIR := ${CURDIR}
DOCKER := docker
#$(MLFLOW_TRACKING_URI) ?= ${MLFLOW_TRACKING_URI}
MLFLOW_EXPERIMENT_NAME=deep-q
#DOCKER_ENV_VARS := -e MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} -e TF_CPP_MIN_LOG_LEVEL=2
DOCKER_ENV_VARS := -e MLFLOW_ARTIFACT_URI=${MLFLOW_TRACKING_URI} \
				   -e MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) \
				   -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
				   -e TF_CPP_MIN_LOG_LEVEL=0
#$(DOCKER_ENV_VARS) := -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} -e TF_CPP_MIN_LOG_LEVEL=3
#$(DOCKER_ENV_VARS) = -e TF_CPP_MIN_LOG_LEVEL=3
DOCKER_RUN_ARGS := --rm --gpus all -v $(DIR):/app -v $(DIR)/logs:/logs -v /artifacts:/artifacts $(DOCKER_ENV_VARS) $(FULL_NAME)
CONTAINER_CMD := python src/main.py
DOCKER_RUN_CMD := $(DOCKER) run $(DOCKER_RUN_ARGS)

build: Dockerfile
	docker build -t $(NS)/$(IMAGE_NAME) -f Dockerfile .

main_double_duel: build
	docker run --rm --gpus all -v $(DIR):/app -v $(DIR)/logs:/logs $(NS)/$(IMAGE_NAME) python src/main.py \
		--environment SpaceInvaders-v4 \
		--frame_skip 3 \
		--learner_type double_duel \
		--name_prefix double_duel

main: build
	docker run --rm --gpus all -v $(DIR):/app -v $(DIR)/logs:/logs $(NS)/$(IMAGE_NAME) python src/main.py

pong: build
	docker run --rm --gpus all -v $(DIR):/app -v $(DIR)/logs:/logs $(NS)/$(IMAGE_NAME) python src/main.py --environment "Pong-v4"

breakout_no_prio: build
	$(DOCKER_RUN_CMD) $(CONTAINER_CMD) \
									   --environment "Breakout-v4" \
									   --name_prefix No_prio \
									   --duel \
									   --double \
									   --clip_reward 


breakout_vanilla_noclipreward: build
	$(DOCKER_RUN_CMD) $(CONTAINER_CMD) \
									   --environment "Breakout-v4" \
									   --name_prefix noClipReward



breakout_all: build
	$(DOCKER_RUN_CMD) $(CONTAINER_CMD) \
									   --environment "Breakout-v4" \
									   --name_prefix All_ \
									   --duel \
									   --double \
									   --clip_reward \
									   --prio


breakout_double: build
	$(DOCKER_RUN_CMD) $(CONTAINER_CMD) --frame_skip 4 \
									   --environment "Breakout-v4" \
									   --name_prefix Double_
									   --double

breakout: build
	$(DOCKER_RUN_CMD) $(CONTAINER_CMD) --frame_skip 4 \
									   --environment "Breakout-v4" \
									   --name_prefix Vanilla_

space: build
	$(DOCKER_RUN_CMD) $(CONTAINER_CMD) --frame_skip 3 --environment "SpaceInvaders-v4" --name_prefix prio

main_cpu: build
	docker run --rm -e CUDA_VISIBLE_DEVICES='-1' -v $(DIR):/app -v $(DIR)/logs:/logs $(NS)/$(IMAGE_NAME) python src/main.py

bash: build
	docker run --rm -v $(DIR):/app -v $(DIR)/logs:/logs $(IMAGE_NAME) bash

test_utils: build
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python src/utils/test_utils.py

brick: build
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m unittest test_agent.TestAgent.test_Breakout

test_log: build
	$(DOCKER_RUN_CMD) $(CONTAINER_CMD) \
			--end_length      11000 \
			--start_length    10000 \
			--random_decay_end 1000 \
			--max_steps       20000 \
			--random_choice_min_rate 0 \
			--name_prefix Test_Log \
			--verbose 1 \
			--environment SpaceInvaders-v4
	#docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m cProfile -o breakout.prof test_agent.py


test_quick: build
	$(DOCKER_RUN_CMD) $(CONTAINER_CMD) \
			--end_length      11000 \
			--start_length    10000 \
			--random_decay_end 1000 \
			--max_steps       20000 \
			--random_choice_min_rate 0 \
			--name_prefix Short_ \
			--verbose 0 \
			--environment SpaceInvaders-v4
	#docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m cProfile -o breakout.prof test_agent.py


profile: build
	docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m cProfile -o profile.prof src/main.py \
			--start_length  1000000 \
			--end_length    1000000 \
			--random_decay_end  10000 \
			--max_steps        200000 \
			--random_choice_min_rate 0 \
			--verbose 1 \
			--name_prefix Profile \
			--environment SpaceInvaders-v4
	#docker run --rm --gpus all -v $(DIR):/app $(NS)/$(IMAGE_NAME) python -m cProfile -o breakout.prof test_agent.py

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
