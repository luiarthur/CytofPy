.PHONY: getData

RESULTS_DIR = results/test/

getData:
	aws s3 cp s3://cytof-cb-data/cb.txt data/cb.txt

run:
	mkdir -p $(RESULTS_DIR)
	python3 cb.py $(RESULTS_DIR) > $(RESULTS_DIR)/log.txt
