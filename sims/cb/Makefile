.PHONY: getData

getData:
	aws s3 cp s3://cytof-cb-data/cb.txt data/cb.txt

run:
	rm -f results/log.txt
	python3 cb.py >> results/log.txt
