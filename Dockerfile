FROM rocm/pytorch:rocm5.0.1_ubuntu18.04_py3.7_pytorch_staging

RUN apt-get install -y  python3-tk 	# Used for matplotlib to display plots on host machine
