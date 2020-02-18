# Jupyter notebook tutorials

Setup:

Cheatsheet
Jupyter Notebook Getting started
Questions? mark.schutera@kit.edu / mark.schutera@mailbox.org


1. You need to install anaconda and then start the anaconda prompt


2. Create a virtual env
 on the base of tensorflow
	>conda create -n tensorflow_env tensorflow



3. Activate the virtual env

	>conda activate tensorflow_env


   From the anaconda prompt just do
	>activate tensorflow_env

   Show all accessible environments
	>conda info --envs


4. Install ipykernel

	>pip install ipykernel



5. create a kernel for tensorflow:

	>python -m ipykernel install --user --name=tensorflow



6. Open juypter notebook in browser
	>jupyter notebook

now when you are in the notebook, select Kernel->Change kernel->tensorflow


