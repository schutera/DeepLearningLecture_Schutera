I love virtual coffee: https://www.paypal.me/MarkSchutera

# Topic overview

1. Backpropagation and an Introduction to Tensorflow
3. Transfer Learning with Tensorflow for Object Classification
5. Segmentation with U-Net
8. Deep-Q Reinforcement Learning with the OpenAI gym
10. Generative Adversarial Neural Networks on MNIST
12. Recurrent Neural Networks for Language Modelling and Generation

# Jupyter notebook tutorial setup

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


