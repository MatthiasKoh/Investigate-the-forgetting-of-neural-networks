# Continual Learning with CIFAR10 and in house AnimalParts Dataset
### Editted and prepared by Matthias Koh

This is a PyTorch implementation of the continual learning experiments that closely follows https://github.com/GMvandeVen/continual-learning 
and takes reference to experients described in the following papers:
* Three scenarios for continual learning ([link](https://arxiv.org/abs/1904.07734))
* Generative replay with feedback connections as a general strategy 
for continual learning ([link](https://arxiv.org/abs/1809.10635))

In my implementation, we focus on the differences in results of 3 Class Incremental learning types; Naive Fine-Tuning, Elastic Weight Consideration(EWC) and Incremental Classifier and Representative Learning and investigate the forgetfullness of neural networks.

## NOTE WHEN USING THE CODE:
* Do run the two lines below as this is how GMvandeVen has implemented it else the code will not run properly, the visdom server may not work on google collab so after it installs the neccessary packages you may to stop the chunk from running

###################

!pip install visdom

!python3 -m visdom.server

###################

## Running the experiments
Individual experiments can be run with `main.py`. Main options are:
- `--experiment`: which task protocol? (`splitMNIST`|`permMNIST`|`CIFAR10`|`ANIMALPART`|`ABLATEDHEAD`|`ABLATEDTORSO`|`ABLATEDTAIL`|`ALLANIMALPART`)
- `--scenario`: according to which scenario? (`task`|`domain`|`class`)
- `--tasks`: how many tasks?
- `--runs`: how many runs do you want i.e. how many different shuffle of classes eg. [1,2,3],[3,2,1],[2,3,1]....

To run specific methods, you can use the following:
- Context-dependent-Gating (XdG): `./main.py --xdg=0.8`
- Elastic Weight Consolidation (EWC): `./main.py --ewc --lambda=5000`
- Online EWC:  `./main.py --ewc --online --lambda=5000 --gamma=1`
- Synaptic Intelligence (SI): `./main.py --si --c=0.1`
- Learning without Forgetting (LwF): `./main.py --replay=current --distill`
- Generative Replay (GR): `./main.py --replay=generative`
- GR with distillation: `./main.py --replay=generative --distill`
- Replay-trough-Feedback (RtF): `./main.py --replay=generative --distill --feedback`
- Experience Replay (ER): `./main.py --replay=exemplars --budget=2000`
- Averaged Gradient Episodic Memory (A-GEM): `./main.py --replay=exemplars --agem --budget=2000`
- iCaRL: `./main.py --icarl --budget=2000`

To run the two baselines (see the papers for details):
- None: `./main.py`
- Offline: `./main.py --replay=offline`

For information on further options: `./main.py -h`.

Apart from the original MNIST, this code has been editted to support CIFAR10 and also the adding of in-house datasets such as ANIMALPARTS dataset.

The ANINALPARTS dataset has been split into a few parts and hence the different experiment names:
1. `ANIMALPART` - to perform class incremental learning, 2 classes for each of the task, on the original animal pictures without mask
2. `ABLATEDHEAD` - to perform class incremental learning on the animal pictures with applied mask that occludes the HEADS of the animals, replacing them with a white pixel background, it is then also tested on the original animal pictures without mask apart from the testset from `ABLATEDHEAD` dataset
3. `ABLATEDTORSO` - to perform class incremental learning on the animal pictures with applied mask that occludes the TORSO of the animals, replacing them with a white pixel background, it is then also tested on the original animal pictures without mask apart from the testset from `ABLATEDTORSO` dataset
4. `ABLATEDTAIL` - to perform class incremental learning on the animal pictures with applied mask that occludes the TAIL of the animals, replacing them with a white pixel background, it is then also tested on the original animal pictures without mask apart from the testset from `ABLATEDTAIL` dataset
5. `ALLANIMALPART` - to perform incremental learning on all the ablated datasets;`ABLATEDHEAD`, `ABLATEDTORSO` and `ABLATEDTAIL`
* Do note that the datasets need to be downloaded locally, implementation may need to be adjusted for each testing purpose. The main 4 files to edit is `main.py`, `data.py`, `train.py` and `evalute.py`.

## Running comparisons from the papers
#### "Three CL scenarios"-paper
[This paper](https://arxiv.org/abs/1904.07734) describes three scenarios for continual learning (Task-IL, Domain-IL &
Class-IL) and provides an extensive comparion of recently proposed continual learning methods. It uses the permuted and
split MNIST task protocols, with both performed according to all three scenarios.

A comparison of all methods included in this paper can be run with `compare_all.py` (this script includes extra
methods and reports additional metrics compared to the paper). The comparison in Appendix B can be run with
`compare_taskID.py`, and Figure C.1 can be recreated with `compare_replay.py`.

#### "Replay-through-Feedback"-paper
The three continual learning scenarios were actually first identified in [this paper](https://arxiv.org/abs/1809.10635),
after which this paper introduces the Replay-through-Feedback framework as a more efficent implementation of generative
replay. 

A comparison of all methods included in this paper can be run with
`compare_time.py`. This includes a comparison of the time these methods take to train (Figures 4 and 5).

Note that the results reported in this paper were obtained with
[this earlier version](https://github.com/GMvandeVen/continual-learning/tree/9c0ca78f43c29594b376ca59516031fcdaa5d7ba)
of the code. 


## On-the-fly plots during training
With this code it is possible to track progress during training with on-the-fly plots. This feature requires `visdom`, 
which can be installed as follows:
```bash
pip install visdom
```
Before running the experiments, the visdom server should be started from the command line:
```bash
python -m visdom.server
```
The visdom server is now alive and can be accessed at `http://localhost:8097` in your browser (the plots will appear
there). The flag `--visdom` should then be added when calling `./main.py` to run the experiments with on-the-fly plots.

For more information on `visdom` see <https://github.com/facebookresearch/visdom>.


### Citation
Please consider citing the original codes papers if you use this code in your research:
```
@article{vandeven2019three,
  title={Three scenarios for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1904.07734},
  year={2019}
}

@article{vandeven2018generative,
  title={Generative replay with feedback connections as a general strategy for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1809.10635},
  year={2018}
}
```

### Acknowledgments
The research projects from which this code originated have been supported by an IBRO-ISN Research Fellowship, by the 
Lifelong Learning Machines (L2M) program of the Defence Advanced Research Projects Agency (DARPA) via contract number 
HR0011-18-2-0025 and by the Intelligence Advanced Research Projects Activity (IARPA) via Department of 
Interior/Interior Business Center (DoI/IBC) contract number D16PC00003. Disclaimer: views and conclusions 
contained herein are those of the authors and should not be interpreted as necessarily representing the official
policies or endorsements, either expressed or implied, of DARPA, IARPA, DoI/IBC, or the U.S. Government.
