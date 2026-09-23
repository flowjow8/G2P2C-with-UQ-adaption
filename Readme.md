G2P2C: Reinforcement Learning based Artificial Pancreas Systems adapted with Uncertainty Quantification.

This repository contains the code used for my Bachelor thesis investigating uncertainty-aware reinforcement learning for automated insulin delivery in Type 1 Diabetes. 

The implementation builds upon the G2P2C framework originally implemented by Hettiarachchi et al. The original PPO controller and experimental pipeline has been adapted to incorporate Monte Carlo Dropout Uncertainty Quantification and uncertainty-based controller handover. 

The main modifications include:
- Monte Carlo Dropout added to the PPO actor network for uncertainty estimation. 
- Action uncertainty estimated as the standard deviation of the actor mean across repeated Monte Carlo forward passes.  
- Uncertainty-based controller handover from PPO controller to basal-bolus controller, once the uncertainty threshold is exceeded. 
- Additional parameters for configuring uncertainty. 
- Logging of uncertainty values and controller handover events during testing. 

The UQ hyperparameters used for my thesis were:
- MC dropout probability = 0.3 
- Number of Monte Carlo samples = 20 
- UQ action threshold = 0.075 

This repository is based on:
@article{hettiarachchi2024g2p2c,
  title={G2P2C—A modular reinforcement learning algorithm for glucose control by glucose prediction and planning in Type 1 Diabetes},
  author={Hettiarachchi, Chirath and Malagutti, Nicolo and Nolan, Christopher J. and Suominen, Hanna and Daskalaki, Elena},
  journal={Biomedical Signal Processing and Control},
  volume={90},
  pages={105839},
  year={2024},
  publisher={Elsevier}
}